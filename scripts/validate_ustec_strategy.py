"""Validate USTEC-only strategy with early kill switch.

Critical question: can we detect losing folds BEFORE hitting -10% halt?

Tests:
1. USTEC-only at 1.75% risk (baseline)
2. USTEC-only with early P&L circuit breaker (halt at -5% instead of -10%)
3. USTEC-only with signal density monitor (count signals in first 200 bars)
4. USTEC-only with combined: density + early PnL gate

Returns detailed per-bar equity curves for each fold to verify timing.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.pipeline import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = load_config()


def load_full_data(symbol: str):
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))
    tf = config["timeframes"]["signal"]
    data_dir = str(PROJECT_ROOT / "data" / "raw")
    df = connector.load_data(symbol, tf, data_dir)
    if df.empty:
        return pd.DataFrame(), [], None
    df = feature_eng.compute_all(df)
    label_cfg = config.get("labeling", {})
    df["target"] = triple_barrier_labels(
        df,
        upper_barrier_atr=label_cfg.get("upper_barrier_atr", 1.5),
        lower_barrier_atr=label_cfg.get("lower_barrier_atr", 1.5),
        max_holding_bars=label_cfg.get("max_holding_bars", 8),
    )
    feature_cols = feature_eng.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    df = df[df["target"] != 0]
    df["target"] = (df["target"] == 1).astype(int)
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    return df, feature_cols, h4_data


def run_fold_with_monitoring(
    df, feature_cols, h4_data, symbol, train_end, val_end, test_end, fold_num,
    strategy="baseline",
    early_halt_pct=0.05,         # For early halt strategy: stop at -5% instead of -10%
    density_check_bar=200,       # For density strategy: check at bar 200
    density_min_signals=50,      # For density strategy: need 50 signals in first 200 bars
    density_min_trades=20,       # For density strategy: need 20 trades in first 200 bars
):
    """Run a single fold with detailed monitoring.

    Strategies:
      - 'baseline': Standard 1.75% risk, normal halts
      - 'early_halt': Tighter halt at early_halt_pct instead of 10%
      - 'density_gate': Check signal density at density_check_bar, halt if below threshold
      - 'combined': density gate + early halt
    """
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]

    if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values
    X_test = test_df[feature_cols].values

    # Train fresh model
    ensemble = StackingEnsemble(config.get("model", {}))
    ensemble.fit(X_train, y_train)

    val_proba = ensemble.predict_proba(X_val)
    median_proba = float(np.median(val_proba))
    ensemble._median_proba = median_proba

    # Regime detection
    regime_scalars = np.ones(len(test_df))
    if h4_data is not None and not h4_data.empty:
        try:
            train_end_time = train_df.index[-1]
            if hasattr(train_end_time, 'tz') and train_end_time.tz is not None:
                train_end_time = train_end_time.tz_localize(None)
            h4_norm = h4_data.copy()
            if h4_norm.index.tz is not None:
                h4_norm.index = h4_norm.index.tz_localize(None)
            h4_train = h4_norm[h4_norm.index <= train_end_time]
            if len(h4_train) > 50:
                regime = RegimeDetector(n_states=3)
                regime.fit(h4_train)
                h4_scalars = regime.get_size_scalar(h4_norm)
                test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
                regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
                regime_scalars = regime_reindexed.values
        except Exception:
            pass

    # Generate signals
    test_proba = ensemble.predict_proba(X_test)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset

    # Risk parameters
    initial_balance = config.get("account", {}).get("initial_balance", 100_000)
    risk_pct = risk_cfg.get("risk_per_trade_pct", 1.75) / 100
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)
    max_daily_loss = risk_cfg.get("daily_loss_halt_pct", 4.0) / 100
    phase1_target = config.get("account", {}).get("phase1_target_pct", 10.0) / 100

    # Choose total halt based on strategy
    if strategy in ("early_halt", "combined"):
        max_total_loss = early_halt_pct
    else:
        max_total_loss = risk_cfg.get("total_drawdown_halt_pct", 9.0) / 100

    # Simulation with per-bar tracking
    balance = initial_balance
    trades = 0
    wins = 0
    signals_count = 0
    halted = False
    density_halted = False
    current_date = None
    day_start_balance = balance
    daily_balances = {}
    equity_by_bar = []  # Track equity at each bar

    for i in range(len(test_df)):
        ts = test_df.index[i]
        p = test_proba[i]
        today = ts.date() if hasattr(ts, "date") else ts

        if today != current_date:
            current_date = today
            day_start_balance = balance

        # Total DD check
        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= max_total_loss:
            halted = True

        # Daily DD check
        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= max_daily_loss:
            equity_by_bar.append({"bar": i, "equity": balance, "trades": trades, "signals": signals_count})
            continue

        equity_by_bar.append({"bar": i, "equity": balance, "trades": trades, "signals": signals_count})

        if halted or density_halted:
            continue

        # Density gate check at specified bar
        if strategy in ("density_gate", "combined") and i == density_check_bar:
            if signals_count < density_min_signals or trades < density_min_trades:
                density_halted = True
                continue

        # Signal
        if p >= buy_thresh:
            signal = 1
            signals_count += 1
        elif p <= sell_thresh:
            signal = -1
            signals_count += 1
        else:
            continue

        atr = test_df["atr"].iloc[i]
        if atr <= 0 or np.isnan(atr):
            continue

        if i + 1 >= len(test_df):
            continue
        entry_price = test_df["open"].iloc[i + 1]

        confidence = max(0.5, min(1.0,
            (p - median_proba) / (1 - median_proba) if signal == 1
            else (median_proba - p) / median_proba
        ))

        effective_risk = risk_pct * confidence * regime_scalars[i]
        risk_amount = balance * effective_risk
        sl_dist = sl_atr_mult * atr
        tp_dist = tp_atr_mult * atr

        # Simulate trade
        trade_pnl = 0.0
        max_bars = 8
        for j in range(1, max_bars + 1):
            idx = i + 1 + j
            if idx >= len(test_df):
                break
            bh = test_df["high"].iloc[idx]
            bl = test_df["low"].iloc[idx]
            if signal == 1:
                if bh >= entry_price + tp_dist:
                    trade_pnl = tp_dist; break
                if bl <= entry_price - sl_dist:
                    trade_pnl = -sl_dist; break
            else:
                if bl <= entry_price - tp_dist:
                    trade_pnl = tp_dist; break
                if bh >= entry_price + sl_dist:
                    trade_pnl = -sl_dist; break
        else:
            final_idx = min(i + 1 + max_bars, len(test_df) - 1)
            trade_pnl = (test_df["close"].iloc[final_idx] - entry_price) * signal

        trade_pnl -= atr * 0.01  # Spread cost

        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
            dollar_pnl = r_multiple * risk_amount
        else:
            dollar_pnl = 0

        balance += dollar_pnl
        trades += 1
        if dollar_pnl > 0:
            wins += 1

        if today not in daily_balances:
            daily_balances[today] = 0.0
        daily_balances[today] += dollar_pnl

    # Compile results
    total_return = (balance - initial_balance) / initial_balance
    eq = pd.DataFrame(equity_by_bar)
    max_dd = max(0, (initial_balance - eq["equity"].min()) / initial_balance) if len(eq) > 0 else 0

    worst_daily_dd = 0.0
    if daily_balances:
        running = initial_balance
        for d in sorted(daily_balances.keys()):
            dp = daily_balances[d]
            if running > 0 and dp < 0:
                worst_daily_dd = max(worst_daily_dd, abs(dp) / running)
            running += dp

    wr = wins / trades if trades > 0 else 0
    market_return = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]

    # Early detection metrics
    if len(eq) > density_check_bar:
        eq_at_check = eq.iloc[density_check_bar]
        signals_at_check = eq_at_check["signals"]
        trades_at_check = eq_at_check["trades"]
        dd_at_check = (initial_balance - eq_at_check["equity"]) / initial_balance
    else:
        signals_at_check = signals_count
        trades_at_check = trades
        dd_at_check = max_dd

    return {
        "fold": fold_num,
        "strategy": strategy,
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "total_return": total_return,
        "max_dd": max_dd,
        "max_daily_dd": worst_daily_dd,
        "win_rate": wr,
        "total_trades": trades,
        "total_signals": signals_count,
        "phase1_passed": total_return >= phase1_target,
        "halted": halted,
        "density_halted": density_halted,
        "market_return": market_return,
        # Early detection
        "signals_at_200": signals_at_check,
        "trades_at_200": trades_at_check,
        "dd_at_200": dd_at_check,
    }


def main():
    symbol = "USTEC"
    min_train_bars = 3000
    val_window_bars = 1000
    test_window_bars = 1000
    step_bars = 1000

    print(f"\n{'='*90}")
    print(f"USTEC-ONLY STRATEGY VALIDATION WITH KILL SWITCH VARIANTS")
    print(f"{'='*90}")

    df, feature_cols, h4_data = load_full_data(symbol)
    if df.empty:
        print("No data!")
        return

    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    strategies = [
        ("baseline", {}),
        ("early_halt", {"early_halt_pct": 0.05}),
        ("density_gate", {"density_check_bar": 200, "density_min_signals": 50, "density_min_trades": 15}),
        ("combined", {"early_halt_pct": 0.05, "density_check_bar": 200, "density_min_signals": 50, "density_min_trades": 15}),
    ]

    all_results = {s[0]: [] for s in strategies}

    n = len(df)
    train_end = min_train_bars
    fold_num = 0

    while train_end + val_window_bars + test_window_bars <= n:
        fold_num += 1
        val_end = train_end + val_window_bars
        test_end = min(val_end + test_window_bars, n)

        test_start_date = df.index[val_end].date()
        test_end_date = df.index[test_end - 1].date()
        print(f"\n  --- Fold {fold_num}: test {test_start_date} to {test_end_date} ---")

        for strat_name, strat_params in strategies:
            result = run_fold_with_monitoring(
                df, feature_cols, h4_data, symbol,
                train_end, val_end, test_end, fold_num,
                strategy=strat_name, **strat_params,
            )
            if result:
                all_results[strat_name].append(result)

        # Print baseline + best comparison for this fold
        base = all_results["baseline"][-1] if all_results["baseline"] else None
        if base:
            p1 = "PASS" if base["phase1_passed"] else "fail"
            halt = "HALTED" if base["halted"] else ""
            print(f"    Baseline:  {base['total_return']:>+8.2%} | DD: {base['max_dd']:.2%} | "
                  f"Trades: {base['total_trades']:>4} | Signals@200: {base['signals_at_200']:>3} | "
                  f"Trades@200: {base['trades_at_200']:>3} | DD@200: {base['dd_at_200']:.2%} | {p1} {halt}")

            for strat_name in ["early_halt", "density_gate", "combined"]:
                r = all_results[strat_name][-1] if all_results[strat_name] else None
                if r:
                    p1 = "PASS" if r["phase1_passed"] else "fail"
                    extra = ""
                    if r["density_halted"]:
                        extra = "DENSITY-HALT"
                    elif r["halted"]:
                        extra = "HALTED"
                    print(f"    {strat_name:<12} {r['total_return']:>+8.2%} | DD: {r['max_dd']:.2%} | "
                          f"Trades: {r['total_trades']:>4} | {p1} {extra}")

        train_end += step_bars

    # Summary comparison
    print(f"\n{'='*90}")
    print(f"STRATEGY COMPARISON SUMMARY — USTEC ({fold_num} folds)")
    print(f"{'='*90}")

    print(f"\n  {'Strategy':<20} {'Pass%':>7} {'Blow%':>7} {'MeanRet':>9} {'MedRet':>9} "
          f"{'MeanDD':>8} {'WorstDD':>8} {'E[$]':>10}")
    print(f"  {'-'*85}")

    for strat_name, _ in strategies:
        results = all_results[strat_name]
        if not results:
            continue
        rdf = pd.DataFrame(results)
        n_folds = len(rdf)
        pass_rate = rdf["phase1_passed"].mean()
        blow_rate = rdf["halted"].mean()
        mean_ret = rdf["total_return"].mean()
        med_ret = rdf["total_return"].median()
        mean_dd = rdf["max_dd"].mean()
        worst_dd = rdf["max_dd"].max()
        ev = mean_ret * 100_000 - 500

        # Flag density-halted folds
        density_stopped = rdf["density_halted"].sum() if "density_halted" in rdf else 0

        extra = f" (density-halt: {density_stopped})" if density_stopped > 0 else ""

        print(f"  {strat_name:<20} {pass_rate:>6.0%} {blow_rate:>6.0%} {mean_ret:>+8.2%} "
              f"{med_ret:>+8.2%} {mean_dd:>7.2%} {worst_dd:>7.2%} ${ev:>+9,.0f}{extra}")

    # Detailed: show early detection metrics for baseline (helps calibrate thresholds)
    print(f"\n{'='*90}")
    print(f"EARLY DETECTION METRICS (at bar 200 of test window)")
    print(f"{'='*90}")

    base_results = all_results["baseline"]
    print(f"\n  {'Fold':>4} {'Return':>9} {'Signals@200':>12} {'Trades@200':>12} {'DD@200':>8} "
          f"{'TotalTrades':>12} {'Outcome':>10}")
    print(f"  {'-'*75}")
    for r in base_results:
        outcome = "PASS" if r["phase1_passed"] else ("BLOW" if r["halted"] else "fail")
        print(f"  {r['fold']:>4} {r['total_return']:>+8.2%} {r['signals_at_200']:>12} "
              f"{r['trades_at_200']:>12} {r['dd_at_200']:>7.2%} "
              f"{r['total_trades']:>12} {outcome:>10}")

    # Find optimal density thresholds
    print(f"\n  THRESHOLD ANALYSIS:")
    for sig_thresh in [30, 50, 70, 100]:
        for trade_thresh in [10, 15, 20, 30]:
            caught_blow = 0
            killed_pass = 0
            for r in base_results:
                would_halt = r["signals_at_200"] < sig_thresh or r["trades_at_200"] < trade_thresh
                if would_halt and r["halted"]:
                    caught_blow += 1
                if would_halt and r["phase1_passed"]:
                    killed_pass += 1
            total_blow = sum(1 for r in base_results if r["halted"])
            total_pass = sum(1 for r in base_results if r["phase1_passed"])
            if caught_blow > 0 and killed_pass == 0:
                print(f"    sig>={sig_thresh:>3} AND trades>={trade_thresh:>2}: "
                      f"catches {caught_blow}/{total_blow} blowups, kills {killed_pass}/{total_pass} winners")

    # Save detailed results
    all_flat = []
    for strat_name, results in all_results.items():
        all_flat.extend(results)
    output_path = str(PROJECT_ROOT / "reports" / "ustec_strategy_validation.csv")
    pd.DataFrame(all_flat).to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
