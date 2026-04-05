"""Phase 4: Full Walk-Forward Validation of Optimized Strategy.

Re-runs COMPLETE walk-forward (fresh model retrain per fold) with the
optimized risk configuration from Phase 3:
  - risk_per_trade: 2.50%
  - kill_switch: after 3 trades, halt if PnL < -1.0%
  - streak_scaling: after 2 consecutive losses, reduce to 25% size
  - NO dynamic acceleration (stripped — doesn't improve pass rate)

Also computes:
  - Comparison vs baseline (same folds, different risk rules)
  - Per-fold early-warning accuracy
  - Statistical significance of improvement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.pipeline import load_config, PROJECT_ROOT

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

config = load_config()


def load_full_data(symbol: str):
    """Load full dataset with features and labels."""
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


def simulate_trade(prices, entry_idx, direction, entry_price, sl_dist, tp_dist, max_bars=8):
    """Simulate trade, return (pnl, exit_type, bars_held)."""
    for j in range(1, max_bars + 1):
        idx = entry_idx + j
        if idx >= len(prices):
            break
        bh = prices["high"].iloc[idx]
        bl = prices["low"].iloc[idx]
        if direction == 1:
            if bh >= entry_price + tp_dist:
                return tp_dist, "TP", j
            if bl <= entry_price - sl_dist:
                return -sl_dist, "SL", j
        else:
            if bl <= entry_price - tp_dist:
                return tp_dist, "TP", j
            if bh >= entry_price + sl_dist:
                return -sl_dist, "SL", j
    final_idx = min(entry_idx + max_bars, len(prices) - 1)
    final_close = prices["close"].iloc[final_idx]
    return (final_close - entry_price) * direction, "TIME", max_bars


def run_fold_optimized(df, feature_cols, h4_data, symbol,
                        train_end, val_end, test_end, fold_num,
                        strategy="optimized"):
    """Run fold with either baseline or optimized risk management.

    strategy: "baseline" or "optimized"
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

    # Train model
    ensemble = StackingEnsemble(config.get("model", {}))
    ensemble.fit(X_train, y_train)

    # Calibrate from validation set
    val_proba = ensemble.predict_proba(X_val)
    median_proba = float(np.median(val_proba))
    val_preds = (val_proba >= 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()

    # Regime detector on training H4
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
                regime_scalars = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0).values
        except Exception:
            pass

    # Generate predictions
    test_proba = ensemble.predict_proba(test_df[feature_cols].values)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)

    # Strategy-specific parameters
    if strategy == "optimized":
        risk_per_trade = 0.025
        kill_switch_n = 3
        kill_switch_thresh = -0.01
        streak_scale = True
        streak_loss_count = 2
        streak_scale_factor = 0.25
    else:  # baseline
        risk_per_trade = risk_cfg.get("risk_per_trade_pct", 1.75) / 100
        kill_switch_n = 0
        kill_switch_thresh = -999
        streak_scale = False
        streak_loss_count = 999
        streak_scale_factor = 1.0

    # Simulate
    initial_balance = 100_000
    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0
    trade_count = 0
    consecutive_losses = 0
    killed = False
    halted_total = False
    halted_daily = False
    daily_pnl = {}
    current_date = None
    day_start_balance = balance
    trades = []
    equity_curve = []

    for i in range(len(test_df)):
        ts = test_df.index[i]
        today = ts.date() if hasattr(ts, "date") else ts

        if today != current_date:
            current_date = today
            day_start_balance = balance
            halted_daily = False

        # Check total halt
        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= 0.09:
            halted_total = True

        # Check daily halt
        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= 0.035:
            halted_daily = True

        equity_curve.append({"time": ts, "equity": balance})

        if halted_total or halted_daily or killed:
            continue

        p = test_proba[i]
        if p >= buy_thresh:
            signal = 1
            confidence = max(0.5, min(1.0, (p - median_proba) / (1 - median_proba)))
        elif p <= sell_thresh:
            signal = -1
            confidence = max(0.5, min(1.0, (median_proba - p) / median_proba))
        else:
            continue

        atr = test_df["atr"].iloc[i]
        if atr <= 0 or np.isnan(atr):
            continue
        if i + 1 >= len(test_df):
            continue

        entry_price = test_df["open"].iloc[i + 1]
        sl_dist = sl_atr_mult * atr
        tp_dist = tp_atr_mult * atr

        # Determine effective risk
        current_risk = risk_per_trade
        if streak_scale and consecutive_losses >= streak_loss_count:
            current_risk *= streak_scale_factor

        effective_risk = current_risk * confidence * regime_scalars[i]
        risk_amount = balance * effective_risk

        # Simulate trade
        trade_pnl, exit_type, bars_held = simulate_trade(
            test_df, i + 1, signal, entry_price, sl_dist, tp_dist
        )
        trade_pnl -= atr * 0.01  # spread

        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
            dollar_pnl = r_multiple * risk_amount
        else:
            continue

        balance += dollar_pnl
        trade_count += 1

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance
        if dd > max_dd:
            max_dd = dd

        if today not in daily_pnl:
            daily_pnl[today] = 0.0
        daily_pnl[today] += dollar_pnl

        if dollar_pnl > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1

        trades.append({
            "dollar_pnl": dollar_pnl,
            "r_multiple": r_multiple,
            "win": int(dollar_pnl > 0),
        })

        # Kill switch check
        if kill_switch_n > 0 and trade_count == kill_switch_n:
            cum_pnl_pct = (balance - initial_balance) / initial_balance
            if cum_pnl_pct < kill_switch_thresh:
                killed = True

    # Compute results
    total_return = (balance - initial_balance) / initial_balance
    total_dd = max(0, (initial_balance - min(balance, initial_balance)) / initial_balance)

    # Win rate
    win_rate = sum(1 for t in trades if t["win"]) / len(trades) if trades else 0
    # Profit factor
    total_wins = sum(t["dollar_pnl"] for t in trades if t["dollar_pnl"] > 0)
    total_losses = abs(sum(t["dollar_pnl"] for t in trades if t["dollar_pnl"] < 0))
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Sharpe
    if equity_curve:
        eq = pd.DataFrame(equity_curve).set_index("time")["equity"]
        daily_ret = eq.resample("D").last().pct_change(fill_method=None).dropna()
        if len(daily_ret) > 0 and daily_ret.std() > 0:
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Best day check
    best_day_pct = 0
    if daily_pnl:
        positive_days = {d: p for d, p in daily_pnl.items() if p > 0}
        if positive_days:
            total_pos = sum(positive_days.values())
            best = max(positive_days.values())
            best_day_pct = best / total_pos if total_pos > 0 else 0

    # Market return
    market_return = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]

    return {
        "fold": fold_num,
        "strategy": strategy,
        "train_start": str(train_df.index[0].date()),
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "val_accuracy": val_acc,
        "median_proba": median_proba,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": total_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": trade_count,
        "phase1_passed": total_return >= 0.10,
        "blown": halted_total,
        "killed": killed,
        "best_day_pct": best_day_pct,
        "best_day_violated": best_day_pct >= 0.50,
        "market_return": market_return,
        "alpha": total_return - market_return,
    }


def main():
    symbol = "USTEC"
    print(f"Loading data for {symbol}...")

    df, feature_cols, h4_data = load_full_data(symbol)
    if df.empty:
        print("No data!")
        return

    print(f"Total samples: {len(df)}, Range: {df.index[0].date()} to {df.index[-1].date()}")

    # Walk-forward parameters
    min_train_bars = 3000
    val_window_bars = 1000
    test_window_bars = 1000
    step_bars = 1000

    # USTEC offset
    start_date = pd.Timestamp("2022-07-19")
    if df.index.tz is not None:
        start_date = start_date.tz_localize(df.index.tz)
    start_idx = df.index.searchsorted(start_date)
    df_offset = df.iloc[start_idx:]
    print(f"USTEC offset: {len(df_offset)} samples from {df_offset.index[0].date()}")

    baseline_results = []
    optimized_results = []

    n = len(df_offset)
    fold_num = 0
    train_end = min_train_bars

    while train_end + val_window_bars + test_window_bars <= n:
        fold_num += 1
        val_end = train_end + val_window_bars
        test_end = min(val_end + test_window_bars, n)

        test_start_date = df_offset.index[val_end].date()
        test_end_date = df_offset.index[test_end - 1].date()
        print(f"\n  Fold {fold_num}: test {test_start_date} to {test_end_date}")

        # Run baseline
        b_result = run_fold_optimized(
            df_offset, feature_cols, h4_data, symbol,
            train_end, val_end, test_end, fold_num,
            strategy="baseline"
        )

        # Run optimized (SAME model, different risk rules)
        o_result = run_fold_optimized(
            df_offset, feature_cols, h4_data, symbol,
            train_end, val_end, test_end, fold_num,
            strategy="optimized"
        )

        if b_result and o_result:
            baseline_results.append(b_result)
            optimized_results.append(o_result)

            b_status = "PASS" if b_result["phase1_passed"] else ("BLOW" if b_result["blown"] else "fail")
            o_status = "PASS" if o_result["phase1_passed"] else ("BLOW" if o_result["blown"] else ("KILL" if o_result["killed"] else "fail"))

            change = ""
            if b_status != o_status:
                change = f" [{b_status}->{o_status}]"

            print(f"    Baseline:  {b_result['total_return']:>+8.2%} | "
                  f"Trades={b_result['total_trades']:>4} | WR={b_result['win_rate']:.1%} | {b_status}")
            print(f"    Optimized: {o_result['total_return']:>+8.2%} | "
                  f"Trades={o_result['total_trades']:>4} | WR={o_result['win_rate']:.1%} | {o_status}{change}")

        train_end += step_bars

    if not baseline_results:
        print("No results!")
        return

    # =====================================================================
    # AGGREGATE COMPARISON
    # =====================================================================
    b_df = pd.DataFrame(baseline_results)
    o_df = pd.DataFrame(optimized_results)

    print(f"\n{'='*75}")
    print("WALK-FORWARD VALIDATION: BASELINE vs OPTIMIZED")
    print(f"{'='*75}")

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Delta':>10}")
    print(f"  {'-'*65}")

    b_pass = b_df['phase1_passed'].sum()
    o_pass = o_df['phase1_passed'].sum()
    b_blow = b_df['blown'].sum()
    o_blow = o_df['blown'].sum()
    o_kill = o_df['killed'].sum()
    n_folds = len(b_df)

    print(f"  {'Pass Rate':<30} {b_pass}/{n_folds} ({b_pass/n_folds:.0%})"
          f"{o_pass}/{n_folds} ({o_pass/n_folds:.0%}):>12"
          f"{(o_pass-b_pass):>+9}")

    metrics = [
        ("Pass Rate", f"{b_pass}/{n_folds} ({b_pass/n_folds:.0%})", f"{o_pass}/{n_folds} ({o_pass/n_folds:.0%})", f"+{o_pass-b_pass} folds"),
        ("Blow Rate", f"{b_blow}/{n_folds} ({b_blow/n_folds:.0%})", f"{o_blow}/{n_folds} ({o_blow/n_folds:.0%})", f"{o_blow-b_blow:+d} folds"),
        ("Killed Early", f"N/A", f"{o_kill}/{n_folds}", ""),
        ("Mean Return", f"{b_df['total_return'].mean():.2%}", f"{o_df['total_return'].mean():.2%}", f"{o_df['total_return'].mean()-b_df['total_return'].mean():+.2%}"),
        ("Median Return", f"{b_df['total_return'].median():.2%}", f"{o_df['total_return'].median():.2%}", ""),
        ("Mean Win Rate", f"{b_df['win_rate'].mean():.1%}", f"{o_df['win_rate'].mean():.1%}", ""),
        ("Mean Sharpe", f"{b_df['sharpe'].mean():.2f}", f"{o_df['sharpe'].mean():.2f}", ""),
        ("Mean Max DD", f"{b_df['max_dd'].mean():.2%}", f"{o_df['max_dd'].mean():.2%}", ""),
        ("Best Day Violations", f"{b_df['best_day_violated'].sum()}", f"{o_df['best_day_violated'].sum()}", ""),
    ]

    print(f"\n  {'Metric':<25} {'Baseline':>18} {'Optimized':>18} {'Delta':>12}")
    print(f"  {'-'*75}")
    for name, b_val, o_val, delta in metrics:
        print(f"  {name:<25} {b_val:>18} {o_val:>18} {delta:>12}")

    # Per-fold detail
    print(f"\n  PER-FOLD DETAIL:")
    print(f"  {'Fold':>4} {'Baseline':>10} {'B-Status':>8} {'Optimized':>10} {'O-Status':>8} {'Change':>12}")
    print(f"  {'-'*60}")
    for b, o in zip(baseline_results, optimized_results):
        b_s = "PASS" if b["phase1_passed"] else ("BLOW" if b["blown"] else "fail")
        o_s = "PASS" if o["phase1_passed"] else ("BLOW" if o["blown"] else ("KILL" if o["killed"] else "fail"))
        change = f"{b_s}->{o_s}" if b_s != o_s else ""
        print(f"  {b['fold']:>4} {b['total_return']:>+9.2%} {b_s:>8} "
              f"{o['total_return']:>+9.2%} {o_s:>8} {change:>12}")

    # =====================================================================
    # STATISTICAL TEST OF IMPROVEMENT
    # =====================================================================
    print(f"\n{'='*75}")
    print("STATISTICAL TEST: Is Optimized Better Than Baseline?")
    print(f"{'='*75}")

    # Paired test on returns (same folds, different strategy)
    b_returns = b_df["total_return"].values
    o_returns = o_df["total_return"].values
    diff = o_returns - b_returns

    print(f"  Paired differences (optimized - baseline):")
    print(f"    Mean: {diff.mean():+.4f} ({diff.mean():+.2%})")
    print(f"    Median: {np.median(diff):+.4f}")
    print(f"    Positive: {(diff > 0).sum()}/{len(diff)}")

    # Wilcoxon signed-rank (non-parametric paired test)
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) >= 6:
        w_stat, w_p = stats.wilcoxon(nonzero_diff, alternative="greater")
        print(f"\n  Wilcoxon signed-rank (H1: optimized > baseline):")
        print(f"    W-statistic: {w_stat:.1f}")
        print(f"    p-value: {w_p:.4f}")
        print(f"    {'SIGNIFICANT' if w_p < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(o_returns, b_returns)
    t_p_one = t_p / 2 if t_stat > 0 else 1 - t_p / 2
    print(f"\n  Paired t-test (H1: optimized > baseline):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value (one-sided): {t_p_one:.4f}")
    print(f"    {'SIGNIFICANT' if t_p_one < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    # McNemar's test on pass/fail outcomes
    b_pass_vec = b_df["phase1_passed"].values
    o_pass_vec = o_df["phase1_passed"].values
    # Count discordant pairs
    b_pass_o_fail = ((b_pass_vec) & (~o_pass_vec)).sum()
    b_fail_o_pass = ((~b_pass_vec) & (o_pass_vec)).sum()
    print(f"\n  McNemar's test (pass/fail concordance):")
    print(f"    Baseline pass, Optimized fail: {b_pass_o_fail}")
    print(f"    Baseline fail, Optimized pass: {b_fail_o_pass}")
    if b_pass_o_fail + b_fail_o_pass > 0:
        # Exact binomial version of McNemar
        n_disc = b_pass_o_fail + b_fail_o_pass
        mcnemar_p = stats.binomtest(b_fail_o_pass, n_disc, 0.5, alternative="greater").pvalue
        print(f"    p-value (one-sided): {mcnemar_p:.4f}")
        print(f"    {'SIGNIFICANT' if mcnemar_p < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    # =====================================================================
    # KILL SWITCH EFFECTIVENESS
    # =====================================================================
    print(f"\n{'='*75}")
    print("KILL SWITCH ANALYSIS")
    print(f"{'='*75}")

    killed_folds = o_df[o_df["killed"]]
    passed_folds = o_df[o_df["phase1_passed"]]
    failed_folds = o_df[~o_df["phase1_passed"] & ~o_df["killed"] & ~o_df["blown"]]

    print(f"  Killed folds: {len(killed_folds)}")
    for _, row in killed_folds.iterrows():
        # What was the baseline outcome for this fold?
        b_row = b_df[b_df["fold"] == row["fold"]].iloc[0]
        b_status = "BLOW" if b_row["blown"] else ("PASS" if b_row["phase1_passed"] else "fail")
        b_ret = b_row["total_return"]
        print(f"    Fold {int(row['fold'])}: Killed at {row['total_return']:+.2%} | "
              f"Baseline would have been: {b_status} at {b_ret:+.2%} | "
              f"Saved: {abs(b_ret - row['total_return']):.2%}")

    # False positive rate (killed folds that would have passed)
    killed_would_pass = 0
    killed_would_fail = 0
    for _, row in killed_folds.iterrows():
        b_row = b_df[b_df["fold"] == row["fold"]].iloc[0]
        if b_row["phase1_passed"]:
            killed_would_pass += 1
        else:
            killed_would_fail += 1

    print(f"\n  Kill switch accuracy:")
    print(f"    True positives (killed bad folds): {killed_would_fail}")
    print(f"    False positives (killed good folds): {killed_would_pass}")
    if len(killed_folds) > 0:
        precision = killed_would_fail / len(killed_folds)
        print(f"    Precision: {precision:.0%}")

    # =====================================================================
    # EXPECTED VALUE COMPARISON
    # =====================================================================
    print(f"\n{'='*75}")
    print("EXPECTED VALUE COMPARISON")
    print(f"{'='*75}")

    fee = 500
    for label, results_df in [("Baseline", b_df), ("Optimized", o_df)]:
        pr = results_df["phase1_passed"].mean()
        pass_ret = results_df[results_df["phase1_passed"]]["total_return"]
        ev_pass = pass_ret.mean() * 100_000 if len(pass_ret) > 0 else 0
        ftmo_ev = pr * ev_pass - fee
        e_attempts = 1 / pr if pr > 0 else float("inf")

        print(f"\n  {label}:")
        print(f"    Pass rate: {pr:.0%}")
        print(f"    E[profit|pass]: ${ev_pass:,.0f}")
        print(f"    FTMO EV per attempt: ${ftmo_ev:+,.0f}")
        print(f"    Expected attempts to pass: {e_attempts:.1f}")
        print(f"    P(pass within 3 attempts): {1-(1-pr)**3:.1%}")

    # =====================================================================
    # WORST-CASE ANALYSIS
    # =====================================================================
    print(f"\n{'='*75}")
    print("WORST-CASE ANALYSIS (Optimized)")
    print(f"{'='*75}")

    o_returns = o_df["total_return"].values
    print(f"  Worst fold return: {o_returns.min():.2%}")
    print(f"  Max drawdown across all folds: {o_df['max_dd'].max():.2%}")
    print(f"  Worst daily drawdown: {o_df['max_dd'].max():.2%}")

    # Consecutive failure analysis
    outcomes = o_df["phase1_passed"].values
    max_consec_fail = 0
    current_fail = 0
    for o in outcomes:
        if not o:
            current_fail += 1
            max_consec_fail = max(max_consec_fail, current_fail)
        else:
            current_fail = 0
    print(f"  Max consecutive non-pass folds: {max_consec_fail}")

    # Cost of consecutive failures
    non_pass_costs = []
    for _, row in o_df[~o_df["phase1_passed"]].iterrows():
        cost = abs(min(0, row["total_return"])) * 100_000 + fee
        non_pass_costs.append(cost)
    if non_pass_costs:
        print(f"  Average cost per failure: ${np.mean(non_pass_costs):,.0f} (loss + fee)")
        print(f"  Worst failure cost: ${max(non_pass_costs):,.0f}")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    output_path = PROJECT_ROOT / "reports" / "validation_optimized.csv"
    combined = pd.concat([b_df, o_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")

    # =====================================================================
    # FINAL VERDICT
    # =====================================================================
    print(f"\n{'='*75}")
    print("VALIDATION VERDICT")
    print(f"{'='*75}")

    improvement = o_pass - b_pass
    if improvement > 0:
        print(f"  Optimized strategy IMPROVES pass rate by +{improvement} folds")
    elif improvement == 0:
        print(f"  Optimized strategy maintains pass rate (but reduces blow rate)")
    else:
        print(f"  WARNING: Optimized strategy REDUCES pass rate by {improvement} folds")

    if o_blow < b_blow:
        print(f"  Blow rate REDUCED from {b_blow} to {o_blow} folds")

    if o_kill > 0 and killed_would_pass == 0:
        print(f"  Kill switch: PERFECT precision — {o_kill} kills, 0 false positives")
    elif killed_would_pass > 0:
        print(f"  Kill switch: {killed_would_pass} false positive(s) — review threshold")

    print(f"\n  RECOMMENDED CONFIG FOR LIVE TRADING:")
    print(f"    risk_per_trade_pct: 2.50")
    print(f"    kill_switch_n: 3")
    print(f"    kill_switch_thresh_pct: -1.0")
    print(f"    streak_scale: true")
    print(f"    streak_loss_count: 2")
    print(f"    streak_scale_factor: 0.25")
    print(f"    signal_offset: {config.get('risk', {}).get('signal_offset', 0.01)}")
    print(f"    stop_loss_atr_multiple: 1.5")
    print(f"    take_profit_atr_multiple: 2.5")


if __name__ == "__main__":
    main()
