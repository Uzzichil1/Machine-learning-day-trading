"""Walk-forward validation: retrain + backtest across multiple time windows.

Tests whether the model has real edge across different market regimes,
not just the single (bullish) OOS period we've been using.

Approach:
  - Expanding training window (all data up to fold boundary)
  - Fixed-size validation window (for median_proba calibration)
  - Fixed-size test window (for FTMO backtest)
  - Step forward by test_window_size each iteration
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
from src.backtest.engine import FTMOBacktester
from src.pipeline import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = load_config()


def load_full_data(symbol: str):
    """Load and prepare full dataset with features and labels."""
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

    # Also load H4 data for regime detection
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)

    return df, feature_cols, h4_data


def run_fold(
    df: pd.DataFrame,
    feature_cols: list,
    h4_data: pd.DataFrame,
    symbol: str,
    train_end: int,
    val_end: int,
    test_end: int,
    fold_num: int,
) -> dict:
    """Train model on fold and backtest the test window."""
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]

    if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
        logger.warning(f"  Fold {fold_num}: insufficient data (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
        return None

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values
    X_test = test_df[feature_cols].values

    # Train fresh model
    ensemble = StackingEnsemble(config.get("model", {}))
    ensemble.fit(X_train, y_train)

    # Calibrate median_proba from VALIDATION set (no test leakage)
    val_proba = ensemble.predict_proba(X_val)
    median_proba = float(np.median(val_proba))
    ensemble._median_proba = median_proba

    # Validation accuracy
    val_preds = (val_proba >= 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()

    # Fit regime detector on TRAINING H4 data only
    regime_scalars = np.ones(len(test_df))
    if h4_data is not None and not h4_data.empty:
        try:
            # Normalize all timestamps to tz-naive for comparison
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
        except Exception as e:
            logger.warning(f"  Fold {fold_num}: regime detection failed: {e}")

    # Generate test predictions and build signals
    test_proba = ensemble.predict_proba(X_test)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset

    signals = pd.DataFrame(index=test_df.index)
    signals["signal"] = 0
    signals.loc[test_proba >= buy_thresh, "signal"] = 1
    signals.loc[test_proba <= sell_thresh, "signal"] = -1

    signals["confidence"] = np.where(
        signals["signal"] == 1,
        (test_proba - median_proba) / (1 - median_proba),
        np.where(signals["signal"] == -1, (median_proba - test_proba) / median_proba, 0.5),
    ).clip(0.5, 1.0)

    signals["atr"] = test_df["atr"].values
    signals["regime_scalar"] = regime_scalars
    signals["symbol"] = symbol

    n_signals = (signals["signal"] != 0).sum()
    n_buy = (signals["signal"] == 1).sum()
    n_sell = (signals["signal"] == -1).sum()

    # Run FTMO backtest
    backtester = FTMOBacktester({
        **config.get("account", {}),
        **config.get("ftmo_limits", {}),
    })

    result = backtester.run(
        signals, test_df,
        risk_per_trade=risk_cfg.get("risk_per_trade_pct", 0.40) / 100,
        sl_atr_mult=risk_cfg.get("stop_loss_atr_multiple", 1.5),
        tp_atr_mult=risk_cfg.get("take_profit_atr_multiple", 2.5),
    )

    # Compute market return for the test period (buy-and-hold baseline)
    market_return = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]

    return {
        "fold": fold_num,
        "train_start": str(train_df.index[0].date()),
        "train_end": str(train_df.index[-1].date()),
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "val_accuracy": val_acc,
        "median_proba": median_proba,
        "n_signals": int(n_signals),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "total_return": result.total_return,
        "sharpe": result.sharpe_ratio,
        "max_dd": result.max_drawdown,
        "max_daily_dd": result.max_daily_drawdown,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "days_to_target": result.days_to_target,
        "phase1_passed": result.ftmo_phase1_passed,
        "daily_breached": result.ftmo_daily_limit_breached,
        "total_breached": result.ftmo_total_limit_breached,
        "market_return": market_return,
        "alpha": result.total_return - market_return,  # Excess return over buy-and-hold
    }


def run_fold_rampup(
    df: pd.DataFrame,
    feature_cols: list,
    h4_data: pd.DataFrame,
    symbol: str,
    train_end: int,
    val_end: int,
    test_end: int,
    fold_num: int,
    probe_trades: int = 15,
    probe_risk_pct: float = 0.50,
    full_risk_pct: float = 1.75,
    min_wr_to_scale: float = 0.38,
    min_wr_to_continue: float = 0.28,
) -> dict:
    """Run fold with ramp-up: probe at low risk, scale up only if edge confirmed.

    Strategy:
      1. First `probe_trades` trades use probe_risk_pct (0.5%)
      2. If WR >= min_wr_to_scale after probe: switch to full_risk_pct
      3. If WR < min_wr_to_continue after probe: STOP trading entirely
      4. Between thresholds: continue at probe risk for another probe period
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
    val_preds = (val_proba >= 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()

    # Regime detection (same tz fix)
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

    # Custom simulation with ramp-up logic
    initial_balance = config.get("account", {}).get("initial_balance", 100_000)
    max_daily_loss = risk_cfg.get("daily_loss_halt_pct", 4.0) / 100
    max_total_loss = risk_cfg.get("total_drawdown_halt_pct", 9.0) / 100
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)
    phase1_target = config.get("account", {}).get("phase1_target_pct", 10.0) / 100

    balance = initial_balance
    equity_curve = []
    trades_completed = 0
    wins = 0
    total_pnl = 0.0
    current_risk_pct = probe_risk_pct / 100
    halted = False
    stopped_no_edge = False
    daily_balances = {}
    current_date = None
    day_start_balance = balance

    for i in range(len(test_df)):
        ts = test_df.index[i]
        p = test_proba[i]
        today = ts.date() if hasattr(ts, "date") else ts

        if today != current_date:
            current_date = today
            day_start_balance = balance

        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= max_total_loss:
            halted = True

        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= max_daily_loss:
            equity_curve.append({"time": ts, "equity": balance})
            continue

        equity_curve.append({"time": ts, "equity": balance})

        if halted or stopped_no_edge:
            continue

        # Signal generation
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

        # Entry on next bar
        if i + 1 >= len(test_df):
            continue
        entry_price = test_df["open"].iloc[i + 1]

        effective_risk = current_risk_pct * confidence * regime_scalars[i]
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

        trade_pnl -= atr * 0.01  # Spread

        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
            dollar_pnl = r_multiple * risk_amount
        else:
            dollar_pnl = 0

        balance += dollar_pnl
        trades_completed += 1
        total_pnl += dollar_pnl
        if dollar_pnl > 0:
            wins += 1

        if today not in daily_balances:
            daily_balances[today] = 0.0
        daily_balances[today] += dollar_pnl

        # Ramp-up decision after probe period
        if trades_completed == probe_trades:
            wr = wins / trades_completed
            if wr >= min_wr_to_scale:
                current_risk_pct = full_risk_pct / 100  # Scale up!
            elif wr < min_wr_to_continue:
                stopped_no_edge = True  # No edge detected, stop

        # Second check at 2x probe trades (for mid-range WR)
        if trades_completed == probe_trades * 2:
            wr = wins / trades_completed
            if wr >= min_wr_to_scale:
                current_risk_pct = full_risk_pct / 100
            elif wr < 0.35:
                stopped_no_edge = True

    # Compile results
    market_return = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]
    total_return = (balance - initial_balance) / initial_balance

    # Max DD
    if equity_curve:
        eq = pd.DataFrame(equity_curve).set_index("time")["equity"]
        min_eq = eq.min()
        max_dd = max(0, (initial_balance - min_eq) / initial_balance)
    else:
        max_dd = 0

    # Daily DD
    worst_daily_dd = 0.0
    if daily_balances:
        running = initial_balance
        for d in sorted(daily_balances.keys()):
            dp = daily_balances[d]
            if running > 0 and dp < 0:
                worst_daily_dd = max(worst_daily_dd, abs(dp) / running)
            running += dp

    wr_final = wins / trades_completed if trades_completed > 0 else 0

    return {
        "fold": fold_num,
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "total_return": total_return,
        "max_dd": max_dd,
        "max_daily_dd": worst_daily_dd,
        "win_rate": wr_final,
        "total_trades": trades_completed,
        "phase1_passed": total_return >= phase1_target,
        "total_breached": halted,
        "stopped_no_edge": stopped_no_edge,
        "market_return": market_return,
        "alpha": total_return - market_return,
        "val_accuracy": val_acc,
    }


def main():
    symbols = ["XAUUSD", "USTEC"]

    # Walk-forward parameters
    min_train_bars = 3000       # ~6 months H1 minimum training
    val_window_bars = 1000      # ~2 months validation
    test_window_bars = 1000     # ~2 months test
    step_bars = 1000            # Step forward by ~2 months each fold

    all_results = []
    all_rampup_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD VALIDATION: {symbol}")
        print(f"{'='*80}")

        df, feature_cols, h4_data = load_full_data(symbol)
        if df.empty:
            print(f"  No data for {symbol}")
            continue

        print(f"  Total samples: {len(df)}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Buy ratio: {df['target'].mean():.2%}")

        n = len(df)
        fold_num = 0
        fold_results = []
        rampup_results = []

        # Generate fold boundaries
        train_end = min_train_bars
        while train_end + val_window_bars + test_window_bars <= n:
            fold_num += 1
            val_end = train_end + val_window_bars
            test_end = min(val_end + test_window_bars, n)

            test_start_date = df.index[val_end].date()
            test_end_date = df.index[test_end - 1].date()
            print(f"\n  --- Fold {fold_num}: test {test_start_date} to {test_end_date} ---")

            # Standard run
            result = run_fold(
                df, feature_cols, h4_data, symbol,
                train_end, val_end, test_end, fold_num,
            )

            # Ramp-up run
            rampup = run_fold_rampup(
                df, feature_cols, h4_data, symbol,
                train_end, val_end, test_end, fold_num,
            )

            if result:
                fold_results.append(result)
                ret = result["total_return"]
                alpha = result["alpha"]
                dd = result["max_dd"]
                trades = result["total_trades"]
                p1 = "PASS" if result["phase1_passed"] else "fail"

                r_ret = rampup["total_return"] if rampup else 0
                r_dd = rampup["max_dd"] if rampup else 0
                r_trades = rampup["total_trades"] if rampup else 0
                r_p1 = "PASS" if rampup and rampup["phase1_passed"] else "fail"
                r_stop = " STOPPED" if rampup and rampup.get("stopped_no_edge") else ""

                print(f"    Standard: {ret:>7.2%} | DD: {dd:.2%} | Trades: {trades:>4} | {p1}")
                print(f"    Ramp-up:  {r_ret:>7.2%} | DD: {r_dd:.2%} | Trades: {r_trades:>4} | {r_p1}{r_stop}")

            if rampup:
                rampup_results.append(rampup)

            train_end += step_bars

        if not fold_results:
            print("  No valid folds!")
            continue

        # Aggregate statistics
        fold_df = pd.DataFrame(fold_results)
        rampup_df = pd.DataFrame(rampup_results) if rampup_results else pd.DataFrame()
        all_results.extend(fold_results)
        all_rampup_results.extend(rampup_results)

        print(f"\n{'='*80}")
        print(f"WALK-FORWARD SUMMARY: {symbol}")
        print(f"{'='*80}")
        print(f"  Total folds:        {len(fold_df)}")
        print(f"  Profitable folds:   {(fold_df['total_return'] > 0).sum()} / {len(fold_df)} "
              f"({(fold_df['total_return'] > 0).mean():.0%})")
        print(f"  Phase 1 passed:     {fold_df['phase1_passed'].sum()} / {len(fold_df)}")
        print(f"  Daily breached:     {fold_df['daily_breached'].sum()} / {len(fold_df)}")
        print(f"  Total breached:     {fold_df['total_breached'].sum()} / {len(fold_df)}")

        print(f"\n  RETURNS:")
        print(f"    Mean:             {fold_df['total_return'].mean():>8.2%}")
        print(f"    Median:           {fold_df['total_return'].median():>8.2%}")
        print(f"    Std:              {fold_df['total_return'].std():>8.2%}")
        print(f"    Worst fold:       {fold_df['total_return'].min():>8.2%}")
        print(f"    Best fold:        {fold_df['total_return'].max():>8.2%}")

        print(f"\n  ALPHA (vs buy-and-hold):")
        print(f"    Mean alpha:       {fold_df['alpha'].mean():>8.2%}")
        print(f"    Positive alpha:   {(fold_df['alpha'] > 0).sum()} / {len(fold_df)} folds")

        print(f"\n  RISK:")
        print(f"    Mean max DD:      {fold_df['max_dd'].mean():>8.2%}")
        print(f"    Worst max DD:     {fold_df['max_dd'].max():>8.2%}")
        print(f"    Mean daily DD:    {fold_df['max_daily_dd'].mean():>8.2%}")
        print(f"    Worst daily DD:   {fold_df['max_daily_dd'].max():>8.2%}")

        print(f"\n  QUALITY:")
        print(f"    Mean win rate:    {fold_df['win_rate'].mean():>8.2%}")
        print(f"    Mean Sharpe:      {fold_df['sharpe'].mean():>8.2f}")
        print(f"    Mean PF:          {fold_df['profit_factor'].mean():>8.2f}")
        print(f"    Mean trades/fold: {fold_df['total_trades'].mean():>8.0f}")

        # Per-fold detail table
        print(f"\n  FOLD DETAILS:")
        print(f"  {'Fold':>4} {'Test Period':>25} {'Return':>8} {'Market':>8} {'Alpha':>8} "
              f"{'WR':>6} {'PF':>6} {'MaxDD':>7} {'Trades':>7} {'P1':>5}")
        print(f"  {'-'*95}")
        for _, row in fold_df.iterrows():
            print(f"  {row['fold']:>4} {row['test_start']+' to '+row['test_end']:>25} "
                  f"{row['total_return']:>7.2%} {row['market_return']:>7.2%} {row['alpha']:>7.2%} "
                  f"{row['win_rate']:>5.1%} {row['profit_factor']:>6.2f} {row['max_dd']:>6.2%} "
                  f"{row['total_trades']:>7} {'PASS' if row['phase1_passed'] else 'fail':>5}")

        # Ramp-up comparison
        if not rampup_df.empty:
            print(f"\n  RAMP-UP vs STANDARD COMPARISON:")
            print(f"  {'':>30} {'Standard':>12} {'Ramp-Up':>12}")
            print(f"  {'Profitable folds':>30} {(fold_df['total_return'] > 0).sum():>8}/{len(fold_df):<3} "
                  f"{(rampup_df['total_return'] > 0).sum():>8}/{len(rampup_df):<3}")
            print(f"  {'Phase 1 passed':>30} {fold_df['phase1_passed'].sum():>8}/{len(fold_df):<3} "
                  f"{rampup_df['phase1_passed'].sum():>8}/{len(rampup_df):<3}")
            print(f"  {'Total breached':>30} {fold_df['total_breached'].sum():>8}/{len(fold_df):<3} "
                  f"{rampup_df['total_breached'].sum():>8}/{len(rampup_df):<3}")
            std_stopped = rampup_df['stopped_no_edge'].sum() if 'stopped_no_edge' in rampup_df else 0
            print(f"  {'Stopped (no edge)':>30} {'N/A':>12} {std_stopped:>8}/{len(rampup_df):<3}")
            print(f"  {'Mean return':>30} {fold_df['total_return'].mean():>11.2%} {rampup_df['total_return'].mean():>11.2%}")
            print(f"  {'Median return':>30} {fold_df['total_return'].median():>11.2%} {rampup_df['total_return'].median():>11.2%}")
            print(f"  {'Mean max DD':>30} {fold_df['max_dd'].mean():>11.2%} {rampup_df['max_dd'].mean():>11.2%}")
            print(f"  {'Worst max DD':>30} {fold_df['max_dd'].max():>11.2%} {rampup_df['max_dd'].max():>11.2%}")

    # Cross-instrument summary
    if all_results:
        full_df = pd.DataFrame(all_results)
        print(f"\n{'='*80}")
        print("OVERALL WALK-FORWARD VERDICT")
        print(f"{'='*80}")
        print(f"  Total folds across all instruments: {len(full_df)}")
        print(f"  Profitable folds: {(full_df['total_return'] > 0).sum()} / {len(full_df)} "
              f"({(full_df['total_return'] > 0).mean():.0%})")
        print(f"  Positive alpha folds: {(full_df['alpha'] > 0).sum()} / {len(full_df)} "
              f"({(full_df['alpha'] > 0).mean():.0%})")
        print(f"  FTMO Phase 1 passed: {full_df['phase1_passed'].sum()} / {len(full_df)}")
        print(f"  FTMO daily breached: {full_df['daily_breached'].sum()} / {len(full_df)}")
        print(f"  FTMO total breached: {full_df['total_breached'].sum()} / {len(full_df)}")
        print(f"  Mean return: {full_df['total_return'].mean():.2%} +/- {full_df['total_return'].std():.2%}")
        print(f"  Mean alpha:  {full_df['alpha'].mean():.2%}")
        print(f"  Mean Sharpe: {full_df['sharpe'].mean():.2f}")

        # Honest assessment
        profitable_pct = (full_df['total_return'] > 0).mean()
        alpha_pct = (full_df['alpha'] > 0).mean()
        breach_rate = (full_df['daily_breached'] | full_df['total_breached']).mean()

        print(f"\n  HONEST ASSESSMENT:")
        if profitable_pct >= 0.7 and alpha_pct >= 0.5 and breach_rate < 0.2:
            print("  >>> Model shows CONSISTENT edge across market regimes <<<")
        elif profitable_pct >= 0.5:
            print("  >>> Model shows SOME edge but inconsistent across regimes <<<")
            print("  >>> Results may be regime-dependent — proceed with caution <<<")
        else:
            print("  >>> Model does NOT show reliable edge across market regimes <<<")
            print("  >>> Single-period backtest results are likely MISLEADING <<<")

        # Ramp-up overall comparison
        if all_rampup_results:
            ramp_df = pd.DataFrame(all_rampup_results)
            print(f"\n  RAMP-UP STRATEGY OVERALL:")
            print(f"    Profitable folds: {(ramp_df['total_return'] > 0).sum()} / {len(ramp_df)} "
                  f"({(ramp_df['total_return'] > 0).mean():.0%})")
            print(f"    Phase 1 passed: {ramp_df['phase1_passed'].sum()} / {len(ramp_df)}")
            print(f"    Total breached: {ramp_df['total_breached'].sum()} / {len(ramp_df)}")
            std_stopped = ramp_df['stopped_no_edge'].sum() if 'stopped_no_edge' in ramp_df else 0
            print(f"    Stopped (no edge): {std_stopped} / {len(ramp_df)}")
            print(f"    Mean return: {ramp_df['total_return'].mean():.2%}")
            print(f"    Mean max DD: {ramp_df['max_dd'].mean():.2%}")
            print(f"    Worst max DD: {ramp_df['max_dd'].max():.2%}")

            # Key metric: expected P&L per challenge attempt
            # Assumes $100K account, results in $ terms
            avg_ret_std = full_df['total_return'].mean()
            avg_ret_ramp = ramp_df['total_return'].mean()
            print(f"\n    Expected $ per challenge attempt:")
            print(f"      Standard: ${avg_ret_std * 100_000:>+10,.0f}")
            print(f"      Ramp-up:  ${avg_ret_ramp * 100_000:>+10,.0f}")

        # Save results
        output_path = str(PROJECT_ROOT / "reports" / "walk_forward_results.csv")
        full_df.to_csv(output_path, index=False)
        print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
