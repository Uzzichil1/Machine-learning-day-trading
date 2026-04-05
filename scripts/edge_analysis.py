"""Phase 2: Edge Decomposition — Understand WHAT drives wins vs losses.

Re-runs walk-forward folds with detailed per-trade logging to answer:
  1. What market conditions (ATR, ADX, regime, session) produce wins vs losses?
  2. Does higher model confidence → higher win rate?
  3. Can we detect hostile regimes in the first N trades?
  4. What is the optimal early-exit heuristic?

Saves trade-level data for all folds to reports/trade_analysis.csv
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
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
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
    """Simulate a single trade, return (pnl_in_atr_units, exit_type, bars_held)."""
    for j in range(1, max_bars + 1):
        idx = entry_idx + j
        if idx >= len(prices):
            break
        bar_high = prices["high"].iloc[idx]
        bar_low = prices["low"].iloc[idx]

        if direction == 1:
            if bar_high >= entry_price + tp_dist:
                return tp_dist, "TP", j
            if bar_low <= entry_price - sl_dist:
                return -sl_dist, "SL", j
        else:
            if bar_low <= entry_price - tp_dist:
                return tp_dist, "TP", j
            if bar_high >= entry_price + sl_dist:
                return -sl_dist, "SL", j

    final_idx = min(entry_idx + max_bars, len(prices) - 1)
    final_close = prices["close"].iloc[final_idx]
    pnl = (final_close - entry_price) * direction
    return pnl, "TIME", max_bars


def run_fold_with_trade_logging(df, feature_cols, h4_data, symbol,
                                 train_end, val_end, test_end, fold_num):
    """Run a single fold and return detailed per-trade records."""
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:test_end]

    if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
        return []

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    # Train model
    ensemble = StackingEnsemble(config.get("model", {}))
    ensemble.fit(X_train, y_train)

    # Calibrate median_proba from validation set
    val_proba = ensemble.predict_proba(X_val)
    median_proba = float(np.median(val_proba))

    # Regime detector on training H4 data
    regime_labels = pd.Series("unknown", index=test_df.index)
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
                h4_regimes = regime.predict_regime(h4_norm)
                h4_scalars = regime.get_size_scalar(h4_norm)
                test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
                regime_labels = h4_regimes.reindex(test_idx, method="ffill").fillna("unknown")
                regime_scalars = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0).values
        except Exception as e:
            logger.warning(f"Fold {fold_num}: regime failed: {e}")

    # Generate predictions
    test_proba = ensemble.predict_proba(test_df[feature_cols].values)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)
    risk_per_trade = risk_cfg.get("risk_per_trade_pct", 0.40) / 100

    # Simulate trades with full logging
    initial_balance = 100_000
    balance = initial_balance
    max_daily_loss = risk_cfg.get("daily_loss_halt_pct", 4.0) / 100
    max_total_loss = risk_cfg.get("total_drawdown_halt_pct", 9.0) / 100
    trades = []
    current_date = None
    day_start_balance = balance
    trade_num = 0

    for i in range(len(test_df)):
        ts = test_df.index[i]
        today = ts.date() if hasattr(ts, "date") else ts

        if today != current_date:
            current_date = today
            day_start_balance = balance

        # Check halts
        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= max_total_loss:
            break
        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= max_daily_loss:
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

        trade_pnl, exit_type, bars_held = simulate_trade(
            test_df, i + 1, signal, entry_price, sl_dist, tp_dist
        )
        trade_pnl -= atr * 0.01  # spread

        effective_risk = risk_per_trade * confidence * regime_scalars[i]
        risk_amount = balance * effective_risk
        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
            dollar_pnl = r_multiple * risk_amount
        else:
            continue

        balance += dollar_pnl
        trade_num += 1

        # Capture features at signal time
        hour = ts.hour if hasattr(ts, 'hour') else 0
        dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0

        # ADX if available
        adx = test_df["adx_14"].iloc[i] if "adx_14" in test_df.columns else np.nan
        rsi = test_df["rsi_14"].iloc[i] if "rsi_14" in test_df.columns else np.nan
        vol_ratio = test_df["vol_ratio"].iloc[i] if "vol_ratio" in test_df.columns else np.nan

        # Rolling market trend (20-bar return)
        if i >= 20:
            trend_20 = (test_df["close"].iloc[i] - test_df["close"].iloc[i-20]) / test_df["close"].iloc[i-20]
        else:
            trend_20 = np.nan

        trades.append({
            "fold": fold_num,
            "trade_num": trade_num,
            "timestamp": str(ts),
            "signal": signal,
            "probability": p,
            "confidence": confidence,
            "median_proba": median_proba,
            "prob_deviation": abs(p - median_proba),
            "regime": regime_labels.iloc[i] if i < len(regime_labels) else "unknown",
            "regime_scalar": regime_scalars[i],
            "atr": atr,
            "adx": adx,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "hour": hour,
            "day_of_week": dow,
            "trend_20": trend_20,
            "exit_type": exit_type,
            "bars_held": bars_held,
            "r_multiple": r_multiple,
            "dollar_pnl": dollar_pnl,
            "effective_risk": effective_risk,
            "balance_after": balance,
            "equity_dd": (initial_balance - balance) / initial_balance,
            "win": int(dollar_pnl > 0),
        })

    return trades


def analyze_trades(all_trades_df):
    """Run comprehensive edge decomposition analysis."""
    df = all_trades_df
    n_trades = len(df)
    n_wins = df["win"].sum()
    total_wr = n_wins / n_trades

    print(f"\n{'='*70}")
    print(f"PHASE 2: EDGE DECOMPOSITION")
    print(f"{'='*70}")
    print(f"Total trades across all folds: {n_trades}")
    print(f"Overall win rate: {total_wr:.1%}")
    print(f"Overall mean R-multiple: {df['r_multiple'].mean():.4f}")

    # =====================================================================
    # 2.1: CONFIDENCE SEGMENTATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.1: EDGE BY MODEL CONFIDENCE")
    print(f"{'='*70}")

    # Bin by probability deviation from median
    df["conf_bucket"] = pd.cut(df["prob_deviation"],
                                bins=[0, 0.01, 0.02, 0.05, 0.1, 1.0],
                                labels=["0-1%", "1-2%", "2-5%", "5-10%", "10%+"])

    conf_stats = df.groupby("conf_bucket", observed=True).agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        mean_r=("r_multiple", "mean"),
        mean_pnl=("dollar_pnl", "mean"),
    ).round(4)
    print(conf_stats.to_string())

    # =====================================================================
    # 2.2: REGIME SEGMENTATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.2: EDGE BY REGIME STATE")
    print(f"{'='*70}")

    regime_stats = df.groupby("regime", observed=True).agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        mean_r=("r_multiple", "mean"),
        mean_pnl=("dollar_pnl", "mean"),
    ).round(4)
    print(regime_stats.to_string())

    # =====================================================================
    # 2.3: SESSION / HOUR SEGMENTATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.3: EDGE BY HOUR OF DAY")
    print(f"{'='*70}")

    hour_stats = df.groupby("hour").agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        mean_r=("r_multiple", "mean"),
    ).round(4)
    print(hour_stats.to_string())

    # Session classification
    df["session"] = "other"
    df.loc[df["hour"].between(7, 12), "session"] = "london"
    df.loc[df["hour"].between(13, 16), "session"] = "ny_overlap"
    df.loc[df["hour"].between(17, 21), "session"] = "ny_afternoon"

    session_stats = df.groupby("session", observed=True).agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        mean_r=("r_multiple", "mean"),
        mean_pnl=("dollar_pnl", "mean"),
    ).round(4)
    print(f"\nBy Session:")
    print(session_stats.to_string())

    # =====================================================================
    # 2.4: TREND CONTEXT SEGMENTATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.4: EDGE BY TREND CONTEXT")
    print(f"{'='*70}")

    # ADX segmentation (if available)
    if df["adx"].notna().sum() > 100:
        df["adx_bucket"] = pd.cut(df["adx"], bins=[0, 15, 20, 25, 30, 100],
                                   labels=["<15", "15-20", "20-25", "25-30", "30+"])
        adx_stats = df.groupby("adx_bucket", observed=True).agg(
            trades=("win", "count"),
            win_rate=("win", "mean"),
            mean_r=("r_multiple", "mean"),
        ).round(4)
        print(f"By ADX (trend strength):")
        print(adx_stats.to_string())

    # 20-bar trend direction
    if df["trend_20"].notna().sum() > 100:
        df["trend_bucket"] = pd.cut(df["trend_20"],
                                     bins=[-1, -0.03, -0.01, 0.01, 0.03, 1],
                                     labels=["strong_down", "mild_down", "flat", "mild_up", "strong_up"])
        trend_stats = df.groupby("trend_bucket", observed=True).agg(
            trades=("win", "count"),
            win_rate=("win", "mean"),
            mean_r=("r_multiple", "mean"),
        ).round(4)
        print(f"\nBy 20-bar Trend:")
        print(trend_stats.to_string())

    # =====================================================================
    # 2.5: EXIT TYPE ANALYSIS
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.5: EXIT TYPE DISTRIBUTION")
    print(f"{'='*70}")

    exit_stats = df.groupby("exit_type").agg(
        trades=("win", "count"),
        pct=("win", lambda x: len(x) / n_trades),
        mean_r=("r_multiple", "mean"),
    ).round(4)
    print(exit_stats.to_string())

    # =====================================================================
    # 2.6: DIRECTION ANALYSIS (BUY vs SELL)
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.6: EDGE BY DIRECTION")
    print(f"{'='*70}")

    dir_stats = df.groupby("signal").agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
        mean_r=("r_multiple", "mean"),
        mean_pnl=("dollar_pnl", "mean"),
    ).round(4)
    dir_stats.index = ["SELL", "BUY"]
    print(dir_stats.to_string())

    # =====================================================================
    # 2.7: EARLY WARNING — FIRST-N TRADES ANALYSIS
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.7: EARLY WARNING — FIRST-N TRADES PERFORMANCE")
    print(f"{'='*70}")

    # For each fold, compute cumulative win rate after N trades
    folds_in_data = df["fold"].unique()
    print(f"\nFolds with trade data: {sorted(folds_in_data)}")

    # Classify folds
    fold_outcomes = {}
    for fold in folds_in_data:
        fold_df = df[df["fold"] == fold]
        final_dd = fold_df["equity_dd"].iloc[-1] if len(fold_df) > 0 else 0
        total_ret = (fold_df["balance_after"].iloc[-1] - 100_000) / 100_000 if len(fold_df) > 0 else 0
        blow = final_dd >= 0.09
        passed = total_ret >= 0.10
        fold_outcomes[fold] = "BLOW" if blow else ("PASS" if passed else "MARGINAL")

    print(f"\nFold outcomes: {fold_outcomes}")

    # First-N trade analysis
    for n_trades_check in [5, 10, 15, 20, 30]:
        print(f"\n  After first {n_trades_check} trades:")
        for fold in sorted(folds_in_data):
            fold_df = df[df["fold"] == fold]
            if len(fold_df) < n_trades_check:
                first_n = fold_df
            else:
                first_n = fold_df.iloc[:n_trades_check]

            wr = first_n["win"].mean() if len(first_n) > 0 else 0
            cum_r = first_n["r_multiple"].sum()
            cum_pnl = first_n["dollar_pnl"].sum()
            pnl_pct = cum_pnl / 100_000

            outcome = fold_outcomes.get(fold, "?")
            marker = "***" if outcome == "BLOW" else "   "
            print(f"    {marker} Fold {fold:>2} [{outcome:>8}]: WR={wr:.0%} | cumR={cum_r:>+6.2f} | "
                  f"PnL={pnl_pct:>+6.2%} | trades={len(first_n)}")

    # =====================================================================
    # 2.8: OPTIMAL EARLY-EXIT THRESHOLD
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.8: OPTIMAL EARLY-EXIT THRESHOLD SEARCH")
    print(f"{'='*70}")

    print(f"\nSweep: After N trades, if cumulative PnL < threshold%, halt trading.")
    print(f"{'N':>4} {'Threshold':>10} {'Blows Caught':>14} {'Passes Killed':>15} {'Net':>5}")
    print("-" * 55)

    best_net = -999
    best_params = (0, 0)

    for n_check in [5, 10, 15, 20, 25, 30]:
        for thresh_pct in [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0]:
            blows_caught = 0
            passes_killed = 0

            for fold in sorted(folds_in_data):
                fold_df = df[df["fold"] == fold]
                if len(fold_df) < n_check:
                    continue
                first_n = fold_df.iloc[:n_check]
                cum_pnl_pct = first_n["dollar_pnl"].sum() / 100_000

                would_halt = cum_pnl_pct < (thresh_pct / 100)
                outcome = fold_outcomes.get(fold, "?")

                if would_halt and outcome == "BLOW":
                    blows_caught += 1
                elif would_halt and outcome == "PASS":
                    passes_killed += 1

            net = blows_caught - passes_killed * 3  # Killing a pass is 3x worse
            if net > best_net:
                best_net = net
                best_params = (n_check, thresh_pct)

            if blows_caught > 0 or passes_killed > 0:
                print(f"{n_check:>4} {thresh_pct:>9.1f}% {blows_caught:>14} {passes_killed:>15} {net:>5}")

    print(f"\nBest params: After {best_params[0]} trades, halt if PnL < {best_params[1]}%")
    print(f"  Net score: {best_net} (blows_caught - 3*passes_killed)")

    # =====================================================================
    # 2.9: TRADE AUTOCORRELATION
    # =====================================================================
    print(f"\n{'='*70}")
    print("2.9: TRADE OUTCOME AUTOCORRELATION")
    print(f"{'='*70}")

    # Per-fold autocorrelation of trade outcomes
    for fold in sorted(folds_in_data):
        fold_df = df[df["fold"] == fold]
        if len(fold_df) < 20:
            continue
        outcomes = fold_df["win"].values
        if outcomes.std() == 0:
            continue
        # Lag-1 autocorrelation
        ac1 = np.corrcoef(outcomes[:-1], outcomes[1:])[0, 1]
        outcome_label = fold_outcomes.get(fold, "?")
        print(f"  Fold {fold:>2} [{outcome_label:>8}]: lag-1 autocorr = {ac1:>+.4f} "
              f"({'clustered' if ac1 > 0.1 else 'anti-clustered' if ac1 < -0.1 else 'independent'})")

    # Pooled autocorrelation
    all_outcomes = df["win"].values
    if len(all_outcomes) > 20:
        ac1_pooled = np.corrcoef(all_outcomes[:-1], all_outcomes[1:])[0, 1]
        print(f"\n  Pooled lag-1 autocorr: {ac1_pooled:>+.4f}")

    return df


def main():
    symbol = "USTEC"
    print(f"Loading data for {symbol}...")

    df, feature_cols, h4_data = load_full_data(symbol)
    if df.empty:
        print("No data!")
        return

    print(f"Total samples: {len(df)}, Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Walk-forward parameters (same as walk_forward.py)
    min_train_bars = 3000
    val_window_bars = 1000
    test_window_bars = 1000
    step_bars = 1000

    # Use USTEC starting offset (2022-07-19 start = different offset)
    # The second run in walk_forward.py starts from a different index
    # We need to match the fold boundaries from the CSV
    # USTEC folds in CSV start from train_start=2022-07-19
    # Find the index where date >= 2022-07-19
    start_date = pd.Timestamp("2022-07-19")
    if df.index.tz is not None:
        start_date = start_date.tz_localize(df.index.tz)
    start_idx = df.index.searchsorted(start_date)

    # Offset the dataframe to match USTEC starting point
    df_offset = df.iloc[start_idx:].copy()
    df_offset_reset = df_offset.reset_index(drop=False)

    # Re-index feature_cols positions relative to original df
    # Actually, we need to use the original df but adjust train_end offsets
    print(f"USTEC offset: starting from index {start_idx} ({df.index[start_idx].date()})")
    print(f"Samples from offset: {len(df_offset)}")

    all_trades = []
    n = len(df_offset)
    fold_num = 0
    train_end = min_train_bars

    while train_end + val_window_bars + test_window_bars <= n:
        fold_num += 1
        val_end = train_end + val_window_bars
        test_end = min(val_end + test_window_bars, n)

        test_start_date = df_offset.index[val_end].date() if hasattr(df_offset.index[val_end], 'date') else df_offset.index[val_end]
        test_end_date = df_offset.index[test_end-1].date() if hasattr(df_offset.index[test_end-1], 'date') else df_offset.index[test_end-1]
        print(f"\n  Fold {fold_num}: test {test_start_date} to {test_end_date}")

        trades = run_fold_with_trade_logging(
            df_offset, feature_cols, h4_data, symbol,
            train_end, val_end, test_end, fold_num,
        )

        print(f"    Trades logged: {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t["win"])
            wr = wins / len(trades) if trades else 0
            total_pnl = sum(t["dollar_pnl"] for t in trades)
            print(f"    WR: {wr:.1%} | PnL: ${total_pnl:+,.0f}")

        all_trades.extend(trades)
        train_end += step_bars

    if not all_trades:
        print("No trades generated!")
        return

    trades_df = pd.DataFrame(all_trades)
    csv_path = PROJECT_ROOT / "reports" / "trade_analysis.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"\nTrade data saved to {csv_path} ({len(trades_df)} trades)")

    # Run analysis
    analyzed = analyze_trades(trades_df)

    # Save summary
    print(f"\n{'='*70}")
    print("PHASE 2 COMPLETE — KEY FINDINGS")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
