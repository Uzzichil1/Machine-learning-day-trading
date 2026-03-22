"""Monte Carlo simulation to stress-test strategy robustness."""

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

logging.basicConfig(level=logging.WARNING)

config = load_config()
N_SIMULATIONS = 1000


def get_trade_pnls(symbol: str) -> list[float]:
    """Get all trade P&Ls by simulating trades WITHOUT halt logic.

    We compute trade outcomes purely based on signals and price movement,
    not constrained by FTMO halts. The Monte Carlo simulation will apply
    halt logic during reshuffling.
    """
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))

    tf = config["timeframes"]["signal"]
    data_dir = str(PROJECT_ROOT / "data" / "raw")
    df = connector.load_data(symbol, tf, data_dir)
    if df.empty:
        return []

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

    data_cfg = config.get("data", {})
    n = len(df)
    val_end = int(n * (data_cfg.get("train_pct", 0.6) + data_cfg.get("validation_pct", 0.2)))
    test_df = df.iloc[val_end:]
    X_test = test_df[feature_cols].values

    ensemble = StackingEnsemble(config.get("model", {}))
    model_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_ensemble.joblib")
    ensemble.load(model_path)

    probas = ensemble.predict_proba(X_test)

    # Load saved regime detector (fitted on training data only)
    regime_scalars = np.ones(len(test_df))
    regime_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_regime.joblib")
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    if os.path.exists(regime_path) and not h4_data.empty:
        regime = RegimeDetector(n_states=3)
        regime.load(regime_path)
        try:
            h4_scalars = regime.get_size_scalar(h4_data)
            h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
            test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
            regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
            regime_scalars = regime_reindexed.values
        except Exception:
            pass

    # Generate signals and compute trade P&Ls directly (no halts)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.01)
    risk_per_trade = risk_cfg.get("risk_per_trade_pct", 1.75) / 100
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)
    median_proba = ensemble._median_proba  # Calibrated from validation set
    initial_balance = config.get("account", {}).get("initial_balance", 100_000)

    trade_pnls = []
    prices = test_df

    for i in range(len(test_df)):
        p = probas[i]
        if p >= median_proba + signal_offset:
            signal = 1
            confidence = (p - median_proba) / (1 - median_proba)
        elif p <= median_proba - signal_offset:
            signal = -1
            confidence = (median_proba - p) / median_proba
        else:
            continue

        confidence = max(0.5, min(1.0, confidence))
        regime_scalar = regime_scalars[i]
        atr = test_df["atr"].iloc[i]

        if atr <= 0 or np.isnan(atr):
            continue

        # Enter on NEXT bar's open (realistic timing)
        if i + 1 >= len(prices):
            continue
        entry_price = prices["open"].iloc[i + 1]

        # Calculate effective risk and R-multiple outcome
        effective_risk = risk_per_trade * confidence * regime_scalar
        sl_dist = sl_atr_mult * atr
        tp_dist = tp_atr_mult * atr

        # Simulate trade outcome starting from bar after entry
        trade_pnl = 0.0
        max_bars = 8
        for j in range(1, max_bars + 1):
            idx = i + 1 + j  # Start from bar after entry
            if idx >= len(prices):
                break
            bar_high = prices["high"].iloc[idx]
            bar_low = prices["low"].iloc[idx]

            if signal == 1:
                if bar_high >= entry_price + tp_dist:
                    trade_pnl = tp_dist
                    break
                if bar_low <= entry_price - sl_dist:
                    trade_pnl = -sl_dist
                    break
            else:
                if bar_low <= entry_price - tp_dist:
                    trade_pnl = tp_dist
                    break
                if bar_high >= entry_price + sl_dist:
                    trade_pnl = -sl_dist
                    break
        else:
            final_idx = min(i + 1 + max_bars, len(prices) - 1)
            final_close = prices["close"].iloc[final_idx]
            trade_pnl = (final_close - entry_price) * signal

        # Spread cost
        spread_cost = atr * 0.01
        trade_pnl -= spread_cost

        # Store as (R-multiple, effective_risk) for proper compounding in MC
        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
        else:
            r_multiple = 0

        trade_pnls.append((r_multiple, effective_risk))

    return trade_pnls


def run_monte_carlo(trade_pnls: list[float], n_sims: int, initial_balance: float = 100_000, **kwargs):
    """Run Monte Carlo simulation with FTMO-accurate halt logic and days-to-target tracking."""
    risk_cfg = config.get("risk", {})
    daily_halt_pct = risk_cfg.get("daily_loss_halt_pct", 4.0) / 100
    total_halt_pct = risk_cfg.get("total_drawdown_halt_pct", 9.0) / 100
    max_trades_per_day = risk_cfg.get("max_trades_per_day", 15)
    phase1_target = config.get("account", {}).get("phase1_target_pct", 10.0) / 100
    phase2_target = config.get("account", {}).get("phase2_target_pct", 5.0) / 100

    rng = np.random.default_rng(42)
    # trade_pnls is a list of (r_multiple, effective_risk) tuples
    trades = np.array(trade_pnls)  # shape: (N, 2)

    avg_trades_per_day = kwargs.get("avg_trades_per_day", 8.0)

    returns = []
    max_dds_ftmo = []
    days_to_target = []
    phase1_passes = 0
    phase2_passes = 0
    blown = 0
    pass_within_14 = 0
    pass_within_21 = 0
    pass_within_30 = 0

    for sim in range(n_sims):
        # Shuffle trade order
        indices = rng.permutation(len(trades))
        shuffled = trades[indices]
        balance = initial_balance
        min_balance = initial_balance
        halted_total = False
        target_hit_day = None
        trade_idx = 0
        trading_day = 0

        while trade_idx < len(shuffled) and not halted_total:
            trading_day += 1
            day_start = balance

            trades_this_day = int(rng.poisson(avg_trades_per_day))
            trades_this_day = max(1, min(trades_this_day, max_trades_per_day))
            halted_daily = False

            for _ in range(trades_this_day):
                if trade_idx >= len(shuffled):
                    break
                if halted_daily:
                    break

                r_multiple, eff_risk = shuffled[trade_idx]
                # Compound: dollar P&L based on CURRENT balance
                dollar_pnl = r_multiple * eff_risk * balance
                balance += dollar_pnl
                trade_idx += 1

                min_balance = min(min_balance, balance)

                # Check daily halt
                daily_dd = (day_start - balance) / day_start if day_start > 0 else 0
                if daily_dd >= daily_halt_pct:
                    halted_daily = True

                # Check total halt
                total_dd = (initial_balance - balance) / initial_balance
                if total_dd >= total_halt_pct:
                    halted_total = True
                    break

            # Check if target hit this day
            if target_hit_day is None and balance >= initial_balance * (1 + phase1_target):
                target_hit_day = trading_day

        # Record results
        total_ret = (balance - initial_balance) / initial_balance
        returns.append(total_ret)

        ftmo_dd = max(0, (initial_balance - min_balance) / initial_balance)
        max_dds_ftmo.append(ftmo_dd)

        if total_ret >= phase1_target:
            phase1_passes += 1
        if total_ret >= phase2_target:
            phase2_passes += 1
        if ftmo_dd >= 0.10:
            blown += 1

        if target_hit_day is not None:
            days_to_target.append(target_hit_day)
            if target_hit_day <= 10:  # 10 trading days ≈ 14 calendar days
                pass_within_14 += 1
            if target_hit_day <= 15:  # 15 trading days ≈ 21 calendar days
                pass_within_21 += 1
            if target_hit_day <= 22:  # 22 trading days ≈ 30 calendar days
                pass_within_30 += 1
        else:
            days_to_target.append(float("inf"))

    returns = np.array(returns)
    max_dds_ftmo = np.array(max_dds_ftmo)
    days_arr = np.array([d for d in days_to_target if d != float("inf")])

    return {
        "returns": returns,
        "max_dds": max_dds_ftmo,
        "days_to_target": days_arr,
        "days_to_target_all": days_to_target,
        "phase1_pass_rate": phase1_passes / n_sims,
        "phase2_pass_rate": phase2_passes / n_sims,
        "blow_rate": blown / n_sims,
        "pass_within_14_cal": pass_within_14 / n_sims,
        "pass_within_21_cal": pass_within_21 / n_sims,
        "pass_within_30_cal": pass_within_30 / n_sims,
        "avg_trades_per_day": avg_trades_per_day,
    }


def main():
    print("=" * 60)
    print("MONTE CARLO STRESS TEST — FTMO 14-DAY CHALLENGE")
    print(f"Simulations: {N_SIMULATIONS}")
    print("=" * 60)

    all_pnls = []
    for instr in config.get("instruments", []):
        if not instr.get("enabled"):
            continue
        symbol = instr["symbol"]
        print(f"\nLoading trades for {symbol}...")
        pnls = get_trade_pnls(symbol)
        n_trades = len(pnls)
        r_multiples = [r for r, _ in pnls] if pnls else []
        avg_r = np.mean(r_multiples) if r_multiples else 0
        win_rate = len([r for r in r_multiples if r > 0]) / n_trades if n_trades else 0
        print(f"  {n_trades} trades | Avg R-multiple: {avg_r:.3f}R | Win rate: {win_rate:.1%}")
        all_pnls.extend(pnls)

    r_all = [r for r, _ in all_pnls]
    print(f"\nCombined: {len(all_pnls)} trades | Avg R: {np.mean(r_all):.3f}R | Expectancy: {np.mean(r_all):.3f}R per trade")

    # Compute realistic trades/day from the actual combined backtest
    # OOS test set is 20% of 4 years ≈ 0.8 years ≈ 200 trading days
    # But signals come from active session hours only, with regime/concurrent limits
    # Use the combined backtest result (4030 trades over ~210 trading days ≈ 19/day)
    # Cap by concurrent position constraint: max 5 open × ~2 turnover cycles = ~10/day
    risk_cfg = config.get("risk", {})
    max_concurrent = risk_cfg.get("max_concurrent_trades", 5)
    # Realistic throughput: concurrent slots × avg turnover per day
    # With 8-bar max hold on H1 and 13 session bars, ~1.6 turnovers per slot
    realistic_trades_per_day = max_concurrent * 1.6
    realistic_trades_per_day = min(realistic_trades_per_day, risk_cfg.get("max_trades_per_day", 15))
    print(f"Realistic trades/day estimate: {realistic_trades_per_day:.1f}")
    print(f"Running {N_SIMULATIONS} Monte Carlo simulations...\n")

    mc = run_monte_carlo(all_pnls, N_SIMULATIONS, avg_trades_per_day=realistic_trades_per_day)

    returns = mc["returns"]
    max_dds = mc["max_dds"]
    days = mc["days_to_target"]

    print("=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
    print(f"  Avg trades/day simulated: {mc['avg_trades_per_day']:.1f}")

    print(f"\nRETURN DISTRIBUTION:")
    print(f"  Mean:      {returns.mean():>8.2%}")
    print(f"  Median:    {np.median(returns):>8.2%}")
    print(f"  Std Dev:   {returns.std():>8.2%}")
    print(f"  5th pct:   {np.percentile(returns, 5):>8.2%}")
    print(f"  25th pct:  {np.percentile(returns, 25):>8.2%}")
    print(f"  75th pct:  {np.percentile(returns, 75):>8.2%}")
    print(f"  95th pct:  {np.percentile(returns, 95):>8.2%}")
    print(f"  Min:       {returns.min():>8.2%}")
    print(f"  Max:       {returns.max():>8.2%}")

    print(f"\nFTMO DRAWDOWN (from initial balance):")
    print(f"  Mean:      {max_dds.mean():>8.2%}")
    print(f"  Median:    {np.median(max_dds):>8.2%}")
    print(f"  5th pct:   {np.percentile(max_dds, 5):>8.2%}")
    print(f"  25th pct:  {np.percentile(max_dds, 25):>8.2%}")
    print(f"  75th pct:  {np.percentile(max_dds, 75):>8.2%}")
    print(f"  95th pct:  {np.percentile(max_dds, 95):>8.2%}")
    print(f"  Max:       {max_dds.max():>8.2%}")

    print(f"\n{'='*60}")
    print(f"DAYS TO PHASE 1 TARGET (10% = $110,000):")
    print(f"{'='*60}")
    if len(days) > 0:
        print(f"  Simulations that passed: {len(days)} / {N_SIMULATIONS} ({len(days)/N_SIMULATIONS:.1%})")
        print(f"  Mean:      {days.mean():>8.1f} trading days")
        print(f"  Median:    {np.median(days):>8.1f} trading days")
        print(f"  5th pct:   {np.percentile(days, 5):>8.1f} trading days (fastest)")
        print(f"  25th pct:  {np.percentile(days, 25):>8.1f} trading days")
        print(f"  75th pct:  {np.percentile(days, 75):>8.1f} trading days")
        print(f"  95th pct:  {np.percentile(days, 95):>8.1f} trading days (slowest)")
        print(f"  Min:       {days.min():>8.0f} trading days")
        print(f"  Max:       {days.max():>8.0f} trading days")

        # Histogram
        print(f"\n  PASS TIMELINE DISTRIBUTION:")
        bins = [(1, 5), (6, 10), (11, 14), (15, 21), (22, 30), (31, 50), (51, 100), (101, 999)]
        for lo, hi in bins:
            count = np.sum((days >= lo) & (days <= hi))
            pct = count / len(days) * 100
            bar = "#" * int(pct / 2)
            label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            print(f"    {label:>7} days: {count:>4} ({pct:>5.1f}%) {bar}")
    else:
        print(f"  No simulations reached target!")

    print(f"\nFTMO PASS RATES:")
    print(f"  Phase 1 (10%):              {mc['phase1_pass_rate']:>6.1%}")
    print(f"  Phase 2 (5%):               {mc['phase2_pass_rate']:>6.1%}")
    print(f"  Pass within 14 cal days:    {mc['pass_within_14_cal']:>6.1%}  (10 trading days)")
    print(f"  Pass within 21 cal days:    {mc['pass_within_21_cal']:>6.1%}  (15 trading days)")
    print(f"  Pass within 30 cal days:    {mc['pass_within_30_cal']:>6.1%}  (22 trading days)")
    print(f"  Account Blown (>9% DD):     {mc['blow_rate']:>6.1%}")

    # Risk of ruin analysis
    consecutive_losses = []
    r_multiples_all = np.array([r for r, _ in all_pnls])
    rng2 = np.random.default_rng(99)
    for _ in range(N_SIMULATIONS):
        shuffled = rng2.permutation(r_multiples_all)
        max_streak = 0
        streak = 0
        for r in shuffled:
            if r < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        consecutive_losses.append(max_streak)

    cl = np.array(consecutive_losses)
    print(f"\nCONSECUTIVE LOSS STREAKS:")
    print(f"  Mean:      {cl.mean():>8.1f}")
    print(f"  Median:    {np.median(cl):>8.1f}")
    print(f"  95th pct:  {np.percentile(cl, 95):>8.1f}")
    print(f"  Max:       {cl.max():>8d}")

    # Risk-adjusted verdict
    print(f"\n{'='*60}")
    blow_rate = mc["blow_rate"]
    pass_14 = mc["pass_within_14_cal"]
    pass_rate = mc["phase1_pass_rate"]

    if pass_14 >= 0.60 and blow_rate <= 0.15:
        print("VERDICT: STRONG — >60% chance of passing within 14 calendar days")
    elif pass_rate >= 0.80 and blow_rate <= 0.20:
        print("VERDICT: VIABLE — High overall pass rate, acceptable blow risk")
    elif pass_rate >= 0.60:
        print("VERDICT: MODERATE — Good pass rate but may take longer than 14 days")
    elif blow_rate >= 0.30:
        print("VERDICT: TOO AGGRESSIVE — High blow rate, reduce risk")
    else:
        print("VERDICT: WEAK — Insufficient edge for FTMO challenge")
    print("=" * 60)


if __name__ == "__main__":
    main()
