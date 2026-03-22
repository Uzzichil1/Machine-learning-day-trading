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
from src.backtest.engine import FTMOBacktester
from src.pipeline import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING)

config = load_config()
N_SIMULATIONS = 1000


def get_trade_pnls(symbol: str) -> list[float]:
    """Get all trade P&Ls from backtesting a symbol."""
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

    # Regime
    regime_scalars = np.ones(len(test_df))
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    if not h4_data.empty:
        regime = RegimeDetector(n_states=3)
        regime.fit(h4_data)
        if regime._fitted:
            try:
                h4_scalars = regime.get_size_scalar(h4_data)
                h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
                test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
                regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
                regime_scalars = regime_reindexed.values
            except Exception:
                pass

    # Generate signals
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    median_proba = np.median(probas)

    signals = pd.DataFrame(index=test_df.index)
    signals["signal"] = 0
    signals.loc[probas >= median_proba + signal_offset, "signal"] = 1
    signals.loc[probas <= median_proba - signal_offset, "signal"] = -1
    signals["confidence"] = np.where(
        signals["signal"] == 1,
        (probas - median_proba) / (1 - median_proba),
        np.where(signals["signal"] == -1, (median_proba - probas) / median_proba, 0.5),
    ).clip(0.3, 1.0)
    signals["atr"] = test_df["atr"].values
    signals["regime_scalar"] = regime_scalars
    signals["symbol"] = symbol

    # Run backtest to get trade results
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

    return [t.pnl for t in result.trades]


def run_monte_carlo(trade_pnls: list[float], n_sims: int, initial_balance: float = 100_000):
    """Run Monte Carlo simulation by reshuffling trade order."""
    rng = np.random.default_rng(42)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    returns = []
    max_dds = []
    sharpes = []
    phase1_passes = 0
    phase2_passes = 0
    blown = 0  # Hit 10% total DD

    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled) + initial_balance

        # Return
        total_ret = (equity[-1] - initial_balance) / initial_balance
        returns.append(total_ret)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = abs(dd.min())
        max_dds.append(max_dd)

        # Phase checks
        if total_ret >= 0.10:
            phase1_passes += 1
        if total_ret >= 0.05:
            phase2_passes += 1
        if max_dd >= 0.10:
            blown += 1

    returns = np.array(returns)
    max_dds = np.array(max_dds)

    return {
        "returns": returns,
        "max_dds": max_dds,
        "phase1_pass_rate": phase1_passes / n_sims,
        "phase2_pass_rate": phase2_passes / n_sims,
        "blow_rate": blown / n_sims,
    }


def main():
    print("=" * 60)
    print("MONTE CARLO STRESS TEST")
    print(f"Simulations: {N_SIMULATIONS}")
    print("=" * 60)

    all_pnls = []
    for instr in config.get("instruments", []):
        if not instr.get("enabled"):
            continue
        symbol = instr["symbol"]
        print(f"\nLoading trades for {symbol}...")
        pnls = get_trade_pnls(symbol)
        print(f"  {len(pnls)} trades, total P&L: ${sum(pnls):,.2f}")
        all_pnls.extend(pnls)

    print(f"\nCombined: {len(all_pnls)} trades, total P&L: ${sum(all_pnls):,.2f}")
    print(f"Running {N_SIMULATIONS} Monte Carlo simulations...\n")

    mc = run_monte_carlo(all_pnls, N_SIMULATIONS)

    returns = mc["returns"]
    max_dds = mc["max_dds"]

    print("=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
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

    print(f"\nMAX DRAWDOWN DISTRIBUTION:")
    print(f"  Mean:      {max_dds.mean():>8.2%}")
    print(f"  Median:    {np.median(max_dds):>8.2%}")
    print(f"  5th pct:   {np.percentile(max_dds, 5):>8.2%}")
    print(f"  25th pct:  {np.percentile(max_dds, 25):>8.2%}")
    print(f"  75th pct:  {np.percentile(max_dds, 75):>8.2%}")
    print(f"  95th pct:  {np.percentile(max_dds, 95):>8.2%}")
    print(f"  Max:       {max_dds.max():>8.2%}")

    print(f"\nFTMO PASS RATES:")
    print(f"  Phase 1 (10%): {mc['phase1_pass_rate']:>6.1%}")
    print(f"  Phase 2 (5%):  {mc['phase2_pass_rate']:>6.1%}")
    print(f"  Account Blown: {mc['blow_rate']:>6.1%}")

    # Risk of ruin analysis
    consecutive_losses = []
    for _ in range(N_SIMULATIONS):
        rng = np.random.default_rng()
        shuffled = rng.permutation(all_pnls)
        max_streak = 0
        streak = 0
        for pnl in shuffled:
            if pnl < 0:
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

    print("\n" + "=" * 60)
    # Overall assessment
    if mc["phase1_pass_rate"] >= 0.95 and mc["blow_rate"] <= 0.05:
        print("VERDICT: STRATEGY IS ROBUST - High confidence FTMO pass")
    elif mc["phase1_pass_rate"] >= 0.80 and mc["blow_rate"] <= 0.10:
        print("VERDICT: STRATEGY IS VIABLE - Good FTMO pass probability")
    elif mc["phase1_pass_rate"] >= 0.60:
        print("VERDICT: STRATEGY NEEDS IMPROVEMENT - Moderate FTMO pass probability")
    else:
        print("VERDICT: STRATEGY IS WEAK - Low FTMO pass probability")
    print("=" * 60)


if __name__ == "__main__":
    main()
