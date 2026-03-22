"""Grid search over risk/reward parameters to find fastest FTMO-safe settings."""

import sys
import os
import itertools
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


def prepare_data(symbol: str, use_validation: bool = True):
    """Prepare data and trained model for a symbol.

    Args:
        use_validation: If True, return VALIDATION set (for grid search).
                       If False, return TEST set (for final evaluation).
    """
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))

    tf = config["timeframes"]["signal"]
    data_dir = str(PROJECT_ROOT / "data" / "raw")
    df = connector.load_data(symbol, tf, data_dir)

    if df.empty:
        return None, None, None, None, None

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
    train_end = int(n * data_cfg.get("train_pct", 0.6))
    val_end = int(n * (data_cfg.get("train_pct", 0.6) + data_cfg.get("validation_pct", 0.2)))

    if use_validation:
        eval_df = df.iloc[train_end:val_end]  # VALIDATION set for grid search
    else:
        eval_df = df.iloc[val_end:]  # TEST set for final evaluation

    X_eval = eval_df[feature_cols].values

    # Load trained model
    ensemble = StackingEnsemble(config.get("model", {}))
    model_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_ensemble.joblib")
    ensemble.load(model_path)

    probas = ensemble.predict_proba(X_eval)

    # Load saved regime detector (fitted on training data only)
    regime_scalars = np.ones(len(eval_df))
    regime_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_regime.joblib")
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    if os.path.exists(regime_path) and not h4_data.empty:
        regime = RegimeDetector(n_states=3)
        regime.load(regime_path)
        try:
            h4_scalars = regime.get_size_scalar(h4_data)
            h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
            eval_idx = eval_df.index.tz_localize(None) if eval_df.index.tz else eval_df.index
            regime_reindexed = h4_scalars.reindex(eval_idx, method="ffill").fillna(1.0)
            regime_scalars = regime_reindexed.values
        except Exception:
            pass

    return eval_df, probas, regime_scalars, feature_cols, ensemble


def run_backtest_with_params(
    test_df, probas, regime_scalars, symbol,
    risk_pct, sl_atr_mult, tp_atr_mult, signal_offset,
    median_proba=0.5,
):
    """Run a single backtest with given parameters."""
    signals = pd.DataFrame(index=test_df.index)
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset

    signals["signal"] = 0
    signals.loc[probas >= buy_thresh, "signal"] = 1
    signals.loc[probas <= sell_thresh, "signal"] = -1

    signals["confidence"] = np.where(
        signals["signal"] == 1,
        (probas - median_proba) / (1 - median_proba),
        np.where(
            signals["signal"] == -1,
            (median_proba - probas) / median_proba,
            0.5,
        ),
    ).clip(0.3, 1.0)

    signals["atr"] = test_df["atr"].values
    signals["regime_scalar"] = regime_scalars
    signals["symbol"] = symbol

    backtester = FTMOBacktester({
        **config.get("account", {}),
        **config.get("ftmo_limits", {}),
    })

    result = backtester.run(
        signals, test_df,
        risk_per_trade=risk_pct / 100,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
    )

    return result


def main():
    symbols = ["XAUUSD", "USTEC"]

    # Grid search parameters
    risk_pcts = [0.25, 0.30, 0.35, 0.40, 0.45]
    sl_atr_mults = [1.5, 2.0]
    tp_atr_mults = [2.0, 2.25, 2.5, 3.0]
    signal_offsets = [0.01, 0.015, 0.02, 0.025, 0.03]

    max_dd_limit = 0.07  # Keep well under FTMO's 10%
    max_daily_dd_limit = 0.035  # Keep under FTMO's 5%

    results = []

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING ON VALIDATION SET: {symbol}")
        print(f"{'='*70}")

        # Grid search on VALIDATION set (not test set — prevents overfitting)
        val_data = prepare_data(symbol, use_validation=True)
        val_df, val_probas, val_regime_scalars, feature_cols, ensemble = val_data

        if val_df is None:
            print(f"  No data for {symbol}, skipping")
            continue

        print(f"  Validation samples: {len(val_df)}")
        print(f"  Probability range: [{val_probas.min():.4f}, {val_probas.max():.4f}]")
        print(f"  Model median_proba (from val calibration): {ensemble._median_proba:.4f}")

        total_combos = len(risk_pcts) * len(sl_atr_mults) * len(tp_atr_mults) * len(signal_offsets)
        print(f"  Running {total_combos} parameter combinations on validation set...")

        combo_num = 0
        for risk_pct, sl_mult, tp_mult, sig_off in itertools.product(
            risk_pcts, sl_atr_mults, tp_atr_mults, signal_offsets
        ):
            combo_num += 1
            if combo_num % 50 == 0:
                print(f"  ... {combo_num}/{total_combos}")

            # Skip if TP < SL (negative R:R)
            if tp_mult < sl_mult:
                continue

            result = run_backtest_with_params(
                val_df, val_probas, val_regime_scalars, symbol,
                risk_pct, sl_mult, tp_mult, sig_off,
                median_proba=ensemble._median_proba,
            )

            results.append({
                "symbol": symbol,
                "risk_pct": risk_pct,
                "sl_atr": sl_mult,
                "tp_atr": tp_mult,
                "signal_offset": sig_off,
                "rr_ratio": tp_mult / sl_mult,
                "total_return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_dd": result.max_drawdown,
                "max_daily_dd": result.max_daily_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "days_to_target": result.days_to_target,
                "phase1_passed": result.ftmo_phase1_passed,
                "best_day_pct": result.best_day_pct,
                "daily_breached": result.ftmo_daily_limit_breached,
                "total_breached": result.ftmo_total_limit_breached,
            })

    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)

    for symbol in symbols:
        sym_df = df[df["symbol"] == symbol].copy()
        if sym_df.empty:
            continue

        print(f"\n{'='*70}")
        print(f"RESULTS: {symbol}")
        print(f"{'='*70}")

        # Filter: FTMO-safe (MaxDD < 7%, daily DD < 3.5%, no breaches)
        safe = sym_df[
            (sym_df["max_dd"] < max_dd_limit)
            & (sym_df["max_daily_dd"] < max_daily_dd_limit)
            & (~sym_df["daily_breached"])
            & (~sym_df["total_breached"])
            & (sym_df["total_trades"] >= 20)  # Need enough trades
        ].copy()

        print(f"\n  Total combinations tested: {len(sym_df)}")
        print(f"  FTMO-safe combinations (MaxDD<{max_dd_limit*100}%, DailyDD<{max_daily_dd_limit*100}%): {len(safe)}")

        if safe.empty:
            print("  No safe combinations found! Relaxing constraints...")
            safe = sym_df[
                (sym_df["max_dd"] < 0.09)
                & (~sym_df["total_breached"])
                & (sym_df["total_trades"] >= 20)
            ].copy()
            print(f"  Relaxed safe combinations (MaxDD<9%): {len(safe)}")

        if not safe.empty:
            # Sort by: passed Phase 1, then fastest (days_to_target), then best Sharpe
            passed = safe[safe["phase1_passed"]].copy()

            if not passed.empty:
                # Among passed: fastest first, then best Sharpe for ties
                passed = passed.sort_values(
                    ["days_to_target", "sharpe"], ascending=[True, False]
                )
                print(f"\n  Phase 1 PASSED combinations: {len(passed)}")
                print("\n  TOP 10 FASTEST FTMO-SAFE CONFIGURATIONS:")
                print("  " + "-" * 120)
                print(f"  {'Risk%':>6} {'SL_ATR':>7} {'TP_ATR':>7} {'R:R':>5} {'SigOff':>7} "
                      f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'DailyDD':>8} "
                      f"{'WR':>6} {'PF':>6} {'Trades':>7} {'Days':>6} {'BestDay':>8}")
                print("  " + "-" * 120)

                for _, row in passed.head(10).iterrows():
                    print(
                        f"  {row['risk_pct']:>6.2f} {row['sl_atr']:>7.1f} {row['tp_atr']:>7.2f} "
                        f"{row['rr_ratio']:>5.2f} {row['signal_offset']:>7.3f} "
                        f"{row['total_return']:>7.2%} {row['sharpe']:>7.2f} "
                        f"{row['max_dd']:>6.2%} {row['max_daily_dd']:>7.2%} "
                        f"{row['win_rate']:>5.1%} {row['profit_factor']:>6.2f} "
                        f"{row['total_trades']:>7d} {row['days_to_target']:>6d} "
                        f"{row['best_day_pct']:>7.1%}"
                    )

                # Print the BEST overall recommendation
                best = passed.iloc[0]
                print(f"\n  RECOMMENDED CONFIG for {symbol}:")
                print(f"    risk_per_trade_pct: {best['risk_pct']}")
                print(f"    stop_loss_atr_multiple: {best['sl_atr']}")
                print(f"    take_profit_atr_multiple: {best['tp_atr']}")
                print(f"    signal_offset: {best['signal_offset']}")
                print(f"    Expected: {best['total_return']:.2%} return in {best['days_to_target']} days, "
                      f"MaxDD={best['max_dd']:.2%}, Sharpe={best['sharpe']:.2f}")
            else:
                print("\n  No combinations passed Phase 1 target.")
                # Show best return instead
                best_ret = safe.sort_values("total_return", ascending=False)
                print("\n  TOP 5 BY RETURN (didn't reach 10% target):")
                for _, row in best_ret.head(5).iterrows():
                    print(
                        f"    Risk={row['risk_pct']:.2f}% SL={row['sl_atr']:.1f} TP={row['tp_atr']:.2f} "
                        f"Sig={row['signal_offset']:.3f} -> "
                        f"Return={row['total_return']:.2%} MaxDD={row['max_dd']:.2%} "
                        f"Sharpe={row['sharpe']:.2f} Trades={row['total_trades']}"
                    )

        # Also show absolute best (ignoring safety) for reference
        print(f"\n  REFERENCE — Absolute best return (any MaxDD):")
        abs_best = sym_df.sort_values("total_return", ascending=False).iloc[0]
        print(
            f"    Risk={abs_best['risk_pct']:.2f}% SL={abs_best['sl_atr']:.1f} "
            f"TP={abs_best['tp_atr']:.2f} Sig={abs_best['signal_offset']:.3f} -> "
            f"Return={abs_best['total_return']:.2%} MaxDD={abs_best['max_dd']:.2%} "
            f"Days={abs_best['days_to_target']}"
        )

    # Save full results
    output_path = str(PROJECT_ROOT / "reports" / "optimization_grid.csv")
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
