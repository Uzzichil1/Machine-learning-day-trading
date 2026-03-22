"""Main ML pipeline — data download, feature engineering, training, backtesting, and visualization."""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.backtest.engine import FTMOBacktester
from src.visualization.charts import (
    plot_equity_curve, plot_monthly_heatmap, plot_trade_distribution,
    plot_win_rate_over_time, plot_feature_importance, plot_confusion_matrix,
    plot_regime_overlay, plot_daily_pnl_bars, plot_cumulative_trades,
    plot_risk_metrics_gauge, plot_hourly_performance, plot_ftmo_summary_card,
    plot_correlation_heatmap, generate_full_report,
)

logger = logging.getLogger(__name__)

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class MLPipeline:
    """End-to-end ML pipeline for FTMO trading system."""

    def __init__(self, config: dict):
        self.config = config
        self.connector = MT5Connector()
        self.feature_eng = FeatureEngineer(config.get("features", {}))
        self.ensemble = StackingEnsemble(config.get("model", {}))
        self.regime = RegimeDetector(n_states=3)
        self.backtester = FTMOBacktester({
            **config.get("account", {}),
            **config.get("ftmo_limits", {}),
        })
        self.data_dir = str(PROJECT_ROOT / "data" / "raw")
        self.model_dir = str(PROJECT_ROOT / "models" / "saved")
        self.report_dir = str(PROJECT_ROOT / "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    # ── DATA ─────────────────────────────────────────────────────────

    def download_data(self):
        """Download historical data for all instruments from MT5."""
        logger.info("=" * 60)
        logger.info("STEP 1: DOWNLOADING HISTORICAL DATA")
        logger.info("=" * 60)

        if not self.connector.connect():
            logger.error("Cannot connect to MT5. Is the terminal running?")
            return False

        years = self.config.get("data", {}).get("history_years", 4)
        date_to = datetime.now()
        date_from = date_to - timedelta(days=years * 365)

        for instr in self.config.get("instruments", []):
            if not instr.get("enabled"):
                continue
            symbol = instr["symbol"]
            for tf_key in ["signal", "regime"]:
                tf = self.config["timeframes"][tf_key]
                logger.info(f"Downloading {symbol} {tf} from {date_from.date()} to {date_to.date()}...")
                df = self.connector.get_rates(symbol, tf, date_from, date_to)
                if not df.empty:
                    self.connector.save_data(df, symbol, tf, self.data_dir)
                else:
                    logger.warning(f"No data returned for {symbol} {tf}")

        self.connector.disconnect()
        logger.info("Data download complete\n")
        return True

    # ── FEATURES ─────────────────────────────────────────────────────

    def prepare_features(self, symbol: str) -> pd.DataFrame:
        """Load data, compute features, and create labels."""
        tf = self.config["timeframes"]["signal"]
        df = self.connector.load_data(symbol, tf, self.data_dir)

        if df.empty:
            logger.error(f"No data for {symbol}. Run download_data() first.")
            return pd.DataFrame()

        logger.info(f"{symbol}: Loaded {len(df)} raw bars")

        # Compute features
        df = self.feature_eng.compute_all(df)

        # Create labels
        label_cfg = self.config.get("labeling", {})
        df["target"] = triple_barrier_labels(
            df,
            upper_barrier_atr=label_cfg.get("upper_barrier_atr", 1.5),
            lower_barrier_atr=label_cfg.get("lower_barrier_atr", 1.5),
            max_holding_bars=label_cfg.get("max_holding_bars", 8),
        )

        # Drop rows with NaN features
        feature_cols = self.feature_eng.get_feature_columns(df)
        df = df.dropna(subset=feature_cols + ["target"])

        # Remove neutral labels (0) for binary classification
        df = df[df["target"] != 0]

        # Remap -1 to 0 for binary classification (1=buy, 0=sell)
        df["target"] = (df["target"] == 1).astype(int)

        logger.info(f"{symbol}: {len(df)} labeled samples | Buy ratio: {df['target'].mean():.2%}")
        return df

    # ── TRAINING ─────────────────────────────────────────────────────

    def train(self, symbol: str) -> dict:
        """Train the ensemble model for a symbol with full metrics and visuals."""
        logger.info("=" * 60)
        logger.info(f"STEP 2: TRAINING MODEL — {symbol}")
        logger.info("=" * 60)

        df = self.prepare_features(symbol)
        if df.empty:
            return {}

        feature_cols = self.feature_eng.get_feature_columns(df)
        X = df[feature_cols].values
        y = df["target"].values

        # Time-based split
        data_cfg = self.config.get("data", {})
        train_pct = data_cfg.get("train_pct", 0.6)
        val_pct = data_cfg.get("validation_pct", 0.2)

        n = len(X)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test(OOS)={len(X_test)}")

        # Train ensemble
        self.ensemble.fit(X_train, y_train)

        # Evaluate on validation
        val_proba = self.ensemble.predict_proba(X_val)
        val_preds = (val_proba >= 0.5).astype(int)
        val_acc = (val_preds == y_val).mean()

        # Calibrate median_proba from validation set (NOT test set — prevents leakage)
        self.ensemble._median_proba = float(np.median(val_proba))
        logger.info(f"Calibrated median_proba from validation set: {self.ensemble._median_proba:.4f}")

        # Evaluate on test (OOS)
        test_proba = self.ensemble.predict_proba(X_test)
        test_preds = (test_proba >= 0.5).astype(int)
        test_acc = (test_preds == y_test).mean()

        logger.info(f"Validation accuracy: {val_acc:.4f}")
        logger.info(f"Test (OOS) accuracy: {test_acc:.4f}")

        # Feature importance
        importance = self.ensemble.get_feature_importance(feature_cols)
        top_features = list(importance.items())[:15]
        logger.info("Top 10 features:")
        for feat, imp in top_features[:10]:
            logger.info(f"  {feat}: {imp}")

        # Save model (includes median_proba)
        save_path = os.path.join(self.model_dir, f"{symbol}_ensemble.joblib")
        self.ensemble.save(save_path)

        # Train regime detector on TRAINING-PERIOD H4 data only (no future leakage)
        h4_data = self.connector.load_data(
            symbol, self.config["timeframes"]["regime"], self.data_dir
        )
        if not h4_data.empty:
            # Use same temporal split ratio for H4 as H1
            h4_train_end = int(len(h4_data) * train_pct)
            h4_train = h4_data.iloc[:h4_train_end]
            self.regime.fit(h4_train)
            regime_path = os.path.join(self.model_dir, f"{symbol}_regime.joblib")
            self.regime.save(regime_path)
            logger.info(f"Regime detector fitted on training H4 data only ({len(h4_train)}/{len(h4_data)} bars)")

        # ── GENERATE TRAINING VISUALS ──
        sym_report_dir = os.path.join(self.report_dir, symbol)
        os.makedirs(sym_report_dir, exist_ok=True)

        # Confusion matrix
        fig_cm = plot_confusion_matrix(y_test, test_preds)
        fig_cm.write_html(os.path.join(sym_report_dir, "confusion_matrix.html"), include_plotlyjs="cdn")

        # Feature importance
        fig_fi = plot_feature_importance(importance)
        fig_fi.write_html(os.path.join(sym_report_dir, "feature_importance.html"), include_plotlyjs="cdn")

        # Feature correlation heatmap (on top 20 features)
        top_feat_names = [f for f, _ in top_features[:20]]
        if len(top_feat_names) > 5:
            fig_corr = plot_correlation_heatmap(df, top_feat_names)
            fig_corr.write_html(os.path.join(sym_report_dir, "feature_correlation.html"), include_plotlyjs="cdn")

        # Regime overlay on price chart
        if not h4_data.empty and self.regime._fitted:
            try:
                regimes = self.regime.predict_regime(h4_data)
                # Use last 500 bars for readability
                n_bars = min(500, len(h4_data), len(regimes))
                fig_regime = plot_regime_overlay(h4_data.iloc[-n_bars:], regimes.iloc[-n_bars:])
                fig_regime.write_html(os.path.join(sym_report_dir, "regime_overlay.html"), include_plotlyjs="cdn")
            except Exception as e:
                logger.warning(f"Regime overlay failed: {e}")

        logger.info(f"Training visuals saved to {sym_report_dir}/\n")

        return {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "top_features": top_features,
            "importance": importance,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "y_test": y_test,
            "test_preds": test_preds,
        }

    # ── BACKTESTING ──────────────────────────────────────────────────

    def backtest(self, symbol: str, metrics: dict = None) -> dict:
        """Run FTMO backtest on OOS data with full visualization report."""
        logger.info("=" * 60)
        logger.info(f"STEP 3: BACKTESTING — {symbol}")
        logger.info("=" * 60)

        df = self.prepare_features(symbol)
        if df.empty:
            return {}

        feature_cols = self.feature_eng.get_feature_columns(df)
        data_cfg = self.config.get("data", {})
        n = len(df)
        test_start = int(n * (data_cfg.get("train_pct", 0.6) + data_cfg.get("validation_pct", 0.2)))

        test_df = df.iloc[test_start:]
        X_test = test_df[feature_cols].values

        # Load trained model
        model_path = os.path.join(self.model_dir, f"{symbol}_ensemble.joblib")
        self.ensemble.load(model_path)

        # Generate signals
        probas = self.ensemble.predict_proba(X_test)

        # Load saved regime detector (fitted on training data only)
        regime_scalars = np.ones(len(test_df))
        regime_path = os.path.join(self.model_dir, f"{symbol}_regime.joblib")
        h4_data = self.connector.load_data(
            symbol, self.config["timeframes"]["regime"], self.data_dir
        )
        if os.path.exists(regime_path) and not h4_data.empty:
            try:
                self.regime.load(regime_path)
                h4_scalars = self.regime.get_size_scalar(h4_data)
                h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
                test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
                regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
                regime_scalars = regime_reindexed.values
                logger.info("Regime scalars integrated (model fitted on training data only)")
            except Exception as e:
                logger.warning(f"Regime integration failed: {e}")

        # Build signal DataFrame — use validation-calibrated median (no test set leakage)
        risk_cfg = self.config.get("risk", {})
        signals = pd.DataFrame(index=test_df.index)
        signals["confidence"] = probas
        signals["signal"] = 0
        median_proba = self.ensemble._median_proba  # From validation set, not test
        signal_offset = risk_cfg.get("signal_offset", 0.02)
        buy_thresh = median_proba + signal_offset
        sell_thresh = median_proba - signal_offset
        signals.loc[probas >= buy_thresh, "signal"] = 1
        signals.loc[probas <= sell_thresh, "signal"] = -1
        signals["confidence"] = np.where(
            signals["signal"] == 1,
            (probas - median_proba) / (1 - median_proba),
            np.where(
                signals["signal"] == -1,
                (median_proba - probas) / median_proba,
                0.5
            )
        ).clip(0.5, 1.0)
        logger.info(f"Probability stats: median={median_proba:.4f} (from validation) | buy_thresh={buy_thresh:.4f} | sell_thresh={sell_thresh:.4f}")
        signals["atr"] = test_df["atr"].values
        signals["regime_scalar"] = regime_scalars
        signals["symbol"] = symbol

        signal_count = (signals["signal"] != 0).sum()
        logger.info(f"Generated {signal_count} signals ({(signals['signal']==1).sum()} buy, {(signals['signal']==-1).sum()} sell)")
        result = self.backtester.run(
            signals, test_df,
            risk_per_trade=risk_cfg.get("risk_per_trade_pct", 0.25) / 100,
            sl_atr_mult=risk_cfg.get("stop_loss_atr_multiple", 1.5),
            tp_atr_mult=risk_cfg.get("take_profit_atr_multiple", 2.25),
        )

        self.backtester.print_summary(result)

        # ── GENERATE BACKTEST VISUALS ──
        sym_report_dir = os.path.join(self.report_dir, symbol)
        os.makedirs(sym_report_dir, exist_ok=True)

        importance = metrics.get("importance", {}) if metrics else {}

        generate_full_report(
            result=result,
            trades=result.trades,
            daily_pnl=result.daily_pnl if result.daily_pnl is not None else pd.Series(dtype=float),
            feature_importance=importance,
            output_dir=sym_report_dir,
        )
        logger.info(f"Full backtest report saved to {sym_report_dir}/full_report.html\n")

        return result

    # ── FULL RUN ─────────────────────────────────────────────────────

    def run_all(self):
        """Execute the complete pipeline: download → train → backtest → report."""
        # Step 1: Download
        if not self.download_data():
            logger.error("Data download failed. Aborting.")
            return

        # Step 2+3: Train and backtest each instrument
        all_results = {}
        for instr in self.config.get("instruments", []):
            if not instr.get("enabled"):
                continue
            symbol = instr["symbol"]

            metrics = self.train(symbol)
            if metrics:
                result = self.backtest(symbol, metrics)
                all_results[symbol] = result

        # Print combined summary
        logger.info("\n" + "=" * 60)
        logger.info("COMBINED RESULTS ACROSS ALL INSTRUMENTS")
        logger.info("=" * 60)
        for symbol, result in all_results.items():
            status = "PASS" if result.ftmo_phase1_passed else "FAIL"
            logger.info(
                f"  {symbol:>8}: {result.total_return:>8.2%} return | "
                f"Sharpe: {result.sharpe_ratio:.2f} | "
                f"MaxDD: {result.max_drawdown:.2%} | "
                f"WR: {result.win_rate:.1%} | "
                f"Trades: {result.total_trades} | "
                f"FTMO: {status}"
            )

        logger.info("\nReports saved to: reports/")
        logger.info("Open reports/<SYMBOL>/full_report.html in your browser for interactive charts.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(PROJECT_ROOT / "logs" / "pipeline.log"), mode="w"),
        ],
    )

    config = load_config()
    pipeline = MLPipeline(config)
    pipeline.run_all()
