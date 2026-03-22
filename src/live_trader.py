"""Live trading loop — runs the ML system on FTMO account via MT5."""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.risk.manager import RiskManager, TradeRequest
from src.execution.mt5_executor import MT5Executor

logger = logging.getLogger(__name__)


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class LiveTrader:
    """Runs the ML trading system live against MT5."""

    def __init__(self, config: dict):
        self.config = config
        self.connector = MT5Connector()
        self.feature_eng = FeatureEngineer(config.get("features", {}))
        self.risk_mgr = RiskManager(config.get("risk", {}))
        self.executor = MT5Executor(config.get("execution", {}))

        # Load models per instrument
        self.models = {}
        self.regimes = {}
        for instr in config.get("instruments", []):
            if not instr.get("enabled"):
                continue
            symbol = instr["symbol"]
            model = StackingEnsemble()
            model.load(f"models/saved/{symbol}_ensemble.joblib")
            self.models[symbol] = model

            regime = RegimeDetector(n_states=3)
            # Regime model would be loaded from saved state
            self.regimes[symbol] = regime

        self.poll_interval = config.get("execution", {}).get("poll_interval_seconds", 5)
        self._last_bar_time = {}

    def run(self):
        """Main trading loop."""
        logger.info("Starting live trading loop...")

        if not self.connector.connect():
            logger.error("Failed to connect to MT5")
            return

        try:
            while True:
                self._tick()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.connector.disconnect()

    def _tick(self):
        """Single iteration of the trading loop."""
        # Update account state
        account = self.connector.get_account_info()
        if not account:
            logger.warning("Cannot get account info — MT5 connection issue")
            return

        self.risk_mgr.update_state(account["balance"], account["equity"])

        # Check if we can trade at all
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.debug(f"Trading blocked: {reason}")
            return

        # Process each instrument
        for instr in self.config.get("instruments", []):
            if not instr.get("enabled"):
                continue
            symbol = instr["symbol"]
            self._process_instrument(symbol, instr)

    def _process_instrument(self, symbol: str, instr_config: dict):
        """Check for new bar and generate signal for one instrument."""
        # Check if we're in the trading session
        now_utc = datetime.now(timezone.utc)
        session = instr_config.get("session_utc", [0, 24])
        if not (session[0] <= now_utc.hour < session[1]):
            return

        # Get latest bars
        tf = self.config["timeframes"]["signal"]
        bars_needed = 250  # Enough for all indicator calculations
        df = self.connector.get_recent_bars(symbol, tf, bars_needed)

        if df.empty or len(df) < bars_needed:
            return

        # Check if this is a new bar (avoid re-processing same bar)
        latest_time = df.index[-1]
        if symbol in self._last_bar_time and latest_time <= self._last_bar_time[symbol]:
            return
        self._last_bar_time[symbol] = latest_time

        logger.info(f"New bar: {symbol} {tf} @ {latest_time}")

        # Compute features
        df = self.feature_eng.compute_all(df)
        feature_cols = self.feature_eng.get_feature_columns(df)

        # Drop NaN rows but keep the last row (current bar)
        last_row = df.iloc[-1:]
        if last_row[feature_cols].isna().any(axis=1).iloc[0]:
            logger.debug(f"{symbol}: Features contain NaN, skipping")
            return

        X = last_row[feature_cols].values

        # Get model prediction
        model = self.models.get(symbol)
        if model is None:
            return

        proba = model.predict_proba(X)[0]

        # Use median-centered thresholds (calibrated during training)
        # The model's median output is ~0.525 for XAUUSD, so fixed thresholds
        # create massive directional bias. Instead, offset from median.
        signal_offset = self.config.get("risk", {}).get("signal_offset", 0.02)
        # Use stored median or estimate from recent predictions
        median_proba = getattr(model, '_median_proba', 0.5)
        buy_thresh = median_proba + signal_offset
        sell_thresh = median_proba - signal_offset

        if proba >= buy_thresh:
            direction = 1  # Buy
            confidence = (proba - median_proba) / (1 - median_proba)
        elif proba <= sell_thresh:
            direction = -1  # Sell
            confidence = (median_proba - proba) / median_proba
        else:
            logger.debug(f"{symbol}: No signal (proba={proba:.3f}, median={median_proba:.3f})")
            return

        confidence = max(0.3, min(1.0, confidence))

        # Check regime
        regime_scalar = 1.0  # Default if regime not trained
        # TODO: integrate regime gating here

        # Check if we can still trade
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.info(f"{symbol}: Signal generated but blocked: {reason}")
            return

        # Calculate position size
        atr = df["atr"].iloc[-1]
        pip_value = instr_config.get("pip_value", 10.0)

        trade_req = TradeRequest(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            stop_distance_pips=atr * self.config["risk"]["stop_loss_atr_multiple"],
            atr=atr,
            regime_scalar=regime_scalar,
        )

        lots, sl_dist, tp_dist = self.risk_mgr.calculate_position_size(trade_req, pip_value)

        if lots <= 0:
            logger.info(f"{symbol}: Position size too small, skipping")
            return

        # Execute trade
        result = self.executor.open_trade(
            symbol=symbol,
            direction=direction,
            lots=lots,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            comment=f"sig_{proba:.2f}",
        )

        if result["success"]:
            risk_pct = lots * sl_dist * pip_value / self.risk_mgr.state.current_balance
            self.risk_mgr.record_trade_opened(risk_pct)
            logger.info(
                f"TRADE OPENED: {symbol} {'BUY' if direction == 1 else 'SELL'} "
                f"{lots} lots | Confidence: {confidence:.2f} | "
                f"Risk: {risk_pct:.3%}"
            )
        else:
            logger.error(f"TRADE FAILED: {symbol} - {result.get('error')}")

    def emergency_close_all(self):
        """Emergency: close all positions immediately."""
        logger.critical("EMERGENCY CLOSE ALL POSITIONS")
        self.executor.close_all_positions()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/trading.log"),
        ],
    )

    config = load_config()

    # Set initial balance from config
    risk_config = config.get("risk", {})
    risk_config["initial_balance"] = config.get("account", {}).get("initial_balance", 100_000)

    trader = LiveTrader(config)
    trader.run()
