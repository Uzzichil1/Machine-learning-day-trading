"""Live trading loop — runs the ML system on FTMO account via MT5."""

import logging
import os
import time
from datetime import datetime, timezone, timedelta

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
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
        model_dir = os.path.join(PROJECT_ROOT, "models", "saved")
        for instr in config.get("instruments", []):
            if not instr.get("enabled"):
                continue
            symbol = instr["symbol"]

            model = StackingEnsemble(config.get("model", {}))
            model.load(os.path.join(model_dir, f"{symbol}_ensemble.joblib"))
            self.models[symbol] = model

            regime_path = os.path.join(model_dir, f"{symbol}_regime.joblib")
            regime = RegimeDetector(n_states=3)
            if os.path.exists(regime_path):
                regime.load(regime_path)
                logger.info(f"Loaded regime detector for {symbol}")
            else:
                logger.warning(f"No regime detector found for {symbol} at {regime_path}")
            self.regimes[symbol] = regime

        self.poll_interval = config.get("execution", {}).get("poll_interval_seconds", 5)
        self._last_bar_time = {}
        # Track open positions: {ticket: {symbol, open_bar_time, risk_pct}}
        self._tracked_positions = {}

    def run(self):
        """Main trading loop."""
        logger.info("Starting live trading loop...")
        logger.info(f"Enabled instruments: {[s for s in self.models.keys()]}")
        logger.info(f"Risk per trade: {self.config.get('risk', {}).get('risk_per_trade_pct', 0)}%")

        if not self.connector.connect():
            logger.error("Failed to connect to MT5")
            return

        try:
            status_counter = 0
            while True:
                self._tick()
                status_counter += 1
                # Print status every 60 ticks (~5 min with 5s poll)
                if status_counter % 60 == 0:
                    self._print_status()
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

        # Sync position state with MT5 (detect closed positions + enforce time barrier)
        self._sync_positions()

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

    def _sync_positions(self):
        """Sync tracked positions with MT5 — detect closes and enforce time barrier."""
        import MetaTrader5 as mt5

        open_tickets = set()
        positions = self.executor.get_open_positions()
        for p in positions:
            open_tickets.add(p.ticket)

        # Detect positions that closed (SL/TP hit by MT5)
        closed_tickets = []
        for ticket, info in list(self._tracked_positions.items()):
            if ticket not in open_tickets:
                closed_tickets.append(ticket)
                self.risk_mgr.record_trade_closed(info["risk_pct"], 0.0)
                logger.info(f"Position #{ticket} ({info['symbol']}) closed by MT5")

        for ticket in closed_tickets:
            del self._tracked_positions[ticket]

        # Enforce time barrier: close positions older than max_holding_bars
        max_bars = self.config.get("labeling", {}).get("max_holding_bars", 8)
        tf = self.config["timeframes"]["signal"]
        # H1 = 1 hour per bar
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
        bar_minutes = tf_minutes.get(tf, 60)
        max_age = timedelta(minutes=max_bars * bar_minutes)

        now = datetime.now(timezone.utc)
        for ticket, info in list(self._tracked_positions.items()):
            age = now - info["open_time"]
            if age >= max_age and ticket in open_tickets:
                # Find the MT5 position object
                for p in positions:
                    if p.ticket == ticket:
                        result = self.executor.close_position(p)
                        if result["success"]:
                            self.risk_mgr.record_trade_closed(info["risk_pct"], 0.0)
                            del self._tracked_positions[ticket]
                            logger.info(f"Time barrier: closed #{ticket} ({info['symbol']}) after {age}")
                        break

        # Update risk manager position count from actual MT5 state
        self.risk_mgr.state.open_positions = len(self._tracked_positions)

    def _print_status(self):
        """Print periodic status summary."""
        status = self.risk_mgr.get_status_summary()
        logger.info(
            f"STATUS | Balance: ${status['balance']:,.2f} | "
            f"Equity: ${status['equity']:,.2f} | "
            f"Daily P&L: {status['daily_pnl_pct']:+.2%} | "
            f"Total P&L: {status['total_pnl_pct']:+.2%} | "
            f"Open: {status['open_positions']} | "
            f"Trades today: {status['trades_today']} | "
            f"{'HALTED' if status['halted_daily'] or status['halted_total'] else 'ACTIVE'}"
        )

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

        X = last_row[feature_cols]

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

        confidence = max(0.5, min(1.0, confidence))

        # Check regime — fetch H4 data and compute scalar
        regime_scalar = 1.0
        regime_detector = self.regimes.get(symbol)
        if regime_detector is not None and regime_detector._fitted:
            try:
                h4_tf = self.config["timeframes"]["regime"]
                h4_bars = self.connector.get_recent_bars(symbol, h4_tf, 100)
                if not h4_bars.empty:
                    h4_scalars = regime_detector.get_size_scalar(h4_bars)
                    if len(h4_scalars) > 0:
                        regime_scalar = float(h4_scalars.iloc[-1])
                        if regime_scalar <= 0:
                            logger.info(f"{symbol}: Regime scalar = 0 (high vol chaos), skipping")
                            return
            except Exception as e:
                logger.warning(f"{symbol}: Regime check failed: {e}")

        # Check if we can still trade
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            logger.info(f"{symbol}: Signal generated but blocked: {reason}")
            return

        # Calculate position size
        atr = df["atr"].iloc[-1]
        pip_value = instr_config.get("pip_value", 10.0)

        # Get broker volume constraints for proper lot rounding
        vol_constraints = self.executor.get_volume_constraints(symbol)

        trade_req = TradeRequest(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            stop_distance_pips=atr * self.config["risk"]["stop_loss_atr_multiple"],
            atr=atr,
            regime_scalar=regime_scalar,
        )

        lots, sl_dist, tp_dist = self.risk_mgr.calculate_position_size(
            trade_req, pip_value, **vol_constraints
        )

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
            # Track position for close detection and time barrier
            self._tracked_positions[result["order_id"]] = {
                "symbol": symbol,
                "open_time": datetime.now(timezone.utc),
                "risk_pct": risk_pct,
                "direction": direction,
                "lots": lots,
            }
            logger.info(
                f"TRADE OPENED: {symbol} {'BUY' if direction == 1 else 'SELL'} "
                f"{lots} lots | Confidence: {confidence:.2f} | "
                f"Risk: {risk_pct:.3%} | Regime: {regime_scalar:.1f}"
            )
        else:
            logger.error(f"TRADE FAILED: {symbol} - {result.get('error')}")

    def emergency_close_all(self):
        """Emergency: close all positions immediately."""
        logger.critical("EMERGENCY CLOSE ALL POSITIONS")
        self.executor.close_all_positions()


if __name__ == "__main__":
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "trading.log")),
        ],
    )

    config = load_config()

    # Set initial balance from config
    risk_config = config.get("risk", {})
    risk_config["initial_balance"] = config.get("account", {}).get("initial_balance", 100_000)

    trader = LiveTrader(config)
    trader.run()
