"""FTMO-compliant risk management with safety buffers."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

logger = logging.getLogger(__name__)


@dataclass
class TradeRequest:
    symbol: str
    direction: int  # 1 = buy, -1 = sell
    confidence: float  # Model confidence (0.5 - 1.0)
    stop_distance_pips: float
    atr: float
    regime_scalar: float = 1.0


@dataclass
class RiskState:
    """Tracks real-time risk state for FTMO compliance."""
    initial_balance: float = 100_000.0
    current_balance: float = 100_000.0
    current_equity: float = 100_000.0
    day_start_balance: float = 100_000.0
    peak_balance: float = 100_000.0
    current_date: date = field(default_factory=date.today)
    trades_today: int = 0
    open_positions: int = 0
    open_risk_pct: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    is_halted_daily: bool = False
    is_halted_total: bool = False
    daily_profits: dict = field(default_factory=dict)  # date → profit


class RiskManager:
    """Enforces FTMO risk limits with safety buffers."""

    def __init__(self, config: dict):
        self.risk_per_trade = config.get("risk_per_trade_pct", 0.25) / 100
        self.max_risk_per_trade = config.get("max_risk_per_trade_pct", 0.50) / 100
        self.max_open_risk = config.get("max_open_risk_pct", 1.0) / 100
        self.daily_halt_pct = config.get("daily_loss_halt_pct", 3.0) / 100
        self.total_halt_pct = config.get("total_drawdown_halt_pct", 8.0) / 100
        self.max_concurrent = config.get("max_concurrent_trades", 3)
        self.max_trades_day = config.get("max_trades_per_day", 8)
        self.sl_atr_multiple = config.get("stop_loss_atr_multiple", 1.5)
        self.tp_atr_multiple = config.get("take_profit_atr_multiple", 2.25)

        self.state = RiskState()

    def update_state(self, balance: float, equity: float):
        """Update risk state from live account info."""
        today = date.today()

        # New day reset
        if today != self.state.current_date:
            self.state.day_start_balance = max(balance, self.state.current_balance)
            self.state.current_date = today
            self.state.trades_today = 0
            self.state.daily_pnl = 0.0
            self.state.is_halted_daily = False
            logger.info(f"New trading day. Day start balance: ${self.state.day_start_balance:,.2f}")

        self.state.current_balance = balance
        self.state.current_equity = equity
        self.state.daily_pnl = equity - self.state.day_start_balance
        self.state.total_pnl = equity - self.state.initial_balance

        if equity > self.state.peak_balance:
            self.state.peak_balance = equity

        self._check_halts()

    def _check_halts(self):
        """Check if daily or total drawdown limits are breached."""
        daily_dd = -self.state.daily_pnl / self.state.day_start_balance
        total_dd = -self.state.total_pnl / self.state.initial_balance

        if daily_dd >= self.daily_halt_pct:
            self.state.is_halted_daily = True
            logger.warning(
                f"DAILY HALT: {daily_dd:.2%} daily drawdown exceeds {self.daily_halt_pct:.2%} limit"
            )

        if total_dd >= self.total_halt_pct:
            self.state.is_halted_total = True
            logger.critical(
                f"TOTAL HALT: {total_dd:.2%} total drawdown exceeds {self.total_halt_pct:.2%} limit"
            )

    def can_trade(self) -> tuple[bool, str]:
        """Check if a new trade is allowed. Returns (allowed, reason)."""
        if self.state.is_halted_total:
            return False, "Total drawdown halt active"
        if self.state.is_halted_daily:
            return False, "Daily loss halt active"
        if self.state.open_positions >= self.max_concurrent:
            return False, f"Max concurrent trades ({self.max_concurrent}) reached"
        if self.state.trades_today >= self.max_trades_day:
            return False, f"Max daily trades ({self.max_trades_day}) reached"
        if self.state.open_risk_pct >= self.max_open_risk:
            return False, f"Max open risk ({self.max_open_risk:.1%}) reached"
        return True, "OK"

    def calculate_position_size(
        self, trade: TradeRequest, pip_value: float
    ) -> tuple[float, float, float]:
        """Calculate lot size, SL price, and TP price.

        Returns:
            (lots, stop_loss_distance, take_profit_distance)
        """
        # Scale risk by confidence and regime
        confidence_scalar = max(0.5, min(1.0, trade.confidence))
        regime_scalar = max(0.0, min(1.0, trade.regime_scalar))

        effective_risk = self.risk_per_trade * confidence_scalar * regime_scalar
        effective_risk = min(effective_risk, self.max_risk_per_trade)

        # Check if adding this risk exceeds open risk limit
        remaining_risk = self.max_open_risk - self.state.open_risk_pct
        effective_risk = min(effective_risk, remaining_risk)

        if effective_risk <= 0:
            return 0.0, 0.0, 0.0

        # Position size
        risk_amount = self.state.current_balance * effective_risk
        stop_distance = self.sl_atr_multiple * trade.atr
        tp_distance = self.tp_atr_multiple * trade.atr

        if stop_distance <= 0 or pip_value <= 0:
            return 0.0, 0.0, 0.0

        lots = risk_amount / (stop_distance * pip_value)

        # Round to 2 decimal places (standard lot precision)
        lots = round(lots, 2)
        lots = max(lots, 0.01)  # Minimum lot size

        logger.info(
            f"Position size: {lots} lots | "
            f"Risk: {effective_risk:.3%} (${risk_amount:,.2f}) | "
            f"SL: {stop_distance:.5f} | TP: {tp_distance:.5f} | "
            f"Confidence: {confidence_scalar:.2f} | Regime: {regime_scalar:.1f}"
        )

        return lots, stop_distance, tp_distance

    def record_trade_opened(self, risk_pct: float):
        """Update state when a trade is opened."""
        self.state.trades_today += 1
        self.state.open_positions += 1
        self.state.open_risk_pct += risk_pct

    def record_trade_closed(self, risk_pct: float, pnl: float):
        """Update state when a trade is closed."""
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.open_risk_pct = max(0.0, self.state.open_risk_pct - risk_pct)

        today = date.today()
        if today not in self.state.daily_profits:
            self.state.daily_profits[today] = 0.0
        self.state.daily_profits[today] += pnl

    def check_best_day_rule(self) -> tuple[bool, float]:
        """Check if Best Day Rule is satisfied for withdrawal.

        Returns (is_ok, best_day_pct of total positive profit).
        """
        positive_days = {d: p for d, p in self.state.daily_profits.items() if p > 0}
        if not positive_days:
            return True, 0.0

        total_positive = sum(positive_days.values())
        best_day = max(positive_days.values())
        best_day_pct = best_day / total_positive if total_positive > 0 else 0

        is_ok = best_day_pct <= 0.50
        return is_ok, best_day_pct

    def get_status_summary(self) -> dict:
        """Return current risk state summary."""
        return {
            "balance": self.state.current_balance,
            "equity": self.state.current_equity,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.state.daily_pnl / self.state.day_start_balance,
            "total_pnl": self.state.total_pnl,
            "total_pnl_pct": self.state.total_pnl / self.state.initial_balance,
            "open_positions": self.state.open_positions,
            "open_risk_pct": self.state.open_risk_pct,
            "trades_today": self.state.trades_today,
            "halted_daily": self.state.is_halted_daily,
            "halted_total": self.state.is_halted_total,
        }
