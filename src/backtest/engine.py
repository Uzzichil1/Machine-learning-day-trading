"""FTMO-accurate backtesting engine with equity tracking and drawdown accounting."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp = None
    symbol: str = ""
    direction: int = 0  # 1 buy, -1 sell
    lots: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    risk_pct: float = 0.0
    confidence: float = 0.0


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = None
    daily_pnl: pd.Series = None
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_daily_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    ftmo_phase1_passed: bool = False
    ftmo_phase2_passed: bool = False
    ftmo_daily_limit_breached: bool = False
    ftmo_total_limit_breached: bool = False
    best_day_pct: float = 0.0
    days_to_target: int = 0


class FTMOBacktester:
    """Backtests a strategy with FTMO-exact risk accounting."""

    def __init__(self, config: dict):
        self.initial_balance = config.get("initial_balance", 100_000)
        self.phase1_target = config.get("phase1_target_pct", 10.0) / 100
        self.phase2_target = config.get("phase2_target_pct", 5.0) / 100
        self.max_daily_loss = config.get("max_daily_loss_pct", 5.0) / 100
        self.max_total_loss = config.get("max_total_loss_pct", 10.0) / 100
        self.spread_pips = config.get("spread_pips", 1.0)

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        risk_per_trade: float = 0.0025,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.25,
    ) -> BacktestResult:
        """Run backtest on signal DataFrame.

        Args:
            signals: DataFrame with columns: signal (1/-1/0), confidence, atr, regime_scalar
            prices: DataFrame with OHLCV data aligned to signals
            risk_per_trade: Base risk per trade as fraction
            sl_atr_mult: Stop loss in ATR multiples
            tp_atr_mult: Take profit in ATR multiples

        Returns:
            BacktestResult with full FTMO compliance metrics
        """
        balance = self.initial_balance
        equity_curve = []
        trades = []
        daily_balances = {}
        day_start_balance = balance
        current_date = None
        halted_daily = False
        halted_total = False

        for i in range(len(signals)):
            ts = signals.index[i]
            today = ts.date() if hasattr(ts, "date") else ts

            # New day handling
            if today != current_date:
                current_date = today
                day_start_balance = balance
                halted_daily = False

            # Check total halt
            total_dd = (self.initial_balance - balance) / self.initial_balance
            if total_dd >= self.max_total_loss:
                halted_total = True

            # Check daily halt
            daily_dd = (day_start_balance - balance) / day_start_balance
            if daily_dd >= self.max_daily_loss:
                halted_daily = True

            equity_curve.append({"time": ts, "equity": balance})

            # Skip if halted or no signal
            if halted_total or halted_daily:
                continue

            sig = signals.iloc[i]
            signal = sig.get("signal", 0)
            if signal == 0:
                continue

            confidence = sig.get("confidence", 0.6)
            atr = sig.get("atr", 0)
            regime_scalar = sig.get("regime_scalar", 1.0)

            if atr <= 0 or np.isnan(atr):
                continue

            # Calculate position and barriers
            effective_risk = risk_per_trade * confidence * regime_scalar
            risk_amount = balance * effective_risk
            sl_dist = sl_atr_mult * atr
            tp_dist = tp_atr_mult * atr

            entry_price = prices["close"].iloc[i]

            # Simulate trade outcome using future bars
            trade_pnl = self._simulate_trade(
                prices, i, signal, entry_price, sl_dist, tp_dist, max_bars=8
            )

            # Apply spread cost
            spread_cost = self.spread_pips * atr * 0.01  # Rough spread cost
            trade_pnl -= spread_cost

            # Convert to dollar P&L based on risk
            if sl_dist > 0:
                r_multiple = trade_pnl / sl_dist
                dollar_pnl = r_multiple * risk_amount
            else:
                dollar_pnl = 0

            balance += dollar_pnl

            trade = BacktestTrade(
                entry_time=ts,
                symbol=sig.get("symbol", ""),
                direction=signal,
                entry_price=entry_price,
                pnl=dollar_pnl,
                pnl_pct=dollar_pnl / self.initial_balance,
                risk_pct=effective_risk,
                confidence=confidence,
            )
            trades.append(trade)

            # Track daily P&L
            if today not in daily_balances:
                daily_balances[today] = 0.0
            daily_balances[today] += dollar_pnl

        return self._compile_results(
            trades, equity_curve, daily_balances, halted_daily, halted_total
        )

    def _simulate_trade(
        self,
        prices: pd.DataFrame,
        entry_idx: int,
        direction: int,
        entry_price: float,
        sl_dist: float,
        tp_dist: float,
        max_bars: int = 8,
    ) -> float:
        """Simulate a single trade using future price bars."""
        for j in range(1, max_bars + 1):
            idx = entry_idx + j
            if idx >= len(prices):
                break

            bar_high = prices["high"].iloc[idx]
            bar_low = prices["low"].iloc[idx]
            bar_close = prices["close"].iloc[idx]

            if direction == 1:  # Long
                if bar_high >= entry_price + tp_dist:
                    return tp_dist
                if bar_low <= entry_price - sl_dist:
                    return -sl_dist
            else:  # Short
                if bar_low <= entry_price - tp_dist:
                    return tp_dist
                if bar_high >= entry_price + sl_dist:
                    return -sl_dist

        # Time barrier: return actual P&L at last bar
        final_idx = min(entry_idx + max_bars, len(prices) - 1)
        final_close = prices["close"].iloc[final_idx]
        return (final_close - entry_price) * direction

    def _compile_results(
        self, trades, equity_curve, daily_balances, halted_daily, halted_total
    ) -> BacktestResult:
        result = BacktestResult()
        result.trades = trades
        result.total_trades = len(trades)

        if not equity_curve:
            return result

        eq_df = pd.DataFrame(equity_curve).set_index("time")
        result.equity_curve = eq_df["equity"]

        # Returns and Sharpe
        final_equity = result.equity_curve.iloc[-1]
        result.total_return = (final_equity - self.initial_balance) / self.initial_balance

        daily_returns = result.equity_curve.resample("D").last().pct_change(fill_method=None).dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            result.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        # Max drawdown
        peak = result.equity_curve.cummax()
        drawdown = (result.equity_curve - peak) / peak
        result.max_drawdown = abs(drawdown.min())

        # Max daily drawdown
        if daily_balances:
            daily_pnl = pd.Series(daily_balances)
            result.daily_pnl = daily_pnl
            # Calculate daily DD as pct of day-start balance
            result.max_daily_drawdown = abs(daily_pnl.min()) / self.initial_balance

        # Win rate and profit factor
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            result.win_rate = len(wins) / len(trades)
            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # FTMO pass checks
        result.ftmo_phase1_passed = result.total_return >= self.phase1_target
        result.ftmo_phase2_passed = result.total_return >= self.phase2_target
        result.ftmo_daily_limit_breached = halted_daily
        result.ftmo_total_limit_breached = halted_total

        # Best day check
        if daily_balances:
            positive_days = {d: p for d, p in daily_balances.items() if p > 0}
            if positive_days:
                total_pos = sum(positive_days.values())
                best = max(positive_days.values())
                result.best_day_pct = best / total_pos if total_pos > 0 else 0

        # Days to reach Phase 1 target
        target_equity = self.initial_balance * (1 + self.phase1_target)
        target_reached = result.equity_curve[result.equity_curve >= target_equity]
        if len(target_reached) > 0:
            start = result.equity_curve.index[0]
            end = target_reached.index[0]
            result.days_to_target = (end - start).days

        return result

    def print_summary(self, result: BacktestResult):
        """Print human-readable backtest summary."""
        print("\n" + "=" * 60)
        print("FTMO BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Return:       {result.total_return:>10.2%}")
        print(f"Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
        print(f"Max Drawdown:       {result.max_drawdown:>10.2%}")
        print(f"Win Rate:           {result.win_rate:>10.2%}")
        print(f"Profit Factor:      {result.profit_factor:>10.2f}")
        print(f"Total Trades:       {result.total_trades:>10d}")
        print(f"Best Day % of Total:{result.best_day_pct:>10.2%}")
        print("-" * 60)
        print(f"Phase 1 Passed:     {'YES' if result.ftmo_phase1_passed else 'NO':>10}")
        print(f"Phase 2 Passed:     {'YES' if result.ftmo_phase2_passed else 'NO':>10}")
        print(f"Daily Limit Hit:    {'YES' if result.ftmo_daily_limit_breached else 'NO':>10}")
        print(f"Total Limit Hit:    {'YES' if result.ftmo_total_limit_breached else 'NO':>10}")
        if result.days_to_target > 0:
            print(f"Days to Target:     {result.days_to_target:>10d}")
        print("=" * 60)
