"""MT5 trade execution — handles order placement, modification, and monitoring."""

import logging
from datetime import datetime

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class MT5Executor:
    """Executes trades through MT5 with full error handling."""

    def __init__(self, config: dict):
        self.magic = config.get("magic_number", 234000)
        self.deviation = config.get("deviation_points", 20)
        self.comment_prefix = config.get("comment_prefix", "ftmo_ml")
        self._filling_mode = self._resolve_filling(config.get("filling_mode", "ioc"))

    def _resolve_filling(self, mode: str) -> int:
        modes = {
            "ioc": mt5.ORDER_FILLING_IOC,
            "fok": mt5.ORDER_FILLING_FOK,
            "return": mt5.ORDER_FILLING_RETURN,
        }
        return modes.get(mode, mt5.ORDER_FILLING_IOC)

    def normalize_lots(self, symbol: str, lots: float) -> float:
        """Normalize lot size to broker's volume constraints."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return lots
        step = info.volume_step
        lots = round(round(lots / step) * step, 8)
        lots = max(info.volume_min, min(lots, info.volume_max))
        return lots

    def get_volume_constraints(self, symbol: str) -> dict:
        """Get broker's volume constraints for a symbol."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return {"volume_min": 0.1, "volume_max": 250.0, "volume_step": 0.1}
        return {
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def preflight_check(self, symbol: str) -> tuple[bool, str]:
        """Verify symbol and account are ready for trading.

        Returns (ok, reason).
        """
        # Ensure symbol is in Market Watch
        if not mt5.symbol_select(symbol, True):
            return False, f"Cannot select {symbol} in Market Watch"

        info = mt5.symbol_info(symbol)
        if info is None:
            return False, f"symbol_info returned None for {symbol}"

        # trade_mode: 0 = FULL, 1 = LONGONLY, 2 = SHORTONLY, 3 = CLOSEONLY, 4 = DISABLED
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, f"{symbol} trade_mode is DISABLED on this broker/account"
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
            return False, f"{symbol} trade_mode is CLOSE_ONLY — no new positions allowed"

        # Check account-level trading permissions
        account = mt5.account_info()
        if account is not None:
            if not account.trade_allowed:
                return False, "Account trade_allowed is False — trading disabled on this account"
            if not account.trade_expert:
                return False, "Account trade_expert is False — enable 'Allow Algo Trading' in MT5"

        return True, "OK"

    def open_trade(
        self,
        symbol: str,
        direction: int,
        lots: float,
        sl_distance: float,
        tp_distance: float,
        comment: str = "",
    ) -> dict:
        """Open a market order with SL and TP.

        Args:
            symbol: Trading instrument
            direction: 1 for buy, -1 for sell
            lots: Position size
            sl_distance: Stop loss distance from entry (in price)
            tp_distance: Take profit distance from entry (in price)
            comment: Order comment

        Returns:
            dict with order result or error info
        """
        # Pre-flight: ensure symbol selected and tradeable
        ok, reason = self.preflight_check(symbol)
        if not ok:
            logger.error(f"Preflight failed for {symbol}: {reason}")
            return {"success": False, "error": reason}

        # Check direction-specific trade mode
        info = mt5.symbol_info(symbol)
        if direction == 1 and info.trade_mode == mt5.SYMBOL_TRADE_MODE_SHORTONLY:
            return {"success": False, "error": f"{symbol} is SHORT_ONLY — cannot open BUY"}
        if direction == -1 and info.trade_mode == mt5.SYMBOL_TRADE_MODE_LONGONLY:
            return {"success": False, "error": f"{symbol} is LONG_ONLY — cannot open SELL"}

        # Normalize lots to broker constraints
        lots = self.normalize_lots(symbol, lots)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": f"Cannot get tick for {symbol}"}

        if direction == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
            sl = round(price - sl_distance, 5)
            tp = round(price + tp_distance, 5)
        elif direction == -1:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
            sl = round(price + sl_distance, 5)
            tp = round(price - tp_distance, 5)
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": f"{self.comment_prefix}_{comment}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": f"order_send returned None: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Order failed: {result.retcode} - {result.comment} | "
                f"{symbol} {'BUY' if direction == 1 else 'SELL'} {lots} lots"
            )
            return {
                "success": False,
                "retcode": int(result.retcode),
                "error": result.comment,
            }

        logger.info(
            f"Order filled: #{result.order} {symbol} "
            f"{'BUY' if direction == 1 else 'SELL'} {lots} lots @ {price} | "
            f"SL: {sl} TP: {tp}"
        )
        return {
            "success": True,
            "order_id": int(result.order),
            "price": price,
            "sl": sl,
            "tp": tp,
            "lots": lots,
            "symbol": symbol,
            "direction": direction,
        }

    def close_position(self, position) -> dict:
        """Close an open position by ticket."""
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return {"success": False, "error": f"Cannot get tick for {position.symbol}"}

        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": close_type,
            "position": int(position.ticket),
            "price": price,
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": f"{self.comment_prefix}_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else str(mt5.last_error())
            logger.error(f"Close failed for #{position.ticket}: {error}")
            return {"success": False, "error": error}

        logger.info(f"Closed #{position.ticket} {position.symbol} @ {price}")
        return {"success": True, "price": price}

    def close_all_positions(self) -> list:
        """Emergency close all positions managed by this bot."""
        positions = mt5.positions_get()
        if positions is None:
            return []

        results = []
        for pos in positions:
            if pos.magic == self.magic:
                result = self.close_position(pos)
                results.append(result)

        if results:
            logger.warning(f"Emergency close: {len(results)} positions closed")
        return results

    def get_open_positions(self) -> list:
        """Get all open positions for this bot."""
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.magic]

    def get_position_pnl(self) -> float:
        """Get total floating P&L for all bot positions."""
        positions = self.get_open_positions()
        return sum(float(p.profit) for p in positions)
