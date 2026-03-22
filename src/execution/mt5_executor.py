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
