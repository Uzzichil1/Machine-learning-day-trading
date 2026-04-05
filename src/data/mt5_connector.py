"""MT5 data connector — handles connection, data retrieval, and symbol management."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}


class MT5Connector:
    """Manages MT5 terminal connection and data operations."""

    def __init__(self, mt5_config: dict = None):
        self._connected = False
        self._mt5_config = mt5_config or {}

    def connect(self) -> bool:
        kwargs = {}
        if self._mt5_config.get("login"):
            kwargs["login"] = int(self._mt5_config["login"])
        if self._mt5_config.get("password"):
            kwargs["password"] = str(self._mt5_config["password"])
        if self._mt5_config.get("server"):
            kwargs["server"] = str(self._mt5_config["server"])

        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
        self._connected = True
        info = mt5.terminal_info()
        acc = mt5.account_info()
        logger.info(f"Connected to MT5: {info.name}, build {info.build}")
        if acc:
            logger.info(f"Account: {acc.login} on {acc.server} (balance: ${acc.balance:,.2f})")
        return True

    def disconnect(self):
        mt5.shutdown()
        self._connected = False
        logger.info("Disconnected from MT5")

    def ensure_symbol(self, symbol: str) -> bool:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return False
        return True

    def get_rates(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
    ) -> pd.DataFrame:
        """Download OHLCV data as a DataFrame."""
        self.ensure_symbol(symbol)
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "tick_volume": "volume",
                "real_volume": "real_volume",
            },
            inplace=True,
        )
        logger.info(f"Downloaded {len(df)} bars for {symbol} {timeframe}")
        return df

    def get_recent_bars(
        self, symbol: str, timeframe: str, count: int = 500
    ) -> pd.DataFrame:
        """Get the most recent N bars."""
        self.ensure_symbol(symbol)
        tf = TIMEFRAME_MAP.get(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.margin_free),
            "profit": float(info.profit),
            "leverage": int(info.leverage),
        }

    def get_current_tick(self, symbol: str) -> dict:
        self.ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "spread": float(tick.ask - tick.bid),
            "time": datetime.fromtimestamp(tick.time),
        }

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, data_dir: str):
        """Save downloaded data to parquet."""
        path = Path(data_dir) / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved {len(df)} bars to {path}")

    def load_data(self, symbol: str, timeframe: str, data_dir: str) -> pd.DataFrame:
        """Load previously saved data."""
        path = Path(data_dir) / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
