"""Feature engineering pipeline — transforms raw OHLCV into ML-ready features."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure numpy / pandas indicator implementations
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range (Wilder smoothing / RMA)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (equivalent to EWM with alpha = 1/length, adjust=False)
    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _bbands(
    close: pd.Series, length: int, std_dev: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands — returns (upper, mid, lower)."""
    mid = close.rolling(length).mean()
    sigma = close.rolling(length).std(ddof=0)
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    return upper, mid, lower


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD — returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Average Directional Index (Wilder smoothing)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    alpha = 1.0 / length
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    plus_di = 100.0 * plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_w
    minus_di = 100.0 * minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr_w

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom

    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return adx


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator — returns (%K, %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / denom

    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d


def _sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(length).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


# ---------------------------------------------------------------------------
# Feature engineering class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Computes all features from raw OHLCV data."""

    def __init__(self, config: dict):
        self.lookback_periods = config.get("lookback_periods", [1, 3, 5, 10, 20])
        self.rsi_periods = config.get("rsi_periods", [7, 14, 21])
        self.ma_periods = config.get("ma_periods", [20, 50, 200])
        self.atr_period = config.get("atr_period", 14)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all feature tiers and return enriched DataFrame."""
        df = df.copy()
        df = self._price_action_features(df)
        df = self._volatility_features(df)
        df = self._momentum_features(df)
        df = self._volume_features(df)
        df = self._time_features(df)
        return df

    def _price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 1: Price action derivatives."""
        close = df["close"]

        # Log returns at multiple lags
        for period in self.lookback_periods:
            df[f"log_return_{period}"] = np.log(close / close.shift(period))

        # Normalized bar range
        atr = _atr(df["high"], df["low"], close, length=self.atr_period)
        df["atr"] = atr
        df["hl_range_norm"] = (df["high"] - df["low"]) / atr

        # Candle body and wick ratios
        body = (close - df["open"]).abs()
        total_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = body / total_range
        df["upper_wick_ratio"] = (df["high"] - df[["close", "open"]].max(axis=1)) / total_range
        df["lower_wick_ratio"] = (df[["close", "open"]].min(axis=1) - df["low"]) / total_range

        # Opening gap
        df["gap"] = df["open"] / close.shift(1) - 1

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 2: Volatility features."""
        close = df["close"]

        # Bollinger Band width
        upper, mid, lower = _bbands(close, length=self.bb_period, std_dev=self.bb_std)
        df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

        # Realized volatility at multiple windows
        log_ret = np.log(close / close.shift(1))
        for window in [5, 10, 20]:
            df[f"realized_vol_{window}"] = log_ret.rolling(window).std() * np.sqrt(252)

        # Volatility ratio (short/long) — regime change detector
        df["vol_ratio"] = df["realized_vol_5"] / df["realized_vol_20"].replace(0, np.nan)

        # ATR percentile rank
        df["atr_percentile"] = df["atr"].rolling(100).rank(pct=True)

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 3: Momentum and mean-reversion indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI at multiple periods
        for period in self.rsi_periods:
            df[f"rsi_{period}"] = _rsi(close, length=period)

        # MACD histogram
        _, _, histogram = _macd(close)
        df["macd_hist"] = histogram

        # ADX
        df["adx_14"] = _adx(high, low, close, length=14)

        # Stochastic %K
        k, _ = _stoch(high, low, close)
        df["stoch_k"] = k

        # Distance from MAs (z-scored by ATR)
        for period in self.ma_periods:
            ma = _sma(close, length=period)
            df[f"dist_from_{period}ma"] = (close - ma) / df["atr"]

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 4: Volume features."""
        if "volume" not in df.columns:
            return df

        vol = df["volume"].astype(float)

        # Volume z-score
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std().replace(0, np.nan)
        df["volume_zscore"] = (vol - vol_mean) / vol_std

        # OBV slope
        obv = _obv(df["close"], vol)
        df["obv_slope"] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan,
            raw=True,
        )

        # Volume-price correlation
        log_ret = np.log(df["close"] / df["close"].shift(1)).abs()
        df["vol_price_corr"] = vol.rolling(20).corr(log_ret)

        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tier 5: Time and session features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        hour = df.index.hour
        dow = df.index.dayofweek

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

        # Session flags
        df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
        df["is_ny"] = ((hour >= 13) & (hour < 21)).astype(int)
        df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return list of feature column names (excludes OHLCV and target)."""
        exclude = {"open", "high", "low", "close", "volume", "real_volume", "spread", "target"}
        return [c for c in df.columns if c not in exclude and not c.startswith("_")]
