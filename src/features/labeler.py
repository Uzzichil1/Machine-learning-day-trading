"""Triple-barrier labeling for target variable creation."""

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    upper_barrier_atr: float = 1.5,
    lower_barrier_atr: float = 1.5,
    max_holding_bars: int = 8,
    atr_column: str = "atr",
) -> pd.Series:
    """Apply triple-barrier labeling method (Lopez de Prado).

    For each bar, check if price hits:
    - Upper barrier (TP): close + upper_barrier_atr * ATR → label = 1 (long win)
    - Lower barrier (SL): close - lower_barrier_atr * ATR → label = -1 (long loss)
    - Time barrier: max_holding_bars elapsed → label based on final P&L

    Returns:
        Series of labels: 1 (buy), -1 (sell), 0 (no trade / time-barrier neutral)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df[atr_column].values
    n = len(df)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n - max_holding_bars):
        if np.isnan(atr[i]) or atr[i] <= 0:
            labels[i] = 0
            continue

        entry_price = close[i]
        upper = entry_price + upper_barrier_atr * atr[i]
        lower = entry_price - lower_barrier_atr * atr[i]

        label = 0
        for j in range(1, max_holding_bars + 1):
            idx = i + j
            if idx >= n:
                break

            # Check if upper barrier hit (using high of future bar)
            if high[idx] >= upper:
                label = 1
                break

            # Check if lower barrier hit (using low of future bar)
            if low[idx] <= lower:
                label = -1
                break

        # Time barrier: if neither barrier hit, use final close vs entry
        if label == 0 and i + max_holding_bars < n:
            final_close = close[i + max_holding_bars]
            pnl = final_close - entry_price
            threshold = 0.3 * atr[i]  # Minimum move to assign direction
            if pnl > threshold:
                label = 1
            elif pnl < -threshold:
                label = -1

        labels[i] = label

    return pd.Series(labels, index=df.index, name="target")
