"""Core indicator calculations without external TA libraries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential smoothing."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of input OHLCV data with trend and momentum indicators."""
    enriched = frame.copy()

    enriched["SMA_20"] = enriched["Close"].rolling(window=20, min_periods=20).mean()
    enriched["SMA_50"] = enriched["Close"].rolling(window=50, min_periods=50).mean()
    enriched["SMA_200"] = enriched["Close"].rolling(window=200, min_periods=200).mean()
    enriched["EMA_20"] = enriched["Close"].ewm(span=20, adjust=False).mean()
    enriched["RSI_14"] = _rsi(enriched["Close"], period=14)

    daily_returns = enriched["Close"].pct_change()
    enriched["VOLATILITY_20"] = daily_returns.rolling(window=20, min_periods=20).std() * np.sqrt(252)

    return enriched
