"""Determine whether a stock behaves like long-term or short-term."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LONG_TERM = "LONG_TERM"
SHORT_TERM = "SHORT_TERM"


@dataclass(frozen=True)
class RegimeAssessment:
    """Explainable regime classification for one stock."""

    regime: str
    confidence: float
    reason: str


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert optional numeric values to finite float defaults."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _direction_changes(close: pd.Series, lookback: int = 30) -> float:
    """Estimate how frequently price direction flips over recent sessions."""
    returns = close.pct_change().dropna().tail(lookback)
    if returns.empty:
        return 1.0

    signs = returns.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    non_zero = signs[signs != 0]
    if len(non_zero) < 2:
        return 0.0

    flips = (non_zero != non_zero.shift(1)).sum() - 1
    flips = max(flips, 0)
    return flips / (len(non_zero) - 1)


def detect_regime(frame: pd.DataFrame) -> RegimeAssessment:
    """Classify the latest market behavior and explain the decision."""
    latest = frame.iloc[-1]
    close = frame["Close"]

    annual_vol = _safe_float(latest.get("VOLATILITY_20", 0.0), default=0.0)
    price_20 = _safe_float(close.iloc[-20], default=0.0) if len(close) >= 20 else _safe_float(close.iloc[0], default=0.0)
    trend_20 = ((float(close.iloc[-1]) / price_20) - 1.0) if price_20 else 0.0

    sma_50 = _safe_float(latest.get("SMA_50"), default=0.0)
    sma_200 = _safe_float(latest.get("SMA_200"), default=0.0)
    latest_close = _safe_float(close.iloc[-1], default=0.0)
    trend_gap = abs((sma_50 - sma_200) / latest_close) if latest_close else 0.0

    swing_rate = _direction_changes(close, lookback=30)

    stable_trend = trend_20 > 0.03 and trend_gap > 0.01
    low_volatility = annual_vol > 0 and annual_vol < 0.28
    frequent_swings = swing_rate > 0.45

    if stable_trend and low_volatility and not frequent_swings:
        confidence = min(0.95, 0.55 + max(0.0, (0.28 - annual_vol)) + trend_20)
        return RegimeAssessment(
            regime=LONG_TERM,
            confidence=confidence,
            reason="Low volatility + steady upward trend = long-term behavior",
        )

    confidence = min(0.95, 0.55 + max(0.0, annual_vol - 0.25) + swing_rate / 2)
    if annual_vol >= 0.28 and frequent_swings:
        reason = "High volatility + frequent price swings = short-term behavior"
    elif annual_vol >= 0.28:
        reason = "High volatility with unstable momentum = short-term behavior"
    else:
        reason = "Weak trend strength with inconsistent direction = short-term behavior"

    return RegimeAssessment(regime=SHORT_TERM, confidence=confidence, reason=reason)
