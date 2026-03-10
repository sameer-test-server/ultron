"""Reasoning engine for multi-signal, explainable analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReasoningReport:
    """Explainable multi-signal reasoning output."""

    score: float
    label: str
    confidence: float
    summary: list[str]
    flags: list[str]


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_reasoning_report(data: pd.DataFrame, regime: str) -> ReasoningReport:
    """
    Build a multi-signal reasoning report.

    Signals used:
    - Trend: close vs SMA50/SMA200 + slope of recent returns
    - Momentum: RSI
    - Volatility: annualized rolling volatility
    - Stability: direction change rate
    """
    if data.empty:
        return ReasoningReport(
            score=0.0,
            label="Weak",
            confidence=0.0,
            summary=["Insufficient data for reasoning."],
            flags=["No data"],
        )

    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        return ReasoningReport(
            score=0.0,
            label="Weak",
            confidence=0.0,
            summary=["No valid close prices available."],
            flags=["Invalid prices"],
        )

    latest = data.iloc[-1]
    rsi = _safe_float(latest.get("RSI_14"), default=50.0)
    vol = _safe_float(latest.get("VOLATILITY_20"), default=0.0)
    sma_50 = _safe_float(latest.get("SMA_50"), default=0.0)
    sma_200 = _safe_float(latest.get("SMA_200"), default=0.0)
    last_close = _safe_float(latest.get("Close"), default=float(close.iloc[-1]))

    returns = close.pct_change().dropna()
    recent_returns = returns.tail(20)
    trend_slope = float(recent_returns.mean()) if not recent_returns.empty else 0.0

    # Stability: direction changes over last 30 sessions
    recent = close.tail(30)
    direction_changes = 0.0
    if len(recent) > 1:
        sign = recent.pct_change().dropna().apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
        sign = sign[sign != 0]
        if len(sign) > 1:
            direction_changes = (sign != sign.shift(1)).sum() - 1
            direction_changes = max(0, direction_changes) / max(1, (len(sign) - 1))

    # Score components (0-100)
    trend_score = 0.0
    if last_close > sma_50 > 0 and last_close > sma_200 > 0:
        trend_score = 35.0
    elif last_close > sma_50 > 0:
        trend_score = 25.0
    elif sma_50 > 0 and last_close < sma_50:
        trend_score = 10.0

    momentum_score = 0.0
    if rsi >= 55 and rsi <= 70:
        momentum_score = 20.0
    elif rsi > 70:
        momentum_score = 12.0
    elif rsi < 30:
        momentum_score = 6.0
    else:
        momentum_score = 14.0

    vol_score = 0.0
    if vol == 0:
        vol_score = 8.0
    elif vol < 0.20:
        vol_score = 22.0
    elif vol < 0.35:
        vol_score = 16.0
    else:
        vol_score = 8.0

    stability_score = 0.0
    if direction_changes < 0.35:
        stability_score = 18.0
    elif direction_changes < 0.5:
        stability_score = 12.0
    else:
        stability_score = 6.0

    score = trend_score + momentum_score + vol_score + stability_score
    score = max(0.0, min(score, 100.0))

    confidence = 0.45 + (score / 200.0)
    confidence = max(0.0, min(confidence, 0.95))

    if score >= 75:
        label = "Strong"
    elif score >= 55:
        label = "Moderate"
    else:
        label = "Weak"

    summary = []
    flags = []

    if trend_score >= 25:
        summary.append("Trend strength is positive (price above key moving averages).")
    else:
        summary.append("Trend strength is mixed or weak (price below key averages).")

    if momentum_score >= 18:
        summary.append("Momentum is constructive based on RSI behavior.")
    elif rsi < 30:
        summary.append("Momentum is weak (RSI oversold zone).")
    elif rsi > 70:
        summary.append("Momentum is stretched (RSI overbought zone).")
    else:
        summary.append("Momentum is neutral.")

    if vol < 0.20:
        summary.append("Volatility is low, supporting steadier price behavior.")
    elif vol < 0.35:
        summary.append("Volatility is moderate, suggesting balanced risk.")
    else:
        summary.append("Volatility is high; short-term swings are more likely.")

    if direction_changes > 0.5:
        flags.append("Frequent direction changes (choppy regime).")
    if trend_slope < 0:
        flags.append("Recent returns slope is negative.")
    if regime == "SHORT_TERM" and vol < 0.2:
        flags.append("Regime says SHORT_TERM but volatility is low; review signals.")

    return ReasoningReport(
        score=round(score, 2),
        label=label,
        confidence=round(confidence, 2),
        summary=summary,
        flags=flags,
    )
