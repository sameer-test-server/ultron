"""Signal reliability ledger calculations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalReliability:
    name: str
    occurrences: int
    hit_rate: float
    avg_return_pct: float


def _evaluate_signal(close: pd.Series, signal: pd.Series, horizon: int = 5) -> SignalReliability:
    occurrences = 0
    hits = 0
    returns: list[float] = []

    for idx in range(len(close) - horizon):
        if bool(signal.iloc[idx]):
            occurrences += 1
            start = float(close.iloc[idx])
            end = float(close.iloc[idx + horizon])
            if start <= 0:
                continue
            ret = (end / start - 1.0) * 100.0
            returns.append(ret)
            if ret > 0:
                hits += 1

    hit_rate = (hits / occurrences * 100.0) if occurrences else 0.0
    avg_return = (sum(returns) / len(returns)) if returns else 0.0
    return SignalReliability("", occurrences, round(hit_rate, 1), round(avg_return, 2))


def compute_signal_reliability(data: pd.DataFrame) -> list[SignalReliability]:
    """Compute reliability for common signals using a 5-day forward horizon."""
    if data.empty or "Close" not in data.columns:
        return []

    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().reset_index(drop=True)
    if close.empty:
        return []

    sma50 = pd.to_numeric(df.get("SMA_50"), errors="coerce").reset_index(drop=True)
    rsi = pd.to_numeric(df.get("RSI_14"), errors="coerce").reset_index(drop=True)

    signals = []

    # SMA50 bullish crossover
    if len(sma50) == len(close):
        cross_up = (close.shift(1) <= sma50.shift(1)) & (close > sma50)
        res = _evaluate_signal(close, cross_up.fillna(False))
        signals.append(SignalReliability("SMA50 Cross Up", res.occurrences, res.hit_rate, res.avg_return_pct))

        cross_down = (close.shift(1) >= sma50.shift(1)) & (close < sma50)
        res = _evaluate_signal(close, cross_down.fillna(False))
        signals.append(SignalReliability("SMA50 Cross Down", res.occurrences, res.hit_rate, res.avg_return_pct))

    # RSI oversold and overbought
    if len(rsi) == len(close):
        oversold = rsi < 30
        res = _evaluate_signal(close, oversold.fillna(False))
        signals.append(SignalReliability("RSI Oversold", res.occurrences, res.hit_rate, res.avg_return_pct))

        overbought = rsi > 70
        res = _evaluate_signal(close, overbought.fillna(False))
        signals.append(SignalReliability("RSI Overbought", res.occurrences, res.hit_rate, res.avg_return_pct))

    return signals
