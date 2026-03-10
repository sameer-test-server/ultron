"""Risk suite: drawdown, tail risk, liquidity risk."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TailRiskEvent:
    date: str
    return_pct: float


@dataclass(frozen=True)
class RiskSummary:
    max_drawdown_pct: float
    tail_risk_events: list[TailRiskEvent]
    liquidity_score: float


def compute_risk_summary(data: pd.DataFrame) -> RiskSummary:
    if data.empty:
        return RiskSummary(0.0, [], 0.0)

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    if df.empty:
        return RiskSummary(0.0, [], 0.0)

    close = df["Close"].reset_index(drop=True)
    returns = close.pct_change().fillna(0)

    # max drawdown
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min() * 100.0)

    # tail risk: daily return < -2 std or < -5%
    std = float(returns.std()) if len(returns) > 2 else 0.0
    threshold = min(-0.05, -2 * std) if std > 0 else -0.05
    events = []
    for idx, r in returns.items():
        if float(r) <= threshold:
            date = df["Date"].iloc[idx].strftime("%Y-%m-%d")
            events.append(TailRiskEvent(date=date, return_pct=round(float(r) * 100.0, 2)))

    # liquidity score: normalized volume (0-100)
    vol = df["Volume"].dropna()
    if vol.empty:
        liquidity = 0.0
    else:
        p50 = float(vol.quantile(0.5))
        p90 = float(vol.quantile(0.9))
        latest = float(vol.iloc[-1])
        if p90 <= 0:
            liquidity = 0.0
        else:
            liquidity = min(100.0, max(0.0, (latest / p90) * 100.0))

    return RiskSummary(round(max_dd, 2), events[:5], round(liquidity, 1))
