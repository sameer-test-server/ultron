"""UI data models for stock dashboard and detail views."""

from __future__ import annotations

from dataclasses import dataclass

from core.analyst import StockAnalysis
from core.paper_trader import PaperTradeResult


@dataclass
class StockViewModel:
    """Dashboard and detail payload for one stock."""

    ticker: str
    regime: str
    confidence: float
    confidence_label: str
    volatility_value: float
    volatility_label: str
    hypothetical_return_pct: float
    explanation: str
    insights: list[str]
    trend_note: str
    momentum_note: str
    volatility_note: str
    simulation: PaperTradeResult
    analysis: StockAnalysis
    reasoning_score: float | None = None
    reasoning_label: str | None = None
    reasoning_summary: list[str] | None = None
    reasoning_flags: list[str] | None = None
    last_data_date: str | None = None
    days_since_update: int | None = None
    scenario_summary: list[dict] | None = None
    signal_reliability: list[dict] | None = None
    risk_summary: dict | None = None
    grid_results: list[dict] | None = None
    data_quality: dict | None = None
    interactive_chart_html: str | None = None
    chart_price_path: str | None = None
    chart_rsi_path: str | None = None
    data_fingerprint: str | None = None
