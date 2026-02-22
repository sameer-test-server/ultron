"""Ultron stock analyst: explainable signals and hypothetical scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.data_reader import read_stock_csv
from core.indicators import add_indicators
from core.regime_detector import RegimeAssessment, detect_regime


@dataclass(frozen=True)
class HypotheticalScenario:
    """One historical what-if entry/exit scenario."""

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    return_pct: float
    note: str


@dataclass(frozen=True)
class SignalPoint:
    """A hypothetical signal marker for simulation and plotting."""

    date: pd.Timestamp
    price: float
    kind: str
    reason: str


@dataclass
class StockAnalysis:
    """Complete analysis package for one stock."""

    ticker: str
    data: pd.DataFrame
    regime: RegimeAssessment
    insights: list[str]
    scenarios: list[HypotheticalScenario]
    signals: list[SignalPoint]
    hypothetical_return_pct: float


class StockAnalyst:
    """Analyze local stock history with automatic data refresh fallback."""

    def __init__(self, refresh_callback: Callable[[], dict] | None = None) -> None:
        self._refresh_callback = refresh_callback

    def _load_with_fallback(self, ticker: str) -> pd.DataFrame:
        """Load local data; refresh all data once if file is missing or empty."""
        def _read() -> pd.DataFrame:
            frame = read_stock_csv(ticker)
            if frame.empty:
                raise ValueError(f"CSV has no valid rows for {ticker}")
            return frame

        try:
            return _read()
        except (FileNotFoundError, ValueError):
            if self._refresh_callback is not None:
                self._refresh_callback()
                return _read()
            raise

    @staticmethod
    def _generate_insights(data: pd.DataFrame, regime: RegimeAssessment) -> list[str]:
        """Generate plain-English insights from latest indicator context."""
        insights: list[str] = [regime.reason]
        if len(data) < 2:
            return insights

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        if pd.notna(latest["SMA_50"]) and pd.notna(prev["SMA_50"]):
            if prev["Close"] <= prev["SMA_50"] and latest["Close"] > latest["SMA_50"]:
                insights.append("Price crossed above the 50-day average, showing improving momentum.")
            elif prev["Close"] >= prev["SMA_50"] and latest["Close"] < latest["SMA_50"]:
                insights.append("Price crossed below the 50-day average, showing weaker momentum.")

        if latest["RSI_14"] < 30:
            insights.append("RSI is in oversold territory, which often reflects short-term stress.")
        elif latest["RSI_14"] > 70:
            insights.append("RSI is in overbought territory, which can precede pullbacks.")
        else:
            insights.append("RSI is neutral, indicating balanced momentum.")

        if pd.notna(latest["VOLATILITY_20"]):
            vol_pct = latest["VOLATILITY_20"] * 100
            insights.append(f"Recent annualized volatility is about {vol_pct:.1f}%, which shapes risk behavior.")

        return insights

    @staticmethod
    def _build_signals(data: pd.DataFrame) -> list[SignalPoint]:
        """Create hypothetical entry/exit markers from SMA20 and RSI context."""
        signals: list[SignalPoint] = []
        in_position = False

        for idx in range(1, len(data)):
            prev = data.iloc[idx - 1]
            row = data.iloc[idx]
            date = pd.Timestamp(row["Date"])
            price = float(row["Close"])

            crossed_up = pd.notna(row["SMA_20"]) and prev["Close"] <= prev["SMA_20"] and row["Close"] > row["SMA_20"]
            crossed_down = pd.notna(row["SMA_20"]) and prev["Close"] >= prev["SMA_20"] and row["Close"] < row["SMA_20"]

            if not in_position and crossed_up and row["RSI_14"] < 65:
                signals.append(
                    SignalPoint(
                        date=date,
                        price=price,
                        kind="ENTRY",
                        reason="Price moved above SMA20 with controlled RSI.",
                    )
                )
                in_position = True
            elif in_position and (crossed_down or row["RSI_14"] > 72):
                exit_reason = "Price moved below SMA20." if crossed_down else "RSI reached extended levels."
                signals.append(SignalPoint(date=date, price=price, kind="EXIT", reason=exit_reason))
                in_position = False

        if in_position and len(data) > 0:
            last = data.iloc[-1]
            signals.append(
                SignalPoint(
                    date=pd.Timestamp(last["Date"]),
                    price=float(last["Close"]),
                    kind="EXIT",
                    reason="Scenario closed at end of available data.",
                )
            )

        return signals

    @staticmethod
    def _build_scenarios(signals: list[SignalPoint]) -> list[HypotheticalScenario]:
        """Pair entry/exit signals into historical what-if trade scenarios."""
        scenarios: list[HypotheticalScenario] = []
        pending_entry: SignalPoint | None = None

        for signal in signals:
            if signal.kind == "ENTRY":
                pending_entry = signal
                continue

            if signal.kind == "EXIT" and pending_entry is not None:
                ret = ((signal.price / pending_entry.price) - 1.0) * 100
                scenarios.append(
                    HypotheticalScenario(
                        entry_date=pending_entry.date,
                        entry_price=pending_entry.price,
                        exit_date=signal.date,
                        exit_price=signal.price,
                        return_pct=ret,
                        note=f"{pending_entry.reason} Then exit because: {signal.reason}",
                    )
                )
                pending_entry = None

        return scenarios

    def analyze_stock(self, ticker: str) -> StockAnalysis:
        """Run full analysis for one ticker with explainable outputs."""
        raw = self._load_with_fallback(ticker)
        data = add_indicators(raw)
        regime = detect_regime(data)
        insights = self._generate_insights(data, regime)

        signals = self._build_signals(data)
        scenarios = self._build_scenarios(signals)

        if scenarios:
            avg_return = sum(item.return_pct for item in scenarios) / len(scenarios)
        else:
            avg_return = 0.0

        return StockAnalysis(
            ticker=ticker,
            data=data,
            regime=regime,
            insights=insights,
            scenarios=scenarios,
            signals=signals,
            hypothetical_return_pct=avg_return,
        )
