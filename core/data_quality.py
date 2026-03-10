"""Data integrity checks for Ultron."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    """Summarize data integrity health for one ticker."""

    ticker: str
    rows: int
    start_date: str | None
    end_date: str | None
    days_since_update: int | None
    missing_days: int
    stale: bool
    gaps: int
    notes: list[str]


def evaluate_data_quality(ticker: str, data: pd.DataFrame) -> DataQualityReport:
    """Compute data quality indicators for a single ticker."""
    notes: list[str] = []
    if data.empty:
        return DataQualityReport(
            ticker=ticker,
            rows=0,
            start_date=None,
            end_date=None,
            days_since_update=None,
            missing_days=0,
            stale=True,
            gaps=0,
            notes=["No data available"],
        )

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    if df.empty:
        return DataQualityReport(
            ticker=ticker,
            rows=0,
            start_date=None,
            end_date=None,
            days_since_update=None,
            missing_days=0,
            stale=True,
            gaps=0,
            notes=["No valid dates"],
        )

    start_date = df["Date"].iloc[0].date()
    end_date = df["Date"].iloc[-1].date()
    days_since_update = (pd.Timestamp.now().date() - end_date).days

    # Trading calendar approximation: weekdays only
    expected_days = pd.date_range(start=start_date, end=end_date, freq="B")
    actual_days = pd.to_datetime(df["Date"]).dt.normalize().unique()
    missing_days = max(0, len(expected_days) - len(actual_days))

    gaps = 0
    prev = df["Date"].iloc[0]
    for current in df["Date"].iloc[1:]:
        if (current - prev).days > 4:  # gap beyond weekend
            gaps += 1
        prev = current

    stale = days_since_update > 3
    if stale:
        notes.append(f"Stale data: {days_since_update} days old")
    if missing_days > 5:
        notes.append(f"Missing {missing_days} expected trading days")
    if gaps > 0:
        notes.append(f"Detected {gaps} date gaps")

    return DataQualityReport(
        ticker=ticker,
        rows=len(df),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        days_since_update=days_since_update,
        missing_days=missing_days,
        stale=stale,
        gaps=gaps,
        notes=notes,
    )
