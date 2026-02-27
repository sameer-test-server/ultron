"""Read and validate local OHLCV CSV files for Ultron."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import RAW_DATA_DIR

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
_NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_column_names(columns: Iterable[str]) -> dict[str, str]:
    """Map case-variant column names to required canonical names."""
    lookup = {str(col).strip().lower(): col for col in columns}
    mapping: dict[str, str] = {}

    for required in REQUIRED_COLUMNS:
        source = lookup.get(required.lower())
        if source is not None:
            mapping[source] = required

    return mapping


def _validate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize required OHLCV columns."""
    rename_map = _normalize_column_names(frame.columns)
    normalized = frame.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    return normalized[REQUIRED_COLUMNS].copy()


def _clean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop corrupted rows and return sorted deduplicated daily candles."""
    cleaned = frame.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")

    for column in _NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[cleaned["High"] >= cleaned["Low"]]
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="last")
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)
    return cleaned


def read_stock_data(ticker: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Read one stock data file from local storage and return a validated DataFrame.
    It first attempts to read from a `.feather` file for performance. If it
    doesn't exist, it reads the `.csv`, and creates a `.feather` file for next time.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When required columns are missing.
    """
    base_dir = Path(data_dir) if data_dir is not None else Path(RAW_DATA_DIR)
    feather_path = base_dir / f"{ticker}.feather"

    if feather_path.exists():
        try:
            frame = pd.read_feather(feather_path)
            # Feather preserves types, so we can often return early.
            # A quick validation is still good practice.
            if all(col in frame.columns for col in REQUIRED_COLUMNS):
                return frame
        except Exception:
            # If feather read fails, fall back to CSV.
            pass

    csv_path = base_dir / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Primary data file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame = _validate_columns(frame)
    frame = _clean_rows(frame)

    # Save to Feather for next time, but don't block on it.
    try:
        frame.to_feather(feather_path)
    except Exception:
        # Don't let caching failures break the main data-reading path.
        pass

    return frame
