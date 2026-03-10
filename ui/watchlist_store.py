"""Local watchlist storage for Ultron."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import BASE_DIR

WATCHLIST_FILE = Path(BASE_DIR) / "data" / "watchlist.json"


def load_watchlist() -> list[dict]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_watchlist(entries: list[dict]) -> None:
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")
