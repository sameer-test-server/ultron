#!/usr/bin/env python3
"""Generate daily summary report (markdown + PDF)."""

import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.data_quality import evaluate_data_quality
from core.data_reader import read_stock_data
from config.nifty50 import NIFTY50_TICKERS
from core.daily_report import generate_markdown, generate_pdf


def main() -> int:
    summary = {
        "Total Tickers": len(NIFTY50_TICKERS),
        "Report Date": datetime.now().strftime("%Y-%m-%d"),
    }

    stale = 0
    for ticker in NIFTY50_TICKERS:
        try:
            data = read_stock_data(ticker)
        except Exception:
            stale += 1
            continue
        dq = evaluate_data_quality(ticker, data)
        if dq.stale:
            stale += 1

    summary["Stale Tickers"] = stale
    summary["Healthy Tickers"] = len(NIFTY50_TICKERS) - stale

    date_stamp = datetime.now().strftime("%Y%m%d")
    md = generate_markdown(summary, f"daily_summary_{date_stamp}.md")
    pdf = generate_pdf(summary, f"daily_summary_{date_stamp}.pdf")

    print(f"Wrote: {md}")
    print(f"Wrote: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
