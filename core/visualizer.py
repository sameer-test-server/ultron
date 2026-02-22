"""Chart generation for Ultron analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import BASE_DIR
from core.analyst import StockAnalysis
from core.paper_trader import PaperTradeResult


def _chart_output_dir() -> Path:
    path = Path(BASE_DIR) / "reports" / "charts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stock_chart(analysis: StockAnalysis, simulation: PaperTradeResult) -> Path:
    """Save price/indicator/RSI chart with hypothetical entry/exit markers."""
    data = analysis.data

    fig, (ax_price, ax_rsi) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(data["Date"], data["Close"], label="Close", linewidth=1.6)
    ax_price.plot(data["Date"], data["SMA_20"], label="SMA 20", linewidth=1.0)
    ax_price.plot(data["Date"], data["SMA_50"], label="SMA 50", linewidth=1.0)
    ax_price.plot(data["Date"], data["SMA_200"], label="SMA 200", linewidth=1.0)
    ax_price.plot(data["Date"], data["EMA_20"], label="EMA 20", linestyle="--", linewidth=1.0)

    for trade in simulation.trades:
        entry_date = pd.to_datetime(trade.entry_date)
        exit_date = pd.to_datetime(trade.exit_date)
        ax_price.scatter(entry_date, trade.entry_price, marker="^", s=90, color="green", label="Entry")
        ax_price.scatter(exit_date, trade.exit_price, marker="v", s=90, color="red", label="Exit")

    handles, labels = ax_price.get_legend_handles_labels()
    dedup_labels = {}
    for handle, label in zip(handles, labels):
        dedup_labels[label] = handle
    ax_price.legend(dedup_labels.values(), dedup_labels.keys(), loc="upper left")
    ax_price.set_title(f"{analysis.ticker} | Regime: {analysis.regime.regime}")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.25)

    ax_rsi.plot(data["Date"], data["RSI_14"], label="RSI 14", color="tab:purple", linewidth=1.2)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.9)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.9)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_xlabel("Date")
    ax_rsi.grid(True, alpha=0.25)

    fig.tight_layout()

    output_path = _chart_output_dir() / f"{analysis.ticker}.png"
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path
