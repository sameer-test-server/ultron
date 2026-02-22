"""Ultron end-to-end offline market analysis and paper trading pipeline."""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass

from config.nifty50 import NIFTY50_TICKERS
from config.settings import BASE_DIR
from core.analyst import StockAnalyst
from core.data_loader import update_all_data
from core.paper_trader import PaperTradeResult, simulate_paper_trades
from core.visualizer import save_stock_chart


@dataclass(frozen=True)
class StockRunResult:
    """Final per-stock result for summary ranking."""

    ticker: str
    regime: str
    confidence: float
    explanation: str
    hypothetical_return_pct: float
    chart_path: str
    simulation: PaperTradeResult


def _configure_logging() -> None:
    """Configure file logging for local and cron runs."""
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "ultron.log")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.6:
        return "MEDIUM"
    return "LOW"


def _print_stock_result(result: StockRunResult) -> None:
    """Print explainable per-stock output in plain English."""
    print(f"\n=== {result.ticker} ===")
    print(f"Regime: {result.regime}")
    print(f"Confidence: {_confidence_label(result.confidence)} ({result.confidence:.2f})")
    print(f"Hypothetical Return: {result.hypothetical_return_pct:.2f}%")
    print(f"Paper Trades: {len(result.simulation.trades)} | Win Rate: {result.simulation.win_rate:.1f}%")
    print(f"Chart: {result.chart_path}")

    print("Explanation:")
    print(f"- {result.explanation}")


def _print_final_summary(results: list[StockRunResult], skipped: list[tuple[str, str]]) -> None:
    """Print roll-up summary across all processed stocks."""
    long_term = sum(1 for item in results if item.regime == "LONG_TERM")
    short_term = sum(1 for item in results if item.regime == "SHORT_TERM")

    print("\n===== ULTRON FINAL SUMMARY =====")
    print(f"Total stocks analyzed: {len(results)}")
    print(f"Long-term count: {long_term}")
    print(f"Short-term count: {short_term}")
    print(f"Skipped count: {len(skipped)}")

    if results:
        best = max(results, key=lambda item: item.hypothetical_return_pct)
        worst = min(results, key=lambda item: item.hypothetical_return_pct)
        print(
            "Best hypothetical performer: "
            f"{best.ticker} ({best.hypothetical_return_pct:.2f}%, {best.simulation.win_rate:.1f}% win rate)"
        )
        print(
            "Worst hypothetical performer: "
            f"{worst.ticker} ({worst.hypothetical_return_pct:.2f}%, {worst.simulation.win_rate:.1f}% win rate)"
        )

    if skipped:
        print("Skipped symbols:")
        for ticker, reason in skipped:
            print(f"- {ticker}: {reason}")


def main() -> int:
    """Run the full Ultron analysis pipeline locally and safely."""
    _configure_logging()
    logger = logging.getLogger("ultron.analysis_runner")

    start = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Ultron analysis started at %s", start.isoformat())

    analyst = StockAnalyst(refresh_callback=update_all_data)

    results: list[StockRunResult] = []
    skipped: list[tuple[str, str]] = []

    for ticker in NIFTY50_TICKERS:
        try:
            analysis = analyst.analyze_stock(ticker)
        except Exception as error:
            reason = str(error)
            logger.warning("Skipping %s: %s", ticker, reason)
            skipped.append((ticker, reason))
            continue

        simulation = simulate_paper_trades(analysis.signals, initial_capital=2000.0)
        chart_path = str(save_stock_chart(analysis, simulation))

        result = StockRunResult(
            ticker=ticker,
            regime=analysis.regime.regime,
            confidence=analysis.regime.confidence,
            explanation=" ".join(analysis.insights),
            hypothetical_return_pct=simulation.total_return_pct,
            chart_path=chart_path,
            simulation=simulation,
        )
        results.append(result)
        _print_stock_result(result)

    _print_final_summary(results, skipped)

    end = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Ultron analysis ended at %s", end.isoformat())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
