"""Enhanced local read-only Flask UI for Ultron analysis visualization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, abort, render_template, request, send_file
from plotly.offline import get_plotlyjs, plot
from plotly.subplots import make_subplots

# Make `python ui/app.py` work without external PYTHONPATH setup.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.nifty50 import NIFTY50_TICKERS
from config.settings import BASE_DIR
from core.analyst import StockAnalysis, StockAnalyst
from core.paper_trader import PaperTradeResult, simulate_paper_trades
from utils.pdf_exporter import PDF_DIR, build_stock_pdf


BASE_PATH = Path(BASE_DIR)
UI_DIR = BASE_PATH / "ui"
STATIC_DIR = UI_DIR / "static"
CHARTS_DIR = STATIC_DIR / "charts"


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
    interactive_chart_html: str
    chart_price_path: str
    chart_rsi_path: str
    simulation: PaperTradeResult


def create_app() -> Flask:
    """Create and configure the local Flask application."""
    app = Flask(
        __name__,
        template_folder=str(UI_DIR / "templates"),
        static_folder=str(STATIC_DIR),
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    analyst = StockAnalyst(refresh_callback=None)
    state: dict[str, Any] = {
        "last_run": None,
        "stocks": {},
        "errors": {},
        "data_signature": None,
    }

    def _compute_data_signature() -> tuple[int, int, int]:
        """
        Compute a lightweight signature of `data/raw` contents.
        Any CSV add/update/removal changes this signature.
        """
        raw_dir = BASE_PATH / "data" / "raw"
        if not raw_dir.exists():
            return (0, 0, 0)

        csv_files = sorted(raw_dir.glob("*.csv"))
        if not csv_files:
            return (0, 0, 0)

        latest_mtime = max(path.stat().st_mtime_ns for path in csv_files)
        total_size = sum(path.stat().st_size for path in csv_files)
        return (len(csv_files), total_size, latest_mtime)

    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert optional numeric values to finite float safely."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def confidence_label(value: float) -> str:
        """Convert numeric confidence to label."""
        if value >= 0.75:
            return "HIGH"
        if value >= 0.60:
            return "MEDIUM"
        return "LOW"

    def volatility_label(value: float) -> str:
        """Classify annualized volatility into intuitive levels."""
        if value < 0.20:
            return "Low"
        if value < 0.35:
            return "Medium"
        return "High"

    def _plot_price_chart(analysis: StockAnalysis, simulation: PaperTradeResult, output_path: Path) -> None:
        """Save static price chart for reuse and PDF export."""
        data = analysis.data
        fig, ax = plt.subplots(figsize=(11, 5.5))

        ax.plot(data["Date"], data["Close"], label="Close", linewidth=1.6)
        ax.plot(data["Date"], data["SMA_20"], label="SMA 20", linewidth=1.0)
        ax.plot(data["Date"], data["SMA_50"], label="SMA 50", linewidth=1.0)
        ax.plot(data["Date"], data["SMA_200"], label="SMA 200", linewidth=1.0)
        ax.plot(data["Date"], data["EMA_20"], label="EMA 20", linestyle="--", linewidth=1.0)

        for trade in simulation.trades:
            entry_date = pd.to_datetime(trade.entry_date)
            exit_date = pd.to_datetime(trade.exit_date)
            ax.scatter(entry_date, trade.entry_price, marker="^", s=80, color="green", label="Hypothetical Entry")
            ax.scatter(exit_date, trade.exit_price, marker="v", s=80, color="red", label="Hypothetical Exit")

        handles, labels = ax.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique[label] = handle

        ax.legend(unique.values(), unique.keys(), loc="upper left")
        ax.set_title(f"{analysis.ticker} Historical Price with Moving Averages")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=130)
        plt.close(fig)

    def _plot_rsi_chart(analysis: StockAnalysis, output_path: Path) -> None:
        """Save static RSI chart for reuse and PDF export."""
        data = analysis.data
        fig, ax = plt.subplots(figsize=(11, 3.2))

        ax.plot(data["Date"], data["RSI_14"], label="RSI 14", color="tab:purple", linewidth=1.3)
        ax.axhline(70, color="red", linestyle="--", linewidth=0.9)
        ax.axhline(30, color="green", linestyle="--", linewidth=0.9)
        ax.set_ylim(0, 100)
        ax.set_title(f"{analysis.ticker} RSI")
        ax.set_ylabel("RSI")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=130)
        plt.close(fig)

    def _ensure_static_charts(analysis: StockAnalysis, simulation: PaperTradeResult) -> tuple[str, str]:
        """Generate static charts once and reuse cached files."""
        price_file = CHARTS_DIR / f"{analysis.ticker}_price.png"
        rsi_file = CHARTS_DIR / f"{analysis.ticker}_rsi.png"

        if not price_file.exists():
            _plot_price_chart(analysis, simulation, price_file)
        if not rsi_file.exists():
            _plot_rsi_chart(analysis, rsi_file)

        return (f"charts/{price_file.name}", f"charts/{rsi_file.name}")

    def _build_interactive_chart(analysis: StockAnalysis, simulation: PaperTradeResult) -> str:
        """Build local interactive Plotly chart with price indicators and RSI."""
        data = analysis.data
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
            subplot_titles=("Price + SMA/EMA", "RSI"),
        )

        figure.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                name="Close",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        for column, label, dash in [
            ("SMA_20", "SMA 20", None),
            ("SMA_50", "SMA 50", None),
            ("SMA_200", "SMA 200", None),
            ("EMA_20", "EMA 20", "dash"),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data[column],
                    mode="lines",
                    name=label,
                    line={"dash": dash} if dash else None,
                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if simulation.trades:
            entry_x = [pd.to_datetime(item.entry_date) for item in simulation.trades]
            entry_y = [item.entry_price for item in simulation.trades]
            exit_x = [pd.to_datetime(item.exit_date) for item in simulation.trades]
            exit_y = [item.exit_price for item in simulation.trades]

            figure.add_trace(
                go.Scatter(
                    x=entry_x,
                    y=entry_y,
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 10, "color": "green"},
                    name="Hypothetical Entry",
                    hovertemplate="Entry: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=exit_x,
                    y=exit_y,
                    mode="markers",
                    marker={"symbol": "triangle-down", "size": 10, "color": "red"},
                    name="Hypothetical Exit",
                    hovertemplate="Exit: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        figure.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["RSI_14"],
                mode="lines",
                name="RSI 14",
                line={"color": "purple"},
                hovertemplate="Date: %{x|%Y-%m-%d}<br>RSI: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        figure.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        figure.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        figure.update_layout(
            title=f"{analysis.ticker} Interactive Historical View",
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "h", "y": 1.12},
            margin={"l": 30, "r": 20, "t": 60, "b": 30},
            xaxis2={"title": "Date"},
            yaxis={"title": "Price"},
            yaxis2={"title": "RSI", "range": [0, 100]},
        )

        return plot(
            figure,
            output_type="div",
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )

    def _build_notes(analysis: StockAnalysis, vol_label: str, vol_value: float) -> tuple[str, str, str]:
        """Create beginner-friendly overview notes for trend/momentum/volatility."""
        data = analysis.data
        latest = data.iloc[-1]

        close = _safe_float(latest["Close"])
        sma_50 = _safe_float(latest.get("SMA_50"))
        sma_200 = _safe_float(latest.get("SMA_200"))
        rsi = _safe_float(latest.get("RSI_14"), default=50.0)

        if close > sma_50 > 0 and close > sma_200 > 0:
            trend_note = "Historical observation: Price is above key moving averages, showing a stronger trend."
        elif close < sma_50 and sma_50 > 0:
            trend_note = "Historical observation: Price is below the 50-day average, indicating weaker trend strength."
        else:
            trend_note = "Historical observation: Trend signals are mixed in the recent data window."

        if rsi < 30:
            momentum_note = "Historical observation: Momentum is weak (oversold RSI zone)."
        elif rsi > 70:
            momentum_note = "Historical observation: Momentum is stretched (overbought RSI zone)."
        else:
            momentum_note = "Historical observation: Momentum is balanced (neutral RSI zone)."

        volatility_note = (
            "Historical observation: "
            f"Volatility is {vol_label.lower()} at about {vol_value * 100:.1f}% annualized."
        )

        return trend_note, momentum_note, volatility_note

    def _run_analysis() -> None:
        """Compute analysis from local CSV files only and cache results in memory."""
        stocks: dict[str, StockViewModel] = {}
        errors: dict[str, str] = {}

        for ticker in NIFTY50_TICKERS:
            try:
                analysis = analyst.analyze_stock(ticker)
                simulation = simulate_paper_trades(analysis.signals, initial_capital=2000.0)
                chart_price_path, chart_rsi_path = _ensure_static_charts(analysis, simulation)

                vol_value = _safe_float(analysis.data.iloc[-1].get("VOLATILITY_20"))
                vol_level = volatility_label(vol_value)
                trend_note, momentum_note, vol_note = _build_notes(analysis, vol_level, vol_value)

                stocks[ticker] = StockViewModel(
                    ticker=ticker,
                    regime=analysis.regime.regime,
                    confidence=analysis.regime.confidence,
                    confidence_label=confidence_label(analysis.regime.confidence),
                    volatility_value=vol_value,
                    volatility_label=vol_level,
                    hypothetical_return_pct=simulation.total_return_pct,
                    explanation=" ".join(analysis.insights),
                    insights=analysis.insights,
                    trend_note=trend_note,
                    momentum_note=momentum_note,
                    volatility_note=vol_note,
                    interactive_chart_html=_build_interactive_chart(analysis, simulation),
                    chart_price_path=chart_price_path,
                    chart_rsi_path=chart_rsi_path,
                    simulation=simulation,
                )
            except Exception as exc:
                errors[ticker] = str(exc)

        state["stocks"] = stocks
        state["errors"] = errors
        state["last_run"] = datetime.now()
        state["data_signature"] = _compute_data_signature()

    def _ensure_state() -> None:
        """Ensure the in-memory cache is populated for current process."""
        current_signature = _compute_data_signature()
        if (
            not state["stocks"]
            and not state["errors"]
        ) or state["data_signature"] != current_signature:
            _run_analysis()

    @app.route("/")
    def index() -> str:
        """Interactive dashboard with filters, sorting, and summary metrics."""
        _ensure_state()

        regime_filter = request.args.get("regime", "ALL").upper()
        sort_by = request.args.get("sort", "highest_return")

        stocks_map: dict[str, StockViewModel] = state["stocks"]
        stocks = list(stocks_map.values())

        if regime_filter in {"LONG_TERM", "SHORT_TERM"}:
            stocks = [item for item in stocks if item.regime == regime_filter]
        else:
            regime_filter = "ALL"

        if sort_by == "lowest_risk":
            stocks.sort(key=lambda item: item.volatility_value)
        elif sort_by == "highest_confidence":
            stocks.sort(key=lambda item: item.confidence, reverse=True)
        else:
            sort_by = "highest_return"
            stocks.sort(key=lambda item: item.hypothetical_return_pct, reverse=True)

        all_stocks = list(stocks_map.values())
        long_term_count = sum(1 for item in all_stocks if item.regime == "LONG_TERM")
        short_term_count = sum(1 for item in all_stocks if item.regime == "SHORT_TERM")

        best_return = max((item.hypothetical_return_pct for item in all_stocks), default=0.0)
        worst_return = min((item.hypothetical_return_pct for item in all_stocks), default=0.0)

        return render_template(
            "index.html",
            stocks=stocks,
            errors=state["errors"],
            total_analyzed=len(all_stocks),
            displayed_count=len(stocks),
            long_term_count=long_term_count,
            short_term_count=short_term_count,
            best_return=best_return,
            worst_return=worst_return,
            last_run=state["last_run"],
            regime_filter=regime_filter,
            sort_by=sort_by,
        )

    @app.route("/stock/<ticker>")
    def stock_detail(ticker: str) -> str:
        """Detailed stock page with interactive chart, explanation, and paper trading panel."""
        _ensure_state()
        stock = state["stocks"].get(ticker)
        if stock is None:
            reason = state["errors"].get(ticker)
            if reason is None:
                abort(404)
            return render_template("stock.html", ticker=ticker, stock=None, reason=reason, plotly_js="")

        return render_template(
            "stock.html",
            ticker=ticker,
            stock=stock,
            reason=None,
            plotly_js=get_plotlyjs(),
        )

    @app.route("/export/pdf/<ticker>")
    def export_pdf(ticker: str):
        """Export one stock analysis as a local PDF report."""
        _ensure_state()
        stock = state["stocks"].get(ticker)
        if stock is None:
            abort(404)

        paper_trade_lines = [
            f"Virtual capital: INR {stock.simulation.initial_capital:.2f}",
            f"Final capital: INR {stock.simulation.final_capital:.2f}",
            f"Total hypothetical return: {stock.hypothetical_return_pct:.2f}%",
            f"Win rate: {stock.simulation.win_rate:.1f}% ({len(stock.simulation.trades)} trades)",
        ]

        for trade in stock.simulation.trades[:6]:
            outcome = "Win" if trade.profit_loss >= 0 else "Loss"
            paper_trade_lines.append(
                f"{trade.entry_date} @ {trade.entry_price:.2f} -> {trade.exit_date} @ {trade.exit_price:.2f} | "
                f"P/L {trade.return_pct:.2f}% ({outcome})"
            )

        pdf_path = build_stock_pdf(
            ticker=stock.ticker,
            regime=stock.regime,
            confidence_label=stock.confidence_label,
            confidence_value=stock.confidence,
            volatility_label=stock.volatility_label,
            explanation_points=[stock.trend_note, stock.momentum_note, stock.volatility_note, *stock.insights],
            paper_trade_lines=paper_trade_lines,
            disclaimer=(
                "Disclaimer: Historical observation and paper trade only. "
                "Ultron does not place real trades."
            ),
            price_chart_path=STATIC_DIR / stock.chart_price_path,
            rsi_chart_path=STATIC_DIR / stock.chart_rsi_path,
        )

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_path.name,
            mimetype="application/pdf",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

# Run UI:
#   cd /home/madara/ultron
#   python ui/app.py
# Export PDF:
#   Open stock page and click "Export Analysis to PDF"
#   or directly visit: http://127.0.0.1:5000/export/pdf/<TICKER>
# PDFs storage:
#   /home/madara/ultron/reports/pdf/
