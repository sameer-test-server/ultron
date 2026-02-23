"""Chart and chart-cache helpers for Ultron UI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from core.analyst import StockAnalysis
from core.paper_trader import PaperTradeResult


def plot_price_chart(analysis: StockAnalysis, simulation: PaperTradeResult, output_path: Path) -> None:
    """Save static price chart for reuse and PDF export."""
    data = analysis.data
    fig = Figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)

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
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle

    ax.legend(unique.values(), unique.keys(), loc="upper left")
    ax.set_title(f"{analysis.ticker} Historical Price with Moving Averages")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)


def plot_rsi_chart(analysis: StockAnalysis, output_path: Path) -> None:
    """Save static RSI chart for reuse and PDF export."""
    data = analysis.data
    fig = Figure(figsize=(11, 3.2))
    ax = fig.add_subplot(111)

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


def ensure_static_charts(analysis: StockAnalysis, simulation: PaperTradeResult, charts_dir: Path) -> tuple[str, str]:
    """Generate static charts once and reuse cached files."""
    price_file = charts_dir / f"{analysis.ticker}_price.png"
    rsi_file = charts_dir / f"{analysis.ticker}_rsi.png"

    if not price_file.exists():
        plot_price_chart(analysis, simulation, price_file)
    if not rsi_file.exists():
        plot_rsi_chart(analysis, rsi_file)

    return (f"charts/{price_file.name}", f"charts/{rsi_file.name}")


def cache_busted_chart_url(
    static_dir: Path,
    relative_path: str | None,
    url_for_fn: Callable[..., str],
) -> str | None:
    """Return static chart URL with cache-busting query parameter."""
    if not relative_path:
        return None
    chart_file = static_dir / relative_path
    if not chart_file.exists():
        return None
    version = chart_file.stat().st_mtime_ns
    return f"{url_for_fn('static', filename=relative_path)}?v={version}"


def build_interactive_chart(analysis: StockAnalysis, simulation: PaperTradeResult) -> str:
    """Build local interactive Plotly chart with price indicators and RSI."""
    data = analysis.data
    date_series = pd.to_datetime(data["Date"], errors="coerce").dropna().sort_values()
    recent_start: pd.Timestamp | None = None
    recent_end: pd.Timestamp | None = None
    if not date_series.empty:
        full_start = date_series.iloc[0]
        recent_end = date_series.iloc[-1]
        recent_start = max(full_start, recent_end - pd.Timedelta(days=180))

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

    if recent_start is not None and recent_end is not None:
        range_pair = [recent_start.to_pydatetime(), recent_end.to_pydatetime()]
        figure.update_xaxes(range=range_pair, row=1, col=1)
        figure.update_xaxes(range=range_pair, row=2, col=1)

    figure.update_layout(
        title=f"{analysis.ticker} Interactive Historical View",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
        xaxis2={
            "title": "Date",
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "ALL"},
                ]
            },
            "rangeslider": {"visible": True, "thickness": 0.06},
            "type": "date",
        },
        yaxis={"title": "Price"},
        yaxis2={"title": "RSI", "range": [0, 100]},
    )

    return plot(
        figure,
        output_type="div",
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
    )


def chart_window_dates(data: pd.DataFrame) -> dict[str, str]:
    """Build date boundaries for recent vs full-history chart controls."""
    date_series = pd.to_datetime(data["Date"], errors="coerce").dropna().sort_values()
    if date_series.empty:
        return {
            "recent_start": "",
            "recent_end": "",
            "full_start": "",
            "full_end": "",
        }

    full_start = date_series.iloc[0]
    full_end = date_series.iloc[-1]
    recent_start = max(full_start, full_end - pd.Timedelta(days=180))

    return {
        "recent_start": recent_start.strftime("%Y-%m-%d"),
        "recent_end": full_end.strftime("%Y-%m-%d"),
        "full_start": full_start.strftime("%Y-%m-%d"),
        "full_end": full_end.strftime("%Y-%m-%d"),
    }
