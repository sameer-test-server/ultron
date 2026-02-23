"""Enhanced local read-only Flask UI for Ultron analysis visualization."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any
from urllib.parse import quote_plus

MPL_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import yfinance as yf
from flask import Flask, abort, got_request_exception, jsonify, render_template, request, send_file, url_for
from plotly.offline import get_plotlyjs

# Make `python ui/app.py` work without external PYTHONPATH setup.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config.nifty50 import NIFTY50_TICKERS
from config.settings import BASE_DIR
from core.analyst import StockAnalysis, StockAnalyst
from core.paper_trader import simulate_paper_trades
from ui.api import parse_float, parse_int, projection_payload, serialize_simulation
from ui.charts import (
    build_interactive_chart,
    cache_busted_chart_url,
    chart_window_dates,
    ensure_static_charts,
)
from ui.models import StockViewModel
from ui.utils.pdf_exporter import PDF_DIR, build_stock_pdf


BASE_PATH = Path(BASE_DIR)
UI_DIR = BASE_PATH / "ui"
STATIC_DIR = UI_DIR / "static"
CHARTS_DIR = STATIC_DIR / "charts"

LIVE_TRACKER_DEFAULT_LIMIT = 50
LIVE_TRACKER_MAX_TICKERS = 50
LIVE_TRACKER_REFRESH_SECONDS = 20
LIVE_TRACKER_CACHE_SECONDS = 8
ANALYSIS_REFRESH_INTERVAL_SECONDS = 300
ANALYSIS_STATUS_POLL_SECONDS = 4

SIM_DEFAULT_INITIAL_CAPITAL = 2000.0
SIM_DEFAULT_POSITION_SIZE_PCT = 100.0
SIM_DEFAULT_BROKERAGE_PCT = 0.08
SIM_DEFAULT_SLIPPAGE_PCT = 0.05
SIM_DEFAULT_MAX_HOLD_DAYS = 60
SIM_DEFAULT_STOP_LOSS_PCT = 8.0
SIM_DEFAULT_TAKE_PROFIT_PCT = 18.0

PROJECTION_DEFAULT_HORIZONS = [1, 5, 20]
PROJECTION_MAX_HORIZON_DAYS = 90

SECTOR_BY_TICKER: dict[str, str] = {
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "Financial Services",
    "BAJAJFINSV.NS": "Financial Services",
    "SBILIFE.NS": "Financial Services",
    "HDFCLIFE.NS": "Financial Services",
    "TCS.NS": "Information Technology",
    "INFY.NS": "Information Technology",
    "HCLTECH.NS": "Information Technology",
    "TECHM.NS": "Information Technology",
    "WIPRO.NS": "Information Technology",
    "LTIM.NS": "Information Technology",
    "MARUTI.NS": "Automobile",
    "TATAMOTORS.NS": "Automobile",
    "BAJAJ-AUTO.NS": "Automobile",
    "M&M.NS": "Automobile",
    "EICHERMOT.NS": "Automobile",
    "HEROMOTOCO.NS": "Automobile",
    "TATASTEEL.NS": "Metals & Mining",
    "JSWSTEEL.NS": "Metals & Mining",
    "HINDALCO.NS": "Metals & Mining",
    "COALINDIA.NS": "Metals & Mining",
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "BPCL.NS": "Energy",
    "POWERGRID.NS": "Utilities",
    "NTPC.NS": "Utilities",
    "SUNPHARMA.NS": "Pharma & Healthcare",
    "DRREDDY.NS": "Pharma & Healthcare",
    "CIPLA.NS": "Pharma & Healthcare",
    "DIVISLAB.NS": "Pharma & Healthcare",
    "APOLLOHOSP.NS": "Pharma & Healthcare",
    "HINDUNILVR.NS": "Consumer",
    "ITC.NS": "Consumer",
    "NESTLEIND.NS": "Consumer",
    "BRITANNIA.NS": "Consumer",
    "TATACONSUM.NS": "Consumer",
    "ASIANPAINT.NS": "Consumer",
    "TITAN.NS": "Consumer",
    "ULTRATECH.NS": "Industrials",
    "GRASIM.NS": "Industrials",
    "LT.NS": "Industrials",
    "ADANIPORTS.NS": "Industrials",
    "ADANIENT.NS": "Industrials",
    "BHARTIARTL.NS": "Telecom",
    "UPL.NS": "Chemicals",
}


def _configure_ui_logger() -> logging.Logger:
    """Configure file logger for UI app and background threads."""
    logs_dir = BASE_PATH / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ui.log"

    logger = logging.getLogger("ultron.ui")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    existing = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = handler
            break

    if existing is None:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)

    return logger


def _warn_if_multi_worker(logger: logging.Logger) -> None:
    """Log warning when likely deployed with multiple workers/processes."""
    worker_envs = {
        "WEB_CONCURRENCY": os.getenv("WEB_CONCURRENCY"),
        "GUNICORN_WORKERS": os.getenv("GUNICORN_WORKERS"),
        "WORKERS": os.getenv("WORKERS"),
    }
    for key, raw_value in worker_envs.items():
        if raw_value is None:
            continue
        try:
            workers = int(raw_value)
        except ValueError:
            continue
        if workers > 1:
            logger.warning(
                "Detected %s=%s. Run UI with a single worker to avoid duplicate scheduler threads.",
                key,
                raw_value,
            )
            return

    server_software = (os.getenv("SERVER_SOFTWARE") or "").lower()
    if "gunicorn" in server_software:
        logger.info(
            "Gunicorn runtime detected. Recommended launch: `gunicorn -w 1 --threads 8 ui.app:app`."
        )


def _data_fingerprint(data: pd.DataFrame) -> str:
    """Create a lightweight fingerprint used for cache retention decisions."""
    if data.empty:
        return "empty"
    dates = pd.to_datetime(data["Date"], errors="coerce")
    closes = pd.to_numeric(data["Close"], errors="coerce")
    last_idx = closes.last_valid_index()
    last_close = float(closes.loc[last_idx]) if last_idx is not None else 0.0
    last_date = ""
    if not dates.empty and pd.notna(dates.iloc[-1]):
        last_date = dates.iloc[-1].strftime("%Y-%m-%d")
    return f"{len(data)}|{last_date}|{last_close:.6f}"


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
    logger = _configure_ui_logger()
    _warn_if_multi_worker(logger)
    logger.info("UI app initialized")

    analyst = StockAnalyst(refresh_callback=None)
    state_lock = threading.RLock()
    analysis_thread: threading.Thread | None = None
    scheduler_thread: threading.Thread | None = None
    scheduler_stop_event = threading.Event()

    state: dict[str, Any] = {
        "last_run": None,
        "stocks": {},
        "errors": {},
        "data_signature": None,
        "live_quotes_cache": None,
        "analysis_status": "idle",
        "analysis_message": "Not started",
        "analysis_started_at": None,
        "analysis_finished_at": None,
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

    def _ensure_static_charts(analysis: Any, simulation: Any) -> tuple[str, str]:
        """Generate static charts once and reuse cached files."""
        return ensure_static_charts(analysis, simulation, CHARTS_DIR)

    def _cache_busted_chart_url(relative_path: str | None) -> str | None:
        """Return static chart URL with cache-busting query parameter."""
        return cache_busted_chart_url(STATIC_DIR, relative_path, url_for)

    def _build_interactive_chart(analysis: Any, simulation: Any) -> str:
        """Build local interactive Plotly chart with price indicators and RSI."""
        return build_interactive_chart(analysis, simulation)

    def _chart_window_dates(data: pd.DataFrame) -> dict[str, str]:
        """Build date boundaries for recent vs full-history chart controls."""
        return chart_window_dates(data)

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

    def _build_nse_quote_snapshot(analysis: StockAnalysis) -> dict[str, Any]:
        """Build NSE-style quote snapshot from local OHLCV history."""
        data = analysis.data.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")

        if data.empty:
            return {}

        latest = data.iloc[-1]
        prev_close = None
        if len(data) >= 2:
            prev_close = float(data.iloc[-2]["Close"])

        closes = data["Close"].dropna()
        highs = data["High"].dropna()
        lows = data["Low"].dropna()
        volumes = data["Volume"].dropna()

        latest_close = float(latest["Close"])
        latest_open = float(latest["Open"])
        latest_high = float(latest["High"])
        latest_low = float(latest["Low"])
        latest_volume = int(float(latest["Volume"])) if pd.notna(latest["Volume"]) else 0
        last_date = pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d")

        day_change_pct = None
        if prev_close is not None and prev_close != 0:
            day_change_pct = ((latest_close - prev_close) / prev_close) * 100.0

        trailing = data.tail(252)
        week_52_high = float(trailing["High"].max()) if not trailing.empty else float(highs.max())
        week_52_low = float(trailing["Low"].min()) if not trailing.empty else float(lows.min())

        distance_to_52w_high = ((latest_close / week_52_high) - 1.0) * 100.0 if week_52_high > 0 else 0.0
        distance_from_52w_low = ((latest_close / week_52_low) - 1.0) * 100.0 if week_52_low > 0 else 0.0

        avg_volume_20 = float(volumes.tail(20).mean()) if not volumes.empty else 0.0
        turnover_estimate = latest_close * latest_volume
        day_vwap_approx = (latest_high + latest_low + latest_close) / 3.0

        near_high = distance_to_52w_high > -2.0
        near_low = distance_from_52w_low < 2.0
        if near_high:
            position_note = "Near 52W High"
        elif near_low:
            position_note = "Near 52W Low"
        else:
            position_note = "Mid-range"

        symbol = analysis.ticker.replace(".NS", "")
        nse_url = f"https://www.nseindia.com/get-quote/equity/{quote_plus(symbol)}"

        return {
            "symbol": symbol,
            "nse_url": nse_url,
            "as_of": last_date,
            "last_price": round(latest_close, 2),
            "day_change_pct": round(day_change_pct, 2) if day_change_pct is not None else None,
            "prev_close": round(prev_close, 2) if prev_close is not None else None,
            "open": round(latest_open, 2),
            "day_high": round(latest_high, 2),
            "day_low": round(latest_low, 2),
            "day_vwap_approx": round(day_vwap_approx, 2),
            "volume": latest_volume,
            "avg_volume_20": int(round(avg_volume_20)),
            "turnover_estimate": round(turnover_estimate, 2),
            "week_52_high": round(week_52_high, 2),
            "week_52_low": round(week_52_low, 2),
            "distance_to_52w_high_pct": round(distance_to_52w_high, 2),
            "distance_from_52w_low_pct": round(distance_from_52w_low, 2),
            "position_note": position_note,
        }

    def _window_return_pct(analysis: StockAnalysis, window_days: int = 20) -> float | None:
        """Compute trailing window return from close prices."""
        closes = pd.to_numeric(analysis.data.get("Close"), errors="coerce").dropna()
        if len(closes) <= window_days:
            return None
        start = float(closes.iloc[-1 - window_days])
        end = float(closes.iloc[-1])
        if start <= 0:
            return None
        return ((end / start) - 1.0) * 100.0

    def _breakout_status(analysis: StockAnalysis) -> dict[str, Any]:
        """Get simple 52-week breakout/near-breakout status."""
        data = analysis.data.copy()
        data["High"] = pd.to_numeric(data["High"], errors="coerce")
        data["Low"] = pd.to_numeric(data["Low"], errors="coerce")
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna(subset=["High", "Low", "Close"])
        if data.empty:
            return {"label": "No data", "distance_to_high_pct": None, "distance_from_low_pct": None}

        trailing = data.tail(252)
        high_52w = float(trailing["High"].max())
        low_52w = float(trailing["Low"].min())
        last_close = float(trailing.iloc[-1]["Close"])

        distance_to_high_pct = ((last_close / high_52w) - 1.0) * 100.0 if high_52w > 0 else None
        distance_from_low_pct = ((last_close / low_52w) - 1.0) * 100.0 if low_52w > 0 else None

        label = "Mid-range"
        if distance_to_high_pct is not None and distance_to_high_pct >= 0:
            label = "52W Breakout"
        elif distance_from_low_pct is not None and distance_from_low_pct <= 0:
            label = "52W Breakdown"
        elif distance_to_high_pct is not None and distance_to_high_pct >= -2.0:
            label = "Near 52W High"
        elif distance_from_low_pct is not None and distance_from_low_pct <= 2.0:
            label = "Near 52W Low"

        return {
            "label": label,
            "distance_to_high_pct": round(distance_to_high_pct, 2) if distance_to_high_pct is not None else None,
            "distance_from_low_pct": round(distance_from_low_pct, 2) if distance_from_low_pct is not None else None,
        }

    def _circuit_proxy(analysis: StockAnalysis) -> dict[str, Any]:
        """Estimate an intraday risk band around last close (proxy, not exchange limits)."""
        closes = pd.to_numeric(analysis.data.get("Close"), errors="coerce").dropna()
        if closes.empty:
            return {}
        daily_returns = closes.pct_change().dropna().tail(20)
        if daily_returns.empty:
            band_pct = 3.0
        else:
            band_pct = float(daily_returns.std(ddof=0) * 100.0 * 2.5)
            band_pct = max(2.0, min(10.0, band_pct))

        last_close = float(closes.iloc[-1])
        return {
            "band_pct": round(band_pct, 2),
            "upper": round(last_close * (1.0 + band_pct / 100.0), 2),
            "lower": round(last_close * (1.0 - band_pct / 100.0), 2),
            "reference_close": round(last_close, 2),
        }

    def _stock_context_cards(ticker: str, stock: StockViewModel, stocks_map: dict[str, StockViewModel]) -> dict[str, Any]:
        """Build compact context cards for stock page."""
        sector_name = SECTOR_BY_TICKER.get(ticker, "Diversified")
        stock_return = _window_return_pct(stock.analysis, window_days=20)

        sector_returns: list[float] = []
        index_returns: list[float] = []
        for peer_ticker, peer_stock in stocks_map.items():
            peer_ret = _window_return_pct(peer_stock.analysis, window_days=20)
            if peer_ret is None:
                continue
            index_returns.append(peer_ret)
            if SECTOR_BY_TICKER.get(peer_ticker, "Diversified") == sector_name:
                sector_returns.append(peer_ret)

        sector_avg = (sum(sector_returns) / len(sector_returns)) if sector_returns else None
        index_avg = (sum(index_returns) / len(index_returns)) if index_returns else None

        relative_vs_sector = (stock_return - sector_avg) if (stock_return is not None and sector_avg is not None) else None
        relative_vs_index = (stock_return - index_avg) if (stock_return is not None and index_avg is not None) else None

        return {
            "sector_name": sector_name,
            "stock_20d_return_pct": round(stock_return, 2) if stock_return is not None else None,
            "sector_20d_return_pct": round(sector_avg, 2) if sector_avg is not None else None,
            "index_20d_return_pct": round(index_avg, 2) if index_avg is not None else None,
            "relative_vs_sector_pct": round(relative_vs_sector, 2) if relative_vs_sector is not None else None,
            "relative_vs_index_pct": round(relative_vs_index, 2) if relative_vs_index is not None else None,
            "breakout": _breakout_status(stock.analysis),
            "circuit_proxy": _circuit_proxy(stock.analysis),
        }

    def _build_dashboard_breakout_watchlist(stocks_map: dict[str, StockViewModel], limit: int = 8) -> list[dict[str, Any]]:
        """Create compact breakout watchlist for dashboard."""
        watch: list[dict[str, Any]] = []
        for ticker, stock in stocks_map.items():
            status = _breakout_status(stock.analysis)
            label = status.get("label")
            if label not in {"52W Breakout", "Near 52W High", "Near 52W Low", "52W Breakdown"}:
                continue

            dist_high = status.get("distance_to_high_pct")
            dist_low = status.get("distance_from_low_pct")
            if label in {"52W Breakout", "Near 52W High"} and dist_high is not None:
                priority = abs(float(dist_high))
            elif label in {"52W Breakdown", "Near 52W Low"} and dist_low is not None:
                priority = abs(float(dist_low))
            else:
                priority = 99.0

            watch.append(
                {
                    "ticker": ticker,
                    "label": label,
                    "distance_to_high_pct": dist_high,
                    "distance_from_low_pct": dist_low,
                    "priority": priority,
                }
            )

        watch.sort(key=lambda item: item["priority"])
        return watch[:limit]

    def _parse_float(raw_value: str | None, default: float, min_value: float, max_value: float) -> float:
        """Parse bounded float from request args."""
        return parse_float(raw_value, default, min_value, max_value)

    def _parse_int(raw_value: str | None, default: int, min_value: int, max_value: int) -> int:
        """Parse bounded int from request args."""
        return parse_int(raw_value, default, min_value, max_value)

    def _serialize_simulation(result: Any) -> dict[str, Any]:
        """Convert simulation dataclass to API payload."""
        return serialize_simulation(result)

    def _projection_payload(analysis: Any, amount: float, horizon_days: int) -> dict[str, Any]:
        """Build forward estimate from historical daily returns."""
        return projection_payload(analysis, amount, horizon_days)

    def _run_analysis_snapshot() -> tuple[dict[str, StockViewModel], dict[str, str]]:
        """Compute analysis from local CSV files only and return fresh payload."""
        stocks: dict[str, StockViewModel] = {}
        errors: dict[str, str] = {}

        for ticker in NIFTY50_TICKERS:
            try:
                analysis = analyst.analyze_stock(ticker)
                simulation = simulate_paper_trades(analysis.signals, initial_capital=SIM_DEFAULT_INITIAL_CAPITAL)

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
                    simulation=simulation,
                    analysis=analysis,
                    data_fingerprint=_data_fingerprint(analysis.data),
                )
            except Exception as exc:
                errors[ticker] = str(exc)
                logger.warning("Analysis skipped for %s: %s", ticker, exc)

        return stocks, errors

    def _analysis_worker(expected_signature: tuple[int, int, int]) -> None:
        """Background worker that refreshes analysis cache."""
        nonlocal analysis_thread
        try:
            logger.info("Background analysis started")
            stocks, errors = _run_analysis_snapshot()
            with state_lock:
                existing_stocks: dict[str, StockViewModel] = state["stocks"]
                for ticker, refreshed in stocks.items():
                    existing = existing_stocks.get(ticker)
                    if existing is None:
                        continue
                    if existing.data_fingerprint != refreshed.data_fingerprint:
                        continue
                    refreshed.interactive_chart_html = existing.interactive_chart_html
                    refreshed.chart_price_path = existing.chart_price_path
                    refreshed.chart_rsi_path = existing.chart_rsi_path

                state["stocks"] = stocks
                state["errors"] = errors
                state["last_run"] = datetime.now()
                state["data_signature"] = expected_signature
                state["analysis_status"] = "ready"
                state["analysis_message"] = "Analysis ready"
                state["analysis_finished_at"] = datetime.now()
            logger.info(
                "Background analysis completed: %s stocks, %s errors",
                len(stocks),
                len(errors),
            )
        except Exception as exc:
            logger.exception("Background analysis crashed: %s", exc)
            with state_lock:
                state["analysis_status"] = "error"
                state["analysis_message"] = f"Analysis failed: {exc}"
                state["analysis_finished_at"] = datetime.now()
        finally:
            with state_lock:
                analysis_thread = None

    def _schedule_analysis_if_needed(force: bool = False) -> bool:
        """Start analysis in background if cache is missing or data changed."""
        nonlocal analysis_thread
        current_signature = _compute_data_signature()

        with state_lock:
            has_cache = bool(state["stocks"]) or bool(state["errors"])
            signature_changed = state["data_signature"] != current_signature
            needs_refresh = force or signature_changed or not has_cache

            if not needs_refresh:
                return False

            if analysis_thread is not None and analysis_thread.is_alive():
                return True

            state["analysis_status"] = "running"
            state["analysis_message"] = "Refreshing data in background"
            state["analysis_started_at"] = datetime.now()

            analysis_thread = threading.Thread(
                target=_analysis_worker,
                args=(current_signature,),
                name="ultron-analysis-worker",
                daemon=True,
            )
            analysis_thread.start()
            logger.info("Scheduled background analysis refresh")
            return True

    def _start_scheduler() -> None:
        """Run periodic non-blocking analysis checks."""
        nonlocal scheduler_thread
        with state_lock:
            if scheduler_thread is not None and scheduler_thread.is_alive():
                return

        def _loop() -> None:
            _schedule_analysis_if_needed(force=True)
            while not scheduler_stop_event.wait(ANALYSIS_REFRESH_INTERVAL_SECONDS):
                _schedule_analysis_if_needed(force=False)

        scheduler_thread = threading.Thread(
            target=_loop,
            name="ultron-analysis-scheduler",
            daemon=True,
        )
        scheduler_thread.start()
        logger.info(
            "Analysis scheduler started (interval=%ss)",
            ANALYSIS_REFRESH_INTERVAL_SECONDS,
        )

    def _snapshot_state() -> dict[str, Any]:
        """Read a consistent snapshot of mutable UI state."""
        with state_lock:
            return {
                "last_run": state["last_run"],
                "stocks": dict(state["stocks"]),
                "errors": dict(state["errors"]),
                "data_signature": state["data_signature"],
                "live_quotes_cache": state["live_quotes_cache"],
                "analysis_status": state["analysis_status"],
                "analysis_message": state["analysis_message"],
                "analysis_started_at": state["analysis_started_at"],
                "analysis_finished_at": state["analysis_finished_at"],
            }

    def _ensure_state() -> None:
        """Ensure scheduler is running and refresh is queued when needed."""
        _start_scheduler()
        _schedule_analysis_if_needed(force=False)

    def _sanitize_requested_tickers(raw_tickers: list[str]) -> list[str]:
        """Validate and deduplicate tickers for live tracking requests."""
        allowed = set(NIFTY50_TICKERS)
        cleaned: list[str] = []
        seen: set[str] = set()

        for ticker in raw_tickers:
            normalized = ticker.strip().upper()
            if not normalized or normalized not in allowed or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
            if len(cleaned) >= LIVE_TRACKER_MAX_TICKERS:
                break

        return cleaned

    def _chunk_tickers(tickers: list[str], chunk_size: int = 10) -> list[list[str]]:
        """Split tickers into fixed chunks for stable live quote fetches."""
        if chunk_size <= 0:
            chunk_size = 10
        return [tickers[idx : idx + chunk_size] for idx in range(0, len(tickers), chunk_size)]

    def _extract_series(dataframe: pd.DataFrame, ticker: str, column: str) -> pd.Series:
        """Extract one numeric series for a ticker from yfinance output."""
        if dataframe.empty:
            return pd.Series(dtype="float64")

        series = pd.Series(dtype="float64")
        if isinstance(dataframe.columns, pd.MultiIndex):
            first_level = dataframe.columns.get_level_values(0)
            if ticker in first_level:
                scoped = dataframe[ticker]
                if column in scoped.columns:
                    series = scoped[column]
            else:
                try:
                    series = dataframe[column][ticker]
                except Exception:
                    series = pd.Series(dtype="float64")
        elif column in dataframe.columns:
            series = dataframe[column]

        if series.empty:
            return pd.Series(dtype="float64")
        return pd.to_numeric(series, errors="coerce").dropna()

    def _build_live_quote(ticker: str, intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict[str, Any] | None:
        """Assemble one live quote payload from intraday + daily market data."""
        intraday_close = _extract_series(intraday_df, ticker, "Close")
        intraday_high = _extract_series(intraday_df, ticker, "High")
        intraday_low = _extract_series(intraday_df, ticker, "Low")
        intraday_volume = _extract_series(intraday_df, ticker, "Volume")

        daily_close = _extract_series(daily_df, ticker, "Close")
        daily_high = _extract_series(daily_df, ticker, "High")
        daily_low = _extract_series(daily_df, ticker, "Low")
        daily_volume = _extract_series(daily_df, ticker, "Volume")

        if not intraday_close.empty:
            last_price = float(intraday_close.iloc[-1])
            as_of_ts = pd.to_datetime(intraday_close.index[-1], errors="coerce")
            mode = "LIVE"
        elif not daily_close.empty:
            last_price = float(daily_close.iloc[-1])
            as_of_ts = pd.to_datetime(daily_close.index[-1], errors="coerce")
            mode = "DELAYED"
        else:
            return None

        prev_close = float(daily_close.iloc[-2]) if len(daily_close) >= 2 else None
        if prev_close and prev_close != 0:
            change_pct = ((last_price - prev_close) / prev_close) * 100.0
        else:
            change_pct = None

        if not intraday_high.empty:
            day_high = float(intraday_high.max())
        elif not daily_high.empty:
            day_high = float(daily_high.iloc[-1])
        else:
            day_high = None

        if not intraday_low.empty:
            day_low = float(intraday_low.min())
        elif not daily_low.empty:
            day_low = float(daily_low.iloc[-1])
        else:
            day_low = None

        if not intraday_volume.empty:
            day_volume = int(intraday_volume.fillna(0).sum())
        elif not daily_volume.empty:
            day_volume = int(float(daily_volume.iloc[-1]))
        else:
            day_volume = 0

        as_of_text = "N/A"
        if pd.notna(as_of_ts):
            as_of_text = pd.Timestamp(as_of_ts).strftime("%Y-%m-%d %H:%M")

        return {
            "ticker": ticker,
            "last_price": round(last_price, 2),
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "day_high": round(day_high, 2) if day_high is not None else None,
            "day_low": round(day_low, 2) if day_low is not None else None,
            "day_volume": day_volume,
            "as_of": as_of_text,
            "mode": mode,
        }

    def _build_local_quote_from_csv(ticker: str) -> dict[str, Any] | None:
        """Fallback quote payload from local raw CSV when live sources are unavailable."""
        csv_path = BASE_PATH / "data" / "raw" / f"{ticker}.csv"
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path, usecols=["Date", "High", "Low", "Close", "Volume"])
        except Exception:
            return None

        if df.empty:
            return None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        if df.empty:
            return None

        latest = df.iloc[-1]
        prev_close = None
        if len(df) >= 2:
            prev_close = pd.to_numeric(df.iloc[-2].get("Close"), errors="coerce")

        last_price = pd.to_numeric(latest.get("Close"), errors="coerce")
        day_high = pd.to_numeric(latest.get("High"), errors="coerce")
        day_low = pd.to_numeric(latest.get("Low"), errors="coerce")
        day_volume = pd.to_numeric(latest.get("Volume"), errors="coerce")
        if pd.isna(last_price):
            return None

        change_pct = None
        if prev_close is not None and not pd.isna(prev_close) and float(prev_close) != 0:
            change_pct = ((float(last_price) - float(prev_close)) / float(prev_close)) * 100.0

        return {
            "ticker": ticker,
            "last_price": round(float(last_price), 2),
            "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
            "day_high": round(float(day_high), 2) if not pd.isna(day_high) else None,
            "day_low": round(float(day_low), 2) if not pd.isna(day_low) else None,
            "day_volume": int(float(day_volume)) if not pd.isna(day_volume) else 0,
            "as_of": pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d"),
            "mode": "LOCAL",
        }

    def _build_local_quotes(tickers: list[str]) -> list[dict[str, Any]]:
        """Build fallback quote list from available local CSV data."""
        quotes: list[dict[str, Any]] = []
        for ticker in tickers:
            payload = _build_local_quote_from_csv(ticker)
            if payload is not None:
                quotes.append(payload)
        return quotes

    def _fetch_live_quotes(tickers: list[str]) -> list[dict[str, Any]]:
        """Fetch live-ish quotes from yfinance for dashboard tracking."""
        if not tickers:
            return []

        quotes: list[dict[str, Any]] = []
        for ticker_chunk in _chunk_tickers(tickers, chunk_size=10):
            joined = " ".join(ticker_chunk)

            try:
                intraday_df = yf.download(
                    joined,
                    period="1d",
                    interval="1m",
                    auto_adjust=False,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    prepost=False,
                    timeout=6,
                )
            except Exception:
                intraday_df = pd.DataFrame()

            try:
                daily_df = yf.download(
                    joined,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    timeout=6,
                )
            except Exception:
                daily_df = pd.DataFrame()

            for ticker in ticker_chunk:
                try:
                    payload = _build_live_quote(ticker, intraday_df, daily_df)
                    if payload is not None:
                        quotes.append(payload)
                except Exception:
                    continue
        return quotes

    def _get_live_quotes_cached(tickers: list[str]) -> tuple[list[dict[str, Any]], datetime]:
        """Cache live quote fetches briefly to keep dashboard refresh stable."""
        now = datetime.now()
        with state_lock:
            cache = state.get("live_quotes_cache")
        key = tuple(tickers)

        if cache:
            cached_key = cache.get("key")
            cached_at = cache.get("fetched_at")
            if cached_key == key and isinstance(cached_at, datetime):
                age = (now - cached_at).total_seconds()
                if age <= LIVE_TRACKER_CACHE_SECONDS:
                    return cache.get("quotes", []), cached_at

        quotes = _fetch_live_quotes(tickers)
        if quotes:
            with state_lock:
                state["live_quotes_cache"] = {
                    "key": key,
                    "fetched_at": now,
                    "quotes": quotes,
                }
            return quotes, now

        local_quotes = _build_local_quotes(tickers)
        if local_quotes:
            with state_lock:
                state["live_quotes_cache"] = {
                    "key": key,
                    "fetched_at": now,
                    "quotes": local_quotes,
                }
            return local_quotes, now

        if cache and cache.get("key") == key and cache.get("quotes"):
            return cache.get("quotes", []), cache.get("fetched_at", now)

        with state_lock:
            state["live_quotes_cache"] = {
                "key": key,
                "fetched_at": now,
                "quotes": [],
            }
        return [], now

    @app.route("/")
    def index() -> str:
        """Interactive dashboard with filters, sorting, and summary metrics."""
        _ensure_state()
        snapshot = _snapshot_state()

        regime_filter = request.args.get("regime", "ALL").upper()
        sort_by = request.args.get("sort", "highest_return")

        stocks_map: dict[str, StockViewModel] = snapshot["stocks"]
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
        live_tracker_tickers = [item.ticker for item in stocks[:LIVE_TRACKER_DEFAULT_LIMIT]]
        if not live_tracker_tickers:
            live_tracker_tickers = NIFTY50_TICKERS[:LIVE_TRACKER_DEFAULT_LIMIT]

        analysis_status = snapshot["analysis_status"]
        analysis_message = snapshot["analysis_message"]
        processing = analysis_status == "running"
        breakout_watchlist = _build_dashboard_breakout_watchlist(stocks_map, limit=8)

        return render_template(
            "index.html",
            stocks=stocks,
            errors=snapshot["errors"],
            total_analyzed=len(all_stocks),
            displayed_count=len(stocks),
            long_term_count=long_term_count,
            short_term_count=short_term_count,
            best_return=best_return,
            worst_return=worst_return,
            last_run=snapshot["last_run"],
            regime_filter=regime_filter,
            sort_by=sort_by,
            live_tracker_tickers=live_tracker_tickers,
            live_refresh_seconds=LIVE_TRACKER_REFRESH_SECONDS,
            analysis_status=analysis_status,
            analysis_message=analysis_message,
            processing=processing,
            analysis_poll_seconds=ANALYSIS_STATUS_POLL_SECONDS,
            breakout_watchlist=breakout_watchlist,
        )

    @app.route("/api/analysis-status")
    def analysis_status():
        """Expose background analysis status for lightweight UI polling."""
        _ensure_state()
        snapshot = _snapshot_state()
        return jsonify(
            {
                "status": snapshot["analysis_status"],
                "message": snapshot["analysis_message"],
                "last_run": snapshot["last_run"].strftime("%Y-%m-%d %H:%M:%S")
                if snapshot["last_run"]
                else None,
                "stocks_count": len(snapshot["stocks"]),
                "errors_count": len(snapshot["errors"]),
            }
        )

    @app.route("/api/live-quotes")
    def live_quotes():
        """Return live-ish quote snapshots for selected dashboard tickers."""
        query_tickers = request.args.get("tickers", "")
        requested = _sanitize_requested_tickers(query_tickers.split(",")) if query_tickers else []

        if not requested:
            _ensure_state()
            snapshot = _snapshot_state()
            fallback = list(snapshot["stocks"].keys())[:LIVE_TRACKER_DEFAULT_LIMIT]
            if not fallback:
                fallback = NIFTY50_TICKERS[:LIVE_TRACKER_DEFAULT_LIMIT]
            requested = _sanitize_requested_tickers(fallback)

        quotes, fetched_at = _get_live_quotes_cached(requested)
        return jsonify(
            {
                "tickers": requested,
                "quotes": quotes,
                "count": len(quotes),
                "updated_at": fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    @app.route("/api/simulation/<ticker>")
    def simulation_api(ticker: str):
        """Run paper-trade simulation with user-tunable controls."""
        _ensure_state()
        snapshot = _snapshot_state()
        stock = snapshot["stocks"].get(ticker)
        if stock is None:
            reason = snapshot["errors"].get(ticker) or "Stock analysis is not ready yet."
            logger.warning("Simulation API requested unavailable ticker %s: %s", ticker, reason)
            return jsonify({"error": reason}), 404

        capital = _parse_float(request.args.get("capital"), SIM_DEFAULT_INITIAL_CAPITAL, 100.0, 100000000.0)
        position_size_pct = _parse_float(
            request.args.get("position_size_pct"),
            SIM_DEFAULT_POSITION_SIZE_PCT,
            5.0,
            100.0,
        )
        brokerage_pct = _parse_float(request.args.get("brokerage_pct"), SIM_DEFAULT_BROKERAGE_PCT, 0.0, 2.0)
        slippage_pct = _parse_float(request.args.get("slippage_pct"), SIM_DEFAULT_SLIPPAGE_PCT, 0.0, 2.0)
        max_holding_days = _parse_int(
            request.args.get("max_holding_days"),
            SIM_DEFAULT_MAX_HOLD_DAYS,
            1,
            365,
        )
        stop_loss_pct = _parse_float(request.args.get("stop_loss_pct"), SIM_DEFAULT_STOP_LOSS_PCT, 0.5, 40.0)
        take_profit_pct = _parse_float(request.args.get("take_profit_pct"), SIM_DEFAULT_TAKE_PROFIT_PCT, 1.0, 120.0)

        simulation = simulate_paper_trades(
            stock.analysis.signals,
            initial_capital=capital,
            position_size_pct=position_size_pct / 100.0,
            brokerage_pct=brokerage_pct / 100.0,
            slippage_pct=slippage_pct / 100.0,
            max_holding_days=max_holding_days,
            stop_loss_pct=stop_loss_pct / 100.0,
            take_profit_pct=take_profit_pct / 100.0,
        )

        return jsonify(
            {
                "ticker": ticker,
                "params": {
                    "capital": round(capital, 2),
                    "position_size_pct": round(position_size_pct, 2),
                    "brokerage_pct": round(brokerage_pct, 4),
                    "slippage_pct": round(slippage_pct, 4),
                    "max_holding_days": max_holding_days,
                    "stop_loss_pct": round(stop_loss_pct, 2),
                    "take_profit_pct": round(take_profit_pct, 2),
                },
                "result": _serialize_simulation(simulation),
            }
        )

    @app.route("/api/projection/<ticker>")
    def projection_api(ticker: str):
        """Estimate forward value for investment horizons and expose validation metrics."""
        _ensure_state()
        snapshot = _snapshot_state()
        stock = snapshot["stocks"].get(ticker)
        if stock is None:
            reason = snapshot["errors"].get(ticker) or "Stock analysis is not ready yet."
            logger.warning("Projection API requested unavailable ticker %s: %s", ticker, reason)
            return jsonify({"error": reason}), 404

        amount = _parse_float(request.args.get("amount"), SIM_DEFAULT_INITIAL_CAPITAL, 100.0, 100000000.0)
        custom_horizon = _parse_int(request.args.get("custom_horizon_days"), 0, 0, PROJECTION_MAX_HORIZON_DAYS)

        horizons: list[int] = []
        raw_horizons = request.args.get("horizons", ",".join(str(item) for item in PROJECTION_DEFAULT_HORIZONS))
        for chunk in raw_horizons.split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                value = int(text)
            except ValueError:
                continue
            if 1 <= value <= PROJECTION_MAX_HORIZON_DAYS and value not in horizons:
                horizons.append(value)

        if custom_horizon > 0 and custom_horizon not in horizons:
            horizons.append(custom_horizon)
        if not horizons:
            horizons = list(PROJECTION_DEFAULT_HORIZONS)

        projections: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for horizon in sorted(horizons):
            try:
                projections.append(_projection_payload(stock.analysis, amount, horizon))
            except Exception as exc:
                failures.append({"horizon_days": str(horizon), "reason": str(exc)})

        if not projections:
            return jsonify(
                {
                    "ticker": ticker,
                    "amount": round(amount, 2),
                    "error": "Projection failed for all requested horizons.",
                    "failures": failures,
                }
            ), 422

        return jsonify(
            {
                "ticker": ticker,
                "amount": round(amount, 2),
                "projections": projections,
                "failures": failures,
                "method_note": (
                    "Projection uses adaptive historical drift (20-day + 120-day blend) with "
                    "rolling backtest validation."
                ),
            }
        )

    @app.route("/stock/<ticker>")
    def stock_detail(ticker: str) -> str:
        """Detailed stock page with interactive chart, explanation, and paper trading panel."""
        _ensure_state()
        snapshot = _snapshot_state()
        stock = snapshot["stocks"].get(ticker)
        if stock is None:
            reason = snapshot["errors"].get(ticker)
            if reason is None and snapshot["analysis_status"] == "running":
                reason = "Analysis is processing in the background. Please refresh shortly."
            if reason is None:
                abort(404)
            logger.warning("Stock page unavailable for %s: %s", ticker, reason)
            return render_template("stock.html", ticker=ticker, stock=None, reason=reason, plotly_js="")

        if stock.interactive_chart_html is None:
            interactive_chart_html = _build_interactive_chart(stock.analysis, stock.simulation)
            with state_lock:
                live_stock = state["stocks"].get(ticker)
                if live_stock is not None:
                    live_stock.interactive_chart_html = interactive_chart_html
            stock.interactive_chart_html = interactive_chart_html

        return render_template(
            "stock.html",
            ticker=ticker,
            stock=stock,
            reason=None,
            plotly_js=get_plotlyjs(),
            interactive_chart_html=stock.interactive_chart_html,
            chart_price_url=_cache_busted_chart_url(stock.chart_price_path),
            chart_rsi_url=_cache_busted_chart_url(stock.chart_rsi_path),
            chart_window=_chart_window_dates(stock.analysis.data),
            nse_quote=_build_nse_quote_snapshot(stock.analysis),
            stock_context_cards=_stock_context_cards(ticker, stock, snapshot["stocks"]),
            sim_defaults={
                "capital": SIM_DEFAULT_INITIAL_CAPITAL,
                "position_size_pct": SIM_DEFAULT_POSITION_SIZE_PCT,
                "brokerage_pct": SIM_DEFAULT_BROKERAGE_PCT,
                "slippage_pct": SIM_DEFAULT_SLIPPAGE_PCT,
                "max_holding_days": SIM_DEFAULT_MAX_HOLD_DAYS,
                "stop_loss_pct": SIM_DEFAULT_STOP_LOSS_PCT,
                "take_profit_pct": SIM_DEFAULT_TAKE_PROFIT_PCT,
            },
            projection_defaults={
                "amount": SIM_DEFAULT_INITIAL_CAPITAL,
                "horizons": PROJECTION_DEFAULT_HORIZONS,
                "custom_horizon_days": 10,
            },
        )

    @app.route("/export/pdf/<ticker>")
    def export_pdf(ticker: str):
        """Export one stock analysis as a local PDF report."""
        _ensure_state()
        snapshot = _snapshot_state()
        stock = snapshot["stocks"].get(ticker)
        if stock is None:
            logger.warning("PDF export requested unavailable ticker %s", ticker)
            abort(404)

        if not stock.chart_price_path or not stock.chart_rsi_path:
            price_path, rsi_path = _ensure_static_charts(stock.analysis, stock.simulation)
            with state_lock:
                live_stock = state["stocks"].get(ticker)
                if live_stock is not None:
                    live_stock.chart_price_path = price_path
                    live_stock.chart_rsi_path = rsi_path
            stock.chart_price_path = price_path
            stock.chart_rsi_path = rsi_path

        paper_trade_lines = [
            f"Virtual capital: INR {stock.simulation.initial_capital:.2f}",
            f"Final capital: INR {stock.simulation.final_capital:.2f}",
            f"Total hypothetical return: {stock.hypothetical_return_pct:.2f}%",
            f"Win rate: {stock.simulation.win_rate:.1f}% ({len(stock.simulation.trades)} trades)",
            f"Max drawdown: {stock.simulation.max_drawdown_pct:.2f}%",
            f"Profit factor: {stock.simulation.profit_factor:.2f}",
            f"Total fees: INR {stock.simulation.total_fees:.2f}",
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

    def _log_unhandled_exception(sender: Flask, exception: Exception, **_: Any) -> None:
        logger.exception("Unhandled UI exception: %s", exception)

    got_request_exception.connect(_log_unhandled_exception, app)

    _start_scheduler()
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
