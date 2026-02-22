import datetime
import io
import logging
import os
import socket
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import smtplib
import os
from email.message import EmailMessage

import pandas as pd
import yfinance as yf

from config.nifty50 import NIFTY50_TICKERS
from config.settings import HISTORY_YEARS, RAW_DATA_DIR

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
NSE_VOLUME_COLUMNS = ("TOTTRDQTY", "TOTTRD_QTY", "TTL_TRD_QNTY", "TOT_QTY")

LOGGER = logging.getLogger("ultron.data_loader")
_NSE_BHAVCOPY_CACHE = {}
_HOST_RESOLUTION_CACHE = {}


def _emit(message, level=logging.INFO):
    """Write status to stdout and log file."""
    print(message)
    LOGGER.log(level, message)


def _ticker_csv_path(ticker):
    """Return the CSV path for a ticker inside data/raw."""
    return os.path.join(RAW_DATA_DIR, f"{ticker}.csv")


def _empty_ohlcv_frame():
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _normalize_downloaded_data(dataframe):
    """Normalize source output to Date/Open/High/Low/Close/Volume."""
    if dataframe is None or dataframe.empty:
        return _empty_ohlcv_frame()

    normalized = dataframe.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [column[0] for column in normalized.columns]

    normalized = normalized.reset_index()
    if "Date" not in normalized.columns and "Datetime" in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Date"})

    if not set(REQUIRED_COLUMNS).issubset(normalized.columns):
        return _empty_ohlcv_frame()

    return normalized[REQUIRED_COLUMNS]


def _clean_and_save(dataframe, csv_path):
    """Deduplicate, sort by date, and overwrite CSV."""
    if dataframe.empty:
        return False

    cleaned = dataframe.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"])
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="last")
    cleaned = cleaned.sort_values(by="Date").reset_index(drop=True)
    cleaned = cleaned[REQUIRED_COLUMNS]

    if cleaned.empty:
        return False

    cleaned.to_csv(csv_path, index=False)
    return True


def _full_history_start_date():
    """Compute the configured historical lookback start date."""
    return datetime.date.today() - datetime.timedelta(days=HISTORY_YEARS * 365)


def _is_network_or_http_error(error):
    """Heuristic check for transport, DNS, timeout, or HTTP failures."""
    if isinstance(error, (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError)):
        return True

    message = str(error).lower()
    indicators = (
        "could not resolve host",
        "name or service not known",
        "failed to establish a new connection",
        "temporary failure in name resolution",
        "max retries exceeded",
        "connection aborted",
        "connection reset",
        "timeout",
        "timed out",
        "http",
        "dns",
        "curl:",
    )
    return any(token in message for token in indicators)


def _read_csv_from_url(url, timeout=20):
    """Read a remote CSV into a DataFrame with a deterministic timeout."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
    )
    attempts = 3
    backoff = 1.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return pd.read_csv(io.BytesIO(response.read()))
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(backoff)
            backoff *= 2


def _host_resolves(host):
    """Resolve host once per process to avoid repeated DNS calls."""
    if host in _HOST_RESOLUTION_CACHE:
        return _HOST_RESOLUTION_CACHE[host]

    try:
        socket.gethostbyname(host)
        _HOST_RESOLUTION_CACHE[host] = True
    except OSError:
        _HOST_RESOLUTION_CACHE[host] = False
    return _HOST_RESOLUTION_CACHE[host]


def _yahoo_dns_unavailable():
    """Detect DNS-level Yahoo unavailability for fallback activation."""
    hosts = ("guce.yahoo.com", "query1.finance.yahoo.com")
    return not any(_host_resolves(host) for host in hosts)


def _download_from_yahoo(ticker, start_date, end_date):
    """Primary source: yfinance daily candles."""
    attempts = 2
    backoff = 1.0
    for attempt in range(1, attempts + 1):
        try:
            downloaded = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            return _normalize_downloaded_data(downloaded)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(backoff)
            backoff *= 2


def _nse_bhavcopy_url(trade_date):
    """Build NSE Bhavcopy ZIP URL for a specific trade date."""
    month = trade_date.strftime("%b").upper()
    return (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        f"{trade_date:%Y}/{month}/cm{trade_date:%d}{month}{trade_date:%Y}bhav.csv.zip"
    )


def _load_nse_bhavcopy_day(trade_date):
    """Load and cache one NSE Bhavcopy daily file."""
    if trade_date in _NSE_BHAVCOPY_CACHE:
        return _NSE_BHAVCOPY_CACHE[trade_date]

    url = _nse_bhavcopy_url(trade_date)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
    )

    attempts = 3
    backoff = 1.0
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    _NSE_BHAVCOPY_CACHE[trade_date] = None
                    return None
                with zipped.open(csv_names[0]) as handle:
                    day_data = pd.read_csv(handle)
            break
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                _NSE_BHAVCOPY_CACHE[trade_date] = None
                return None
            _NSE_BHAVCOPY_CACHE[trade_date] = None
            return None
        except Exception:
            if attempt == attempts:
                _NSE_BHAVCOPY_CACHE[trade_date] = None
                return None
            time.sleep(backoff)
            backoff *= 2

    day_data.columns = [str(column).strip().upper() for column in day_data.columns]
    _NSE_BHAVCOPY_CACHE[trade_date] = day_data
    return day_data


def _to_number(value):
    """Convert numeric-like values safely."""
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    return pd.to_numeric(value, errors="coerce")


def _download_from_nse_bhavcopy(ticker, start_date, end_date):
    """Fallback source #1: NSE Bhavcopy daily archives."""
    if not _host_resolves("nsearchives.nseindia.com"):
        return _empty_ohlcv_frame()

    symbol = ticker.replace(".NS", "").upper()
    records = []
    current_date = start_date

    while current_date < end_date:
        if current_date.weekday() < 5:
            day_data = _load_nse_bhavcopy_day(current_date)
            if day_data is not None and "SYMBOL" in day_data.columns:
                rows = day_data[day_data["SYMBOL"].astype(str).str.upper() == symbol]
                if not rows.empty:
                    volume_column = next((name for name in NSE_VOLUME_COLUMNS if name in day_data.columns), None)
                    if volume_column is not None:
                        row = rows.iloc[0]
                        records.append(
                            {
                                "Date": current_date,
                                "Open": _to_number(row.get("OPEN")),
                                "High": _to_number(row.get("HIGH")),
                                "Low": _to_number(row.get("LOW")),
                                "Close": _to_number(row.get("CLOSE")),
                                "Volume": _to_number(row.get(volume_column)),
                            }
                        )
        current_date += datetime.timedelta(days=1)

    if not records:
        return _empty_ohlcv_frame()

    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def _stooq_symbol_candidates(ticker):
    """Generate likely Stooq symbol candidates for NSE equities."""
    base = ticker.replace(".NS", "").lower()
    compact = "".join(char for char in base if char.isalnum())
    variants = [base, base.replace("-", ""), base.replace("&", ""), compact]

    candidates = []
    for variant in variants:
        if variant:
            candidates.append(f"{variant}.in")
            candidates.append(f"{variant}.ns")
    return list(dict.fromkeys(candidates))


def _download_from_stooq(ticker, start_date, end_date):
    """Fallback source #2: Stooq daily CSV endpoint."""
    if not _host_resolves("stooq.com"):
        return _empty_ohlcv_frame()

    for symbol in _stooq_symbol_candidates(ticker):
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        try:
            raw_df = _read_csv_from_url(url)
        except Exception:
            continue

        if raw_df.empty:
            continue

        renamed = {column: str(column).strip().lower() for column in raw_df.columns}
        raw_df = raw_df.rename(columns=renamed)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(raw_df.columns)):
            continue

        normalized = raw_df[["date", "open", "high", "low", "close", "volume"]].copy()
        normalized.columns = REQUIRED_COLUMNS
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"])

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        normalized = normalized[(normalized["Date"] >= start_ts) & (normalized["Date"] < end_ts)]
        if normalized.empty:
            continue

        return normalized

    return _empty_ohlcv_frame()


def _download_from_alpha_vantage(ticker, start_date, end_date):
    """Optional fallback: AlphaVantage CSV export if `ALPHAVANTAGE_API_KEY` is set.

    AlphaVantage may not support all NSE symbols; this is a best-effort attempt.
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return _empty_ohlcv_frame()

    # AlphaVantage expects symbol without country suffix in many cases; try variants.
    candidates = [ticker, ticker.replace('.NS', ''), ticker.replace('.NS', '.BSE')]
    for symbol in candidates:
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={symbol}&outputsize=full&datatype=csv&apikey={api_key}"
        )
        try:
            df = _read_csv_from_url(url, timeout=20)
        except Exception:
            continue

        if df.empty:
            continue

        # AlphaVantage CSV columns: timestamp, open, high, low, close, adjusted_close, volume, ...
        renamed = {c: c for c in df.columns}
        # Normalize to required columns
        normalized = df.rename(columns={
            df.columns[0]: "Date",
            df.columns[1]: "Open",
            df.columns[2]: "High",
            df.columns[3]: "Low",
            df.columns[4]: "Close",
            df.columns[6]: "Volume",
        })
        normalized = normalized[["Date", "Open", "High", "Low", "Close", "Volume"]]
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        normalized = normalized[(normalized["Date"] >= start_ts) & (normalized["Date"] < end_ts)]
        if normalized.empty:
            continue
        return normalized

    return _empty_ohlcv_frame()


def _download_with_fallback(ticker, start_date, end_date, allow_empty):
    """Attempt Yahoo first, then NSE Bhavcopy, then Stooq."""
    try:
        yahoo_df = _download_from_yahoo(ticker, start_date, end_date)
        if not yahoo_df.empty:
            return yahoo_df, "yahoo"

        days_requested = (end_date - start_date).days
        should_fallback = (not allow_empty) or _yahoo_dns_unavailable() or days_requested >= 3
        if not should_fallback:
            return yahoo_df, "empty"

        _emit(f"Yahoo failed for {ticker}, attempting fallback")
    except Exception as error:
        if _is_network_or_http_error(error):
            _emit(f"Yahoo failed for {ticker}, attempting fallback")
        else:
            _emit(f"Yahoo failed for {ticker}, attempting fallback")
    # If requested history is very large, limit per-source fallback window to recent 365 days
    fallback_start = start_date
    if days_requested > 365:
        fallback_start = max(start_date, end_date - datetime.timedelta(days=365))
        _emit(f"Fallback window limited to last 365 days for {ticker}")

    # Try NSE Bhavcopy using a limited window to avoid long per-day loops for multi-year rebuilds
    nse_df = _download_from_nse_bhavcopy(ticker, fallback_start, end_date)
    if not nse_df.empty:
        _emit(f"Recovered via NSE for {ticker}")
        return nse_df, "nse"

    stooq_df = _download_from_stooq(ticker, start_date, end_date)
    if not stooq_df.empty:
        _emit(f"Recovered via Stooq for {ticker}")
        return stooq_df, "stooq"
    # Try AlphaVantage if API key is set (optional premium fallback)
    av_df = _download_from_alpha_vantage(ticker, fallback_start, end_date)
    if not av_df.empty:
        _emit(f"Recovered via AlphaVantage for {ticker}")
        return av_df, "alphavantage"
    _emit(f"All data sources failed for {ticker}", level=logging.WARNING)
    return _empty_ohlcv_frame(), "failed"


def _rebuild_ticker_file(ticker, csv_path):
    """Re-download full history and overwrite an invalid file."""
    _emit(f"Rebuilding corrupted file for {ticker}")
    full_history, source = _download_with_fallback(
        ticker=ticker,
        start_date=_full_history_start_date(),
        end_date=datetime.date.today() + datetime.timedelta(days=1),
        allow_empty=False,
    )
    if _clean_and_save(full_history, csv_path):
        return "updated"
    return "failed" if source == "failed" else "updated"


def _update_single_ticker(ticker):
    """Update one ticker file using full or incremental mode."""
    csv_path = _ticker_csv_path(ticker)
    today = datetime.date.today()

    if not os.path.exists(csv_path):
        _emit(f"Downloading full history for {ticker}")
        full_history, source = _download_with_fallback(
            ticker=ticker,
            start_date=_full_history_start_date(),
            end_date=today + datetime.timedelta(days=1),
            allow_empty=False,
        )
        if _clean_and_save(full_history, csv_path):
            return "updated"
        return "failed" if source == "failed" else "updated"

    try:
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
    except Exception:
        return _rebuild_ticker_file(ticker, csv_path)

    if existing.empty or "Date" not in existing.columns:
        return _rebuild_ticker_file(ticker, csv_path)

    existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
    last_date = existing["Date"].dropna().max()
    if pd.isna(last_date):
        return _rebuild_ticker_file(ticker, csv_path)

    next_date = last_date.date() + datetime.timedelta(days=1)
    if next_date > today:
        _emit(f"Up to date: {ticker}")
        return "up_to_date"

    _emit(f"Updating {ticker}")
    missing, source = _download_with_fallback(
        ticker=ticker,
        start_date=next_date,
        end_date=today + datetime.timedelta(days=1),
        allow_empty=True,
    )

    if missing.empty:
        if source == "failed":
            return "failed"
        _emit(f"Up to date: {ticker}")
        return "up_to_date"

    updated = pd.concat([existing, missing], ignore_index=True)
    if _clean_and_save(updated, csv_path):
        return "updated"
    return "failed" if source == "failed" else "updated"


def update_all_data(workers: int | None = None):
    """
    Main entry point.
    Updates all NIFTY50 ticker files without stopping on single-ticker failures.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    summary = {
        "total": len(NIFTY50_TICKERS),
        "updated": 0,
        "up_to_date": 0,
        "failed": 0,
        "status": "unknown",
    }

    # If workers is provided and >1, perform parallel updates.
    if workers and workers > 1:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ticker in NIFTY50_TICKERS:
                futures[executor.submit(_update_single_ticker, ticker)] = ticker
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    result = fut.result()
                except Exception as error:
                    _emit(f"FAILED {ticker}: {error}", level=logging.WARNING)
                    result = "failed"

                if result == "updated":
                    summary["updated"] += 1
                elif result == "up_to_date":
                    summary["up_to_date"] += 1
                else:
                    summary["failed"] += 1
    else:
        for ticker in NIFTY50_TICKERS:
            try:
                result = _update_single_ticker(ticker)
            except Exception as error:
                _emit(f"FAILED {ticker}: {error}", level=logging.WARNING)
                result = "failed"

            if result == "updated":
                summary["updated"] += 1
            elif result == "up_to_date":
                summary["up_to_date"] += 1
            else:
                summary["failed"] += 1

    if summary["failed"] == 0:
        summary["status"] = "success"
    elif summary["failed"] == summary["total"]:
        summary["status"] = "failed"
    else:
        summary["status"] = "partial"

    _emit(
        "Final summary: "
        f"{summary['status']} | total={summary['total']} "
        f"updated={summary['updated']} up_to_date={summary['up_to_date']} failed={summary['failed']}"
    )
    # Persist failed ticker list for diagnostics
    failed_names = [t for t in NIFTY50_TICKERS if _ticker_status(t, summary) == "failed"]
    try:
        if failed_names:
            logs_dir = os.path.join(os.path.dirname(RAW_DATA_DIR), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            fname = os.path.join(logs_dir, f"failed_tickers_{datetime.date.today().isoformat()}.txt")
            with open(fname, "w", encoding="utf-8") as fh:
                for t in failed_names:
                    fh.write(t + "\n")
            _emit(f"Wrote failed ticker list to {fname}", level=logging.WARNING)
            # Attempt to send alert email if configured
            alert_body = f"Ultron failed to update the following tickers on {datetime.date.today()}:\n\n"
            for t in failed_names:
                alert_body += f"  - {t}\n"
            alert_body += f"\nCheck logs/failed_tickers_{datetime.date.today().isoformat()}.txt for details."
            _send_alert_email(f"Ultron Alert: {len(failed_names)} Failed Tickers", alert_body)
    except Exception:
        pass
    return summary


def _ticker_status(ticker, summary):
    """Helper used to approximate per-ticker status when writing diagnostics."""
    # This helper is a lightweight placeholder; callers should prefer direct results per-ticker.
    # If a file exists and has recent data, treat as updated/up_to_date; otherwise failed.
    csv_path = _ticker_csv_path(ticker)
    if not os.path.exists(csv_path):
        return "failed"
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        last = df["Date"].dropna().max()
        if pd.isna(last):
            return "failed"
        if (datetime.date.today() - last.date()).days <= 3:
            return "updated"
        return "failed"
    except Exception:
        return "failed"


def _send_alert_email(subject: str, body: str) -> bool:
    """Send an alert email if SMTP credentials are configured via environment variables.

    Expected env vars: ULTRON_SMTP_HOST, ULTRON_SMTP_PORT, ULTRON_SMTP_USER, ULTRON_SMTP_PASS, ULTRON_ALERT_TO
    Returns True if email sent, False if not configured or on error.
    """
    smtp_host = os.environ.get("ULTRON_SMTP_HOST")
    smtp_port = os.environ.get("ULTRON_SMTP_PORT", "587")
    smtp_user = os.environ.get("ULTRON_SMTP_USER")
    smtp_pass = os.environ.get("ULTRON_SMTP_PASS")
    alert_to = os.environ.get("ULTRON_ALERT_TO")

    if not all([smtp_host, smtp_user, smtp_pass, alert_to]):
        return False

    try:
        port = int(smtp_port)
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = alert_to
        msg.set_content(body)

        with smtplib.SMTP(smtp_host, port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        _emit(f"Alert email sent to {alert_to}", level=logging.INFO)
        return True
    except Exception as e:
        _emit(f"Failed to send alert email: {e}", level=logging.WARNING)
        return False
