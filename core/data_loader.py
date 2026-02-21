import datetime
import io
import logging
import os
import socket
import urllib.error
import urllib.request
import zipfile

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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return pd.read_csv(io.BytesIO(response.read()))


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
    downloaded = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    return _normalize_downloaded_data(downloaded)


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
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            _NSE_BHAVCOPY_CACHE[trade_date] = None
            return None
        _NSE_BHAVCOPY_CACHE[trade_date] = None
        return None
    except Exception:
        _NSE_BHAVCOPY_CACHE[trade_date] = None
        return None

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

    nse_df = _download_from_nse_bhavcopy(ticker, start_date, end_date)
    if not nse_df.empty:
        _emit(f"Recovered via NSE for {ticker}")
        return nse_df, "nse"

    stooq_df = _download_from_stooq(ticker, start_date, end_date)
    if not stooq_df.empty:
        _emit(f"Recovered via Stooq for {ticker}")
        return stooq_df, "stooq"

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


def update_all_data():
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
    return summary
