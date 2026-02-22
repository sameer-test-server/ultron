# Ultron Data Loader: Setup & Configuration

This guide helps you set up Ultron for daily automated stock data updates with alerts and multi-source fallback recovery.

## Quick Start (Local Testing)

```bash
cd /home/madara/ultron

# Activate virtualenv
source .venv/bin/activate

# Run a quick test on a few tickers (no network stress)
python scripts/run_ultron.py --tickers RELIANCE.NS,INFY.NS,TCS.NS --parallel 2

# Check logs
tail -f logs/ultron.log
```

## Optional: Email Alerts on Failed Tickers

Create a `.env` file in the project root to enable email alerts:

```bash
cat > /home/madara/ultron/.env << 'EOF'
# Gmail example (use App Password from myaccount.google.com/apppasswords)
ULTRON_SMTP_HOST=smtp.gmail.com
ULTRON_SMTP_PORT=587
ULTRON_SMTP_USER=your-email@gmail.com
ULTRON_SMTP_PASS=your-app-specific-password
ULTRON_ALERT_TO=alerts@example.com

# Optional: AlphaVantage API for premium fallback (free tier available)
ALPHAVANTAGE_API_KEY=demo
EOF

# Make it private
chmod 600 .env
```

Then test email sending:

```bash
python -c "
from core.data_loader import _send_alert_email
import os
os.environ['ULTRON_SMTP_HOST']='smtp.gmail.com'
os.environ['ULTRON_SMTP_PORT']='587'
os.environ['ULTRON_SMTP_USER']='your-email@gmail.com'
os.environ['ULTRON_SMTP_PASS']='your-app-pass'
os.environ['ULTRON_ALERT_TO']='alerts@example.com'
_send_alert_email('Test', 'This is a test alert from Ultron')
"
```

## Daily Cron Setup

Edit your crontab:

```bash
crontab -e
```

Add this line to run updates daily at 6:30 AM:

```bash
30 6 * * * /home/madara/ultron/scripts/cron_runner.sh
```

The cron runner automatically:
- Loads `.env` variables
- Acquires a file lock to prevent overlapping runs
- Logs to `logs/cron.log`
- Runs with 6 parallel worker threads

## Data Sources (Fallback Chain)

Ultron tries sources in this order:

1. **Yahoo Finance** (yfinance) — primary, fast, ~99% coverage
2. **NSE Bhavcopy** (nsearchives.nseindia.com) — direct NSE, very slow per-day loops but authoritative
3. **Stooq** (stooq.com) — free CSV API, covers many symbols
4. **AlphaVantage** (optional, requires API key) — premium, covers global + some NSE symbols

## Troubleshooting Failed Tickers

After each run, if tickers fail, a file `logs/failed_tickers_YYYY-MM-DD.txt` is created with the list.

**Common causes:**
- **ULTRATECH / TATAMOTORS**: Yahoo may rate-limit or return 404 (symbol delisted from Yahoo). AlphaVantage can recover these if API key is set.
- **Network timeouts**: Retries with exponential backoff are built-in; usually resolved on next run.
- **NSE archives missing**: Specific dates may not have bhavcopy files; fallback to Stooq/AlphaVantage.

**Example recovery:**
```bash
# Set AlphaVantage API key and retry
export ALPHAVANTAGE_API_KEY=YOUR_API_KEY
python scripts/run_ultron.py --tickers ULTRATECH.NS,TATAMOTORS.NS --parallel 2
```

## Performance Tuning

Adjust `--parallel` workers based on your CPU/network:

```bash
# Fast (4 workers, good for 50 tickers, ~2-3 min)
python scripts/run_ultron.py --parallel 4

# Aggressive (8 workers, max throughput, ~1-2 min)
python scripts/run_ultron.py --parallel 8

# Conservative (1 worker, no parallelism, uses least bandwidth)
python scripts/run_ultron.py --parallel 1
```

## File Locations

- **Raw data**: `data/raw/*.csv` (50 NIFTY50 stocks)
- **Logs**: `logs/ultron.log` (main), `logs/cron.log` (cron runs)
- **Failed tickers**: `logs/failed_tickers_YYYY-MM-DD.txt`
- **Config**: `.env` (optional, for alerts and API keys)

## Verifying Data Freshness

Check if all tickers are current:

```bash
python - << 'PY'
import pandas as pd, os
from config.settings import RAW_DATA_DIR
for f in sorted(os.listdir(RAW_DATA_DIR)):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, f), parse_dates=['Date'])
        last = df['Date'].dropna().max()
        is_recent = (pd.Timestamp('2026-02-22') - last).days <= 1
        status = '✓' if is_recent else '✗'
        print(f'{status} {f}: {last.date()}')
PY
```

## Next Steps (Optional Enhancements)

- **Real-time streaming**: Integrate intraday WebSocket feeds (e.g., NSE's GFDL or broker APIs)
- **Advanced indicators**: Add momentum, trend, volatility indicators in `core/indicators.py`
- **Database persistence**: Move from CSVs to SQLite for faster queries and historical tracking
- **Backtesting framework**: Test strategies on historical data with parameter sweeps
- **Multi-timeframe analysis**: Daily + weekly + monthly signal confluence
