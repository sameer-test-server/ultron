# Ultron

Ultron is a local-only automated stock data ingestion system focused on the Indian market (NIFTY 50).

## Features

- Daily OHLCV ingestion for NIFTY 50 symbols
- Primary source: Yahoo Finance (`yfinance`)
- Automatic fallback sequence:
  - NSE Bhavcopy
  - Stooq
- Per-ticker fault isolation (one ticker failure does not stop the run)
- Incremental updates + CSV deduplication and sorting
- Cron-ready runner with execution logging

## Project Structure

```text
ultron/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── nifty50.py
├── core/
│   ├── __init__.py
│   └── data_loader.py
├── data/
│   └── raw/
├── logs/
├── scripts/
│   ├── run_ultron.py
│   └── cron_setup.txt
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Linux/macOS
- Internet access for data providers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Manually

```bash
python scripts/run_ultron.py
```

Alternative direct run:

```bash
python -c "from core.data_loader import update_all_data; update_all_data()"
```

## Logging

- Main run log: `logs/ultron.log`
- Cron stdout/stderr log: `logs/cron.log`

`scripts/run_ultron.py` logs:
- run start time
- per-ticker status
- final summary (`success` / `partial` / `failed`)
- run end time

## Cron (Daily 6:30 AM)

Use the template in `scripts/cron_setup.txt`.

Example:

```cron
30 6 * * * /home/sameer/ultron/.venv/bin/python /home/sameer/ultron/scripts/run_ultron.py >> /home/sameer/ultron/logs/cron.log 2>&1
```

## Notes

- This project is local-only (no cloud, no Docker).
- Paths are BASE_DIR-driven for cron safety.
- `.venv` and runtime logs are git-ignored.
