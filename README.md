# Ultron

Ultron is a local-only stock market analyst and paper trading simulator for the NIFTY 50 universe.

It downloads OHLCV data, computes indicators, classifies market regime, generates explainable analysis, simulates hypothetical trades, and visualizes everything in a read-only Flask UI.

## Safety Model

- No real trading
- No broker API integration
- No money movement
- Localhost-only UI
- Read-only analysis and historical simulation outputs

## Core Capabilities

### 1) Data ingestion with fallback
- Downloads OHLCV data per ticker to `data/raw/*.csv`
- Primary source: Yahoo Finance (`yfinance`)
- Fallback order:
  - NSE Bhavcopy
  - Stooq
- Fault isolation per ticker
- Incremental updates, dedupe, and date sorting

### 2) Local analysis pipeline
- CSV validation and cleaning (`core/data_reader.py`)
- Indicator engine (`core/indicators.py`)
  - SMA 20/50/200
  - EMA 20
  - RSI 14
  - Rolling volatility
- Regime detection (`core/regime_detector.py`)
  - `LONG_TERM` vs `SHORT_TERM`
  - Confidence score + plain-English reason
- Explainable analyst output (`core/analyst.py`)
  - Momentum/trend observations
  - Hypothetical scenarios (no buy/sell orders)
- Paper trade simulation (`core/paper_trader.py`)
  - Virtual capital (default INR 2000)
  - Entry/exit, P/L, win rate

### 3) Reporting
- CLI pipeline runner (`run_ultron_analysis.py`)
  - Prints per-stock summary
  - Generates charts
  - Final portfolio-level summary
- Static chart generation (`core/visualizer.py`)
  - Saved under `reports/charts/`

### 4) Read-only web UI (Flask)
- App entry: `ui/app.py`
- Dashboard route `/`
  - Regime filter and sorting
  - Summary cards
  - Clickable stock rows
- Stock detail route `/stock/<ticker>`
  - Overview (regime/confidence/volatility)
  - Interactive Plotly chart (price + SMA/EMA + RSI)
  - Analysis explanation
  - Paper trade simulation table
- PDF export route `/export/pdf/<ticker>`
  - Exports report to `reports/pdf/`

## Project Structure

```text
ultron/
├── config/
│   ├── settings.py
│   └── nifty50.py
├── core/
│   ├── data_loader.py
│   ├── data_reader.py
│   ├── indicators.py
│   ├── regime_detector.py
│   ├── analyst.py
│   ├── paper_trader.py
│   └── visualizer.py
├── data/
│   └── raw/
├── logs/
├── reports/
│   ├── charts/
│   └── pdf/
├── scripts/
│   ├── run_ultron.py
│   └── cron_setup.txt
├── ui/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   │   ├── css/
│   │   └── charts/
│   └── utils/
│       └── pdf_exporter.py
├── run_ultron_analysis.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- Linux/macOS
- Internet access for data download sources

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### A) Update market data

```bash
python scripts/run_ultron.py
```

Alternative:

```bash
python -c "from core.data_loader import update_all_data; update_all_data()"
```

### B) Run full analysis + paper simulation (CLI)

```bash
python run_ultron_analysis.py
```

### C) Run Flask UI (localhost only)

```bash
python ui/app.py
```

Open:

```text
http://127.0.0.1:5000
```

### D) Export PDF from UI

- Use the **Export Analysis to PDF** button on a stock page
- Or open:

```text
http://127.0.0.1:5000/export/pdf/<TICKER>
```

PDF output path:

```text
reports/pdf/<TICKER>_analysis_<DATE>.pdf
```

## Logging

- Main run log: `logs/ultron.log`
- Optional cron stdout/stderr log: `logs/cron.log`

## Cron

Use template in `scripts/cron_setup.txt`.

Example:

```cron
30 6 * * * /home/sameer/ultron/.venv/bin/python /home/sameer/ultron/scripts/run_ultron.py >> /home/sameer/ultron/logs/cron.log 2>&1
```

## Notes

- All paths are based on `BASE_DIR` from `config/settings.py`.
- UI is intentionally read-only and analysis-only.
- No cloud services, no authentication, no external broker integrations.
