# 🛡️ Ultron: Quantitative Trading Research Platform

**A private, local-first platform for quantitative trading research on the NIFTY 50.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/Trading-Paper--Only-lightgrey" alt="Paper Trading Only">
  <img src="https://img.shields.io/badge/Automation-Cron%20%26%20Systemd-blueviolet" alt="Automation Ready">
</p>

---

Ultron is a self-hosted, automated system designed for robust, daily analysis of the NIFTY 50 stock universe. It handles everything from data ingestion and cleaning to technical analysis, paper trading simulation, and visualization, all while running securely on your local machine.

## ✨ Key Features

*   **🔄 Resilient Multi-Source Data:** Automatically fetches data from Yahoo Finance with fallbacks to NSE Bhavcopy, Stooq, and AlphaVantage, ensuring you never miss a day.
*   **🚀 High-Performance Parallelism:** Updates all 50 tickers in seconds using a configurable multi-worker architecture.
*   **📊 Comprehensive Technical Analysis:** Computes SMAs, EMAs, RSI, Volatility, and a proprietary Market Regime model.
*   **💼 Paper Trading Simulation:** Simulates trades with virtual capital, position sizing, and detailed P/L tracking.
*   **📈 Interactive Dashboard:** A local Flask web UI to visualize data, indicators, and analysis for each stock.
*   **🤖 Zero-Touch Automation:** Designed for daily execution via Cron or Systemd, with lock-based safety and optional email alerts for failures.

---

## 🚀 Quick Start

Get up and running in under 5 minutes.

### 1. Activate Environment & Run Tests
```bash
# Navigate to the project directory
cd /path/to/ultron

# Activate your Python virtual environment
source .venv/bin/activate

# Run the integration test suite to validate the entire pipeline
python scripts/test_integration.py
```

### 2. Perform a Manual Data Update
```bash
# Update all 50 tickers using 6 parallel workers
python scripts/run_ultron.py --parallel 6

# (Optional) Run a health check to verify data freshness
python scripts/health_check.py
```

### 3. Explore the Analysis & UI
```bash
# Run the full analysis pipeline and generate reports
python run_ultron_analysis.py

# Launch the Flask web dashboard
python ui/app.py

# Open your browser to http://127.0.0.1:5000
```

### UI Runtime Notes

- UI logs are written to `logs/ui.log`.
- If serving with Gunicorn, use a single worker so only one background scheduler thread runs:

```bash
gunicorn -w 1 --threads 8 ui.app:app
```

---

## ⚙️ Automated Daily Execution

Set up Ultron for fully automated daily runs.

### 1. Configure Environment (Optional but Recommended)

Create a `.env` file for email alerts and the AlphaVantage API key.
```bash
# Create the .env file in the project root by copying the example
cp .env.example .env

# Edit .env to add your credentials
# nano .env
```
*Ensure you set `ULTRON_SMTP_...` variables for email alerts and `ALPHAVANTAGE_API_KEY` for the premium data fallback.*

### 2. Schedule with Cron

Add the provided `cron_runner.sh` script to your crontab to enable daily updates, data commits, and Git pushes.

```bash
# Open your crontab for editing
crontab -e

# Add this line to run daily at 6:30 AM
# 30 6 * * * /path/to/ultron/scripts/cron_runner.sh
```
*For this to work, ensure your machine has Git configured with an SSH key or PAT for the repository.*

---

## 🗼 System Architecture

### Data Ingestion Pipeline

| Source | Speed | Used As | Notes |
| :--- | :--- | :--- | :--- |
| **Yahoo Finance** | ⚡ Fast | **Primary** | Main data source with high availability. |
| **NSE Bhavcopy** | 🐌 Slow | **Fallback 1** | Authoritative daily archives from the exchange. |
| **Stooq** | 💨 Medium | **Fallback 2** | Reliable international data provider. |
| **AlphaVantage** | 💨 Medium | **Fallback 3** | Optional premium source for stubborn tickers. |

### Project Structure
```
ultron/
├── config/             # System-wide settings and ticker lists
├── core/               # Core logic: data loading, analysis, trading
├── data/raw/           # Local storage for raw .csv stock data
├── logs/               # Application, cron, and failure logs
├── reports/            # Generated PDF reports and PNG charts
├── scripts/            # Runners for CLI, cron, and health checks
├── ui/                 # Flask web application (templates, static assets)
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ✨ Recent Updates (February 2026)

*   **⚡️ Performance Boost (Feather Integration):** The data loading and analysis pipelines have been significantly accelerated by replacing CSV parsing with the high-speed Feather binary format. This results in a much faster and more responsive UI experience.
*   **🧠 Smarter Ultron Calculator:** The investment calculator has been reworked to produce more conservative and reliable predictions. It now incorporates a "pessimism factor" based on historical errors to temper overly optimistic forecasts and a more stringent decision-making model to better align with a loss-avoidance strategy.

---

## 🛠️ Technology Stack

*   **Core:** Python 3.11+
*   **Data & Analysis:** pandas, numpy, yfinance
*   **Web UI:** Flask, Plotly
*   **Reporting:** ReportLab
*   **Concurrency:** `concurrent.futures.ThreadPoolExecutor`
*   **Scheduling:** Cron, Systemd (`flock` for safety)
*   **Alerting:** `smtplib`

---

## 🔍 Monitoring & Troubleshooting

Your first stop for diagnosing issues.

*   **Health Dashboard**: `python scripts/health_check.py`
*   **Integration Tests**: `python scripts/test_integration.py`
*   **Application Logs**: `tail -f logs/ultron.log`
*   **Cron Logs**: `tail -f logs/cron.log`
*   **Failed Tickers**: `cat logs/failed_tickers_YYYY-MM-DD.txt`

---

**Last Updated**: 2026-02-27 | **Status**: ✅ **PRODUCTION READY**
