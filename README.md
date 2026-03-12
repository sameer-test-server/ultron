# Ultron — Local-First Market Intelligence

**Ultron is a private, local-only research platform for NIFTY 50 analysis, simulation, and decision support.**

It ingests daily OHLCV data, computes indicators, classifies regimes, runs multi‑strategy backtests, and presents results in a modern offline UI with explainable chat and PDF reports. No real trades. No broker APIs. No cloud services.

---

## Why Ultron

- **Local-only, privacy-first**: Runs on your machine, no cloud dependencies.
- **Explainable analysis**: Every signal and decision includes clear reasoning.
- **Multi-strategy intelligence**: Trend, mean‑reversion, and breakout scenarios compared side‑by‑side.
- **Research-grade**: Signal reliability, risk suite, and parameter grid lab built-in.
- **Operator-ready**: Cron automation, logs, health checks, and daily summary reports.

---

## What You’ll See (UI)

- **Dashboard**: Ranked tickers with regime, confidence, and hypothetical returns
- **Focus Mode**: Deep dive on one ticker with interactive charts
- **Watchlist**: Your short list, stored locally
- **Explainable Chat**: Ask “Explain RELIANCE.NS” with local Ollama
- **PDF Export**: Shareable analysis reports (local only)

---

## Core Capabilities

### Data Engine
- Yahoo Finance (primary)
- NSE Bhavcopy → Stooq (fallbacks)
- Local CSV + Feather caching for fast reads

### Analyst Engine
- SMA 20/50/200, EMA 20, RSI 14, volatility
- Market regime detection (LONG_TERM / SHORT_TERM)
- Reasoning engine with confidence + evidence
- Signal reliability ledger
- Risk suite (drawdown, tail risk, liquidity score)

### Simulation & Research
- Paper trading simulator with costs + risk controls
- Scenario engine (trend / mean‑reversion / breakout)
- Parameter grid runner (top configurations)

### UI/UX
- Interactive dashboard + filters + watchlist
- Focus Mode (single‑ticker deep view)
- Explainable chat with local Ollama (Mistral)
- Offline‑ready assets (Bootstrap, icons, fonts vendored locally)
 - PDF export per stock (reportlab)

### Reporting
- PDF exports per stock
- Daily summary Markdown + PDF reports

---

## Quick Start

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Update Data
```bash
python scripts/run_ultron.py --parallel 6
```

### 3) Run Analysis (CLI)
```bash
python run_ultron_analysis.py
```

### 4) Run UI
```bash
export ULTRON_OFFLINE_MODE=true
python ui/app.py
```
Open: `http://127.0.0.1:5000`

---

## Chat (Local LLM)

Ultron can answer in natural language using local Ollama (no cloud).

```bash
export OLLAMA_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="mistral"
python ui/app.py
```

Try:
- “Explain RELIANCE.NS”
- “Summarize risks for TCS.NS”
- “Top picks”

---

## Daily Summary Report

Generate a daily Markdown + PDF summary:

```bash
python scripts/generate_daily_report.py
```
Outputs in `reports/daily/`.

---

## Safety Guarantees

- **No real trades**
- **No broker APIs**
- **Read-only analysis + simulation**
- **Local-only execution**

---

## Project Structure

```
ultron/
├── config/
├── core/
│   ├── analyst.py
│   ├── data_loader.py
│   ├── data_reader.py
│   ├── indicators.py
│   ├── regime_detector.py
│   ├── reasoning_engine.py
│   ├── scenario_engine.py
│   ├── signal_reliability.py
│   ├── risk_suite.py
│   └── research_lab.py
├── data/
├── reports/
├── scripts/
├── ui/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── watchlist_store.py
└── README.md
```

---

## Safety Guarantees

- **No real trades**
- **No broker APIs**
- **Read-only analysis + simulation**
- **Local-only execution**

---

## Recent Additions (March 2026)

- Reasoning engine with explainable evidence
- Scenario engine (3 strategies)
- Signal reliability ledger
- Risk suite + tail risk alerts
- Research parameter grid (top configs)
- Focus mode + watchlist + chat memory
- Daily summary reports (MD + PDF)
- Fully offline UI assets
 - PDF export per stock

---

## Status

**Stable and local‑only.**

If you want production hardening (CI tests, stricter offline enforcement, or advanced analytics), open an issue or message the maintainer.

---

## Changelog

See `CHANGELOG.md` for a human-readable history of what changed and what was added.
