# Changelog

All notable changes to Ultron are documented here.

## 2026-03-12

### Added
- Local-first UI with dashboard, focus mode, and watchlist
- Explainable chat with local Ollama (safe fallback when offline)
- PDF export per stock (`reports/pdf/`)
- Scenario engine (trend / mean-reversion / breakout)
- Reasoning engine with evidence-backed confidence
- Signal reliability ledger and risk suite
- Research lab parameter grid runner
- Daily summary reports (Markdown + PDF)
- Offline assets bundled locally (Bootstrap, icons, fonts)

### Improved
- Cached interactive charts and lighter dashboard pagination
- Data quality checks and freshness indicators
- Safer local-only defaults with offline mode

### Notes
- Ultron remains **read-only** and **simulation-only**. No real trading, no broker APIs, no cloud services.
