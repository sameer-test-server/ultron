# Ultron UI/UX Roadmap

Date: 2026-03-10

## Ratings (Brutally Honest)

Scale: 1–10 (10 = excellent)

- Visual design (layout, styling consistency): **6/10**
- UX flow (clarity of task paths, hierarchy): **5/10**
- Information architecture (panel organization, scanning): **5/10**
- Performance (page weight, chart load, responsiveness): **6/10**
- Accessibility (contrast, keyboard nav, semantics): **4/10**
- Local-only compliance (no cloud/CDNs): **3/10**
- Reliability (error handling, offline behavior): **6/10**
- Responsiveness (mobile/desktop fit): **6/10**

## Remarks

### What’s working
- The UI is feature-rich (dashboard, live tracker, stock view, simulation lab, projection lab).
- Visuals are modern and attractive.
- Chat integration adds interactivity and explainability.
- Analysis and simulation data are surfaced clearly on detail pages.

### What’s hurting UX
- Too much content on the main dashboard without a clear primary action.
- Mixed visual language (Bootstrap + custom theme + external CDNs).
- Live tracker and multiple tables compete for attention.
- The UI does not clearly separate **observation** vs **hypothesis** vs **simulation**.

### Risky/Outdated
- External CDNs break local-only requirement.
- Live quotes use network calls (yfinance) unless explicitly disabled.
- Accessibility and keyboard navigation are not reviewed.

---

## What’s Done So Far

### Core functionality
- Data ingestion with multi-source fallback (Yahoo → NSE Bhavcopy → Stooq).
- Local analysis pipeline with indicators, regime detection, explainable insights.
- Paper trading simulation with risk controls and metrics.
- Chart generation + PDF export.

### UI functionality
- Dashboard with filters/sorting, summary cards, and ranked watchlist.
- Stock detail with interactive Plotly charts, simulation lab, projection lab.
- Chat endpoint + UI panel (with Ollama + fallback logic).
- Offline mode switch for live quotes.

---

## What Needs To Be Done

### Critical (must do for production-grade)
1. **Remove all CDN dependencies**
   - Bundle Bootstrap, fonts, and icons locally.
   - Eliminate external Google Fonts/CDN scripts.

2. **Make local-only mode fully enforced**
   - Disable live quotes by default.
   - Ensure no network calls happen when `ULTRON_OFFLINE_MODE=true`.

3. **Simplify dashboard hierarchy**
   - Reduce visible panels on first view.
   - Add a clear “primary” section (e.g., watchlist + analysis summary).

4. **Accessibility pass**
   - Improve contrast, keyboard focus states, aria labels.
   - Validate table headings and landmark structure.

---

## Recommended Enhancements (High Impact)

1. **UX restructuring**
   - Split dashboard into tabs: Overview | Live | Models | Labs.
   - Add tooltips for risk/regime/simulation terms.

2. **Performance**
   - Lazy-load heavy tables and chart content.
   - Cache interactive chart HTML and only regenerate on data change.

3. **Chat upgrade**
   - Add short-term memory (last N messages) in context.
   - Provide guided quick questions on the chat panel.

4. **Consistency**
   - Unify typography and color system.
   - Remove unused elements and keep a clean visual grid.

---

## Optional Enhancements (Nice to Have)

- Portfolio-level PDF summary.
- Per-stock “data freshness” badge and health warnings.
- Exportable CSV for analysis outputs.
- UI theme toggle (light/dark) with local assets only.

---

## Next Actions (Suggested)

1. Local asset bundling for Bootstrap/fonts/icons.
2. Disable live quotes by default and gate with explicit toggle.
3. Dashboard redesign: clarify primary flow and reduce panels.
4. Accessibility improvements and contrast audit.

If you want, I can start implementing these in order.
