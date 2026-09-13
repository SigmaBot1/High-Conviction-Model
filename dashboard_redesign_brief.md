# HIGH-CONVICTION COMMAND CENTER — REDESIGN BRIEF
**For: Claude Opus 4.6 + Claude Code (VS Code Terminal)**
**Stack: Python / Streamlit**
**Scope: Visual redesign + F6 bug fix. Logic/calculations untouched unless fixing F6.**

---

## CONTEXT

This is a live Streamlit trading dashboard called the **High-Conviction Command Center**. It runs a QVM (Quality + Value + Momentum) stock rotation model with a crash-tier override system. The app has six tabs: QVM Portfolio, Crash Watchlist, Stock Screener, Backtest Results, Monte Carlo, Settings.

**DO NOT TOUCH:** Backtest Results, Monte Carlo, Settings pages. Redesign only the first three tabs and the global chrome (header, nav, market status cards).

---

## THE CORE PROBLEM

The current UI looks like a default Streamlit prototype. The owner wants something that looks like a premium institutional trading terminal — think Bloomberg meets modern fintech. Specific issues:

1. Market Status metric cards are truncating critical data
2. Tab navigation uses childish emojis
3. Page titles use large emoji decorations that look unprofessional
4. Buttons are default Streamlit blue blobs
5. The Full Diagnose output prints as raw terminal text — needs a structured report card
6. F6 filter (PEG ratio) is broken — not computing or displaying
7. The Crash Tier table is readable but unsophisticated
8. Color palette has no visual hierarchy or intentional design system

---

## DESIGN SYSTEM

### Color Palette
Apply globally via `st.markdown()` CSS injection at app startup.

```css
:root {
  /* Backgrounds */
  --bg-primary:    #080C14;   /* main page background */
  --bg-surface:    #0F1724;   /* card / panel background */
  --bg-elevated:   #162035;   /* hover states, selected rows */
  --bg-border:     #1E2D45;   /* subtle borders */

  /* Accent */
  --accent-blue:   #3B7DD8;   /* primary interactive — buttons, links */
  --accent-blue-dim: #1E3F6E; /* muted blue for backgrounds */
  --accent-gold:   #C9A84C;   /* premium highlight — scores, key metrics */

  /* Semantic */
  --pass-green:    #22C55E;
  --pass-bg:       #052E16;
  --fail-red:      #EF4444;
  --fail-bg:       #2D0A0A;
  --warn-amber:    #F59E0B;
  --warn-bg:       #2D1A00;
  --neutral-gray:  #64748B;

  /* Typography */
  --text-primary:  #E2E8F0;
  --text-secondary:#94A3B8;
  --text-muted:    #475569;
  --text-mono:     'JetBrains Mono', 'Courier New', monospace;

  /* Borders */
  --border-subtle: 1px solid #1E2D45;
  --border-active: 1px solid #3B7DD8;
}
```

### Typography
- **Headers:** `font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: -0.02em`
- **Body:** `font-family: 'Inter', sans-serif; font-weight: 400`
- **Data/numbers:** `font-family: 'JetBrains Mono', monospace`
- **Load via Google Fonts** in the CSS injection block

### Streamlit Global Overrides
```css
/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Page background */
.stApp { background-color: #080C14; }

/* Remove default padding */
.block-container { padding-top: 1.5rem; padding-bottom: 0; max-width: 1400px; }

/* Default text */
body, p, div { color: #E2E8F0; font-family: 'Inter', sans-serif; }
```

---

## COMPONENT SPECIFICATIONS

### 1. APP HEADER

Replace the current title block. Use a full-width dark banner:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚡ HIGH-CONVICTION COMMAND CENTER          Sep 13, 2026 | 02:28 PM │
│  SP900 Universe  ·  QVM Rotation  ·  Crash Override  ·  Live        │
└─────────────────────────────────────────────────────────────────────┘
```

- Background: `#0F1724` with a 1px bottom border in `#1E2D45`
- The `⚡` bolt can stay but should be rendered as a styled HTML element, not raw emoji in a title — use `<span style="color: #C9A84C">⚡</span>`
- Date/time right-aligned in `#64748B` monospace
- Subtitle row in smaller `#475569` text

---

### 2. NAVIGATION TABS

**Current:** Uses `st.tabs()` with emoji labels like `🗂️ QVM Portfolio`, `🚨 Crash Watchlist`

**Replace with:** Custom HTML tab bar injected above the Streamlit tab widget, OR replace emoji with clean SVG icon inline in the label string. Streamlit tab labels support minimal HTML via markdown tricks — use clean text labels only.

**New tab labels (no emojis):**
- `Portfolio` (was: 🗂️ QVM Portfolio)
- `Crash Watchlist` (was: 🚨 Crash Tier Watchlist)
- `Screener` (was: 🔍 Stock Screener)
- `Backtest` — unchanged
- `Monte Carlo` — unchanged
- `Settings` — unchanged

**Tab bar styling:**
```css
.stTabs [data-baseweb="tab-list"] {
    background: #0F1724;
    border-bottom: 1px solid #1E2D45;
    gap: 0;
    padding: 0 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 12px 20px;
    border-bottom: 2px solid transparent;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    background: transparent;
    color: #E2E8F0;
    border-bottom: 2px solid #3B7DD8;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #94A3B8;
    background: #162035;
}
```

---

### 3. MARKET STATUS CARDS

**Current problem:** Cards cut off values. VIX shows fine but SPY value truncates to `$764...`, SPY vs 200D shows `6.6%` but the label is clipped, QVM P&L shows "Set entry pr..." and "Crash P&L" shows "+17...." These are `st.metric()` components in columns that are too narrow.

**Solution:** Replace `st.metric()` with custom HTML cards using `st.markdown(unsafe_allow_html=True)`.

**Card layout:** 6 cards in one row, plus the F8 Status badge. On smaller screens, allow wrap.

**Card template:**
```html
<div class="metric-card">
  <div class="metric-label">VIX</div>
  <div class="metric-value">15.84</div>
  <div class="metric-tag calm">● CALM</div>
</div>
```

**Card CSS:**
```css
.metric-card {
    background: #0F1724;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 16px 20px;
    min-width: 160px;
    flex: 1;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #E2E8F0;
    line-height: 1;
    margin-bottom: 8px;
    white-space: nowrap; /* PREVENTS TRUNCATION */
}
.metric-sub {
    font-size: 12px;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
}
.metric-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.metric-tag.calm     { color: #22C55E; background: #052E16; }
.metric-tag.fear     { color: #F59E0B; background: #2D1A00; }
.metric-tag.caution  { color: #F59E0B; background: #2D1A00; }
.metric-tag.negative { color: #EF4444; background: #2D0A0A; }
.metric-tag.positive { color: #22C55E; background: #052E16; }
```

**F8 Status Badge** — render as a separate full-width strip below the cards:
```html
<div class="f8-strip caution">
  <span class="f8-label">F8 FEAR REGIME</span>
  <span class="f8-status">ELEVATED CAUTION — 1 of 4 conditions met</span>
</div>
```
```css
.f8-strip {
    width: 100%;
    padding: 10px 20px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 12px;
}
.f8-strip.caution { background: #2D1A00; border: 1px solid #92400E; }
.f8-strip.active  { background: #2D0A0A; border: 1px solid #7F1D1D; }
.f8-strip.clear   { background: #052E16; border: 1px solid #14532D; }
.f8-label { font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: #F59E0B; }
.f8-status { font-size: 13px; color: #D97706; }
```

---

### 4. QVM PORTFOLIO PAGE

**Section title:** Replace `🗂️ QVM Rotation Portfolio` with plain styled HTML:
```html
<div class="section-header">
  <div class="section-title">QVM Rotation Portfolio</div>
  <div class="section-sub">Top 10 quality stocks ranked by composite value + momentum · Rebalances quarterly</div>
</div>
```

**"Run QVM Ranking" Button:** Replace with styled version:
```css
.stButton > button {
    background: #1E3F6E;
    color: #93C5FD;
    border: 1px solid #2563EB;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 8px 20px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2563EB;
    color: #FFFFFF;
    border-color: #3B82F6;
}
```

**Next Rebalance Date:** Style as a subtle chip, not bold text:
```html
<span class="date-chip">Next rebalance: Jun 30, 2026</span>
```

---

### 5. CRASH TIER WATCHLIST PAGE

**Page title:** Replace `🚨 Crash Tier Watchlist` with:
```html
<div class="section-header">
  <div class="section-title">Crash Tier Watchlist</div>
  <div class="section-sub">Stocks passing quality filters (F1–F7) monitored for crash-tier entry signals (F8–F12) · Needs 12/12 to be actionable</div>
</div>
```

**"Diagnose All" Button:**
- Same button CSS as above
- Add a small indicator icon before text — use `▶` unicode or `⟳` for refresh style
- Label: `Run Diagnose All`

**Watchlist Table redesign:**
Replace the default `st.dataframe()` with a custom HTML table. Style it as follows:

```css
.watchlist-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.watchlist-table th {
    background: #0F1724;
    color: #475569;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 14px;
    border-bottom: 1px solid #1E2D45;
    text-align: left;
}
.watchlist-table td {
    padding: 11px 14px;
    border-bottom: 1px solid #0F1724;
    color: #CBD5E1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.watchlist-table tr:hover td {
    background: #162035;
}
/* Ticker column */
.ticker-cell {
    font-weight: 700;
    color: #E2E8F0;
    font-size: 13px;
}
/* Score badges */
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
}
.score-high   { background: #052E16; color: #22C55E; } /* 11-12 */
.score-medium { background: #1A1A2D; color: #94A3B8; } /* 8-10 */
.score-low    { background: #2D0A0A; color: #EF4444; } /* <8 */

/* Conviction badges */
.conviction-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.conviction-medium   { background: #162035; color: #60A5FA; border: 1px solid #1E3F6E; }
.conviction-high     { background: #052E16; color: #22C55E; border: 1px solid #14532D; }
.conviction-donot    { background: #2D0A0A; color: #EF4444; border: 1px solid #7F1D1D; }

/* Action badges */
.action-watch        { color: #64748B; }
.action-near-signal  { color: #F59E0B; font-weight: 600; }
.action-enter        { color: #22C55E; font-weight: 700; }
.action-wait         { color: #475569; }

/* Filter status — replace ✅/❌ */
.filter-pass { color: #22C55E; font-weight: 700; }
.filter-fail { color: #EF4444; font-weight: 700; }
/* Render as: <span class="filter-pass">F9</span> <span class="filter-fail">F8</span> */
```

**Key Filters column:** Instead of `F8❌ F9✅ F12✅`, render as colored text labels:
- PASS filter: `<span class="filter-pass">F9</span>`
- FAIL filter: `<span class="filter-fail">F8</span>`

**"Off High" column:** Color negative values red, positive green automatically.

**Monitoring count line:** Style as a subtle data strip, not plain text:
```html
<div class="data-strip">
  <span class="data-label">UNIVERSE</span>
  <span class="data-value">43 tickers</span>
  <span class="data-sep">·</span>
  <span class="data-label">SOURCE</span>
  <span class="data-value">watchlist.txt</span>
  <span class="data-sep">·</span>
  <span class="data-label">LAST UPDATED</span>
  <span class="data-value">2026-09-02 11:17</span>
</div>
```

---

### 6. STOCK SCREENER PAGE

**Page title:** Replace `🔍 Stock Screener` with:
```html
<div class="section-header">
  <div class="section-title">Stock Screener</div>
  <div class="section-sub">Evaluate any stock against the full 12-filter model</div>
</div>
```

**Search input + buttons:**
```css
/* Input field */
.stTextInput > div > div > input {
    background: #0F1724;
    border: 1px solid #1E2D45;
    border-radius: 6px;
    color: #E2E8F0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 10px 14px;
}
.stTextInput > div > div > input:focus {
    border-color: #3B7DD8;
    box-shadow: 0 0 0 2px rgba(59, 125, 216, 0.15);
}
```

**Quick Scan / Full Diagnose buttons:**
- Quick Scan: Secondary style (outlined blue)
- Full Diagnose: Primary style (solid blue)
```css
/* Differentiate primary vs secondary via nth-child or custom class if possible */
/* Primary button — Full Diagnose */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
}
```

**Quick Result Cards (Price, Off High, EMA, Momentum, P/S):**
Replace the current metric boxes with a horizontal strip of compact stat cards:
```html
<div class="stat-strip">
  <div class="stat-item">
    <div class="stat-label">PRICE</div>
    <div class="stat-value">$1,633.35</div>
  </div>
  <div class="stat-item negative">
    <div class="stat-label">OFF 52W HIGH</div>
    <div class="stat-value">−30.0%</div>
    <div class="stat-tag">DISLOCATED</div>
  </div>
  <div class="stat-item positive">
    <div class="stat-label">VS 50 EMA</div>
    <div class="stat-value">+5.4%</div>
    <div class="stat-tag">STABILIZING</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">6M MOMENTUM</div>
    <div class="stat-value">146.9%</div>
  </div>
  <div class="stat-item">
    <div class="stat-label">P/S RATIO</div>
    <div class="stat-value">11.8x</div>
  </div>
</div>
```

**Quick Filter Check table:**
Replace with styled pass/fail rows (same CSS as watchlist filter styling).

---

### 7. FULL DIAGNOSE REPORT — COMPLETE REDESIGN

**This is the most critical change.** Currently the Full Diagnose output prints as raw `st.text()` / terminal output. Replace it entirely with a structured HTML report.

**Report structure:**

```
┌─────────────────────────────────────────────────────┐
│ SNDK · Sandisk Corporation · Technology              │
│ $1,633.35 · 52W High $2,335.00 · As of 2026-09-11  │
│ VIX 14.5 · SPY 1.0% off peak                        │
├──────────────────────────────────────────────────────┤
│ VERDICT: DO NOT ENTER          Score: 8/12           │
│ Failing: F3 · F7 · F8 · F11                         │
└──────────────────────────────────────────────────────┘

QUALITY FILTERS (F1–F7)
┌────────────────────────────┬────────┬──────────────────┐
│ Filter                     │ Status │ Value            │
├────────────────────────────┼────────┼──────────────────┤
│ F1a Revenue Growth >15%    │  PASS  │ 61.2% YoY        │
│ F1b Revenue Growth (prior) │   N/A  │ No prior data    │
│ F2a Gross Margin >40%      │  PASS  │ 50.9% ↑ 29.8%   │
│ F2b Gross Margin stable    │  PASS  │ +21.2% vs prior  │
│ F3  ROIC >15%              │  FAIL  │ 10.4% (need 15%) │
│ F4  D/E Ratio <1x          │  PASS  │ 0.08x            │
│ F5  Positive FCF           │  PASS  │ $1.45B TTM       │
│ F6  PEG Ratio              │  ----  │ SEE BUG NOTE     │
│ F7  FCF Yield >1.5%        │  FAIL  │ 1.16%            │
└────────────────────────────┴────────┴──────────────────┘

ENTRY TIMING (F8–F12)
[similar table structure]

INSIDER ACTIVITY
[clean card]

TRIGGER CONDITIONS
[what needs to change for entry to trigger]
```

**HTML/CSS for the report:**
```css
.diagnose-report {
    background: #0F1724;
    border: 1px solid #1E2D45;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 16px;
}

/* Header block */
.report-header {
    padding: 20px 24px;
    border-bottom: 1px solid #1E2D45;
}
.report-ticker { font-size: 22px; font-weight: 700; color: #E2E8F0; }
.report-company { font-size: 14px; color: #64748B; margin-left: 8px; }
.report-meta { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #475569; margin-top: 6px; }

/* Verdict block */
.report-verdict {
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.report-verdict.fail { background: #1A0A0A; border-bottom: 1px solid #2D0A0A; }
.report-verdict.pass { background: #0A1A0A; border-bottom: 1px solid #052E16; }
.verdict-label { font-size: 18px; font-weight: 700; }
.verdict-label.fail { color: #EF4444; }
.verdict-label.pass { color: #22C55E; }
.verdict-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #C9A84C;
}
.verdict-failing {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 4px;
}
.verdict-failing span { color: #EF4444; font-weight: 600; }

/* Filter section */
.filter-section { padding: 20px 24px; border-bottom: 1px solid #1E2D45; }
.filter-section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 12px;
}
.filter-row {
    display: flex;
    align-items: center;
    padding: 9px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    gap: 12px;
}
.filter-row:hover { background: #162035; }
.filter-row.pass { border-left: 3px solid #22C55E; }
.filter-row.fail { border-left: 3px solid #EF4444; background: #1A0808; }
.filter-row.na   { border-left: 3px solid #475569; }

.filter-name {
    flex: 1;
    font-size: 13px;
    color: #CBD5E1;
}
.filter-status-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    min-width: 44px;
    text-align: center;
    letter-spacing: 0.05em;
}
.filter-status-badge.pass { background: #052E16; color: #22C55E; }
.filter-status-badge.fail { background: #2D0A0A; color: #EF4444; }
.filter-status-badge.na   { background: #1A1A2D; color: #64748B; }

.filter-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94A3B8;
    text-align: right;
    min-width: 200px;
}
.filter-value .need {
    color: #475569;
    font-size: 11px;
    margin-left: 8px;
}

/* Trigger conditions */
.trigger-section { padding: 20px 24px; }
.trigger-item {
    display: flex;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1E2D45;
    font-size: 13px;
    color: #94A3B8;
}
.trigger-item:last-child { border-bottom: none; }
.trigger-filter { color: #F59E0B; font-weight: 600; min-width: 30px; }
.trigger-desc { color: #CBD5E1; }
.trigger-current { color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* Insider activity */
.insider-section {
    padding: 16px 24px;
    border-bottom: 1px solid #1E2D45;
    background: #080C14;
}
.insider-none { color: #475569; font-size: 13px; font-style: italic; }
```

**Implementation note:** Build a Python function `render_diagnose_report(result_dict)` that takes the existing diagnose output dictionary and renders it as `st.markdown(html, unsafe_allow_html=True)`. Do not change the underlying diagnose logic — only the presentation layer.

---

### 8. F6 FILTER BUG FIX

**Current behavior:** The Full Diagnose output shows F6 as `P/S <15.0 + rev >30%` — this is not the correct F6 filter. F6 should be the **PEG Ratio** filter but it is either (a) displaying the wrong filter, (b) the PEG data fetch is failing silently and falling back to P/S logic, or (c) the filter was renamed/replaced and the numbering shifted.

**Investigation steps (in this order):**
1. Locate the filter definition for F6 in the codebase — search for `F6`, `peg`, `PEG`, `price_to_earnings_growth`
2. Check the data source for PEG — likely SimFin API or a computed field. Confirm if the API is returning a value for SNDK
3. If PEG data is unavailable for a ticker, the filter should display: `F6 PEG Ratio — DATA UNAVAILABLE — N/A` rather than silently substituting a different filter
4. If F6 was intentionally replaced with the P/S + revenue growth composite, update the filter name and description in the report to reflect that accurately (`F6 — Valuation Check (P/S < 15x + Rev Growth > 30%)`)
5. In the report output, F6 must always display — never silently skip. If data is missing, show the filter row with `N/A` status and a note: `PEG data not available from SimFin for this ticker`

**Fix the report display:**
Regardless of the underlying data issue, ensure F6 always renders a row in the diagnose report with the correct filter name matching what the model actually evaluates.

---

## SECTION HEADERS — GLOBAL PATTERN

All page titles should use this pattern instead of emoji + large st.title():

```python
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style="
        padding: 0 0 20px 0;
        border-bottom: 1px solid #1E2D45;
        margin-bottom: 24px;
    ">
        <div style="
            font-size: 20px;
            font-weight: 600;
            color: #E2E8F0;
            letter-spacing: -0.02em;
        ">{title}</div>
        {f'<div style="font-size: 13px; color: #64748B; margin-top: 4px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)
```

---

## BUTTONS — GLOBAL PATTERN

All action buttons should follow this hierarchy:

| Type | Use | Style |
|---|---|---|
| Primary | Run QVM, Full Diagnose, Diagnose All | Solid blue `#2563EB`, white text |
| Secondary | Quick Scan, Watch, Filter actions | Outlined blue, `#3B7DD8` border |
| Destructive | Remove, Reset | Outlined red |
| Ghost | Minor actions | No border, `#64748B` text |

Apply via CSS targeting `.stButton > button` with nth-child selectors where needed, or by wrapping buttons in `<div class="btn-primary">` custom containers.

---

## WHAT NOT TO CHANGE

- **Backtest Results tab:** Leave completely as-is — layout, colors, data, everything.
- **Monte Carlo tab:** Leave completely as-is.
- **Settings tab:** Leave completely as-is.
- **All underlying model logic:** F1–F12 filter calculations, QVM scoring, ranking algorithms, data fetching from SimFin/OpenInsider — do not touch unless fixing the F6 display bug.
- **File structure:** Do not reorganize the project. Apply changes within existing files.

---

## DELIVERY CHECKLIST

Before considering the redesign complete, verify each item:

- [ ] Global CSS injection applied at app startup
- [ ] Market status cards show full values without truncation
- [ ] No emoji in tab labels
- [ ] No emoji in page section titles (replaced with styled HTML)
- [ ] Crash Tier table uses styled HTML with color-coded badges
- [ ] "Diagnose All" button styled as primary button
- [ ] "Run QVM Ranking" button styled as primary button
- [ ] Stock Screener search input styled
- [ ] Full Diagnose output renders as structured HTML report (not terminal text)
- [ ] F6 filter always shows in the report output (data or N/A, never silent)
- [ ] F6 filter name is accurate to what the model actually evaluates
- [ ] Backtest Results, Monte Carlo, Settings pages visually unchanged
- [ ] App tested on a stock with a full 12/12 result and a DO NOT ENTER result
- [ ] No Python errors introduced

---

*High-Conviction Command Center — Redesign Brief v1.0*
*Generated September 13, 2026*
