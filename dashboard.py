"""
HIGH-CONVICTION MODEL — COMMAND CENTER
=======================================
Run with:  streamlit run dashboard.py
Install:   pip install streamlit plotly yfinance requests beautifulsoup4

Place this file in your high-conviction-model/ folder alongside main.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import re
import json
import glob
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="High-Conviction Model",
    page_icon="\u26A1",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg-primary: #080C14;
  --bg-surface: #0F1724;
  --bg-elevated: #162035;
  --bg-border: #1E2D45;
  --accent-blue: #3B7DD8;
  --accent-gold: #C9A84C;
  --pass-green: #22C55E;
  --fail-red: #EF4444;
  --warn-amber: #F59E0B;
  --text-primary: #E2E8F0;
  --text-secondary: #94A3B8;
  --text-muted: #475569;
}

.stApp { background-color: #080C14; }
.block-container { padding-top: 1.5rem; padding-bottom: 0; max-width: 1400px; }
body, p, div { color: #E2E8F0; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #0F1724; border-bottom: 1px solid #1E2D45; gap: 0; padding: 0 1rem; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748B; font-size: 13px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; padding: 12px 20px; border-bottom: 2px solid transparent; border-radius: 0; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #E2E8F0 !important; border-bottom: 2px solid #3B7DD8 !important; }
.stTabs [data-baseweb="tab"]:hover { color: #94A3B8; background: #162035; }

/* Button styling */
.stButton > button { background: #1E3F6E; color: #93C5FD; border: 1px solid #2563EB; border-radius: 6px; font-size: 13px; font-weight: 600; letter-spacing: 0.04em; padding: 8px 20px; transition: all 0.15s; }
.stButton > button:hover { background: #2563EB; color: #FFFFFF; border-color: #3B82F6; }

/* Input styling */
.stTextInput > div > div > input { background: #0F1724; border: 1px solid #1E2D45; border-radius: 6px; color: #E2E8F0; font-family: 'JetBrains Mono', monospace; font-size: 14px; padding: 10px 14px; }
.stTextInput > div > div > input:focus { border-color: #3B7DD8; box-shadow: 0 0 0 2px rgba(59, 125, 216, 0.15); }

/* Expander styling */
[data-testid="stExpander"] { background: #0F1724; border: 1px solid #1E2D45; border-radius: 8px; }

/* Dataframe styling */
[data-testid="stDataFrame"] { border: 1px solid #1E2D45; border-radius: 8px; }

/* Status cards (legacy inline HTML) */
.status-card {
    background: #0F1724;
    border: 1px solid #1E2D45;
    border-radius: 8px;
    padding: 20px;
    margin: 8px 0;
}
.status-card-green { border-left: 4px solid #22C55E; }
.status-card-red { border-left: 4px solid #EF4444; }
.status-card-yellow { border-left: 4px solid #F59E0B; }
.status-card-blue { border-left: 4px solid #3B7DD8; }

/* Divider */
.section-divider { border-top: 1px solid #1E2D45; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BASE DIR (resolve all paths relative to this script's location)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(relative):
    """Resolve a path relative to the script directory."""
    return os.path.join(BASE_DIR, relative)

# ---------------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_vix():
    """Fetch current VIX from yfinance."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 2)
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_spy_data():
    """Fetch SPY data for market status."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2y")
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            high_52w = hist['Close'].rolling(252).max().iloc[-1]
            ema200 = hist['Close'].ewm(span=200).mean().iloc[-1]
            off_high = ((current - high_52w) / high_52w) * 100
            vs_ema200 = ((current - ema200) / ema200) * 100
            return {
                'price': round(current, 2),
                'high_52w': round(high_52w, 2),
                'off_high': round(off_high, 1),
                'ema200': round(ema200, 2),
                'vs_ema200': round(vs_ema200, 1),
            }
    except:
        pass
    return None

@st.cache_data(ttl=86400)
def fetch_fear_greed():
    """Fetch CNN Fear & Greed Index."""
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            score = data.get('fear_and_greed', {}).get('score', None)
            rating = data.get('fear_and_greed', {}).get('rating', None)
            if score is not None:
                return {'score': round(score, 1), 'rating': rating}
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """Fetch current stock data."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        if not hist.empty:
            current = hist['Close'].iloc[-1]
            high_52w = hist['Close'].max()
            ema50 = hist['Close'].ewm(span=50).mean().iloc[-1]
            off_high = ((current - high_52w) / high_52w) * 100
            vs_ema50 = ((current - ema50) / ema50) * 100
            
            # 6-month momentum
            if len(hist) >= 126:
                mom_6m = ((current / hist['Close'].iloc[-126]) - 1) * 100
            else:
                mom_6m = None
            
            return {
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', 'N/A'),
                'price': round(current, 2),
                'high_52w': round(high_52w, 2),
                'off_high': round(off_high, 1),
                'ema50': round(ema50, 2),
                'vs_ema50': round(vs_ema50, 1),
                'mom_6m': round(mom_6m, 1) if mom_6m else None,
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
            }
    except:
        pass
    return None

def load_trade_log():
    """Load the most recent trade log."""
    patterns = [
        _path('output/trade_log_sp900_combined.csv'),
        _path('output/trade_log_sp500_combined.csv'),
        _path('output/trade_log.csv'),
    ]
    for pattern in patterns:
        if os.path.exists(pattern):
            return pd.read_csv(pattern)
    # Also try glob for any trade_log file
    for f in glob.glob(_path('output/trade_log*.csv')):
        return pd.read_csv(f)
    return None

def load_metrics():
    """Load backtest metrics."""
    patterns = [
        _path('output/metrics_sp900_combined.csv'),
        _path('output/metrics_sp500_combined.csv'),
        _path('output/metrics.csv'),
    ]
    for pattern in patterns:
        if os.path.exists(pattern):
            return pd.read_csv(pattern)
    for f in glob.glob(_path('output/metrics*.csv')):
        return pd.read_csv(f)
    return None

def load_monte_carlo_stats():
    """Load Monte Carlo simulation stats."""
    path = _path('output/monte_carlo_stats.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_watchlist():
    """Load watchlist.txt."""
    wl_path = _path('watchlist.txt')
    if os.path.exists(wl_path):
        with open(wl_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return tickers
    return []

def get_fear_regime_status(vix, fg_score, spy_data):
    """Check how many F8 conditions are met."""
    conditions_met = 0
    details = []
    
    if vix is not None:
        if vix > 25:
            conditions_met += 1
            details.append(f"VIX {vix} > 25 \u2705")
        else:
            details.append(f"VIX {vix} < 25 \u274C")
    
    if fg_score is not None:
        if fg_score < 35:
            conditions_met += 1
            details.append(f"F&G {fg_score} < 35 \u2705")
        else:
            details.append(f"F&G {fg_score} > 35 \u274C")
    
    if spy_data is not None:
        if spy_data['vs_ema200'] < -10:
            conditions_met += 1
            details.append(f"SPY {spy_data['vs_ema200']}% below 200d \u2705")
        else:
            details.append(f"SPY {spy_data['vs_ema200']}% vs 200d \u274C")
        
        if spy_data['off_high'] < -10:
            conditions_met += 1
            details.append(f"SPY {spy_data['off_high']}% off high \u2705")
        else:
            details.append(f"SPY {spy_data['off_high']}% off high \u274C")
    
    return conditions_met, details


# ---------------------------------------------------------------------------
# SECTION HEADER HELPER
# ---------------------------------------------------------------------------
def section_header(title, subtitle=""):
    st.markdown(f"""<div style="padding:0 0 20px 0; border-bottom:1px solid #1E2D45; margin-bottom:24px;">
        <div style="font-size:20px; font-weight:600; color:#E2E8F0; letter-spacing:-0.02em;">{title}</div>
        {f'<div style="font-size:13px; color:#64748B; margin-top:4px;">{subtitle}</div>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RENDER HELPERS  (presentation-layer only — no model logic)
# ---------------------------------------------------------------------------

def render_watchlist_table(watchlist_data, total_count=None, last_updated="—"):
    """Return styled HTML for the crash watchlist table."""
    count = total_count if total_count is not None else len(watchlist_data)

    css = """<style>
.wl-row:hover { background: #162035 !important; }
</style>"""

    strip = (
        f'<div style="background:#0A1220; border:1px solid #1E2D45; border-radius:6px 6px 0 0; '
        f'padding:8px 16px; display:flex; gap:20px; align-items:center; '
        f'font-size:11px; font-family:\'JetBrains Mono\',monospace;">'
        f'<span><span style="color:#475569; text-transform:uppercase; letter-spacing:0.08em;">Universe</span>'
        f'&nbsp;<span style="color:#94A3B8;">{count} tickers</span></span>'
        f'<span style="color:#1E2D45;">&middot;</span>'
        f'<span><span style="color:#475569; text-transform:uppercase; letter-spacing:0.08em;">Source</span>'
        f'&nbsp;<span style="color:#94A3B8;">watchlist.txt</span></span>'
        f'<span style="color:#1E2D45;">&middot;</span>'
        f'<span><span style="color:#475569; text-transform:uppercase; letter-spacing:0.08em;">Last Updated</span>'
        f'&nbsp;<span style="color:#94A3B8;">{last_updated}</span></span>'
        f'</div>'
    )

    th_s = ("padding:8px 12px; font-size:11px; font-weight:600; letter-spacing:0.08em; "
            "text-transform:uppercase; color:#475569; background:#0F1724; "
            "border-bottom:1px solid #1E2D45; text-align:left;")
    thead = "".join(
        f'<th style="{th_s}">{h}</th>'
        for h in ["TICKER", "NAME", "PRICE", "OFF HIGH", "VS EMA50",
                  "KEY FILTERS", "SCORE", "CONVICTION", "ACTION"]
    )

    def _score_badge(s):
        if not s or s == "---":
            return '<span style="color:#475569; font-family:\'JetBrains Mono\',monospace;">—</span>'
        m = re.match(r'(\d+)/(\d+)', s)
        if not m:
            return f'<span style="color:#64748B; font-size:12px;">{s}</span>'
        n = int(m.group(1))
        bg, fg = ("#052E16", "#22C55E") if n >= 11 else (("#1A1A2D", "#94A3B8") if n >= 8 else ("#2D0A0A", "#EF4444"))
        return (f'<span style="background:{bg}; color:{fg}; font-size:11px; font-weight:600; '
                f'font-family:\'JetBrains Mono\',monospace; padding:3px 8px; border-radius:4px;">{s}</span>')

    def _conviction_badge(c):
        cu = c.upper()
        if cu == "HIGH":
            return ('<span style="color:#22C55E; border:1px solid #22C55E; font-size:11px; font-weight:600; '
                    'padding:2px 8px; border-radius:4px; font-family:\'JetBrains Mono\',monospace;">HIGH</span>')
        if cu == "MEDIUM":
            return ('<span style="color:#3B7DD8; border:1px solid #3B7DD8; font-size:11px; font-weight:600; '
                    'padding:2px 8px; border-radius:4px; font-family:\'JetBrains Mono\',monospace;">MEDIUM</span>')
        if "DO NOT" in cu or "ENTER" in cu:
            return ('<span style="color:#EF4444; border:1px solid #EF4444; font-size:11px; font-weight:600; '
                    'padding:2px 8px; border-radius:4px; font-family:\'JetBrains Mono\',monospace;">NO ENTRY</span>')
        return f'<span style="color:#475569; font-size:11px;">{c}</span>'

    def _action_cell(a):
        au = a.upper()
        if "SIGNAL" in au and "NEAR" not in au:
            return '<span style="color:#22C55E; font-weight:700; font-size:12px;">SIGNAL</span>'
        if "NEAR" in au:
            return '<span style="color:#F59E0B; font-weight:600; font-size:12px;">Near Signal</span>'
        if "WATCH" in au:
            return '<span style="color:#94A3B8; font-size:12px;">Watch</span>'
        if "WAIT" in au:
            return '<span style="color:#475569; font-size:12px;">Wait</span>'
        return f'<span style="color:#475569; font-style:italic; font-size:12px;">{a}</span>'

    def _filters_cell(fs):
        out = []
        for p in fs.split():
            if "✅" in p:
                name = p.replace("✅", "")
                out.append(f'<span style="color:#22C55E; font-size:11px; font-weight:600; '
                           f'margin-right:6px; font-family:\'JetBrains Mono\',monospace;">{name}&thinsp;PASS</span>')
            elif "❌" in p:
                name = p.replace("❌", "")
                out.append(f'<span style="color:#EF4444; font-size:11px; font-weight:600; '
                           f'margin-right:6px; font-family:\'JetBrains Mono\',monospace;">{name}&thinsp;FAIL</span>')
            else:
                out.append(f'<span style="color:#475569; font-size:11px;">{p}</span>')
        return "".join(out)

    td_s = "padding:9px 12px; border-bottom:1px solid #0D1A2D; vertical-align:middle;"
    rows = ""
    for row in watchlist_data:
        off_h = row.get("Off High", "")
        try:
            oh_num = float(off_h.replace("%", ""))
            oh_color = "#EF4444" if oh_num < 0 else "#22C55E"
        except Exception:
            oh_color = "#CBD5E1"
        rows += (
            f'<tr class="wl-row" style="background:#080C14;">'
            f'<td style="{td_s} color:#E2E8F0; font-weight:700; font-size:13px;">{row.get("Ticker","")}</td>'
            f'<td style="{td_s} color:#64748B; font-size:12px; max-width:130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{row.get("Name","")}</td>'
            f'<td style="{td_s} font-family:\'JetBrains Mono\',monospace; font-size:13px; color:#CBD5E1;">{row.get("Price","")}</td>'
            f'<td style="{td_s} font-family:\'JetBrains Mono\',monospace; font-size:13px; color:{oh_color};">{off_h}</td>'
            f'<td style="{td_s} font-family:\'JetBrains Mono\',monospace; font-size:13px; color:#CBD5E1;">{row.get("vs EMA50","")}</td>'
            f'<td style="{td_s}">{_filters_cell(row.get("Key Filters",""))}</td>'
            f'<td style="{td_s}">{_score_badge(row.get("Score","---"))}</td>'
            f'<td style="{td_s}">{_conviction_badge(row.get("Conviction",""))}</td>'
            f'<td style="{td_s}">{_action_cell(row.get("Action",""))}</td>'
            f'</tr>'
        )

    table = (
        f'<table style="width:100%; border-collapse:collapse; background:#080C14; '
        f'border:1px solid #1E2D45; border-top:none; border-radius:0 0 8px 8px;">'
        f'<thead><tr>{thead}</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )
    return css + strip + table


def render_diagnose_report(text):
    """Parse diagnose.py stdout and return a structured HTML report card."""
    if not text or not text.strip():
        return '<div style="padding:20px; color:#475569; font-style:italic;">No diagnose output to display.</div>'

    lines = text.splitlines()

    ticker = company = sector = price = high_52w = date_str = vix_str = spy_str = ""
    sub_checks = {}   # id → {label, passed, values}
    current_id = None
    group_pass = {}   # F1..F12 → True/False/None
    failing_groups = []
    na_groups = []
    conviction = ""
    n_groups_pass_str = ""
    insider_lines = []
    trigger_lines = []

    # State machine matching _print_research_report output structure
    # ════ #1 → header_name | ════ #2 → header_context | ════ #3 → filters
    # INSIDER ACTIVITY → insider | ════ #4 → verdict | TRIGGER/NEXT ACTION → triggers
    section = "pre"

    for line in lines:
        stripped = line.strip()
        is_double = bool(re.fullmatch(r'[═]+', stripped)) and len(stripped) > 10
        is_single = bool(re.fullmatch(r'[─]+', stripped)) and len(stripped) > 10

        if is_double:
            if section == "pre":
                section = "header_name"
            elif section == "header_name":
                section = "header_context"
            elif section == "header_context":
                section = "filters"
            elif section in ("filters", "insider"):
                section = "verdict"
            # closing ════ after verdict → ignore
            continue

        if is_single:
            continue

        # ── Header name block (ticker · company · sector) ──
        if section == "header_name":
            if not stripped:
                continue
            if '·' in stripped:
                parts = [p.strip() for p in stripped.split('·')]
                ticker = parts[0].strip()
                company = parts[1].strip() if len(parts) > 1 else ""
                sector = parts[2].strip() if len(parts) > 2 else ""
            elif not ticker:
                ticker = stripped.split()[0] if stripped.split() else ""
            continue

        # ── Header context block (price, vix, date) ──
        if section == "header_context":
            if not stripped:
                continue
            m = re.search(r'Price\s+(\$[\d.,]+)', stripped)
            if m: price = m.group(1)
            m = re.search(r'52w High\s+(\$[\d.,]+)', stripped)
            if m: high_52w = m.group(1)
            m = re.search(r'VIX\s+([\d.]+)', stripped)
            if m: vix_str = f"VIX {m.group(1)}"
            m = re.search(r'SPY\s+([\S]+)\s+off peak', stripped)
            if m: spy_str = f"SPY {m.group(1)} off peak"
            m = re.search(r'As of:\s*(\S+)', stripped)
            if m: date_str = m.group(1)
            continue

        # ── Filter sections ──
        if section == "filters":
            if not stripped:
                continue
            # Detect insider section transition
            if 'INSIDER ACTIVITY' in stripped.upper():
                section = "insider"
                current_id = None
                continue
            # Skip section titles
            if 'QUALITY FILTERS' in stripped.upper() or 'ENTRY TIMING FILTERS' in stripped.upper():
                current_id = None
                continue
            # F8/F9 explicit group notes:  "  F8 group (2-of-4 required): PASS"
            gm = re.match(r'\s*(F\d+)\s+group\s+\([^)]+\):\s*(PASS|FAIL|N/A)', line)
            if gm:
                gv = gm.group(2).strip()
                group_pass[gm.group(1)] = (True if gv == 'PASS' else (None if gv == 'N/A' else False))
                continue
            # Filter row: split on 3+ dots (dots fill label → PASS/FAIL/N/A)
            dot_parts = re.split(r'\.{3,}', stripped, maxsplit=1)
            if len(dot_parts) == 2:
                left, right = dot_parts[0].strip(), dot_parts[1].strip()
                if right in ('PASS', 'FAIL', 'N/A'):
                    m = re.match(r'(F\d+[a-z]?)\s+(.*)', left)
                    if m:
                        fid = m.group(1)
                        flabel = m.group(2).strip()
                        fpassed = True if right == 'PASS' else (None if right == 'N/A' else False)
                        sub_checks[fid] = {'label': flabel, 'passed': fpassed, 'values': []}
                        current_id = fid
                        continue
            # Value lines (6-space indent from _val helper)
            if line.startswith('      ') and current_id and stripped:
                sub_checks[current_id]['values'].append(stripped)
            continue

        # ── Insider section ──
        if section == "insider":
            if stripped:
                insider_lines.append(stripped)
            continue

        # ── Verdict section ──
        if section == "verdict":
            if not stripped:
                continue
            m = re.search(r'PASSES\s+\d+\s+of\s+\d+.*?\((\d+)\s+of\s+12', stripped, re.IGNORECASE)
            if m: n_groups_pass_str = f"{m.group(1)}/12"
            m = re.search(r'CONVICTION:\s+(.+)', stripped, re.IGNORECASE)
            if m: conviction = m.group(1).strip()
            m = re.search(r'Failing filters:\s+(.+)', stripped, re.IGNORECASE)
            if m: failing_groups = [f.strip() for f in m.group(1).split(',')]
            m = re.search(r'Insufficient data:\s+(.+)', stripped, re.IGNORECASE)
            if m: na_groups = [f.strip() for f in m.group(1).split(',')]
            if stripped.upper() == 'TRIGGER' or 'NEXT ACTION:' in stripped.upper():
                section = 'triggers'
            continue

        # ── Triggers section ──
        if section == "triggers":
            if stripped and not re.fullmatch(r'[═]+', stripped):
                if 'NEXT ACTION:' not in stripped.upper() and stripped.upper() != 'TRIGGER':
                    trigger_lines.append(stripped)
            continue

    # ── Derive group pass/fail from sub-checks where not explicitly stated ──
    _sub_ids = {
        'F1': ['F1a', 'F1b'], 'F2': ['F2a', 'F2b'],
        'F3': ['F3'], 'F4': ['F4'], 'F5': ['F5'], 'F6': ['F6'], 'F7': ['F7'],
        'F8': ['F8a', 'F8b', 'F8c', 'F8d'], 'F9': ['F9a', 'F9b'],
        'F10': ['F10'], 'F11': ['F11'], 'F12': ['F12'],
    }
    _logic = {
        'F1': 'lenient_all', 'F2': 'lenient_all',
        'F3': 'all', 'F4': 'all', 'F5': 'all', 'F6': 'all', 'F7': 'all',
        'F8': 'two_or_more', 'F9': 'any',
        'F10': 'all', 'F11': 'all', 'F12': 'all',
    }
    for g, ids in _sub_ids.items():
        if g in group_pass:
            continue
        vals = [sub_checks[i]['passed'] for i in ids if i in sub_checks]
        if not vals:
            group_pass[g] = None
            continue
        logi = _logic.get(g, 'all')
        if logi == 'two_or_more':
            n_t = sum(1 for v in vals if v is True)
            n_k = sum(1 for v in vals if v is not None)
            group_pass[g] = True if n_t >= 2 else (None if n_k == 0 else False)
        elif logi == 'any':
            group_pass[g] = True if any(v is True for v in vals) else (None if all(v is None for v in vals) else False)
        elif logi == 'lenient_all':
            group_pass[g] = False if any(v is False for v in vals) else (True if any(v is True for v in vals) else None)
        else:  # all
            group_pass[g] = True if all(v is True for v in vals) else (False if any(v is False for v in vals) else None)

    # Override with explicit failing/na from verdict block
    for g in failing_groups:
        if g in _sub_ids:
            group_pass[g] = False
    for g in na_groups:
        if g in _sub_ids and g not in group_pass:
            group_pass[g] = None

    # ── F6 display label fix (task requirement 3) ──
    if 'F6' not in sub_checks:
        sub_checks['F6'] = {
            'label': 'Valuation', 'passed': None,
            'values': ['Insufficient data from SimFin'],
            '_f6_disp': 'Valuation',
        }
    else:
        lbl_up = sub_checks['F6']['label'].upper()
        if 'PEG N/A' in lbl_up or ('PEG' in lbl_up and 'N/A' in lbl_up and 'UNAVAILABLE' in lbl_up):
            sub_checks['F6']['_f6_disp'] = 'Valuation (P/S <15 + Growth >30%)'
        elif 'PEG' in lbl_up and 'N/A' not in lbl_up:
            sub_checks['F6']['_f6_disp'] = 'PEG Ratio <2.0'
        elif 'P/S' in lbl_up or 'PRE-PROFITABLE' in lbl_up:
            sub_checks['F6']['_f6_disp'] = 'Valuation (P/S <15 + Growth >30%)'
        else:
            sub_checks['F6']['_f6_disp'] = sub_checks['F6']['label']

    # ── Conviction display ──
    cu = conviction.upper()
    if 'HIGH' in cu:
        v_color, v_label = '#22C55E', 'HIGH CONVICTION'
    elif 'MEDIUM' in cu:
        v_color, v_label = '#3B7DD8', 'MEDIUM'
    elif 'DO NOT' in cu or 'ENTER' in cu:
        v_color, v_label = '#EF4444', 'DO NOT ENTER'
    else:
        v_color, v_label = '#94A3B8', (conviction or 'UNKNOWN')

    if not n_groups_pass_str:
        n_gp = sum(1 for v in group_pass.values() if v is True)
        n_groups_pass_str = f"{n_gp}/12"

    actual_fail = [g for g, v in sorted(group_pass.items()) if v is False]
    fail_list = actual_fail if actual_fail else failing_groups
    failing_html = ""
    if fail_list:
        spans = " &middot; ".join(
            f'<span style="color:#EF4444; font-family:\'JetBrains Mono\',monospace; font-size:12px;">{f}</span>'
            for f in fail_list
        )
        failing_html = f'<div style="margin-top:6px; font-size:12px; color:#64748B;">Failing: {spans}</div>'

    # ── Filter row builder ──
    def _frow(gid, ids):
        subs = [(i, sub_checks[i]) for i in ids if i in sub_checks]
        gp = group_pass.get(gid)
        border = '#22C55E' if gp is True else ('#EF4444' if gp is False else '#475569')
        bb, bc = ('#052E16', '#22C55E') if gp is True else (('#2D0A0A', '#EF4444') if gp is False else ('#1A1A2D', '#94A3B8'))
        badge = 'PASS' if gp is True else ('FAIL' if gp is False else 'N/A')

        # Main label
        _lbl_map = {
            'F1': 'Revenue Growth >15% YoY',
            'F2': 'Gross Margin >40%',
            'F8': 'Fear Regime (2 of 4 conditions)',
            'F9': 'Price Dislocation (1 of 2)',
        }
        if len(ids) == 1 and subs:
            sid, sc_data = subs[0]
            main_lbl = sc_data.get('_f6_disp', sc_data.get('label', gid)) if sid == 'F6' else sc_data.get('label', gid)
        else:
            main_lbl = _lbl_map.get(gid, gid)

        # Sub-check detail lines
        detail = ""
        for sid, sc_data in subs:
            sp = sc_data.get('passed')
            sc_c = '#22C55E' if sp is True else ('#EF4444' if sp is False else '#64748B')
            si = '✓' if sp is True else ('✗' if sp is False else '—')
            vals = sc_data.get('values', [])
            if len(ids) > 1:
                sub_lbl = sc_data.get('_f6_disp', sc_data.get('label', sid)) if sid == 'F6' else sc_data.get('label', sid)
                detail += (f'<div style="font-size:11px; color:{sc_c}; padding-left:8px; margin-top:3px;">'
                           f'{si} {sub_lbl}</div>')
            if vals:
                detail += (f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:11px; '
                           f'color:#475569; padding-left:8px; margin-top:1px; white-space:pre-wrap;">{vals[0]}</div>')

        return (
            f'<div style="display:flex; flex-direction:column; padding:10px 16px; '
            f'border-left:3px solid {border}; margin-bottom:3px; background:#0F1724; border-radius:0 4px 4px 0;">'
            f'<div style="display:flex; align-items:center;">'
            f'<div style="width:40px; font-family:\'JetBrains Mono\',monospace; font-size:12px; font-weight:700; color:#64748B; flex-shrink:0;">{gid}</div>'
            f'<div style="flex:1; font-size:13px; color:#CBD5E1;">{main_lbl}</div>'
            f'<span style="background:{bb}; color:{bc}; font-size:10px; font-weight:700; padding:2px 8px; '
            f'border-radius:4px; letter-spacing:0.06em; font-family:\'JetBrains Mono\',monospace; flex-shrink:0;">{badge}</span>'
            f'</div>{detail}</div>'
        )

    # ── Insider ──
    if insider_lines:
        ins_html = '<br>'.join(
            f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:12px; color:#CBD5E1;">{l}</span>'
            for l in insider_lines
        )
    else:
        ins_html = '<span style="font-style:italic; color:#475569; font-size:13px;">No open-market insider purchases found in the last 90 days.</span>'

    # ── Triggers ──
    if trigger_lines:
        trig_parts = []
        for tl_item in trigger_lines:
            m = re.match(r'(F\d+)\s+→\s+(.*)', tl_item)
            if m:
                trig_parts.append(
                    f'<div style="padding:8px 0; border-bottom:1px solid #1E2D45; display:flex; gap:12px;">'
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:12px; font-weight:700; '
                    f'color:#F59E0B; min-width:32px; flex-shrink:0;">{m.group(1)}</span>'
                    f'<span style="font-size:13px; color:#CBD5E1;">{m.group(2).strip()}</span>'
                    f'</div>'
                )
            else:
                trig_parts.append(f'<div style="padding:6px 0; font-size:13px; color:#CBD5E1;">{tl_item}</div>')
        trig_html = "".join(trig_parts)
    else:
        trig_html = '<span style="font-style:italic; color:#475569; font-size:13px;">No trigger conditions — all entry filters passing or none parsed.</span>'

    # ── Header meta ──
    meta_parts = []
    if price:
        meta_parts.append(f'Price: <span style="color:#E2E8F0; font-family:\'JetBrains Mono\',monospace;">{price}</span>')
    if high_52w:
        meta_parts.append(f'52W High: <span style="color:#E2E8F0; font-family:\'JetBrains Mono\',monospace;">{high_52w}</span>')
    if vix_str:
        meta_parts.append(f'<span style="color:#64748B;">{vix_str}</span>')
    if spy_str:
        meta_parts.append(f'<span style="color:#64748B;">{spy_str}</span>')
    if date_str:
        meta_parts.append(f'<span style="color:#475569; font-size:11px;">As of {date_str}</span>')
    meta_html = "".join(f'<div>{p}</div>' for p in meta_parts)

    quality_rows = "".join(_frow(g, _sub_ids[g]) for g in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'])
    entry_rows = "".join(_frow(g, _sub_ids[g]) for g in ['F8', 'F9', 'F10', 'F11', 'F12'])
    company_sector = (company or '') + ('&nbsp;&nbsp;&middot;&nbsp;&nbsp;' + sector if sector else '')

    return f"""<div style="border:1px solid #1E2D45; border-radius:8px; overflow:hidden; margin-top:16px;">

  <div style="background:#0F1724; padding:20px 24px; border-bottom:1px solid #1E2D45; display:flex; justify-content:space-between; align-items:flex-start;">
    <div>
      <div style="font-size:22px; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em; font-family:'JetBrains Mono',monospace;">{ticker or '—'}</div>
      <div style="font-size:14px; color:#64748B; margin-top:4px;">{company_sector}</div>
    </div>
    <div style="text-align:right; font-size:12px; color:#64748B; line-height:2.0;">{meta_html}</div>
  </div>

  <div style="background:#0F1724; padding:16px 24px; border-bottom:1px solid #1E2D45; display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div style="font-size:18px; font-weight:700; color:{v_color}; letter-spacing:0.04em;">{v_label}</div>
      {failing_html}
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:32px; font-weight:500; color:#C9A84C;">{n_groups_pass_str}</div>
  </div>

  <div style="background:#080C14; padding:16px 24px; border-bottom:1px solid #1E2D45;">
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:10px;">Quality Filters F1–F7</div>
    {quality_rows}
  </div>

  <div style="background:#080C14; padding:16px 24px; border-bottom:1px solid #1E2D45;">
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:10px;">Entry Filters F8–F12</div>
    {entry_rows}
  </div>

  <div style="background:#080C14; padding:16px 24px; border-bottom:1px solid #1E2D45;">
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:10px;">Insider Activity (Last 90 Days)</div>
    <div style="line-height:1.8;">{ins_html}</div>
  </div>

  <div style="background:#080C14; padding:16px 24px; border-radius:0 0 8px 8px;">
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:10px;">Trigger Conditions</div>
    {trig_html}
  </div>

</div>"""


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:flex-start; justify-content:space-between; padding:8px 0 16px 0; border-bottom:1px solid #1E2D45; margin-bottom:20px;">
  <div>
    <div style="font-size:22px; font-weight:700; color:#E2E8F0; letter-spacing:-0.02em; line-height:1.2;">
      <span style="color:#C9A84C;">&#9889;</span> HIGH-CONVICTION COMMAND CENTER
    </div>
    <div style="font-size:13px; color:#475569; margin-top:6px; letter-spacing:0.02em;">
      SP900 Universe &middot; QVM Rotation &middot; Crash Override &middot; Live
    </div>
  </div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:13px; color:#475569; text-align:right; padding-top:4px; white-space:nowrap;">
    {datetime.now().strftime('%b %d, %Y &nbsp;|&nbsp; %I:%M %p')}
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MARKET STATUS BAR
# ---------------------------------------------------------------------------
vix = fetch_vix()
fg_data = fetch_fear_greed()
spy_data = fetch_spy_data()
fg_score = fg_data['score'] if fg_data else None

conditions_met, fear_details = get_fear_regime_status(vix, fg_score, spy_data)

# Compute QVM P&L — mirrors the same fallback logic used in the Portfolio tab.
# If the saved file exists use it; otherwise fall back to the hardcoded defaults
# so the card always shows live P&L even before the user clicks "Save Positions".
_QVM_DEFAULTS = {
    'Ticker':        ['CF', 'MU', 'INCY', 'GOOGL', 'WDC', 'LRCX', 'PTC', 'ARES', 'MSFT', 'MNST'],
    'Shares':        [3, 1, 3, 1, 1, 1, 2, 3, 1, 4],
    'Entry Price':   [112.93, 453.55, 97.40, 338.49, 372.16, 267.13, 139.73, 116.98, 421.23, 76.59],
}

qvm_pos_file = _path('data_cache/qvm_positions.csv')
qvm_total_pnl_pct = 0.0
qvm_total_pnl_dollar = 0.0
qvm_total_cost = 0.0
qvm_total_value = 0.0
qvm_has_data = False
try:
    if os.path.exists(qvm_pos_file):
        qvm_pos = pd.read_csv(qvm_pos_file)
    else:
        qvm_pos = pd.DataFrame(_QVM_DEFAULTS)
    for _, row in qvm_pos.iterrows():
        ticker = str(row.get('Ticker', '')).strip()
        try:
            entry = float(row.get('Entry Price', 0) or 0)
            shares = float(row.get('Shares', 0) or 0)
        except (TypeError, ValueError):
            entry, shares = 0.0, 0.0
        if entry > 0 and shares > 0 and ticker:
            sd = fetch_stock_data(ticker)
            if sd:
                qvm_total_cost += entry * shares
                qvm_total_value += sd['price'] * shares
    if qvm_total_cost > 0:
        qvm_total_pnl_dollar = qvm_total_value - qvm_total_cost
        qvm_total_pnl_pct = (qvm_total_pnl_dollar / qvm_total_cost) * 100
        qvm_has_data = True
except Exception as _e:
    print(f"[QVM P&L] Error: {_e}")

# Compute Crash P&L from live crash positions only (not backtest trade log)
crash_pnl_pct = 0
crash_pnl_dollar = 0
crash_has_data = False
crash_pos_file = _path('data_cache/crash_positions.csv')
if os.path.exists(crash_pos_file):
    try:
        crash_pos = pd.read_csv(crash_pos_file)
        crash_cost, crash_value = 0.0, 0.0
        for _, row in crash_pos.iterrows():
            ticker = str(row.get('Ticker', '')).strip()
            entry = row.get('Entry Price', 0)
            shares = row.get('Shares', 0)
            try:
                entry = float(entry)
                shares = float(shares)
            except (TypeError, ValueError):
                entry, shares = 0, 0
            if entry > 0 and shares > 0 and ticker:
                sd = fetch_stock_data(ticker)
                if sd:
                    crash_cost += entry * shares
                    crash_value += sd['price'] * shares
        if crash_cost > 0:
            crash_pnl_dollar = crash_value - crash_cost
            crash_pnl_pct = (crash_pnl_dollar / crash_cost) * 100
            crash_has_data = True
    except Exception as _e:
        print(f"[Crash P&L] Error: {_e}")

def _card(label, value, sub=""):
    sub_html = f'<div style="font-size:12px; color:#64748B; font-family:\'JetBrains Mono\',monospace;">{sub}</div>' if sub else ''
    return f"""<div style="background:#0F1724; border:1px solid #1E2D45; border-radius:8px; padding:16px 20px; min-width:160px;">
  <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569; margin-bottom:8px;">{label}</div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:500; color:#E2E8F0; white-space:nowrap;">{value}</div>
  {sub_html}
</div>"""

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    vix_val = f"{vix}" if vix is not None else "N/A"
    vix_sub = "ELEVATED" if (vix and vix > 25) else ("CALM" if vix else "")
    st.markdown(_card("VIX", vix_val, vix_sub), unsafe_allow_html=True)

with col2:
    fg_val = f"{fg_score}" if fg_data else "N/A"
    fg_sub = fg_data.get('rating', '') if fg_data else ""
    st.markdown(_card("Fear &amp; Greed", fg_val, fg_sub), unsafe_allow_html=True)

with col3:
    spy_val = f"${spy_data['price']}" if spy_data else "N/A"
    spy_sub = f"{spy_data['off_high']}% off high" if spy_data else ""
    st.markdown(_card("SPY", spy_val, spy_sub), unsafe_allow_html=True)

with col4:
    spy200_val = f"{spy_data['vs_ema200']}%" if spy_data else "N/A"
    spy200_sub = f"EMA ${spy_data['ema200']}" if spy_data else ""
    st.markdown(_card("SPY vs 200d", spy200_val, spy200_sub), unsafe_allow_html=True)

with col5:
    qvm_val = f"{qvm_total_pnl_pct:+.2f}%" if qvm_has_data else "---"
    qvm_sub = f"${qvm_total_pnl_dollar:+,.0f}" if qvm_has_data else "Set entry prices"
    st.markdown(_card("QVM P&amp;L", qvm_val, qvm_sub), unsafe_allow_html=True)

with col6:
    crash_val = f"{crash_pnl_pct:+.1f}%" if crash_has_data else "---"
    crash_sub = f"${crash_pnl_dollar:+,.0f} total" if crash_has_data else "No active trades"
    st.markdown(_card("Crash P&amp;L", crash_val, crash_sub), unsafe_allow_html=True)

with col7:
    if conditions_met >= 2:
        f8_val = "ACTIVE"
        f8_color = "#EF4444"
    elif conditions_met == 1:
        f8_val = "CAUTION"
        f8_color = "#F59E0B"
    else:
        f8_val = "CALM"
        f8_color = "#22C55E"
    st.markdown(f"""<div style="background:#0F1724; border:1px solid #1E2D45; border-radius:8px; padding:16px 20px; min-width:160px;">
  <div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569; margin-bottom:8px;">F8 Status</div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:500; color:{f8_color}; white-space:nowrap;">{f8_val}</div>
  <div style="font-size:12px; color:#64748B; font-family:'JetBrains Mono',monospace;">{conditions_met}/4 conditions</div>
</div>""", unsafe_allow_html=True)

# F8 status strip
if conditions_met >= 2:
    _f8_bg, _f8_border = "#2D0A0A", "#7F1D1D"
    _f8_status = "FEAR REGIME ACTIVE"
elif conditions_met == 1:
    _f8_bg, _f8_border = "#2D1A00", "#92400E"
    _f8_status = "ELEVATED CAUTION"
else:
    _f8_bg, _f8_border = "#052E16", "#14532D"
    _f8_status = "MARKET CALM"

st.markdown(f"""
<div style="width:100%; padding:10px 20px; border-radius:6px; display:flex; align-items:center; gap:16px; margin-top:12px; background:{_f8_bg}; border:1px solid {_f8_border};">
  <span style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#F59E0B;">F8 FEAR REGIME</span>
  <span style="font-size:13px; color:#D97706;">{_f8_status} &mdash; {conditions_met} of 4 conditions met</span>
</div>
""", unsafe_allow_html=True)

# F8 condition detail expander
with st.expander("F8 Conditions"):
    def _f8_row(label, passed, value_str, threshold_str):
        color = "#22C55E" if passed else ("#EF4444" if passed is False else "#64748B")
        badge_bg = "#052E16" if passed else ("#2D0A0A" if passed is False else "#1A1A2D")
        badge_text = "PASS" if passed else ("FAIL" if passed is False else "N/A")
        return (
            f'<div style="display:flex; align-items:center; padding:8px 12px; '
            f'border-left:3px solid {color}; margin-bottom:3px; background:#0F1724; border-radius:0 4px 4px 0;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:12px; font-weight:700; '
            f'color:#64748B; width:40px; flex-shrink:0;">{label}</div>'
            f'<div style="flex:1; font-size:13px; color:#CBD5E1;">{value_str}</div>'
            f'<span style="background:{badge_bg}; color:{color}; font-size:10px; font-weight:700; '
            f'padding:2px 8px; border-radius:4px; letter-spacing:0.06em; '
            f'font-family:\'JetBrains Mono\',monospace; margin-right:12px;">{badge_text}</span>'
            f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:11px; color:#475569; '
            f'text-align:right; min-width:160px;">{threshold_str}</div>'
            f'</div>'
        )

    f8a_pass = (vix > 25) if vix is not None else None
    f8a_val  = f"VIX: {vix:.1f}" if vix is not None else "VIX: n/a"
    f8a_thr  = "need >25"

    f8b_pass = (spy_data['vs_ema200'] < -10) if spy_data else None
    f8b_val  = f"SPY vs 200d EMA: {spy_data['vs_ema200']:+.1f}%" if spy_data else "SPY vs 200d: n/a"
    f8b_thr  = "need < −10%"

    f8c_pass = (spy_data['off_high'] < -10) if spy_data else None
    f8c_val  = f"SPY off high: {spy_data['off_high']:+.1f}%" if spy_data else "SPY off high: n/a"
    f8c_thr  = "need < −10%"

    f8d_pass = (fg_score < 35) if fg_score is not None else None
    f8d_val  = f"Fear &amp; Greed: {fg_score:.1f}" if fg_score is not None else "Fear &amp; Greed: n/a"
    f8d_thr  = "need <35"

    rows_html = (
        _f8_row("F8a", f8a_pass, f8a_val, f8a_thr) +
        _f8_row("F8b", f8b_pass, f8b_val, f8b_thr) +
        _f8_row("F8c", f8c_pass, f8c_val, f8c_thr) +
        _f8_row("F8d", f8d_pass, f8d_val, f8d_thr)
    )
    st.markdown(
        f'<div style="padding:4px 0;">{rows_html}'
        f'<div style="font-size:11px; color:#475569; margin-top:8px; font-family:\'JetBrains Mono\',monospace;">'
        f'2 of 4 conditions required to activate crash-tier entries</div></div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Portfolio",
    "Crash Watchlist",
    "Screener",
    "Backtest",
    "Monte Carlo",
    "Settings"
])


# ---------------------------------------------------------------------------
# TAB 1: QVM PORTFOLIO
# ---------------------------------------------------------------------------
with tabs[0]:
    section_header("QVM Rotation Portfolio", "Top 10 quality stocks ranked by composite value + momentum. Rebalances quarterly.")
    
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        refresh_qvm = st.button("\U0001F504 Run QVM Ranking", key="refresh_qvm")
    with col_info:
        next_rebalance = "June 30, 2026"
        st.markdown(f"<p style='color:#8b949e; padding-top:8px;'>Next rebalance: <strong style='color:#58a6ff;'>{next_rebalance}</strong></p>", unsafe_allow_html=True)
    
    # Run QVM ranking if button pressed
    if refresh_qvm:
        import subprocess
        scanner_script = _path('scanner.py')
        if os.path.exists(scanner_script):
            with st.spinner("Running QVM ranking on full universe..."):
                try:
                    result = subprocess.run(
                        [sys.executable, scanner_script, '--qvm-rank'],
                        capture_output=True, text=True, timeout=180,
                        cwd=BASE_DIR, encoding='utf-8', errors='replace'
                    )
                    if result.stdout:
                        st.markdown("#### Latest QVM Ranking")
                        st.code(result.stdout, language="text")
                except subprocess.TimeoutExpired:
                    st.error("QVM ranking timed out.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("scanner.py not found.")
    st.markdown("---")
    
    # Manual portfolio entry
    st.markdown("#### Current Positions")
    st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>Enter your QVM positions below. These are tracked for trailing stop monitoring.</p>", unsafe_allow_html=True)
    
    # Default positions from the QVM ranking (update with your actual holdings)
    default_positions = {
        'Ticker': ['CF', 'MU', 'INCY', 'GOOGL', 'WDC', 'LRCX', 'PTC', 'ARES', 'MSFT', 'MNST'],
        'Shares': [3, 1, 3, 1, 1, 1, 2, 3, 1, 4],
        'Entry Price': [112.93, 453.55, 97.40, 338.49, 372.16, 267.13, 139.73, 116.98, 421.23, 76.59],
        'Entry Date': ['2026-04-17'] * 10,
        'Trailing Stop %': [15.0] * 10,
    }
    
    # Check if positions file exists
    positions_file = _path('data_cache/qvm_positions.csv')
    if os.path.exists(positions_file):
        positions_df = pd.read_csv(positions_file)
        # Migrate old format: add Shares column if missing
        if 'Shares' not in positions_df.columns:
            positions_df['Shares'] = 0
        # Remove old Position % column if present
        if 'Position %' in positions_df.columns:
            positions_df = positions_df.drop(columns=['Position %'])
    else:
        positions_df = pd.DataFrame(default_positions)
    
    # Display positions with live data
    total_cost = 0
    total_value = 0
    total_pnl = 0
    
    if not positions_df.empty:
        live_data = []
        for _, row in positions_df.iterrows():
            ticker = row['Ticker']
            shares = row.get('Shares', 0)
            entry_price = row.get('Entry Price', 0)
            stock_data = fetch_stock_data(ticker)
            if stock_data and entry_price > 0 and shares > 0:
                current_price = stock_data['price']
                cost_basis = entry_price * shares
                current_val = current_price * shares
                pnl_dollar = current_val - cost_basis
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                stop_pct = row.get('Trailing Stop %', 15)
                stop_price = entry_price * (1 - stop_pct / 100)
                
                total_cost += cost_basis
                total_value += current_val
                total_pnl += pnl_dollar
                
                live_data.append({
                    'Ticker': ticker,
                    'Shares': int(shares),
                    'Entry': f"${entry_price:.2f}",
                    'Price': f"${current_price:.2f}",
                    'Cost': f"${cost_basis:.2f}",
                    'Value': f"${current_val:.2f}",
                    'P&L $': f"${pnl_dollar:+,.2f}",
                    'P&L %': f"{pnl_pct:+.1f}%",
                    'Stop': f"${stop_price:.2f}",
                })
            elif stock_data:
                live_data.append({
                    'Ticker': ticker,
                    'Shares': int(shares) if shares else 0,
                    'Entry': f"${entry_price:.2f}" if entry_price > 0 else '---',
                    'Price': f"${stock_data['price']:.2f}",
                    'Cost': '---',
                    'Value': '---',
                    'P&L $': '---',
                    'P&L %': '---',
                    'Stop': '---',
                })
        
        # Portfolio summary above table
        if total_cost > 0:
            total_pnl_pct = (total_pnl / total_cost) * 100
            col_tv, col_tc, col_tp, col_tpp = st.columns(4)
            with col_tv:
                st.metric("Portfolio Value", f"${total_value:,.2f}")
            with col_tc:
                st.metric("Cost Basis", f"${total_cost:,.2f}")
            with col_tp:
                st.metric("Total P&L", f"${total_pnl:+,.2f}",
                          delta_color="normal" if total_pnl >= 0 else "inverse")
            with col_tpp:
                st.metric("Return", f"{total_pnl_pct:+.2f}%",
                          delta_color="normal" if total_pnl_pct >= 0 else "inverse")
        
        if live_data:
            live_df = pd.DataFrame(live_data)
            st.dataframe(live_df, use_container_width=True, hide_index=True)
    
    # Live P&L chart for current positions
    if not positions_df.empty and total_cost > 0:
        st.markdown("#### Position Performance")
        import plotly.graph_objects as go
        
        pnl_tickers = []
        pnl_dollars = []
        pnl_colors = []
        
        for _, row in positions_df.iterrows():
            ticker = row['Ticker']
            entry_price = row.get('Entry Price', 0)
            shares = row.get('Shares', 0)
            if entry_price > 0 and shares > 0:
                stock_data = fetch_stock_data(ticker)
                if stock_data:
                    pnl_dollar = (stock_data['price'] - entry_price) * shares
                    pnl_tickers.append(ticker)
                    pnl_dollars.append(round(pnl_dollar, 2))
                    pnl_colors.append('#3fb950' if pnl_dollar >= 0 else '#f85149')
        
        if pnl_tickers:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Bar(
                x=pnl_tickers,
                y=pnl_dollars,
                marker_color=pnl_colors,
                text=[f"${v:+,.2f}" for v in pnl_dollars],
                textposition='outside',
                textfont=dict(color='#c9d1d9', size=11, family='JetBrains Mono'),
            ))
            fig_pnl.add_hline(y=0, line_color='#30363d', line_width=1)
            fig_pnl.update_layout(
                template="plotly_dark",
                paper_bgcolor='#0d1117',
                plot_bgcolor='#161b22',
                font=dict(family="JetBrains Mono, monospace", color="#c9d1d9"),
                yaxis_title="P&L ($)",
                height=300,
                margin=dict(l=40, r=40, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Position editor
    with st.expander("Edit Positions"):
        edited_df = st.data_editor(positions_df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Positions", key="save_pos"):
            os.makedirs(_path('data_cache'), exist_ok=True)
            edited_df.to_csv(positions_file, index=False)
            st.success("Positions saved!")
            st.rerun()
    
    # Sector allocation check
    with st.expander("Sector Allocation"):
        sector_data = {}
        for _, row in positions_df.iterrows():
            stock_data = fetch_stock_data(row['Ticker'])
            if stock_data:
                sector = stock_data.get('sector', 'Unknown')
                sector_data[sector] = sector_data.get(sector, 0) + row.get('Position %', 10)
        
        if sector_data:
            sector_df = pd.DataFrame([
                {'Sector': k, 'Allocation %': v, 'Status': '\u26A0\uFE0F Over 35%' if v > 35 else '\u2705 OK'}
                for k, v in sorted(sector_data.items(), key=lambda x: -x[1])
            ])
            st.dataframe(sector_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: CRASH WATCHLIST
# ---------------------------------------------------------------------------
with tabs[1]:
    section_header("Crash Tier Watchlist", "Stocks passing quality filters (F1-F7) monitored for crash-tier entry signals (F8-F12). A stock needs 12/12 to be actionable.")
    
    # Load watchlist
    watchlist = load_watchlist()
    
    if watchlist:
        col_wl_count, col_wl_diag = st.columns([3, 1])
        with col_wl_count:
            st.markdown(f"**Monitoring {len(watchlist)} tickers from watchlist.txt**")
        with col_wl_diag:
            run_all_diag = st.button("\U0001F9EA Diagnose All", key="diag_all")
        
        # Check for cached diagnose scores
        scores_cache = _path('data_cache/watchlist_scores.json')
        cached_scores = {}
        if os.path.exists(scores_cache):
            try:
                with open(scores_cache, 'r') as f:
                    cached_scores = json.load(f)
            except:
                cached_scores = {}
        
        # Run full diagnose on all watchlist stocks (uses SimFin)
        if run_all_diag:
            import subprocess, re
            diag_script = _path('diagnose.py')
            if os.path.exists(diag_script):
                progress_bar = st.progress(0, text="Starting diagnose...")
                all_outputs = []
                
                for i, ticker in enumerate(watchlist):
                    progress_bar.progress((i + 1) / len(watchlist), text=f"Diagnosing {ticker} ({i+1}/{len(watchlist)})...")
                    try:
                        result = subprocess.run(
                            [sys.executable, diag_script, ticker],
                            capture_output=True, text=True, timeout=60,
                            cwd=BASE_DIR, encoding='utf-8', errors='replace'
                        )
                        raw = result.stdout if result.stdout else ""
                        all_outputs.append(f"--- {ticker} ---\n{raw}")
                        
                        # Parse score from this single ticker's output
                        score = 0
                        total = 12
                        conv = 'DO NOT ENTER'
                        
                        for line in raw.split('\n'):
                            m = re.search(r'PASSES\s+(\d+)\s+of\s+(\d+)', line, re.IGNORECASE)
                            if m:
                                score = int(m.group(1))
                                total = int(m.group(2))
                            m2 = re.search(r'(\d+)\s*/\s*(\d+)\s*filter', line, re.IGNORECASE)
                            if m2:
                                score = int(m2.group(1))
                                total = int(m2.group(2))
                            m3 = re.search(r'(\d+)\s+of\s+(\d+)\s*filter', line, re.IGNORECASE)
                            if m3:
                                score = int(m3.group(1))
                                total = int(m3.group(2))
                            m4 = re.search(r'Score[:\s]+(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
                            if m4:
                                score = int(m4.group(1))
                                total = int(m4.group(2))
                            
                            upper = line.upper()
                            if 'DO NOT ENTER' in upper:
                                conv = 'DO NOT ENTER'
                            elif 'HIGH' in upper and ('CONVICTION' in upper or 'SIGNAL' in upper):
                                conv = 'HIGH'
                            elif 'MEDIUM' in upper and 'CONVICTION' in upper:
                                conv = 'MEDIUM'
                        
                        # Fallback conviction from score
                        if conv == 'DO NOT ENTER' and score >= 9:
                            if score >= 11:
                                conv = 'HIGH'
                            else:
                                conv = 'MEDIUM'
                        
                        cached_scores[ticker] = {
                            'score': score,
                            'total': total,
                            'conviction': conv,
                            'updated': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        
                    except subprocess.TimeoutExpired:
                        cached_scores[ticker] = {'score': 0, 'total': 12, 'conviction': 'TIMEOUT', 'updated': datetime.now().strftime('%Y-%m-%d %H:%M')}
                    except Exception as e:
                        cached_scores[ticker] = {'score': 0, 'total': 12, 'conviction': 'ERROR', 'updated': datetime.now().strftime('%Y-%m-%d %H:%M')}
                
                progress_bar.empty()
                
                # Save cache
                os.makedirs(os.path.dirname(scores_cache), exist_ok=True)
                with open(scores_cache, 'w') as f:
                    json.dump(cached_scores, f, indent=2)
                
                # Show raw output
                with st.expander("Full Diagnosis Output"):
                    st.code('\n'.join(all_outputs), language="text")
                
                st.success(f"Scores updated for {len(cached_scores)} tickers (SimFin data)")
        
        # Build watchlist table
        watchlist_data = []
        for ticker in watchlist:
            stock_data = fetch_stock_data(ticker)
            if stock_data:
                # Get score from cache (real SimFin-based score from diagnose.py)
                score_info = cached_scores.get(ticker, None)
                
                if score_info:
                    total_score = score_info['score']
                    score_total = score_info['total']
                    conviction = score_info['conviction']
                else:
                    total_score = None
                    score_total = 12
                    conviction = "Not scanned"
                
                # Key filter status from price data (always available)
                status_items = []
                f8_pass = conditions_met >= 2
                f9_pass = stock_data['off_high'] < -20
                f12_pass = abs(stock_data['vs_ema50']) < 10
                
                status_items.append("F8\u2705" if f8_pass else "F8\u274C")
                status_items.append("F9\u2705" if f9_pass else "F9\u274C")
                status_items.append("F12\u2705" if f12_pass else "F12\u274C")
                
                # Action based on score
                if total_score is not None:
                    if total_score >= 12:
                        action = '\u26A1 SIGNAL'
                    elif total_score >= 11:
                        action = '\U0001F525 Near Signal'
                    elif total_score >= 9:
                        action = '\U0001F440 Watch'
                    else:
                        action = '\u23F3 Wait'
                    score_display = f"{total_score}/{score_total}"
                else:
                    action = '\u2753 Scan needed'
                    score_display = "---"
                    conviction = "Click Diagnose"
                
                watchlist_data.append({
                    'Ticker': ticker,
                    'Name': stock_data['name'][:20],
                    'Price': f"${stock_data['price']:.2f}",
                    'Off High': f"{stock_data['off_high']}%",
                    'vs EMA50': f"{stock_data['vs_ema50']}%",
                    'Key Filters': ' '.join(status_items),
                    'Score': score_display,
                    'Conviction': conviction,
                    'Action': action,
                })
        
        # Resolve last_update for data strip
        last_update = "\u2014"
        if cached_scores:
            sample = next(iter(cached_scores.values()), {})
            last_update = sample.get('updated', '\u2014')

        if watchlist_data:
            st.markdown(
                render_watchlist_table(watchlist_data, total_count=len(watchlist), last_updated=last_update),
                unsafe_allow_html=True
            )

        if not cached_scores:
            st.markdown(
                "<p style='color:#F59E0B; font-size:0.8rem; margin-top:8px;'>"
                "No scores cached. Click \"Diagnose All\" to run full model evaluation using SimFin data.</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<p style='color:#475569; font-size:0.8rem; margin-top:8px;'>"
                f"Scores from SimFin via diagnose.py &middot; Click \"Diagnose All\" to refresh</p>",
                unsafe_allow_html=True
            )
    else:
        st.warning("No watchlist.txt found. Create one in your project folder with one ticker per line.")
    
    # Edit watchlist
    with st.expander("Edit Watchlist"):
        current_wl = '\n'.join(watchlist) if watchlist else 'MSFT\nAPPF\nADMA\nAMBA\nNVDA\nPLTR'
        new_wl = st.text_area("Tickers (one per line, # for comments)", value=current_wl, height=200)
        if st.button("Save Watchlist", key="save_wl"):
            with open(_path('watchlist.txt'), 'w') as f:
                f.write(new_wl)
            st.success("Watchlist saved!")
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 3: STOCK SCREENER
# ---------------------------------------------------------------------------
with tabs[2]:
    section_header("Stock Screener", "Evaluate any stock against the model's filters.")
    
    col_input, col_btn, col_diag = st.columns([3, 1, 1])
    with col_input:
        screen_ticker = st.text_input("Enter ticker symbol", placeholder="MSFT", key="screener_input").upper().strip()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        screen_btn = st.button("\U0001F50D Quick Scan", key="screen_btn")
    with col_diag:
        st.markdown("<br>", unsafe_allow_html=True)
        diag_btn = st.button("\U0001F9EA Full Diagnosis", key="diag_btn")
    
    # Full diagnose (runs diagnose.py via subprocess)
    if screen_ticker and diag_btn:
        import subprocess
        diag_script = _path('diagnose.py')
        if os.path.exists(diag_script):
            with st.spinner(f"Running full diagnose on {screen_ticker}..."):
                try:
                    result = subprocess.run(
                        [sys.executable, diag_script, screen_ticker],
                        capture_output=True, text=True, timeout=120,
                        cwd=BASE_DIR, encoding='utf-8', errors='replace'
                    )
                    output = result.stdout if result.stdout else ""
                    errors = result.stderr if result.stderr else ""
                    
                    if output:
                        st.markdown(render_diagnose_report(output), unsafe_allow_html=True)
                    if errors and "error" in errors.lower():
                        st.error(f"Diagnose errors:\n{errors[:500]}")
                except subprocess.TimeoutExpired:
                    st.error("Diagnose timed out (>120 seconds). Try again or run in terminal.")
                except Exception as e:
                    st.error(f"Error running diagnose: {e}")
        else:
            st.error(f"diagnose.py not found at {diag_script}")
    
    # Quick scan (price-based only)
    if screen_ticker and (screen_btn or (screen_ticker and not diag_btn)):
        stock_data = fetch_stock_data(screen_ticker)
        
        if stock_data:
            # Header card
            st.markdown(f"""
            <div class='status-card status-card-blue'>
                <h2 style='margin:0; font-size:1.4rem;'>{stock_data['name']} ({screen_ticker})</h2>
                <p style='color:#8b949e; margin:4px 0;'>{stock_data.get('sector', 'N/A')}</p>
                <h1 style='margin:8px 0; font-size:2rem; color:#f0f6fc;'>${stock_data['price']:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                off_high = stock_data['off_high']
                st.metric("Off 52W High", f"{off_high}%", 
                         delta="Dislocated" if off_high < -20 else "Near High",
                         delta_color="inverse" if off_high < -20 else "normal")
            with col_b:
                vs_ema = stock_data['vs_ema50']
                st.metric("vs 50 EMA", f"{vs_ema}%",
                         delta="Stabilizing" if abs(vs_ema) < 10 else "Freefall" if vs_ema < -10 else "Trending Up",
                         delta_color="normal" if abs(vs_ema) < 10 else "inverse")
            with col_c:
                if stock_data['mom_6m']:
                    st.metric("6M Momentum", f"{stock_data['mom_6m']}%",
                             delta_color="normal" if stock_data['mom_6m'] > 0 else "inverse")
                else:
                    st.metric("6M Momentum", "N/A")
            with col_d:
                if stock_data.get('ps_ratio'):
                    st.metric("P/S Ratio", f"{stock_data['ps_ratio']:.1f}x")
                else:
                    st.metric("P/S Ratio", "N/A")
            
            # Quick filter assessment
            st.markdown("#### Quick Filter Check (Price-Based)")
            
            checks = []
            
            # F8 - Fear regime
            checks.append({
                'Filter': 'F8 - Fear Regime',
                'Status': '\u2705 PASS' if conditions_met >= 2 else '\u274C FAIL',
                'Value': f'{conditions_met}/4 conditions met',
                'Trigger': f'Need {2 - conditions_met} more fear signals' if conditions_met < 2 else 'Active'
            })
            
            # F9 - Price dislocation
            f9_pass = stock_data['off_high'] < -20
            checks.append({
                'Filter': 'F9 - Price Dislocation',
                'Status': '\u2705 PASS' if f9_pass else '\u274C FAIL',
                'Value': f"{stock_data['off_high']}% off high",
                'Trigger': f"Need to fall to ${stock_data['high_52w'] * 0.80:.2f} (20% off high)" if not f9_pass else 'Dislocated'
            })
            
            # F12 - Stabilization
            f12_pass = abs(stock_data['vs_ema50']) < 10
            checks.append({
                'Filter': 'F12 - EMA50 Stabilization',
                'Status': '\u2705 PASS' if f12_pass else '\u274C FAIL',
                'Value': f"{stock_data['vs_ema50']}% from EMA50 (${stock_data['ema50']:.2f})",
                'Trigger': f"Need price to reach ${stock_data['ema50'] * 0.90:.2f}" if stock_data['vs_ema50'] < -10 else ('Stabilized' if f12_pass else f"Price at ${stock_data['ema50'] * 1.10:.2f} cap")
            })
            
            checks_df = pd.DataFrame(checks)
            st.dataframe(checks_df, use_container_width=True, hide_index=True)
            
            # Action recommendation
            st.markdown("#### Recommended Action")
            passing = sum(1 for ch in checks if '\u2705' in ch['Status'])
            
            if passing == 3:
                st.markdown("""
                <div class='status-card status-card-green'>
                    <strong>\u26A1 CRASH-TIER CANDIDATE</strong><br>
                    <span style='color:#8b949e;'>Price-based filters passing. Click "Full Diagnosis" above for fundamental filters (F1-F7).
                    If 12/12: read last 2 earnings transcripts, run dip/crash framework, make buy/no-buy decision.</span>
                </div>
                """, unsafe_allow_html=True)
            elif passing >= 1:
                st.markdown(f"""
                <div class='status-card status-card-yellow'>
                    <strong>\U0001F440 WATCHLIST</strong><br>
                    <span style='color:#8b949e;'>{passing}/3 price-based filters passing. Monitor for remaining conditions.
                    Add to watchlist.txt for daily tracking.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='status-card status-card-red'>
                    <strong>\u23F3 NO ACTION</strong><br>
                    <span style='color:#8b949e;'>No entry conditions met. Market calm, stock near highs or in freefall.
                    Click "Full Diagnosis" for complete fundamental + technical analysis.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Could not fetch data for {screen_ticker}. Check the ticker symbol.")


# ---------------------------------------------------------------------------
# TAB 4: BACKTEST RESULTS
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### \U0001F4C8 Backtest Performance")
    
    col_bt_refresh, col_bt_run = st.columns([3, 1])
    with col_bt_run:
        if st.button("\u25B6 Run Backtest", key="run_backtest"):
            import subprocess
            main_script = _path('main.py')
            if os.path.exists(main_script):
                with st.spinner("Running backtest... (this takes 5-10 minutes)"):
                    try:
                        result = subprocess.run(
                            [sys.executable, main_script],
                            capture_output=True, text=True, timeout=900,
                            cwd=BASE_DIR, encoding='utf-8', errors='replace'
                        )
                        if result.returncode == 0:
                            st.success("Backtest complete! Refreshing...")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Backtest failed:\n{result.stderr[:500]}")
                    except subprocess.TimeoutExpired:
                        st.error("Backtest timed out (>15 min).")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("main.py not found.")
    
    trade_log = load_trade_log()
    
    if trade_log is not None:
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_trades = len(trade_log)
        if 'return_pct' in trade_log.columns:
            returns = trade_log['return_pct']
            winners = returns[returns > 0]
            losers = returns[returns <= 0]
            win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
            avg_return = returns.mean()
            best_trade = returns.max()
            worst_trade = returns.min()
        else:
            win_rate = avg_return = best_trade = worst_trade = 0
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Avg Return", f"{avg_return:+.1f}%")
        with col4:
            st.metric("Best Trade", f"+{best_trade:.1f}%")
        with col5:
            st.metric("Worst Trade", f"{worst_trade:.1f}%")
        
        st.markdown("---")
        
        # Trade log
        col_qvm, col_crash = st.columns(2)
        
        with col_qvm:
            st.markdown("#### QVM Trades")
            qvm_trades = trade_log[trade_log['tier'] == 'qvm'] if 'tier' in trade_log.columns else trade_log
            if not qvm_trades.empty:
                display_cols = [c for c in ['ticker', 'entry_date', 'exit_date', 'return_pct', 'hold_days', 'exit_reason'] if c in qvm_trades.columns]
                st.dataframe(qvm_trades[display_cols].tail(20), use_container_width=True, hide_index=True)
                st.markdown(f"<p style='color:#8b949e;'>Showing last 20 of {len(qvm_trades)} QVM trades</p>", unsafe_allow_html=True)
        
        with col_crash:
            st.markdown("#### Crash Trades")
            crash_trades = trade_log[trade_log['tier'] == 'crash'] if 'tier' in trade_log.columns else pd.DataFrame()
            if not crash_trades.empty:
                display_cols = [c for c in ['ticker', 'entry_date', 'exit_date', 'return_pct', 'hold_days', 'exit_reason'] if c in crash_trades.columns]
                st.dataframe(crash_trades[display_cols], use_container_width=True, hide_index=True)
                
                crash_returns = crash_trades['return_pct'] if 'return_pct' in crash_trades.columns else pd.Series()
                if not crash_returns.empty:
                    st.markdown(f"""
                    <div class='status-card status-card-blue'>
                        <strong>Crash Tier Stats:</strong> {len(crash_trades)} trades | 
                        Win rate: {(crash_returns > 0).mean()*100:.0f}% | 
                        Avg return: {crash_returns.mean():+.1f}% | 
                        Avg hold: {crash_trades['hold_days'].mean():.0f} days
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No crash-tier trades in the log.")
        
        # Equity Curve & Drawdown from trade log
        if 'return_pct' in trade_log.columns and 'entry_date' in trade_log.columns:
            st.markdown("---")
            st.markdown("#### Equity Curve & Drawdown")
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Build equity curve from trades
            sorted_trades = trade_log.sort_values('entry_date').copy()
            sorted_trades['return_dec'] = sorted_trades['return_pct'] / 100.0
            
            # Simulate portfolio value: each trade uses ~10% of portfolio
            equity = [1_000_000]
            dates = [pd.Timestamp('2014-01-01')]
            
            for _, trade in sorted_trades.iterrows():
                position_size = 0.10
                port_return = trade['return_dec'] * position_size
                new_val = equity[-1] * (1 + port_return)
                equity.append(new_val)
                try:
                    exit_date = pd.to_datetime(trade.get('exit_date', trade['entry_date']))
                except:
                    exit_date = dates[-1] + pd.Timedelta(days=90)
                dates.append(exit_date)
            
            equity_series = pd.Series(equity, index=dates).sort_index()
            
            # Calculate drawdown
            running_max = equity_series.expanding().max()
            drawdown = ((equity_series - running_max) / running_max) * 100
            
            # SPY benchmark
            try:
                import yfinance as yf
                spy_hist = yf.Ticker("SPY").history(start="2014-01-01", end="2024-12-31")
                if not spy_hist.empty:
                    spy_normalized = (spy_hist['Close'] / spy_hist['Close'].iloc[0]) * 1_000_000
                    spy_running_max = spy_hist['Close'].expanding().max()
                    spy_drawdown = ((spy_hist['Close'] - spy_running_max) / spy_running_max) * 100
                else:
                    spy_normalized = None
                    spy_drawdown = None
            except:
                spy_normalized = None
                spy_drawdown = None
            
            # Create subplot with equity curve and drawdown
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=("Portfolio Value", "Drawdown")
            )
            
            # Equity curve
            fig.add_trace(go.Scatter(
                x=equity_series.index, y=equity_series.values,
                mode='lines', name='Model',
                line=dict(color='#58a6ff', width=2.5),
            ), row=1, col=1)
            
            if spy_normalized is not None:
                fig.add_trace(go.Scatter(
                    x=spy_normalized.index, y=spy_normalized.values,
                    mode='lines', name='SPY',
                    line=dict(color='#d29922', width=1.5, dash='dot'),
                ), row=1, col=1)
            
            # Drawdown
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                mode='lines', name='Model DD',
                fill='tozeroy',
                line=dict(color='#f85149', width=1),
                fillcolor='rgba(248,81,73,0.2)',
                showlegend=False,
            ), row=2, col=1)
            
            if spy_drawdown is not None:
                fig.add_trace(go.Scatter(
                    x=spy_drawdown.index, y=spy_drawdown.values,
                    mode='lines', name='SPY DD',
                    fill='tozeroy',
                    line=dict(color='#d29922', width=1, dash='dot'),
                    fillcolor='rgba(210,153,34,0.1)',
                    showlegend=False,
                ), row=2, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='#0d1117',
                plot_bgcolor='#161b22',
                font=dict(family="JetBrains Mono, monospace", color="#c9d1d9"),
                height=550,
                margin=dict(l=60, r=40, t=40, b=40),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            )
            fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1, tickformat="$,.0f")
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, ticksuffix="%")
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trade log found in output/ folder.")
        # Debug: show what files exist
        output_dir = _path('output')
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                st.info(f"CSV files found in output/: {', '.join(csv_files)}")
            else:
                st.info("No CSV files in output/. Run the backtest: `python main.py`")
        else:
            st.info(f"Output directory not found at: {output_dir}")
    
    # Load and display equity curve image if exists
    equity_img = _path('output/universe_comparison.png')
    if os.path.exists(equity_img):
        with st.expander("Equity Curve Chart"):
            st.image(equity_img, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 5: MONTE CARLO
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### \U0001F3B2 Monte Carlo Simulation")
    
    col_mc_info, col_mc_run = st.columns([3, 1])
    with col_mc_run:
        if st.button("\u25B6 Run Monte Carlo", key="run_mc"):
            import subprocess
            mc_script = _path('monte_carlo.py')
            if os.path.exists(mc_script):
                with st.spinner("Running 10,000 simulations..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, mc_script],
                            capture_output=True, text=True, timeout=300,
                            cwd=BASE_DIR, encoding='utf-8', errors='replace'
                        )
                        if result.returncode == 0:
                            st.success("Monte Carlo complete! Refreshing...")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"Error:\n{result.stderr[:500]}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("monte_carlo.py not found.")
    
    mc_stats = load_monte_carlo_stats()
    
    if mc_stats is not None:
        # Key probability metrics
        st.markdown("#### Probability Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prob_profit = (mc_stats['final_value'] > 1_000_000).mean() * 100
            st.metric("Prob of Profit", f"{prob_profit:.1f}%")
        with col2:
            prob_spy = (mc_stats['cagr'] > 0.13).mean() * 100
            st.metric("Prob > SPY", f"{prob_spy:.1f}%")
        with col3:
            prob_10 = (mc_stats['cagr'] > 0.10).mean() * 100
            st.metric("Prob > 10% CAGR", f"{prob_10:.1f}%")
        with col4:
            prob_loss = (mc_stats['final_value'] < 1_000_000).mean() * 100
            st.metric("Prob of Loss", f"{prob_loss:.1f}%")
        
        st.markdown("---")
        
        # Distribution stats
        st.markdown("#### Distribution Summary")
        
        summary_data = {
            'Metric': ['CAGR', 'Final Value', 'Max Drawdown', 'Sharpe'],
            '5th Pctl': [
                f"{np.percentile(mc_stats['cagr'], 5):.1%}",
                f"${np.percentile(mc_stats['final_value'], 5):,.0f}",
                f"{np.percentile(mc_stats['max_drawdown'], 5):.1%}",
                f"{np.percentile(mc_stats['sharpe'], 5):.2f}",
            ],
            'Median': [
                f"{mc_stats['cagr'].median():.1%}",
                f"${mc_stats['final_value'].median():,.0f}",
                f"{mc_stats['max_drawdown'].median():.1%}",
                f"{mc_stats['sharpe'].median():.2f}",
            ],
            '95th Pctl': [
                f"{np.percentile(mc_stats['cagr'], 95):.1%}",
                f"${np.percentile(mc_stats['final_value'], 95):,.0f}",
                f"{np.percentile(mc_stats['max_drawdown'], 95):.1%}",
                f"{np.percentile(mc_stats['sharpe'], 95):.2f}",
            ],
            'SPY': ['13.2%', '$3.45M', '-33.7%', '0.64'],
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        # CAGR distribution chart
        st.markdown("---")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mc_stats['cagr'] * 100,
            nbinsx=80,
            marker_color='#3fb950',
            marker_line_color='#30363d',
            marker_line_width=0.5,
            opacity=0.85,
            name='Simulated CAGR'
        ))
        fig.add_vline(x=13.19, line_dash="dash", line_color="#f85149", line_width=2,
                     annotation_text="SPY 13.2%", annotation_font_color="#f85149")
        fig.add_vline(x=mc_stats['cagr'].median() * 100, line_dash="solid", line_color="#58a6ff", line_width=2,
                     annotation_text=f"Median {mc_stats['cagr'].median()*100:.1f}%", annotation_font_color="#58a6ff")
        
        fig.update_layout(
            title="CAGR Distribution (10,000 simulations)",
            template="plotly_dark",
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(family="JetBrains Mono, monospace", color="#c9d1d9"),
            xaxis_title="CAGR (%)",
            yaxis_title="Frequency",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No Monte Carlo results found.")
        mc_path = _path('output/monte_carlo_stats.csv')
        st.info(f"Looking for: {mc_path}")
        output_dir = _path('output')
        if os.path.exists(output_dir):
            mc_files = [f for f in os.listdir(output_dir) if 'monte' in f.lower()]
            if mc_files:
                st.info(f"Monte Carlo files found: {', '.join(mc_files)}")
            else:
                st.info("Run Monte Carlo: `python monte_carlo.py`")
    
    # Show image if exists
    mc_img = _path('output/monte_carlo_results.png')
    if os.path.exists(mc_img):
        with st.expander("Full Monte Carlo Chart"):
            st.image(mc_img, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 6: SETTINGS
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### \u2699\uFE0F Model Configuration")
    
    st.markdown("#### Terminal Commands")
    st.markdown("""
    | Command | Purpose |
    |---------|---------|
    | `python diagnose.py` | Check all watchlist stocks |
    | `python diagnose.py MSFT PLTR` | Check specific tickers |
    | `python scanner.py --qvm-rank` | Show QVM top 20 ranking |
    | `python scanner.py --dry-run` | Test notification pipeline |
    | `python main.py` | Run full backtest |
    | `python monte_carlo.py` | Run Monte Carlo simulation |
    | `streamlit run dashboard.py` | Launch this dashboard |
    """)
    
    st.markdown("---")
    st.markdown("#### Model Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Quality Filters (F1-F7):**
        - F1: Revenue Growth > 15% YoY
        - F2: Gross Margin > 40%
        - F3: ROIC > 15%
        - F4: D/E < 1.0
        - F5: FCF Positive / 24+ mo cash
        - F6: PEG < 2 or P/S < 15 + growth > 30%
        - F7: FCF Yield > 1.5%
        """)
    with col2:
        st.markdown("""
        **Entry Filters (F8-F12):**
        - F8: Fear Regime (2/4 conditions)
        - F9: Price Dislocation (>20% off high or >1.5 SD)
        - F10: Momentum not bottom decile
        - F11: P/S bottom 25% of 3yr range
        - F12: Within 10% of 50 EMA
        """)
    
    st.markdown("---")
    st.markdown("#### Position Sizing")
    st.markdown("""
    - **QVM Tier:** 10% per position, top 10 stocks, 15% trailing stop, quarterly rebalance
    - **Crash Tier:** 8-10% per position, no trailing stop, thesis-break exits, 3yr max hold
    - **Sector Cap:** 35% max in any single GICS sector
    - **Max Positions:** 20 total (QVM + Crash combined)
    """)
    
    st.markdown("---")
    st.markdown("#### Data Sources")
    
    simfin_key = os.environ.get('SIMFIN_API_KEY', '')
    st.markdown(f"""
    - **Price Data:** yfinance (free)
    - **Fundamentals:** SimFin {'(\u2705 Key set)' if simfin_key else '(\u274C Key not set)'}
    - **Fear & Greed:** CNN API (free)
    - **VIX:** Yahoo Finance (free)
    - **Insider Data:** OpenInsider (free, via diagnose.py)
    """)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:20px 0;'>
    <p style='color:#30363d; font-family:monospace; font-size:0.75rem;'>
    High-Conviction Model v1.0 | QVM + Crash Override | Not Financial Advice
    </p>
</div>
""", unsafe_allow_html=True)
