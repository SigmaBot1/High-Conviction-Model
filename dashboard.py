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
# DARK THEME CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #388bfd;
        box-shadow: 0 0 15px rgba(31,111,235,0.3);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #0d1117;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin: 8px 0;
    }
    .status-card-green {
        border-left: 4px solid #3fb950;
    }
    .status-card-red {
        border-left: 4px solid #f85149;
    }
    .status-card-yellow {
        border-left: 4px solid #d29922;
    }
    .status-card-blue {
        border-left: 4px solid #58a6ff;
    }
    
    /* Signal badge */
    .signal-active {
        background: #3fb950;
        color: #0d1117;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    .signal-inactive {
        background: #30363d;
        color: #8b949e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    .signal-warning {
        background: #d29922;
        color: #0d1117;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }
    
    /* Mono text */
    .mono {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    
    /* Glow effect for important numbers */
    .glow-green { color: #3fb950; text-shadow: 0 0 10px rgba(63,185,80,0.3); }
    .glow-red { color: #f85149; text-shadow: 0 0 10px rgba(248,81,73,0.3); }
    .glow-blue { color: #58a6ff; text-shadow: 0 0 10px rgba(88,166,255,0.3); }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider */
    .section-divider {
        border-top: 1px solid #30363d;
        margin: 20px 0;
    }
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
@st.cache_data(ttl=3600)
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
# HEADER
# ---------------------------------------------------------------------------
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0; font-size:1.8rem;'>\u26A1 HIGH-CONVICTION COMMAND CENTER</h1>
    <p style='color:#8b949e; font-family: monospace; font-size:0.85rem; margin-top:4px;'>
    QVM Rotation + Crash Override | SP900 Universe | Live Monitoring
    </p>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style='text-align:right; padding-top:12px;'>
        <span style='color:#8b949e; font-family:monospace; font-size:0.8rem;'>
        {datetime.now().strftime('%B %d, %Y | %I:%M %p')}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MARKET STATUS BAR
# ---------------------------------------------------------------------------
st.markdown("### \U0001F4CA Market Status")

vix = fetch_vix()
fg_data = fetch_fear_greed()
spy_data = fetch_spy_data()
fg_score = fg_data['score'] if fg_data else None

conditions_met, fear_details = get_fear_regime_status(vix, fg_score, spy_data)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    if vix is not None:
        vix_color = "normal" if vix < 20 else ("inverse" if vix > 25 else "off")
        st.metric("VIX", f"{vix}", delta=f"{'ELEVATED' if vix > 25 else 'CALM'}", delta_color=vix_color)
    else:
        st.metric("VIX", "N/A")

with col2:
    if fg_data:
        fg_label = fg_data.get('rating', 'Unknown')
        delta_color = "inverse" if fg_score < 35 else ("off" if fg_score < 50 else "normal")
        st.metric("Fear & Greed", f"{fg_score}", delta=fg_label, delta_color=delta_color)
    else:
        st.metric("Fear & Greed", "N/A")

with col3:
    if spy_data:
        st.metric("SPY", f"${spy_data['price']}", delta=f"{spy_data['off_high']}% off high", 
                  delta_color="inverse" if spy_data['off_high'] < -5 else "normal")
    else:
        st.metric("SPY", "N/A")

with col4:
    if spy_data:
        st.metric("SPY vs 200d EMA", f"{spy_data['vs_ema200']}%", 
                  delta=f"EMA: ${spy_data['ema200']}", delta_color="off")
    else:
        st.metric("SPY vs 200d", "N/A")

# QVM Portfolio P&L
with col5:
    qvm_pos_file = _path('data_cache/qvm_positions.csv')
    qvm_total_pnl_pct = 0
    qvm_total_pnl_dollar = 0
    qvm_total_cost = 0
    qvm_total_value = 0
    qvm_has_data = False
    if os.path.exists(qvm_pos_file):
        try:
            qvm_pos = pd.read_csv(qvm_pos_file)
            for _, row in qvm_pos.iterrows():
                entry = row.get('Entry Price', 0)
                shares = row.get('Shares', 0)
                if entry > 0 and shares > 0:
                    sd = fetch_stock_data(row['Ticker'])
                    if sd:
                        cost = entry * shares
                        val = sd['price'] * shares
                        qvm_total_cost += cost
                        qvm_total_value += val
            if qvm_total_cost > 0:
                qvm_total_pnl_dollar = qvm_total_value - qvm_total_cost
                qvm_total_pnl_pct = (qvm_total_pnl_dollar / qvm_total_cost) * 100
                qvm_has_data = True
        except:
            pass
    
    if qvm_has_data:
        st.metric("QVM P&L", f"{qvm_total_pnl_pct:+.2f}%",
                  delta=f"${qvm_total_pnl_dollar:+,.2f}",
                  delta_color="normal" if qvm_total_pnl_dollar >= 0 else "inverse")
    else:
        st.metric("QVM P&L", "---", delta="Set entry prices")

# Crash Tier P&L
with col6:
    # Pull from trade log for active crash positions
    crash_pnl_pct = 0
    crash_pnl_dollar = 0
    crash_has_data = False
    trade_log_file = None
    for p in [_path('output/trade_log_sp900_combined.csv'), _path('output/trade_log_sp500_combined.csv')]:
        if os.path.exists(p):
            trade_log_file = p
            break
    if trade_log_file:
        try:
            tl = pd.read_csv(trade_log_file)
            crash_trades = tl[(tl['tier'] == 'crash') & (tl['return_pct'].notna())]
            if not crash_trades.empty:
                crash_pnl_pct = crash_trades['return_pct'].mean()
                crash_pnl_dollar = crash_trades['pnl_dollars'].sum() if 'pnl_dollars' in crash_trades.columns else 0
                crash_has_data = True
        except:
            pass
    
    if crash_has_data:
        st.metric("Crash P&L", f"{crash_pnl_pct:+.1f}% avg",
                  delta=f"${crash_pnl_dollar:+,.0f} total",
                  delta_color="normal" if crash_pnl_pct >= 0 else "inverse")
    else:
        st.metric("Crash P&L", "---", delta="No crash trades")

with col7:
    if conditions_met >= 2:
        status_html = "<span class='signal-active'>FEAR REGIME ACTIVE</span>"
    elif conditions_met == 1:
        status_html = "<span class='signal-warning'>ELEVATED CAUTION</span>"
    else:
        status_html = "<span class='signal-inactive'>MARKET CALM</span>"
    
    st.markdown(f"""
    <div style='text-align:center; padding-top:8px;'>
        <p style='color:#8b949e; font-size:0.75rem; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>F8 Status ({conditions_met}/4)</p>
        {status_html}
    </div>
    """, unsafe_allow_html=True)

# Fear regime detail expander
with st.expander("F8 Fear Regime Details"):
    for detail in fear_details:
        st.markdown(f"`{detail}`")
    st.markdown(f"**Conditions met: {conditions_met}/4** (need 2 for crash-tier activation)")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "\U0001F4BC QVM Portfolio",
    "\U0001F6A8 Crash Watchlist", 
    "\U0001F50D Stock Screener",
    "\U0001F4C8 Backtest Results",
    "\U0001F3B2 Monte Carlo",
    "\u2699\uFE0F Settings"
])


# ---------------------------------------------------------------------------
# TAB 1: QVM PORTFOLIO
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### \U0001F4BC QVM Rotation Portfolio")
    st.markdown("<p style='color:#8b949e;'>Top 10 quality stocks ranked by composite value + momentum. Rebalances quarterly.</p>", unsafe_allow_html=True)
    
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
                        ['python', scanner_script, '--qvm-rank'],
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
    st.markdown("### \U0001F6A8 Crash Tier Watchlist")
    st.markdown("<p style='color:#8b949e;'>Stocks passing quality filters (F1-F7) monitored for crash-tier entry signals (F8-F12). A stock needs 12/12 to be actionable.</p>", unsafe_allow_html=True)
    
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
                            ['python', diag_script, ticker],
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
                with st.expander("Full Diagnose Output"):
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
        
        if watchlist_data:
            wl_df = pd.DataFrame(watchlist_data)
            st.dataframe(wl_df, use_container_width=True, hide_index=True)
        
        # Show when scores were last updated
        if cached_scores:
            sample = next(iter(cached_scores.values()), {})
            last_update = sample.get('updated', 'Unknown')
            st.markdown(f"<p style='color:#8b949e; font-size:0.8rem;'>\u2139\uFE0F Scores from SimFin via diagnose.py | Last updated: {last_update} | Click \"Diagnose All\" to refresh</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#d29922; font-size:0.8rem;'>\u26A0\uFE0F No scores cached. Click \"Diagnose All\" to run full model evaluation using SimFin data.</p>", unsafe_allow_html=True)
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
    st.markdown("### \U0001F50D Stock Screener")
    st.markdown("<p style='color:#8b949e;'>Evaluate any stock against the model's filters.</p>", unsafe_allow_html=True)
    
    col_input, col_btn, col_diag = st.columns([3, 1, 1])
    with col_input:
        screen_ticker = st.text_input("Enter ticker symbol", placeholder="MSFT", key="screener_input").upper().strip()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        screen_btn = st.button("\U0001F50D Quick Scan", key="screen_btn")
    with col_diag:
        st.markdown("<br>", unsafe_allow_html=True)
        diag_btn = st.button("\U0001F9EA Full Diagnose", key="diag_btn")
    
    # Full diagnose (runs diagnose.py via subprocess)
    if screen_ticker and diag_btn:
        import subprocess
        diag_script = _path('diagnose.py')
        if os.path.exists(diag_script):
            with st.spinner(f"Running full diagnose on {screen_ticker}..."):
                try:
                    result = subprocess.run(
                        ['python', diag_script, screen_ticker],
                        capture_output=True, text=True, timeout=120,
                        cwd=BASE_DIR, encoding='utf-8', errors='replace'
                    )
                    output = result.stdout if result.stdout else ""
                    errors = result.stderr if result.stderr else ""
                    
                    if output:
                        st.markdown("#### Full Model Evaluation")
                        st.code(output, language="text")
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
                    <span style='color:#8b949e;'>Price-based filters passing. Click "Full Diagnose" above for fundamental filters (F1-F7).
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
                    Click "Full Diagnose" for complete fundamental + technical analysis.</span>
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
                            ['python', main_script],
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
                            ['python', mc_script],
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
