"""
Stock research tool – complete model evaluation for any ticker.

New usage (live research):
    python diagnose.py PLTR
    python diagnose.py AAPL MSFT NVDA        # multiple tickers

Legacy usage (backtest diagnostics):
    python diagnose.py --ticker AAPL --date 2022-10-15
    python diagnose.py --scan-fear-dates

The research report evaluates 17 quantitative sub-checks mapped across
the 12 model filters, plus an insider-buying summary from OpenInsider.
Conviction level is based on how many of the 12 filter groups pass.
"""
import argparse
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import DataLoader
from fear_greed import fetch_fear_greed_score
from features import (
    FundamentalAccessor,
    compute_price_features,
    compute_spy_features,
    get_row,
    _available_cols,
)
from signals import SignalGenerator
from universe import get_sector_map, get_sp500_universe

logging.basicConfig(level=logging.WARNING)

# ── Layout constants ──────────────────────────────────────────────────────────
_W       = 72          # total line width
_COL     = 55          # left column width (label + value)
_PASS    = "  PASS"
_FAIL    = "  FAIL"
_NA      = "  N/A "

# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v: Optional[float], decimals: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{decimals}f}%"

def _x(v: Optional[float], decimals: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{decimals}f}x"

def _m(v: Optional[float]) -> str:
    """Format a dollar amount in millions/billions."""
    if v is None:
        return "n/a"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"

def _row(label: str, verdict: Optional[bool]) -> str:
    tag = _PASS if verdict is True else (_NA if verdict is None else _FAIL)
    dots = "." * max(1, _COL - len(label))
    return f"  {label}{dots}{tag}"

def _val(text: str) -> str:
    return f"      {text}"

def _sep(char: str = "─") -> str:
    return char * _W

def _hdr(text: str) -> str:
    return f"\n{text}\n{_sep()}"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_research_data(
    ticker: str,
    cfg: Config,
) -> Tuple[
    Optional[pd.DataFrame],  # price_feat
    Optional[pd.DataFrame],  # spy_feat
    pd.Series,               # vix_series
    Optional[dict],          # fundamentals raw dict
    Optional[dict],          # yf info (name, sector, etc.)
]:
    """Download/load all data needed for a single-ticker research report."""
    today   = datetime.today().strftime("%Y-%m-%d")
    start   = (datetime.today() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    tickers = [ticker, cfg.BENCHMARK_TICKER, "^VIX"]

    loader = DataLoader(cfg)
    # Price data – always fresh from yfinance for current data
    price_data = loader.download_prices(tickers, start=start, end=today)

    # Fundamentals – use FMP cache if present, otherwise download via yfinance
    fund_data = loader.load_fundamentals(ticker)
    if fund_data is None:
        fund_data = loader._download_single_fundamentals(ticker)

    raw_df  = price_data.get(ticker)
    spy_df  = price_data.get(cfg.BENCHMARK_TICKER)
    vix_df  = price_data.get("^VIX")

    # Flatten MultiIndex columns from cached stale yfinance batch downloads
    for _df in [raw_df, spy_df, vix_df]:
        if _df is not None and isinstance(_df.columns, pd.MultiIndex):
            _df.columns = _df.columns.get_level_values(-1)

    price_feat = compute_price_features(raw_df) if raw_df is not None and len(raw_df) >= 60 else None
    spy_feat   = compute_spy_features(spy_df) if spy_df is not None else pd.DataFrame()
    vix_series = vix_df["Close"].dropna() if vix_df is not None else pd.Series(dtype=float)

    # Grab yfinance info (name, sector) separately – lightweight call
    yf_info: Optional[dict] = None
    try:
        yf_info = yf.Ticker(ticker).info
    except Exception:
        pass

    return price_feat, spy_feat, vix_series, fund_data, yf_info


# ── Insider activity (OpenInsider) ────────────────────────────────────────────

def _fetch_insider_buys(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Scrape open-market purchase transactions for the last `days` days.
    Returns a DataFrame with columns [trade_date, name, title, price, qty, value]
    or None if the request fails or no purchases exist.
    """
    url = (
        f"http://openinsider.com/screener?s={ticker}&o=&pl=&ph=&st=&lt=&lk="
        f"&ex=&c=&t=&a=&d=&dd=&ta=&tb=&lg=&fg="
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return None

    # pandas.read_html parses all tables; the transactions table is the largest
    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception:
        return None

    # Find the table that has "Trade Type" or "Insider Name" column.
    # Normalise non-breaking spaces before matching.
    df = None
    for t in tables:
        cols_norm = [str(c).replace("\xa0", " ").lower() for c in t.columns]
        if any("trade" in c for c in cols_norm) and any("name" in c for c in cols_norm):
            df = t
            break

    if df is None or df.empty:
        return None

    # Normalise column names — replace non-breaking spaces, strip whitespace
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "trade date":
            col_map[c] = "trade_date"
        elif lc == "insider name":
            col_map[c] = "name"
        elif lc == "title":
            col_map[c] = "title"
        elif lc == "trade type":
            col_map[c] = "trade_type"
        elif lc == "price" and "price" not in col_map.values():
            col_map[c] = "price"
        elif lc in ("qty", "quantity"):
            col_map[c] = "qty"
        elif lc == "value" and "value" not in col_map.values():
            col_map[c] = "value"
    df = df.rename(columns=col_map)

    needed = {"trade_date", "name", "trade_type"}
    if not needed.issubset(df.columns):
        return None

    # Filter to open-market purchases only
    purchases = df[df["trade_type"].astype(str).str.startswith("P -")]

    # Filter to within `days`
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    try:
        purchases = purchases.copy()
        purchases["trade_date"] = pd.to_datetime(
            purchases["trade_date"].astype(str).str[:10], errors="coerce"
        )
        purchases = purchases[purchases["trade_date"] >= cutoff]
    except Exception:
        pass

    return purchases if not purchases.empty else None


# ── P/S history helper ────────────────────────────────────────────────────────

def _ps_history_and_percentile(
    ticker: str,
    date: pd.Timestamp,
    fa: FundamentalAccessor,
    price_feat: pd.DataFrame,
    market_cap: Optional[float],
    cfg: Config,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int, list]:
    """
    Returns (current_ps, ps_min, ps_max, ps_pct_rank, n_data_points, ps_history_list).
    ps_pct_rank is 0–100 (0 = cheapest ever).
    """
    info   = fa.raw.get("info") or {}
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    lookback_start = date - pd.Timedelta(days=365 * cfg.PS_HISTORY_YEARS)
    lag    = pd.Timedelta(days=fa.lag_days)

    def _build(income_df, ann_factor):
        rev_row = get_row(income_df, FundamentalAccessor.REVENUE_ROWS)
        if rev_row is None:
            return []
        hist = []
        for period_end in rev_row.index:
            avail = pd.Timestamp(period_end) + lag
            if avail < lookback_start or avail > date:
                continue
            rev = float(rev_row[period_end])
            if pd.isna(rev) or rev <= 0:
                continue
            try:
                close_h = price_feat["close"].asof(avail)
                if shares and pd.notna(close_h):
                    ps = float(close_h) * float(shares) / (rev * ann_factor)
                    hist.append(ps)
            except Exception:
                pass
        return hist

    qi = fa.raw.get("quarterly_income")
    hist = _build(qi, 4) if qi is not None else []
    if len(hist) < 4:
        ai = fa.raw.get("annual_income")
        if ai is not None:
            ah = _build(ai, 1)
            if len(ah) > len(hist):
                hist = ah

    if len(hist) < 2 or market_cap is None:
        return None, None, None, None, len(hist), hist

    current_ps = fa.ps_ratio(date, market_cap)
    if current_ps is None:
        return None, None, None, None, len(hist), hist

    ps_arr    = np.array(hist)
    ps_min    = float(ps_arr.min())
    ps_max    = float(ps_arr.max())
    pct_rank  = int(np.searchsorted(np.sort(ps_arr), current_ps) / len(ps_arr) * 100)

    return current_ps, ps_min, ps_max, pct_rank, len(hist), hist


# ── Filter evaluation ─────────────────────────────────────────────────────────

def _evaluate_all(
    ticker: str,
    date: pd.Timestamp,
    fa: FundamentalAccessor,
    price_feat: pd.DataFrame,
    spy_feat: pd.DataFrame,
    vix_series: pd.Series,
    sector_rank: Optional[float],
    market_cap: Optional[float],
    cfg: Config,
) -> List[dict]:
    """
    Evaluate all 17 sub-checks and return a list of result dicts:
      {id, label, passed, lines: [str]}

    17 sub-checks:
      F1a, F1b  — revenue growth (latest, prior quarter)
      F2a, F2b  — gross margin level, stability
      F3        — ROIC
      F4        — D/E ratio
      F5        — FCF / runway
      F6        — PEG or P/S valuation
      F7        — FCF yield
      F8a–F8c   — fear regime (VIX, SPY vs 200d, SPY drawdown)
      F9a, F9b  — price dislocation (z-score, % off 52w high)
      F10       — momentum not bottom decile
      F11       — P/S in bottom quartile
      F12       — near 50-day EMA
    """
    results: List[dict] = []

    def check(id_, label, passed, lines):
        results.append({"id": id_, "label": label, "passed": passed, "lines": lines})

    # ── F1: Revenue Growth ────────────────────────────────────────────────────
    rg = fa.revenue_growth_yoy(date)
    if rg is None:
        g1, g2 = None, None
        src = "no data"
    else:
        g1, g2 = rg
        src = ""

    pass_f1a = None if g1 is None else g1 > cfg.MIN_REVENUE_GROWTH_YOY
    pass_f1b = None if g2 is None else g2 > cfg.MIN_REVENUE_GROWTH_YOY

    check("F1a", "Revenue Growth >15% YoY (latest period)", pass_f1a,
          [f"Latest: {_pct(g1)}  [need >{_pct(cfg.MIN_REVENUE_GROWTH_YOY)}]"
           + (f"  — {src}" if src else "")])
    check("F1b", "Revenue Growth >15% YoY (prior period)", pass_f1b,
          [f"Prior:  {_pct(g2)}  [need >{_pct(cfg.MIN_REVENUE_GROWTH_YOY)}]"])

    # ── F2: Gross Margin ──────────────────────────────────────────────────────
    gm = fa.gross_margin(date)
    gm0 = gm[0] if gm else None
    gm1 = gm[1] if gm and len(gm) > 1 else None

    pass_f2a = None if gm0 is None else gm0 > cfg.MIN_GROSS_MARGIN
    pass_f2b = None if (gm0 is None or gm1 is None) else (gm0 >= gm1 - 0.05)

    trend = ""
    if gm0 is not None and gm1 is not None:
        diff = gm0 - gm1
        trend = f"  ({'↑ improving' if diff >= 0 else f'↓ {_pct(abs(diff))} decline'})"

    check("F2a", "Gross Margin >40%", pass_f2a,
          [f"Latest: {_pct(gm0)}  |  Prior: {_pct(gm1)}{trend}"
           f"  [need >{_pct(cfg.MIN_GROSS_MARGIN)}]"])
    check("F2b", "Gross Margin not deteriorating (>5% drop)", pass_f2b,
          [f"Δ vs prior period: {_pct(gm0 - gm1) if gm0 is not None and gm1 is not None else 'n/a'}"
           f"  [need ≥ −5%]"])

    # ── F3: ROIC ──────────────────────────────────────────────────────────────
    roic_val = fa.roic(date)
    check("F3", "ROIC >15%",
          None if roic_val is None else roic_val > cfg.MIN_ROIC,
          [f"ROIC: {_pct(roic_val)}  [need >{_pct(cfg.MIN_ROIC)}]"])

    # ── F4: D/E Ratio ─────────────────────────────────────────────────────────
    de_val = fa.de_ratio(date)
    bs     = fa.balance_sheet_snapshot(date)
    de_detail = ""
    if bs.get("total_debt") is not None and bs.get("equity") is not None:
        de_detail = f"  (Debt {_m(bs['total_debt'])}  |  Equity {_m(bs['equity'])})"
    check("F4", "D/E ratio <1×",
          None if de_val is None else de_val < cfg.MAX_DE_RATIO,
          [f"D/E: {_x(de_val)}{de_detail}  [need <{cfg.MAX_DE_RATIO}×]"])

    # ── F5: FCF / Runway ──────────────────────────────────────────────────────
    fcf_val = fa.fcf(date)
    if fcf_val is not None and fcf_val > 0:
        pass_f5 = True
        f5_lines = [f"TTM FCF: {_m(fcf_val)}  (positive ✓)"]
    else:
        runway = fa.cash_runway_months(date)
        pass_f5 = None if runway is None else runway >= cfg.MIN_CASH_RUNWAY_MONTHS
        f5_lines = [
            f"TTM FCF: {_m(fcf_val) if fcf_val is not None else 'n/a'}"
            f"  |  Cash runway: {f'{runway:.0f} months' if runway else 'n/a'}"
            f"  [need ≥{cfg.MIN_CASH_RUNWAY_MONTHS}m]"
        ]
    check("F5", "Positive FCF or ≥24-month cash runway", pass_f5, f5_lines)

    # ── F6: PEG or P/S ───────────────────────────────────────────────────────
    profitable  = fa.is_profitable(date)
    ni_ttm      = fa.net_income_ttm(date)
    pe_val      = market_cap / ni_ttm if (market_cap and ni_ttm and ni_ttm > 0) else None
    peg_val     = fa.peg_ratio(date, pe_val) if pe_val else None
    ps_val      = fa.ps_ratio(date, market_cap) if market_cap else None

    if profitable:
        pass_f6 = None if peg_val is None else peg_val < cfg.MAX_PEG_RATIO
        check("F6", f"PEG <{cfg.MAX_PEG_RATIO} (profitable co.)", pass_f6,
              [f"P/E: {_x(pe_val)}  |  PEG: {_x(peg_val)}  [need <{cfg.MAX_PEG_RATIO}]"])
    else:
        rev_g1 = rg[0] if rg else None
        if ps_val is None:
            pass_f6 = None
        else:
            pass_f6 = (
                ps_val < cfg.MAX_PS_PRE_PROFITABLE
                and rev_g1 is not None
                and rev_g1 > cfg.MIN_GROWTH_PRE_PROFITABLE
            )
        check("F6", f"P/S <{cfg.MAX_PS_PRE_PROFITABLE} + rev >30% (pre-profitable)", pass_f6,
              [f"P/S: {_x(ps_val)}  |  Rev growth: {_pct(rev_g1)}"
               f"  [need P/S <{cfg.MAX_PS_PRE_PROFITABLE} and growth >{_pct(cfg.MIN_GROWTH_PRE_PROFITABLE)}]"])

    # ── F7: FCF Yield ─────────────────────────────────────────────────────────
    debt  = bs.get("total_debt")
    cash  = bs.get("cash")
    fy    = fa.fcf_yield(date, market_cap, debt, cash) if market_cap else None
    ev    = (market_cap or 0) + (debt or 0) - (cash or 0)
    check("F7", "FCF Yield >1.5%",
          None if fy is None else fy > cfg.MIN_FCF_YIELD,
          [f"FCF Yield: {_pct(fy, 2)}  (FCF {_m(fcf_val)}  |  EV {_m(ev)})"
           f"  [need >{_pct(cfg.MIN_FCF_YIELD)}]"])

    # ── F8: Fear Regime ───────────────────────────────────────────────────────
    vix_val = None
    if len(vix_series) > 0:
        v = vix_series.asof(date)
        vix_val = float(v) if pd.notna(v) else None

    spy_below_pct, spy_drawdown = None, None
    if len(spy_feat) > 0:
        spy_row = spy_feat.asof(date)
        if spy_row is not None and isinstance(spy_row, pd.Series):
            b = spy_row.get("spy_below_200d_pct")
            d = spy_row.get("spy_drawdown")
            spy_below_pct = float(b) if pd.notna(b) else None
            spy_drawdown  = float(d) if pd.notna(d) else None

    # CNN F&G only meaningful for live/recent dates (not historical backtest mode)
    today_ts = pd.Timestamp.today().normalize()
    fg_score: Optional[float] = None
    if abs((date.normalize() - today_ts).days) <= 3:
        fg_score = fetch_fear_greed_score(cfg.CACHE_DIR)

    pass_f8a = None if vix_val is None else vix_val > cfg.VIX_FEAR_THRESHOLD
    pass_f8b = None if spy_below_pct is None else spy_below_pct > cfg.SPY_BELOW_200D_PCT
    pass_f8c = None if spy_drawdown is None else spy_drawdown > cfg.MARKET_DRAWDOWN_PCT
    pass_f8d = None if fg_score is None else fg_score < cfg.CNN_FG_FEAR_THRESHOLD

    fear_met = sum(1 for p in [pass_f8a, pass_f8b, pass_f8c, pass_f8d] if p is True)
    fear_avail = sum(1 for p in [pass_f8a, pass_f8b, pass_f8c, pass_f8d] if p is not None)

    check("F8a", f"VIX >{cfg.VIX_FEAR_THRESHOLD:.0f} (fear condition 1/4)", pass_f8a,
          [f"VIX: {f'{vix_val:.1f}' if vix_val else 'n/a'}  [need >{cfg.VIX_FEAR_THRESHOLD:.0f}]"])
    check("F8b", "SPY >10% below 200-day MA (fear condition 2/4)", pass_f8b,
          [f"SPY vs 200d MA: {_pct(spy_below_pct)}  [need >{_pct(cfg.SPY_BELOW_200D_PCT)}]"])
    check("F8c", "SPY >10% off recent high (fear condition 3/4)", pass_f8c,
          [f"SPY drawdown: {_pct(spy_drawdown)}  [need >{_pct(cfg.MARKET_DRAWDOWN_PCT)}]"])
    fg_note = (f"F&G: {fg_score:.1f}  [need <{cfg.CNN_FG_FEAR_THRESHOLD:.0f}]"
               if fg_score is not None else "F&G: unavailable  [need <35]")
    check("F8d", f"CNN Fear & Greed <{cfg.CNN_FG_FEAR_THRESHOLD:.0f} (fear condition 4/4)", pass_f8d,
          [f"{fg_note}  → {fear_met}/{fear_avail} fear conditions met"])

    # ── F9: Price Dislocation ────────────────────────────────────────────────
    feat_row  = price_feat.asof(date) if price_feat is not None else None
    zscore    = feat_row.get("price_zscore_12m") if feat_row is not None else None
    high_52w  = feat_row.get("high_52w") if feat_row is not None else None
    close_px  = feat_row.get("close") if feat_row is not None else None
    pct_off   = (
        (float(high_52w) - float(close_px)) / float(high_52w)
        if high_52w and close_px and float(high_52w) > 0
        else None
    )

    pass_f9a = None if zscore is None or pd.isna(zscore) else float(zscore) < cfg.PRICE_ZSCORE_LOWER
    pass_f9b = None if pct_off is None else pct_off > cfg.PRICE_BELOW_52W_HIGH_PCT

    check("F9a", "Price z-score <−1.5 (12-month mean)", pass_f9a,
          [f"Z-score: {f'{float(zscore):.2f}' if zscore is not None and not pd.isna(zscore) else 'n/a'}"
           f"  [need <{cfg.PRICE_ZSCORE_LOWER}]"])
    check("F9b", "Price >20% below 52-week high", pass_f9b,
          [f"Off 52w high: {_pct(pct_off)}  "
           f"(High {f'${float(high_52w):.2f}' if high_52w else 'n/a'}  |  "
           f"Current {f'${float(close_px):.2f}' if close_px else 'n/a'})"
           f"  [need >{_pct(cfg.PRICE_BELOW_52W_HIGH_PCT)}]"])

    # ── F10: Momentum ─────────────────────────────────────────────────────────
    mom_val = feat_row.get("ret_12_1") if feat_row is not None else None
    if sector_rank is not None:
        pass_f10  = sector_rank > cfg.MOMENTUM_BOTTOM_DECILE
        rank_str  = f"Sector rank: {sector_rank * 100:.0f}th pct  [need >{cfg.MOMENTUM_BOTTOM_DECILE * 100:.0f}th]"
    else:
        # Model defaults to pass when sector data is unavailable (per signals.py logic)
        pass_f10  = True
        rank_str  = "Sector rank: n/a — assumed PASS (model default when sector data unavailable)"
    check("F10", "Momentum not in bottom 10% of sector", pass_f10,
          [f"12-1 momentum: {_pct(float(mom_val) if mom_val is not None and not pd.isna(mom_val) else None)}"
           f"  |  {rank_str}"])

    # ── F11: P/S in Bottom Quartile ───────────────────────────────────────────
    current_ps, ps_min, ps_max, ps_pct, n_pts, ps_hist = _ps_history_and_percentile(
        ticker, date, fa, price_feat, market_cap, cfg
    )
    if current_ps is not None and len(ps_hist) >= 2:
        threshold = np.percentile(ps_hist, cfg.PS_QUARTILE_MAX * 100)
        pass_f11  = bool(current_ps <= threshold)
    else:
        pass_f11  = None

    if current_ps is not None and ps_pct is not None:
        ps11_line = (
            f"P/S: {_x(current_ps, 1)}  |  "
            f"3yr range: {_x(ps_min, 1)} – {_x(ps_max, 1)}  |  "
            f"Percentile: {ps_pct}th  ({n_pts} data pts)  "
            f"[need ≤{int(cfg.PS_QUARTILE_MAX * 100)}th pct]"
        )
    else:
        ps11_line = f"P/S history: {n_pts} data point(s) — insufficient  [need ≥2]"

    check("F11", f"P/S in bottom quartile of {cfg.PS_HISTORY_YEARS}-year range", pass_f11,
          [ps11_line])

    # ── F12: Near 50-day EMA ──────────────────────────────────────────────────
    ema50 = feat_row.get("ema_50") if feat_row is not None else None
    ema_gap = (
        (float(close_px) - float(ema50)) / float(ema50)
        if close_px and ema50
        else None
    )
    pass_f12 = (
        None if ema_gap is None
        else float(close_px) >= float(ema50) * 0.90
    )
    buf_note = "  (up to −10% below EMA allowed)"
    check("F12", "Price within −10% of 50-day EMA", pass_f12,
          [f"Price: {f'${float(close_px):.2f}' if close_px else 'n/a'}  |  "
           f"EMA50: {f'${float(ema50):.2f}' if ema50 else 'n/a'}  |  "
           f"Gap: {_pct(ema_gap, 1)}{buf_note}  [need ≥−10%]"])

    return results


# ── Report printing ───────────────────────────────────────────────────────────

_FILTER_GROUPS = [
    ("QUALITY FILTERS  (F1–F7)  — evaluated at last available period", [
        "F1a", "F1b", "F2a", "F2b", "F3", "F4", "F5", "F6", "F7",
    ]),
    ("ENTRY TIMING FILTERS  (F8–F12)  — current market conditions", [
        "F8a", "F8b", "F8c", "F8d", "F9a", "F9b", "F10", "F11", "F12",
    ]),
]

# Map each check-id to its parent filter number for the group pass/fail tally
_FILTER_GROUPS_MAP: Dict[str, str] = {
    "F1a": "F1", "F1b": "F1",
    "F2a": "F2", "F2b": "F2",
    "F3":  "F3", "F4":  "F4", "F5": "F5", "F6": "F6", "F7": "F7",
    "F8a": "F8", "F8b": "F8", "F8c": "F8", "F8d": "F8",
    "F9a": "F9", "F9b": "F9",
    "F10": "F10", "F11": "F11", "F12": "F12",
}

# How each filter group passes:
#   "all"          – every sub-check must be True (None = insufficient data → group is None)
#   "lenient_all"  – passes if no sub-check is False AND at least one is True
#                    (None is acceptable for optional sub-checks, matching signals.py F1/F2 logic)
#   "any"          – at least one sub-check must be True
#   "two_or_more"  – at least 2 sub-checks must be True (N/A conditions are skipped)
_FILTER_GROUP_LOGIC: Dict[str, str] = {
    "F1": "lenient_all",  # prior quarter (F1b) may be None — model accepts this
    "F2": "lenient_all",  # prior margin (F2b) may be None — model accepts this
    "F3": "all", "F4": "all", "F5": "all", "F6": "all", "F7": "all",
    "F8": "two_or_more",  # 2-of-4 fear conditions (CNN F&G skipped if unavailable → 2-of-3)
    "F9": "any",          # 1-of-2 dislocation conditions
    "F10": "all", "F11": "all", "F12": "all",
}


def _score_filters(results: List[dict]) -> Tuple[int, int, Dict[str, bool]]:
    """
    Returns (n_subchecks_passed, total_subchecks, filter_group_pass_dict).
    filter_group_pass_dict maps F1–F12 → True/False/None.
    """
    n_pass = sum(1 for r in results if r["passed"] is True)
    total  = len(results)

    # Group results by parent filter
    by_group: Dict[str, List[Optional[bool]]] = {}
    for r in results:
        g = _FILTER_GROUPS_MAP.get(r["id"], r["id"])
        by_group.setdefault(g, []).append(r["passed"])

    group_pass: Dict[str, Optional[bool]] = {}
    for g, vals in by_group.items():
        logic = _FILTER_GROUP_LOGIC.get(g, "all")
        if logic == "two_or_more":
            n_true = sum(1 for v in vals if v is True)
            n_known = sum(1 for v in vals if v is not None)
            if n_true >= 2:
                group_pass[g] = True
            elif n_known == 0:
                group_pass[g] = None
            else:
                group_pass[g] = False
        elif logic == "any":
            if any(v is True for v in vals):
                group_pass[g] = True
            elif all(v is None for v in vals):
                group_pass[g] = None
            else:
                group_pass[g] = False
        elif logic == "lenient_all":
            # Fails if any sub-check is explicitly False; passes if no False and ≥1 True;
            # None if all are None (no data at all)
            if any(v is False for v in vals):
                group_pass[g] = False
            elif any(v is True for v in vals):
                group_pass[g] = True
            else:
                group_pass[g] = None
        else:  # "all"
            if all(v is True for v in vals):
                group_pass[g] = True
            elif any(v is False for v in vals):
                group_pass[g] = False
            else:
                group_pass[g] = None

    return n_pass, total, group_pass


def _conviction(group_pass: Dict[str, Optional[bool]]) -> Tuple[str, str]:
    """Returns (level, description)."""
    n_pass = sum(1 for v in group_pass.values() if v is True)
    n_total = len(group_pass)
    if n_pass == n_total:
        return "HIGH CONVICTION", "All 12 model filters pass — this is a valid entry signal."
    if n_pass >= 9:
        return "MEDIUM", f"{n_pass}/{n_total} filters pass — monitor closely, not yet an entry."
    return "DO NOT ENTER", f"Only {n_pass}/{n_total} filters pass — does not meet model criteria."


def _next_action(n_groups_pass: int, group_pass: Dict[str, Optional[bool]]) -> str:
    failing = [g for g, v in group_pass.items() if v is False]
    if n_groups_pass == 12:
        return ("CRASH-TIER SIGNAL ACTIVE. Read last 2 earnings transcripts. "
                "Run dip/crash framework. Make buy/no-buy decision today.")
    if n_groups_pass == 11:
        if failing == ["F8"]:
            return ("WATCHLIST PRIORITY. One fear event away from crash signal. "
                    "Set price alerts. Monitor F8 daily.")
        if failing == ["F12"]:
            return ("STABILIZATION WATCH. Price in freefall. "
                    "Monitor for EMA50 convergence before entry.")
    if n_groups_pass >= 9:
        needed = 12 - n_groups_pass
        return (f"WATCH LIST. Not actionable yet. "
                f"{needed} filter(s) need to change before this becomes a signal.")
    return "NO ACTION. Does not meet minimum quality or entry criteria."


def _build_triggers(
    group_pass: Dict[str, Optional[bool]],
    cfg: "Config",
    vix_val: Optional[float],
    feat_row,
    price_feat: Optional[pd.DataFrame],
    date: pd.Timestamp,
    by_id: Dict[str, dict],
) -> List[str]:
    """Return trigger strings for each failing entry filter (F8, F9, F12 only)."""
    triggers = []

    # F8: show VIX and/or F&G thresholds
    if group_pass.get("F8") is False:
        parts = []
        if vix_val is not None:
            parts.append(
                f"VIX needs to rise above {cfg.VIX_FEAR_THRESHOLD:.0f} "
                f"(currently {vix_val:.1f})"
            )
        fg_line = (by_id.get("F8d") or {}).get("lines", [""])[0]
        fg_val: Optional[float] = None
        if fg_line.startswith("F&G: ") and "unavailable" not in fg_line:
            try:
                fg_val = float(fg_line.split("F&G: ")[1].split()[0])
            except (ValueError, IndexError):
                pass
        if fg_val is not None:
            parts.append(
                f"F&G needs to drop below {cfg.CNN_FG_FEAR_THRESHOLD:.0f} "
                f"(currently {fg_val:.0f})"
            )
        elif cfg.CNN_FG_FEAR_THRESHOLD:
            parts.append(f"F&G needs to drop below {cfg.CNN_FG_FEAR_THRESHOLD:.0f} (unavailable)")
        if parts:
            triggers.append("F8 → " + " OR ".join(parts))

    # F9: show price targets for both sub-checks
    if group_pass.get("F9") is False:
        close_px = feat_row.get("close") if feat_row is not None else None
        high_52w = feat_row.get("high_52w") if feat_row is not None else None
        parts = []
        if high_52w and close_px:
            target = float(high_52w) * (1.0 - cfg.PRICE_BELOW_52W_HIGH_PCT)
            parts.append(
                f"${target:.2f} ({cfg.PRICE_BELOW_52W_HIGH_PCT*100:.0f}% off 52w high "
                f"of ${float(high_52w):.2f})"
            )
        if price_feat is not None and "close" in price_feat.columns:
            close_series = price_feat["close"].dropna()
            if len(close_series) >= 120:
                roll_mean = close_series.rolling(252, min_periods=120).mean()
                roll_std  = close_series.rolling(252, min_periods=120).std()
                m = roll_mean.asof(date)
                s = roll_std.asof(date)
                if pd.notna(m) and pd.notna(s) and float(s) > 0:
                    target_z = float(m) + cfg.PRICE_ZSCORE_LOWER * float(s)
                    parts.append(
                        f"${target_z:.2f} ({abs(cfg.PRICE_ZSCORE_LOWER):.1f} SD below "
                        f"12m mean)"
                    )
        if parts:
            triggers.append("F9 → Price needs to fall to " + " or ".join(parts))

    # F12: show price needed to re-enter EMA50 band
    if group_pass.get("F12") is False:
        close_px = feat_row.get("close") if feat_row is not None else None
        ema50    = feat_row.get("ema_50") if feat_row is not None else None
        if ema50 and close_px:
            floor = float(ema50) * 0.90
            triggers.append(
                f"F12 → Price needs to rise to ${floor:.2f} "
                f"(within 10% of EMA50 at ${float(ema50):.2f})"
            )

    return triggers


def _print_research_report(
    ticker: str,
    yf_info: Optional[dict],
    price_feat: Optional[pd.DataFrame],
    vix_series: pd.Series,
    spy_feat: pd.DataFrame,
    date: pd.Timestamp,
    filter_results: List[dict],
    insider_df: Optional[pd.DataFrame],
    cfg: "Config" = None,
) -> None:
    name   = (yf_info or {}).get("longName") or (yf_info or {}).get("shortName") or ticker
    sector = (yf_info or {}).get("sector") or "Unknown"
    feat_row = price_feat.asof(date) if price_feat is not None else None

    close_px = feat_row.get("close") if feat_row is not None else None
    high_52w = feat_row.get("high_52w") if feat_row is not None else None
    low_52w  = None  # yfinance doesn't give 52w low in features; compute from price data
    pct_off  = (
        (float(high_52w) - float(close_px)) / float(high_52w)
        if high_52w and close_px and float(high_52w) > 0
        else None
    )
    vix_val = None
    if len(vix_series) > 0:
        v = vix_series.asof(date)
        vix_val = float(v) if pd.notna(v) else None

    n_sub, total_sub, group_pass = _score_filters(filter_results)
    n_groups_pass = sum(1 for v in group_pass.values() if v is True)
    conviction, conv_desc = _conviction(group_pass)

    print()
    print("═" * _W)
    # Header line
    header = f"  {ticker}  ·  {name}"
    if sector and sector != "Unknown":
        header += f"  ·  {sector}"
    print(header)
    print("═" * _W)

    # Price row
    price_str = f"${float(close_px):.2f}" if close_px else "n/a"
    high_str  = f"${float(high_52w):.2f}" if high_52w else "n/a"
    off_str   = f"(−{_pct(pct_off)})" if pct_off else ""
    print(f"  Price     {price_str:<10}  52w High {high_str} {off_str}")

    # Market context row
    vix_str = f"VIX {vix_val:.1f}" if vix_val else "VIX n/a"
    spy_row = spy_feat.asof(date) if len(spy_feat) > 0 else None
    spy_draw = spy_row.get("spy_drawdown") if spy_row is not None else None
    spy_str = f"SPY {_pct(spy_draw)} off peak" if spy_draw else ""
    print(f"  {vix_str:<20}  {spy_str}")
    print(f"  As of: {date.date()}")
    print("═" * _W)

    # ── Filter groups ─────────────────────────────────────────────────────────
    by_id = {r["id"]: r for r in filter_results}

    for section_title, ids in _FILTER_GROUPS:
        print(_hdr(section_title))
        for id_ in ids:
            r = by_id.get(id_)
            if r is None:
                continue
            # Format short label (strip parenthetical detail for display)
            display_label = r["label"].split("(")[0].strip()
            full_label    = f"{id_}  {display_label}"
            print(_row(full_label, r["passed"]))
            for line in r["lines"]:
                print(_val(line))

    # ── F8 / F9 group-level note ──────────────────────────────────────────────
    f8_pass = group_pass.get("F8")
    f9_pass = group_pass.get("F9")
    print()
    print(f"  F8 group (2-of-4 required): "
          f"{'PASS' if f8_pass else ('N/A' if f8_pass is None else 'FAIL')}")
    print(f"  F9 group (1-of-2 required): "
          f"{'PASS' if f9_pass else ('N/A' if f9_pass is None else 'FAIL')}")

    # ── Insider Activity ──────────────────────────────────────────────────────
    print(_hdr("INSIDER ACTIVITY  (last 90 days, open-market purchases only)"))
    if insider_df is None or insider_df.empty:
        print("  No open-market insider purchases found in the last 90 days.")
        print("  (Source: OpenInsider.com)")
    else:
        n = len(insider_df)
        total_val = None
        if "value" in insider_df.columns:
            def _parse_val(s):
                s = str(s).replace("$", "").replace(",", "").replace("+", "").strip()
                try:
                    return float(s)
                except Exception:
                    return None
            vals = [_parse_val(v) for v in insider_df["value"] if _parse_val(v) is not None]
            total_val = sum(vals) if vals else None

        print(f"  {n} purchase(s)"
              + (f"  |  Total value: {_m(total_val)}" if total_val else ""))
        print()

        # Show up to 5 most recent
        show = insider_df.head(5)
        for _, row in show.iterrows():
            date_str  = str(row.get("trade_date", ""))[:10]
            name_str  = str(row.get("name", "?"))[:28]
            title_str = str(row.get("title", ""))[:20]
            price_str = str(row.get("price", ""))[:8]
            qty_str   = str(row.get("qty", ""))[:10]
            val_str   = str(row.get("value", ""))[:12]
            print(f"  {date_str}  {name_str:<28}  {title_str:<20}"
                  f"  {qty_str:>10} sh @ {price_str:<8}  {val_str}")
        if len(insider_df) > 5:
            print(f"  ... and {len(insider_df) - 5} more")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print("═" * _W)

    failed_groups = [g for g, v in sorted(group_pass.items()) if v is False]
    na_groups     = [g for g, v in sorted(group_pass.items()) if v is None]

    print(f"  PASSES {n_sub} of {total_sub} individual sub-checks  "
          f"({n_groups_pass} of 12 filter groups)")
    print()
    print(f"  CONVICTION: {conviction}")
    print(f"  {conv_desc}")
    if failed_groups:
        print(f"  Failing filters: {', '.join(failed_groups)}")
    if na_groups:
        print(f"  Insufficient data: {', '.join(na_groups)}")

    # ── Next Action ───────────────────────────────────────────────────────────
    print()
    action = _next_action(n_groups_pass, group_pass)
    print(f"  NEXT ACTION: {action}")

    # ── Triggers (failing entry filters only) ─────────────────────────────────
    if cfg is not None:
        feat_row_trig = price_feat.asof(date) if price_feat is not None else None
        triggers = _build_triggers(group_pass, cfg, vix_val, feat_row_trig, price_feat, date, by_id)
        if triggers:
            print()
            print("  TRIGGER")
            for t in triggers:
                print(f"    {t}")

    print("═" * _W)
    print()


# ── Research entry point ──────────────────────────────────────────────────────

def research(ticker: str, cfg: Config) -> None:
    """Run and print the full research report for one ticker."""
    ticker = ticker.upper()
    print(f"\nLoading data for {ticker}...", end=" ", flush=True)

    price_feat, spy_feat, vix_series, fund_data, yf_info = _load_research_data(ticker, cfg)

    if price_feat is None:
        print(f"\nERROR: No price data available for {ticker}.")
        return
    if fund_data is None:
        print(f"\nWARNING: No fundamental data for {ticker}. Filters F1–F7 and F11 will show N/A.")
        fund_data = {}

    print("done.")

    # Use the most recent available trading date
    date = price_feat.index[-1]

    # Market cap
    info   = fund_data.get("info") or {}
    mc_info = info.get("marketCap")
    if mc_info:
        market_cap = float(mc_info)
    else:
        feat_row = price_feat.asof(date)
        close_px = feat_row.get("close") if feat_row is not None else None
        shares   = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if yf_info:
            mc_info = yf_info.get("marketCap")
            if mc_info:
                market_cap = float(mc_info)
                shares = shares or yf_info.get("sharesOutstanding")
            else:
                shares = shares or yf_info.get("sharesOutstanding")
                market_cap = float(close_px) * float(shares) if close_px and shares else None
        else:
            market_cap = float(close_px) * float(shares) if close_px and shares else None

    # Sector rank (best-effort; uses S&P 500 universe)
    sector_rank: Optional[float] = None
    try:
        universe    = get_sp500_universe()
        sector_map  = get_sector_map(universe)
        sp_tickers  = [t for t, _ in universe]
        ticker_sect = sector_map.get(ticker) or (yf_info or {}).get("sector")
        if ticker_sect:
            same_sect = [t for t, s in sector_map.items() if s == ticker_sect]
            # We only have price_feat for this ticker; fall back to no rank
            if price_feat is not None and "ret_12_1" in price_feat.columns:
                mom_val = price_feat["ret_12_1"].asof(date)
                if pd.notna(mom_val):
                    # rank against S&P 500 sector without re-downloading all tickers
                    sector_rank = None  # would need full universe price data
    except Exception:
        pass

    fa = FundamentalAccessor(fund_data, lag_days=cfg.FUNDAMENTAL_LAG_DAYS)

    # Evaluate all 17 sub-checks
    filter_results = _evaluate_all(
        ticker, date, fa, price_feat, spy_feat, vix_series,
        sector_rank, market_cap, cfg
    )

    # Insider activity
    print("Fetching insider activity from OpenInsider...", end=" ", flush=True)
    insider_df = _fetch_insider_buys(ticker, days=90)
    print("done." if insider_df is not None else "no data.")

    _print_research_report(
        ticker, yf_info, price_feat, vix_series, spy_feat,
        date, filter_results, insider_df, cfg,
    )


# ── Legacy backtest diagnostic functions (unchanged) ─────────────────────────

def _legacy_load_data(cfg, tickers):
    loader = DataLoader(cfg)
    all_tickers = tickers + [cfg.BENCHMARK_TICKER, "^VIX"]
    price_data  = loader.download_prices(all_tickers, cfg.START_DATE, cfg.END_DATE)
    fund_data   = loader.download_fundamentals(tickers)
    return price_data, fund_data


def _legacy_build_signal_gen(cfg, tickers, price_data, fund_data, custom_tickers=False):
    price_features = {}
    for t in tickers:
        df = price_data.get(t)
        if df is not None and len(df) >= 60:
            feat = compute_price_features(df)
            if len(feat):
                price_features[t] = feat

    spy_prices = price_data.get(cfg.BENCHMARK_TICKER)
    vix_prices = price_data.get("^VIX")
    spy_feat   = compute_spy_features(spy_prices) if spy_prices is not None else pd.DataFrame()
    universe   = get_sp500_universe()
    sector_map = get_sector_map(universe) if not custom_tickers else {t: "Unknown" for t in tickers}

    sig = SignalGenerator(
        config=cfg,
        price_features=price_features,
        spy_features=spy_feat,
        vix_prices=vix_prices,
        fundamentals=fund_data,
        sector_map=sector_map,
    )
    sig.precompute_sector_momentum()
    return sig, price_features


def _legacy_show_filter_breakdown(sig, ticker, date_str):
    date = pd.Timestamp(date_str)
    passed, results = sig.check_all_filters(ticker, date)
    status = "PASS" if passed else "FAIL"
    print(f"\n[{ticker}] on {date_str}  -->  {status}")
    print(f"{'Filter':<25} {'Result':>10}")
    print("-" * 38)
    for k, v in results.items():
        icon = "PASS" if v is True else ("SKIP" if v is None else "FAIL")
        print(f"  {k:<23} {icon:>10}")


def _legacy_scan_fear_dates(cfg, price_data):
    spy_prices = price_data.get(cfg.BENCHMARK_TICKER)
    vix_prices = price_data.get("^VIX")
    if spy_prices is None:
        print("No SPY data available.")
        return

    spy_feat = compute_spy_features(spy_prices)
    vix      = vix_prices["Close"].dropna() if vix_prices is not None else pd.Series(dtype=float)
    results  = []
    for date in spy_feat.index:
        conditions = []
        if len(vix):
            v = vix.asof(date)
            conditions.append(bool(pd.notna(v) and v > cfg.VIX_FEAR_THRESHOLD))
        else:
            conditions.append(False)
        row = spy_feat.loc[date]
        conditions.append(float(row.get("spy_below_200d_pct", 0)) > cfg.SPY_BELOW_200D_PCT)
        conditions.append(float(row.get("spy_drawdown", 0)) > cfg.MARKET_DRAWDOWN_PCT)
        if sum(conditions) >= 1:
            results.append({
                "date": date.date(),
                "conditions_met": sum(conditions),
                "vix": vix.asof(date) if len(vix) else None,
            })

    if not results:
        print("No fear-regime dates found.")
        return

    df = pd.DataFrame(results)
    print(f"\nFear-regime active on {len(df)} trading days:")
    print(df.head(10).to_string(index=False))
    if len(df) > 20:
        print(f"  ... {len(df)-20} more dates ...")
        print(df.tail(10).to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock research tool / backtest diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python diagnose.py PLTR              # live research report\n"
            "  python diagnose.py AAPL MSFT NVDA    # multiple tickers\n"
            "  python diagnose.py --ticker AAPL --date 2022-10-15   # backtest mode\n"
            "  python diagnose.py --scan-fear-dates                  # find fear dates\n"
        ),
    )
    # Positional: one or more ticker symbols → research mode
    parser.add_argument("tickers", nargs="*", metavar="TICKER",
                        help="Ticker(s) for live research report")

    # Legacy flags
    parser.add_argument("--ticker",          default=None, help="[legacy] ticker for backtest filter check")
    parser.add_argument("--date",            default="2022-10-15", help="[legacy] date for backtest check")
    parser.add_argument("--scan-fear-dates", action="store_true")
    parser.add_argument("--all-tickers",     action="store_true")
    parser.add_argument("--start",           default=None)
    parser.add_argument("--end",             default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.start:
        cfg.START_DATE = args.start
    if args.end:
        cfg.END_DATE = args.end

    # ── New research mode (positional tickers or watchlist.txt) ──────────────
    tickers_from_args = list(args.tickers)
    if not tickers_from_args:
        watchlist_path = Path(__file__).parent / "watchlist.txt"
        if watchlist_path.exists():
            lines = watchlist_path.read_text().splitlines()
            tickers_from_args = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if tickers_from_args:
        for t in tickers_from_args:
            research(t, cfg)
        return

    # ── Legacy backtest diagnostic mode ──────────────────────────────────────
    if args.ticker:
        tickers_to_use = [args.ticker]
        custom = True
    elif args.all_tickers:
        universe       = get_sp500_universe()
        tickers_to_use = [t for t, _ in universe][:50]
        custom = False
    else:
        tickers_to_use = ["AAPL", "MSFT", "NVDA", "AMZN", "META",
                          "JPM", "V", "UNH", "HD", "PG"]
        custom = False

    print(f"Loading backtest data for {len(tickers_to_use)} tickers...")
    price_data, fund_data = _legacy_load_data(cfg, tickers_to_use)

    if args.scan_fear_dates:
        _legacy_scan_fear_dates(cfg, price_data)
        return

    sig, price_features = _legacy_build_signal_gen(
        cfg, tickers_to_use, price_data, fund_data, custom
    )

    if args.ticker:
        _legacy_show_filter_breakdown(sig, args.ticker, args.date)
    else:
        print(f"\nFilter breakdown for all tickers on {args.date}:")
        for t in tickers_to_use:
            if t in price_features and t in fund_data:
                _legacy_show_filter_breakdown(sig, t, args.date)


if __name__ == "__main__":
    main()
