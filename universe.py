"""
Stock universe management.

Supports three scopes:
  "sp500"       — S&P 500 large-cap  (~503 tickers)
  "sp400"       — S&P 400 mid-cap    (~400 tickers)
  "sp500+sp400" — Combined           (~900 tickers, deduplicated)

⚠️  SURVIVORSHIP BIAS NOTE
    This module fetches *current* constituent lists from Wikipedia.
    Stocks removed from the index before 2024 are NOT included.
    Returns are likely overstated compared to a live implementation.
    A production backtest would source point-in-time historical lists
    from a data vendor (e.g. Compustat, Sharadar).
"""
import io
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_SP400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

_CACHE_DIR   = Path("data_cache")
_CACHE_SP500 = _CACHE_DIR / "sp500_tickers.json"
_CACHE_SP400 = _CACHE_DIR / "sp400_tickers.json"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ------------------------------------------------------------------ #
# Fallback lists (used when Wikipedia is unreachable)
# ------------------------------------------------------------------ #

_FALLBACK_SP500: List[Tuple[str, str]] = [
    # Information Technology
    ("AAPL","Information Technology"),("MSFT","Information Technology"),
    ("NVDA","Information Technology"),("AVGO","Information Technology"),
    ("ORCL","Information Technology"),("CSCO","Information Technology"),
    ("ACN","Information Technology"),("ADBE","Information Technology"),
    ("CRM","Information Technology"),("AMD","Information Technology"),
    ("INTC","Information Technology"),("TXN","Information Technology"),
    ("QCOM","Information Technology"),("IBM","Information Technology"),
    ("NOW","Information Technology"),("INTU","Information Technology"),
    ("AMAT","Information Technology"),("LRCX","Information Technology"),
    ("ADI","Information Technology"),("KLAC","Information Technology"),
    ("MU","Information Technology"),("MRVL","Information Technology"),
    ("PANW","Information Technology"),("SNPS","Information Technology"),
    ("CDNS","Information Technology"),("FTNT","Information Technology"),
    ("CTSH","Information Technology"),("HPQ","Information Technology"),
    ("GLW","Information Technology"),("KEYS","Information Technology"),
    # Communication Services
    ("GOOGL","Communication Services"),("META","Communication Services"),
    ("NFLX","Communication Services"),("DIS","Communication Services"),
    ("CMCSA","Communication Services"),("T","Communication Services"),
    ("VZ","Communication Services"),("CHTR","Communication Services"),
    ("TMUS","Communication Services"),("TTWO","Communication Services"),
    ("EA","Communication Services"),("OMC","Communication Services"),
    # Consumer Discretionary
    ("AMZN","Consumer Discretionary"),("TSLA","Consumer Discretionary"),
    ("HD","Consumer Discretionary"),("MCD","Consumer Discretionary"),
    ("NKE","Consumer Discretionary"),("SBUX","Consumer Discretionary"),
    ("LOW","Consumer Discretionary"),("TJX","Consumer Discretionary"),
    ("BKNG","Consumer Discretionary"),("CMG","Consumer Discretionary"),
    ("MAR","Consumer Discretionary"),("HLT","Consumer Discretionary"),
    ("ORLY","Consumer Discretionary"),("AZO","Consumer Discretionary"),
    ("DHI","Consumer Discretionary"),("LEN","Consumer Discretionary"),
    ("F","Consumer Discretionary"),("GM","Consumer Discretionary"),
    # Consumer Staples
    ("PG","Consumer Staples"),("KO","Consumer Staples"),
    ("PEP","Consumer Staples"),("COST","Consumer Staples"),
    ("WMT","Consumer Staples"),("PM","Consumer Staples"),
    ("MO","Consumer Staples"),("MDLZ","Consumer Staples"),
    ("CL","Consumer Staples"),("KMB","Consumer Staples"),
    ("GIS","Consumer Staples"),("SYY","Consumer Staples"),
    # Health Care
    ("UNH","Health Care"),("JNJ","Health Care"),
    ("LLY","Health Care"),("ABT","Health Care"),
    ("MRK","Health Care"),("TMO","Health Care"),
    ("DHR","Health Care"),("ABBV","Health Care"),
    ("PFE","Health Care"),("AMGN","Health Care"),
    ("GILD","Health Care"),("ISRG","Health Care"),
    ("VRTX","Health Care"),("REGN","Health Care"),
    ("BSX","Health Care"),("MDT","Health Care"),
    ("ELV","Health Care"),("HUM","Health Care"),
    ("CVS","Health Care"),("CI","Health Care"),
    ("DXCM","Health Care"),("IDXX","Health Care"),
    ("IQV","Health Care"),("A","Health Care"),
    # Financials
    ("JPM","Financials"),("BAC","Financials"),
    ("WFC","Financials"),("GS","Financials"),
    ("MS","Financials"),("BLK","Financials"),
    ("C","Financials"),("AXP","Financials"),
    ("SPGI","Financials"),("MCO","Financials"),
    ("CB","Financials"),("PGR","Financials"),
    ("V","Financials"),("MA","Financials"),
    ("COF","Financials"),("USB","Financials"),
    ("TFC","Financials"),("SCHW","Financials"),
    ("ICE","Financials"),("CME","Financials"),
    ("AON","Financials"),("MMC","Financials"),
    # Industrials
    ("HON","Industrials"),("UPS","Industrials"),
    ("CAT","Industrials"),("DE","Industrials"),
    ("RTX","Industrials"),("LMT","Industrials"),
    ("GE","Industrials"),("BA","Industrials"),
    ("MMM","Industrials"),("FDX","Industrials"),
    ("NSC","Industrials"),("UNP","Industrials"),
    ("CSX","Industrials"),("EMR","Industrials"),
    ("ITW","Industrials"),("ETN","Industrials"),
    ("GWW","Industrials"),("CTAS","Industrials"),
    ("PCAR","Industrials"),("FAST","Industrials"),
    ("VRSK","Industrials"),("CPRT","Industrials"),
    # Energy
    ("XOM","Energy"),("CVX","Energy"),
    ("COP","Energy"),("EOG","Energy"),
    ("SLB","Energy"),("PSX","Energy"),
    ("VLO","Energy"),("MPC","Energy"),
    ("OXY","Energy"),("PXD","Energy"),
    ("DVN","Energy"),("HES","Energy"),
    # Materials
    ("LIN","Materials"),("APD","Materials"),
    ("ECL","Materials"),("SHW","Materials"),
    ("FCX","Materials"),("NEM","Materials"),
    ("NUE","Materials"),("ALB","Materials"),
    # Utilities
    ("NEE","Utilities"),("DUK","Utilities"),
    ("SO","Utilities"),("D","Utilities"),
    ("AEP","Utilities"),("EXC","Utilities"),
    ("SRE","Utilities"),("XEL","Utilities"),
    # Real Estate
    ("PLD","Real Estate"),("AMT","Real Estate"),
    ("CCI","Real Estate"),("EQIX","Real Estate"),
    ("PSA","Real Estate"),("O","Real Estate"),
    ("WELL","Real Estate"),("SPG","Real Estate"),
    # High-growth software / SaaS
    ("SNOW","Information Technology"),("DDOG","Information Technology"),
    ("ZS","Information Technology"),("CRWD","Information Technology"),
    ("NET","Information Technology"),("MDB","Information Technology"),
    ("BILL","Information Technology"),("HUBS","Information Technology"),
    ("ZM","Communication Services"),("UBER","Industrials"),
    ("ABNB","Consumer Discretionary"),("LYFT","Industrials"),
    ("SHOP","Consumer Discretionary"),("SQ","Financials"),
    ("PYPL","Financials"),("COIN","Financials"),
]

# Representative S&P 400 mid-cap fallback (~80 tickers across sectors)
_FALLBACK_SP400: List[Tuple[str, str]] = [
    ("ESNT","Financials"),("RLI","Financials"),("HALO","Health Care"),
    ("PCVX","Health Care"),("CSWI","Industrials"),("ACLS","Information Technology"),
    ("ACLX","Health Care"),("AAON","Industrials"),("ABCB","Financials"),
    ("ADMA","Health Care"),("AEIS","Information Technology"),("AFG","Financials"),
    ("AGIO","Health Care"),("ALKS","Health Care"),("AMR","Energy"),
    ("AMSF","Financials"),("ANF","Consumer Discretionary"),("APAM","Financials"),
    ("ARWR","Health Care"),("ASGN","Industrials"),("ASTE","Industrials"),
    ("ATRC","Health Care"),("AVNT","Materials"),("AWI","Industrials"),
    ("BCC","Materials"),("BCPC","Materials"),("BDC","Financials"),
    ("BKU","Financials"),("BMI","Industrials"),("BOX","Information Technology"),
    ("BPOP","Financials"),("CADE","Financials"),("CALM","Consumer Staples"),
    ("CARA","Health Care"),("CASH","Financials"),("CBU","Financials"),
    ("CENTA","Consumer Staples"),("CHE","Health Care"),("CHRD","Energy"),
    ("CIVI","Energy"),("CLH","Industrials"),("CLW","Materials"),
    ("CNMD","Health Care"),("CNS","Financials"),("COOP","Financials"),
    ("CRC","Energy"),("CRVL","Health Care"),("CSL","Industrials"),
    ("CUBI","Financials"),("CVCO","Consumer Discretionary"),("CWT","Utilities"),
    ("DORM","Consumer Discretionary"),("EAT","Consumer Discretionary"),
    ("ENOV","Industrials"),("EPAC","Industrials"),("EPRT","Real Estate"),
    ("ESAB","Industrials"),("ESTA","Financials"),("EVTC","Information Technology"),
    ("EXP","Materials"),("EXPO","Information Technology"),("FBRT","Financials"),
    ("FHB","Financials"),("FIZZ","Consumer Staples"),("FLO","Consumer Staples"),
    ("FNB","Financials"),("FORM","Information Technology"),("FRPT","Consumer Staples"),
    ("FSS","Industrials"),("GFF","Industrials"),("GKOS","Health Care"),
    ("GLNG","Energy"),("HAYW","Consumer Discretionary"),("HCC","Materials"),
    ("HCCI","Energy"),("HCI","Financials"),("HIMS","Health Care"),
    ("HIW","Real Estate"),("HRI","Industrials"),("HTLF","Financials"),
]


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _extract_tickers_sectors(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Pull (ticker, GICS sector) from a Wikipedia index DataFrame.
    Handles both 'Symbol' (S&P 500) and 'Ticker symbol' (S&P 400) column names.
    """
    # Ticker column
    ticker_col = next(
        (c for c in df.columns
         if c.strip().lower() in ("symbol", "ticker", "ticker symbol")),
        None,
    )
    # GICS Sector column
    sector_col = next(
        (c for c in df.columns if "gics" in c.lower() and "sector" in c.lower()),
        None,
    )
    if ticker_col is None:
        raise ValueError(f"No ticker column found. Available: {list(df.columns)}")
    if sector_col is None:
        # Try any "sector" column as fallback
        sector_col = next(
            (c for c in df.columns if "sector" in c.lower()), None
        )
    if sector_col is None:
        raise ValueError(f"No sector column found. Available: {list(df.columns)}")

    tickers = df[ticker_col].astype(str).str.replace(".", "-", regex=False).tolist()
    sectors = df[sector_col].astype(str).tolist()
    return list(zip(tickers, sectors))


def _fetch_index(url: str, cache_file: Path, label: str) -> List[Tuple[str, str]]:
    """Fetch an S&P index constituent list from Wikipedia and cache it."""
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            if data:
                logger.info(f"Loaded {len(data)} {label} tickers from cache.")
                return [tuple(x) for x in data]
        except Exception:
            pass

    logger.info(f"Fetching {label} constituent list from Wikipedia...")

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text), header=0)
            # First table is always the constituent table on these pages
            result = _extract_tickers_sectors(tables[0])
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(result, f)
            logger.info(f"Fetched {len(result)} {label} tickers.")
            return result
        except Exception as e:
            logger.warning(f"{label} fetch attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    return []


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def get_sp500_universe(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """Return (ticker, gics_sector) list for current S&P 500 constituents."""
    if force_refresh and _CACHE_SP500.exists():
        _CACHE_SP500.unlink()
    result = _fetch_index(_WIKI_SP500, _CACHE_SP500, "S&P 500")
    if not result:
        logger.warning("S&P 500 Wikipedia fetch failed – using embedded fallback list.")
        return _FALLBACK_SP500
    return result


def get_sp400_universe(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """Return (ticker, gics_sector) list for current S&P 400 mid-cap constituents."""
    if force_refresh and _CACHE_SP400.exists():
        _CACHE_SP400.unlink()
    result = _fetch_index(_WIKI_SP400, _CACHE_SP400, "S&P 400")
    if not result:
        logger.warning("S&P 400 Wikipedia fetch failed – using embedded fallback list.")
        return _FALLBACK_SP400
    return result


def get_universe(scope: str = "sp500+sp400",
                 force_refresh: bool = False) -> List[Tuple[str, str]]:
    """
    Return combined (ticker, gics_sector) list for the requested scope.

    scope:
      "sp500"       — S&P 500 only
      "sp400"       — S&P 400 only
      "sp500+sp400" — both combined, deduplicated (sp500 entry wins on overlap)
    """
    if scope == "sp500":
        return get_sp500_universe(force_refresh)

    if scope == "sp400":
        return get_sp400_universe(force_refresh)

    # "sp500+sp400": merge, sp500 wins on ticker overlap
    sp500 = get_sp500_universe(force_refresh)
    sp400 = get_sp400_universe(force_refresh)
    sp500_tickers = {t for t, _ in sp500}
    combined = list(sp500)
    for t, s in sp400:
        if t not in sp500_tickers:
            combined.append((t, s))
    logger.info(
        f"Combined universe: {len(sp500)} SP500 + "
        f"{len(sp400)} SP400 = {len(combined)} unique tickers"
    )
    return combined


def get_sector_map(universe: List[Tuple[str, str]]) -> dict:
    """Return {ticker: sector} dict."""
    return {t: s for t, s in universe}
