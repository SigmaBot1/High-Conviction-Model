"""
Financial Modeling Prep (FMP) fundamental data downloader.

Free tier: 250 requests/day, ~5 years of historical quarterly data.
Each ticker requires 6 API calls (income / balance / cashflow × quarterly + annual).
For 503 S&P 500 tickers: ~3018 calls → spread over ~13 days automatically.

Data is returned in the same {ticker: dict} format as the yfinance loader so
features.py and all downstream code work without modification.

Rate limiting
-------------
  - Max 4 requests/second (free tier is lenient; be polite).
  - Daily cap tracked in data_cache/fmp_rate.json.
  - If the daily cap is hit mid-run, the script stops cleanly and prints
    how many remain; re-running the next day resumes from where it left off
    because each completed ticker is pickled before moving to the next.

Field mapping
-------------
FMP camelCase fields → row names expected by FundamentalAccessor (features.py).
Multiple aliases are already handled by get_row() in features.py, so we just
need at least one name from each ROWS list to appear as an index entry.
"""
import json
import logging
import pickle
import time
from datetime import date as dt_date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# /stable/ is the current FMP endpoint (works for all plans including free tier).
# /api/v3/ is the legacy endpoint, restricted to pre-Aug-2025 subscribers only.
_BASE_URL = "https://financialmodelingprep.com/stable"

# ---------------------------------------------------------------------------
# Field mappings: FMP camelCase key → row name used by FundamentalAccessor
# ---------------------------------------------------------------------------
_INCOME_FIELD_MAP = {
    "revenue":          "Total Revenue",
    "grossProfit":      "Gross Profit",
    "operatingIncome":  "Operating Income",
    "netIncome":        "Net Income",
    # Tax rate row added via _tax_rate_fn below
}

_BALANCE_FIELD_MAP = {
    "totalAssets":             "Total Assets",
    "totalCurrentLiabilities": "Current Liabilities",
    "cashAndCashEquivalents":  "Cash And Cash Equivalents",
    "totalDebt":               "Total Debt",
    "totalStockholdersEquity": "Stockholders Equity",
}

_CASHFLOW_FIELD_MAP = {
    "operatingCashFlow":  "Operating Cash Flow",
    "capitalExpenditure": "Capital Expenditure",   # FMP stores as negative (cash outflow)
}


class FMPLoader:
    """Downloads and caches fundamental data from the FMP API."""

    CALLS_PER_TICKER = 6     # income + balance + cashflow  ×  (quarterly + annual)
    MAX_PER_SEC      = 4     # be polite; free tier doesn't enforce a hard per-second cap

    def __init__(self, config: Config):
        if not config.FMP_API_KEY:
            raise ValueError(
                "\n  FMP_API_KEY environment variable is not set.\n"
                "  Get a free key at: https://site.financialmodelingprep.com/developer/docs\n"
                "  Then run:  set FMP_API_KEY=your_key_here   (Windows)\n"
                "         or: export FMP_API_KEY=your_key_here (Mac/Linux)"
            )
        self.api_key    = config.FMP_API_KEY
        self.daily_limit = config.FMP_DAILY_LIMIT
        self.fund_dir   = Path(config.CACHE_DIR) / "fundamentals"
        self.fund_dir.mkdir(parents=True, exist_ok=True)
        self._rate_file  = Path(config.CACHE_DIR) / "fmp_rate.json"
        self._last_call  = 0.0
        self._min_gap    = 1.0 / self.MAX_PER_SEC
        self._limit_hit  = False

    # ------------------------------------------------------------------ #
    # Daily quota tracking
    # ------------------------------------------------------------------ #

    def _load_rate_state(self) -> dict:
        today = str(dt_date.today())
        if self._rate_file.exists():
            try:
                state = json.loads(self._rate_file.read_text())
                if state.get("date") == today:
                    return state
            except Exception:
                pass
        return {"date": today, "count": 0}

    def _save_rate_state(self, state: dict):
        try:
            self._rate_file.write_text(json.dumps(state))
        except Exception:
            pass

    def _consume_one(self) -> bool:
        """Increment daily counter. Returns False if limit is reached."""
        state = self._load_rate_state()
        if state["count"] >= self.daily_limit:
            return False
        state["count"] += 1
        self._save_rate_state(state)
        return True

    def requests_remaining_today(self) -> int:
        state = self._load_rate_state()
        return max(0, self.daily_limit - state["count"])

    # ------------------------------------------------------------------ #
    # HTTP helper
    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, params: dict = None) -> Optional[list]:
        """
        Make one FMP stable-API call.
        Symbol is passed as a query param (?symbol=AAPL), not in the URL path.
        Returns the parsed JSON list, or None on failure / limit hit.
        Sets self._limit_hit = True when the daily cap is reached.
        """
        if not self._consume_one():
            self._limit_hit = True
            return None

        # Enforce per-second rate
        now = time.monotonic()
        gap = now - self._last_call
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last_call = time.monotonic()

        url    = f"{_BASE_URL}/{endpoint}"
        params = {"apikey": self.api_key, **(params or {})}
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                self._limit_hit = True
                return None
            resp.raise_for_status()
            if not resp.text.strip():
                return []     # empty response = no data for this ticker
            data = resp.json()
            if isinstance(data, dict):
                msg = data.get("Error Message") or data.get("message") or str(data)
                logger.warning(f"FMP API error for {endpoint}: {msg}")
                return None
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.warning(f"FMP request failed ({endpoint}): {e}")
            return None

    # ------------------------------------------------------------------ #
    # Statement → DataFrame conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tax_rate_fn(rec: dict) -> float:
        """Compute effective tax rate from FMP record fields."""
        tax    = rec.get("incomeTaxExpense") or 0
        before = rec.get("incomeBeforeTax")
        if before and abs(float(before)) > 1e-6:
            rate = float(tax) / float(before)
            return max(0.0, min(rate, 0.6))
        return 0.21

    @staticmethod
    def _records_to_df(
        records: list,
        field_map: dict,
        extra_rows: dict = None,
    ) -> Optional[pd.DataFrame]:
        """
        Convert a list of FMP statement records to a DataFrame.
        Index = row names, Columns = period-end date Timestamps (most recent first).
        extra_rows = {row_name: callable(record) -> float}
        """
        if not records:
            return None

        col_data: Dict[pd.Timestamp, Dict[str, float]] = {}
        for rec in records:
            date_str = rec.get("date")
            if not date_str:
                continue
            col = pd.Timestamp(str(date_str)[:10])

            row_data: Dict[str, float] = {}
            for fmp_field, row_name in field_map.items():
                val = rec.get(fmp_field)
                row_data[row_name] = float(val) if val is not None else float("nan")

            if extra_rows:
                for row_name, fn in extra_rows.items():
                    try:
                        row_data[row_name] = fn(rec)
                    except Exception:
                        row_data[row_name] = float("nan")

            col_data[col] = row_data

        if not col_data:
            return None

        df = pd.DataFrame(col_data)
        # Most-recent-first column order (consistent with yfinance format)
        df = df[sorted(df.columns, reverse=True)]
        return df if not df.empty else None

    # ------------------------------------------------------------------ #
    # Per-ticker download
    # ------------------------------------------------------------------ #

    def _fund_path(self, ticker: str) -> Path:
        return self.fund_dir / f"{ticker.replace('/', '_')}.pkl"

    def _fmp_symbol(self, ticker: str) -> str:
        """Convert yfinance-style ticker to FMP style (BRK-B → BRK.B)."""
        return ticker.replace("-", ".")

    def _fetch_ticker(self, ticker: str) -> Optional[dict]:
        """
        Download all 6 statements for one ticker from FMP.
        Returns None if the daily limit is hit (self._limit_hit set to True)
        or if all statements are empty (no data for this ticker).
        """
        sym = self._fmp_symbol(ticker)
        inc_extra = {"Tax Rate For Calcs": self._tax_rate_fn}

        def get_stmt(endpoint, period, limit=40):
            # stable API: symbol is a query param; period uses "quarterly"/"annual"
            fmp_period = "quarterly" if period == "quarter" else period
            return self._get(endpoint, {"symbol": sym, "period": fmp_period, "limit": limit})

        # Download all 6 (2 periods × 3 statements)
        q_inc_raw = get_stmt("income-statement",       "quarter", 40)
        if self._limit_hit:
            return None
        a_inc_raw = get_stmt("income-statement",       "annual",  20)
        if self._limit_hit:
            return None
        q_bal_raw = get_stmt("balance-sheet-statement","quarter", 40)
        if self._limit_hit:
            return None
        a_bal_raw = get_stmt("balance-sheet-statement","annual",  20)
        if self._limit_hit:
            return None
        q_cf_raw  = get_stmt("cash-flow-statement",    "quarter", 40)
        if self._limit_hit:
            return None
        a_cf_raw  = get_stmt("cash-flow-statement",    "annual",  20)
        if self._limit_hit:
            return None

        # Build DataFrames
        q_inc = self._records_to_df(q_inc_raw or [], _INCOME_FIELD_MAP,  inc_extra)
        a_inc = self._records_to_df(a_inc_raw or [], _INCOME_FIELD_MAP,  inc_extra)
        q_bal = self._records_to_df(q_bal_raw or [], _BALANCE_FIELD_MAP)
        a_bal = self._records_to_df(a_bal_raw or [], _BALANCE_FIELD_MAP)
        q_cf  = self._records_to_df(q_cf_raw  or [], _CASHFLOW_FIELD_MAP)
        a_cf  = self._records_to_df(a_cf_raw  or [], _CASHFLOW_FIELD_MAP)

        # Shares outstanding from most recent quarterly income record
        shares = None
        for rec in (q_inc_raw or []):
            v = rec.get("weightedAverageShsOut") or rec.get("weightedAverageShsOutDil")
            if v:
                shares = float(v)
                break

        data = {
            "quarterly_income":   q_inc,
            "quarterly_balance":  q_bal,
            "quarterly_cashflow": q_cf,
            "annual_income":      a_inc,
            "annual_balance":     a_bal,
            "annual_cashflow":    a_cf,
            "info": {"sharesOutstanding": shares},
        }

        # Only cache if we got at least some statement data
        has_data = any(
            v is not None and isinstance(v, pd.DataFrame)
            for v in data.values()
        )
        return data if has_data else None

    # ------------------------------------------------------------------ #
    # Public interface (mirrors DataLoader.download_fundamentals)
    # ------------------------------------------------------------------ #

    def download_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, dict]:
        """
        Download fundamentals for all tickers using FMP.

        Already-cached tickers are served from disk (skip API calls).
        If the daily request limit would be exceeded, downloads as many as
        possible and stops — re-running the next day resumes automatically.

        Returns {ticker: data_dict} for all tickers with available data.
        """
        result: Dict[str, dict] = {}
        need:   List[str]       = []

        for t in tickers:
            p = self._fund_path(t)
            if not force_refresh and p.exists():
                try:
                    with open(p, "rb") as f:
                        result[t] = pickle.load(f)
                    continue
                except Exception:
                    pass
            need.append(t)

        n_cached = len(result)

        if not need:
            logger.info(f"FMP: all {n_cached} tickers loaded from cache.")
            return result

        remaining_req = self.requests_remaining_today()
        can_download  = remaining_req // self.CALLS_PER_TICKER
        to_download   = need[:can_download]
        deferred      = need[can_download:]

        print(f"\n  FMP fundamentals:  {n_cached} cached  |  {len(need)} to download")

        if deferred:
            days_left = -(-len(need) // max(can_download, 1))   # ceiling div
            print(
                f"  Daily limit:       {self.daily_limit} req/day  "
                f"({remaining_req} remaining today, {self.CALLS_PER_TICKER} per ticker)\n"
                f"  Downloading:       {len(to_download)} tickers now\n"
                f"  Deferred:          {len(deferred)} tickers  "
                f"(re-run tomorrow; ~{days_left} day(s) total at this rate)\n"
                f"  Tip: upgrade FMP plan for higher daily limits."
            )
        elif to_download:
            print(f"  Downloading:       {len(to_download)} tickers now")

        if not to_download:
            if deferred:
                print(
                    f"\n  FMP daily limit reached ({self.daily_limit} req/day). "
                    f"Re-run tomorrow to continue.\n"
                    f"  Returning {n_cached} cached tickers."
                )
            return result

        print()
        failed = 0
        for ticker in tqdm(to_download, desc="FMP fundamentals"):
            if self._limit_hit:
                logger.warning(
                    f"FMP daily limit hit mid-download. "
                    f"Cached {len(result) - n_cached} new tickers this run."
                )
                break

            data = self._fetch_ticker(ticker)

            if self._limit_hit:
                break

            if data is None:
                failed += 1
                logger.debug(f"FMP: no data for {ticker}")
                continue

            p = self._fund_path(ticker)
            with open(p, "wb") as f:
                pickle.dump(data, f)
            result[ticker] = data

        newly_downloaded = len(result) - n_cached
        if failed:
            logger.info(f"FMP: {newly_downloaded} downloaded, {failed} had no data.")

        return result
