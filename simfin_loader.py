"""
SimFin fundamental data downloader.

Free tier: 2,000 requests/day, full historical data (5+ years of quarters).
Each ticker uses 3 API calls (income + balance + cashflow — one call each,
then quarterly/annual rows are split in code).
For 503 S&P 500 tickers: ~1,509 calls → ~1 day.  Resumes automatically
on re-run because each completed ticker is pickled before moving to the next.

API reference: https://backend.simfin.com/api/v3
Authentication: Authorization: api-key <key>  (header, not query param)

Data is returned in the same {ticker: dict} format as FMPLoader / yfinance
so features.py and all downstream code work without modification.

Response format
---------------
SimFin compact endpoint returns:
  [{
    "ticker": "AAPL",
    "statements": [{
      "statement": "PL",
      "columns": [...field names...],
      "data": [[row values], ...]
    }]
  }]

Each row covers one reporting period. "Fiscal Period" distinguishes
Q1/Q2/Q3/Q4 (quarterly) from FY (annual). "Report Date" is the
period-end date used as the column timestamp in the resulting DataFrames.

Field mapping
-------------
SimFin field names → row names expected by FundamentalAccessor (features.py).
Multiple aliases are already handled by get_row() so we just need one matching
name per ROWS list to appear as an index entry.
"""
import json
import logging
import os
import pickle
import time
from datetime import date as dt_date
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

_BASE_URL    = "https://backend.simfin.com/api/v3"
_MAX_PER_SEC = 4

# Rows that count as a pure quarter (used to filter the combined API response)
_QUARTERLY_PERIODS: Set[str] = {"Q1", "Q2", "Q3", "Q4"}
_ANNUAL_PERIODS:    Set[str] = {"FY"}

# ---------------------------------------------------------------------------
# Field mappings: SimFin column name → row name used by FundamentalAccessor
# ---------------------------------------------------------------------------

_INCOME_FIELD_MAP: Dict[str, str] = {
    "Revenue":                  "Total Revenue",
    "Gross Profit":             "Gross Profit",
    "Operating Income (Loss)":  "Operating Income",
    "Net Income":               "Net Income",
    # Tax Rate For Calcs is computed via _tax_rate_fn below
}

_BALANCE_FIELD_MAP: Dict[str, str] = {
    "Total Assets":                                    "Total Assets",
    "Total Current Liabilities":                       "Current Liabilities",
    "Cash, Cash Equivalents & Short Term Investments": "Cash And Cash Equivalents",
    # Total Debt = Long-Term + Short-Term — handled via _total_debt_fn
    "Total Equity":                                    "Stockholders Equity",
}

_CASHFLOW_FIELD_MAP: Dict[str, str] = {
    "Cash from Operating Activities":            "Operating Cash Flow",
    "Acquisition of Fixed Assets & Intangibles": "Capital Expenditure",
    # SimFin stores CapEx as a negative value (cash outflow) — matches FMP convention
}


class SimFinLoader:
    """Downloads and caches fundamental data from the SimFin v3 API."""

    CALLS_PER_TICKER = 3   # income + balance + cashflow (one call each; both periods in one response)
    MAX_PER_SEC      = _MAX_PER_SEC

    def __init__(self, config: Config, api_key: str = ""):
        self.api_key = api_key or os.environ.get("SIMFIN_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "\n  SIMFIN_API_KEY is not set.\n"
                "  Get a free key at: https://app.simfin.com/settings\n"
                "  Then run:  set SIMFIN_API_KEY=your_key   (Windows)\n"
                "         or: export SIMFIN_API_KEY=your_key (Mac/Linux)"
            )
        self.daily_limit  = int(os.environ.get("SIMFIN_DAILY_LIMIT", "1990"))
        self.fund_dir     = Path(config.CACHE_DIR) / "fundamentals"
        self.fund_dir.mkdir(parents=True, exist_ok=True)
        self._rate_file   = Path(config.CACHE_DIR) / "simfin_rate.json"
        self._last_call   = 0.0
        self._min_gap     = 1.0 / self.MAX_PER_SEC
        self._limit_hit   = False

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

    def _get(self, endpoint: str, params: dict) -> Optional[object]:
        """
        Make one SimFin API call.
        Returns the parsed JSON (list or dict), or None on failure / limit hit.
        """
        if not self._consume_one():
            self._limit_hit = True
            return None

        now = time.monotonic()
        gap = now - self._last_call
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last_call = time.monotonic()

        url = f"{_BASE_URL}/{endpoint}"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"api-key {self.api_key}"},
                params=params,
                timeout=30,
            )
            if resp.status_code == 429:
                self._limit_hit = True
                return None
            if resp.status_code == 401:
                logger.error("SimFin: invalid API key (401). Check SIMFIN_API_KEY.")
                self._limit_hit = True
                return None
            resp.raise_for_status()
            if not resp.text.strip():
                return []
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                logger.debug(f"SimFin {endpoint}: {data['error']}")
                return None
            return data
        except Exception as e:
            logger.warning(f"SimFin request failed ({endpoint}): {e}")
            return None

    # ------------------------------------------------------------------ #
    # Response parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap_response(response: object, period_filter: Set[str]) -> Optional[dict]:
        """
        Unwrap a SimFin compact API response to a plain {columns, data} dict.

        The API returns:
          [{..., "statements": [{"columns": [...], "data": [...], ...}]}]

        Rows are filtered to only those whose "Fiscal Period" is in period_filter
        (e.g. {"Q1","Q2","Q3","Q4"} for quarterly, {"FY"} for annual).
        Returns None if the response is empty or malformed.
        """
        if not isinstance(response, list) or not response:
            return None
        stmts = response[0].get("statements", [])
        if not stmts:
            return None
        stmt = stmts[0]
        columns = stmt.get("columns", [])
        data    = stmt.get("data",    [])
        if not columns or not data:
            return None

        fp_idx = next((i for i, c in enumerate(columns) if c == "Fiscal Period"), None)
        if fp_idx is not None:
            data = [row for row in data if fp_idx < len(row) and row[fp_idx] in period_filter]

        return {"columns": columns, "data": data} if data else None

    @staticmethod
    def _tax_rate_fn(row: list, col_idx: Dict[str, int]) -> float:
        """Compute effective tax rate from SimFin row data."""
        # SimFin stores tax as "Income Tax (Expense) Benefit, net" — negative = expense
        tax_i = col_idx.get("Income Tax (Expense) Benefit, net")
        pre_i = col_idx.get("Pretax Income (Loss), Adjusted")
        if tax_i is None or pre_i is None:
            return 0.21
        tax = row[tax_i] if tax_i < len(row) else None
        pre = row[pre_i] if pre_i < len(row) else None
        if tax is None or pre is None:
            return 0.21
        try:
            tax, pre = float(tax), float(pre)
            if abs(pre) > 1e-6:
                rate = -tax / pre   # tax is stored as negative expense
                return max(0.0, min(rate, 0.6))
        except (TypeError, ValueError):
            pass
        return 0.21

    @staticmethod
    def _total_debt_fn(row: list, col_idx: Dict[str, int]) -> float:
        """Total Debt = Long-Term Debt + Short-Term Debt."""
        lt_i  = col_idx.get("Long Term Debt")
        st_i  = col_idx.get("Short Term Debt")
        total = 0.0
        for i in (lt_i, st_i):
            if i is not None and i < len(row) and row[i] is not None:
                try:
                    total += float(row[i])
                except (TypeError, ValueError):
                    pass
        return total

    @staticmethod
    def _compact_to_df(
        inner: dict,
        field_map: Dict[str, str],
        extra_fns: Dict[str, callable] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Convert a plain {columns, data} dict to a DataFrame.
        Index   = row names (FundamentalAccessor format)
        Columns = period-end date Timestamps (most recent first)
        """
        if not isinstance(inner, dict):
            return None
        columns = inner.get("columns", [])
        data    = inner.get("data",    [])
        if not columns or not data:
            return None

        col_idx = {c: i for i, c in enumerate(columns)}
        date_i  = col_idx.get("Report Date")
        if date_i is None:
            return None

        period_data: Dict[pd.Timestamp, Dict[str, float]] = {}
        for row in data:
            if date_i >= len(row) or not row[date_i]:
                continue
            dt = pd.Timestamp(str(row[date_i])[:10])

            row_vals: Dict[str, float] = {}
            for src, dst in field_map.items():
                i = col_idx.get(src)
                v = row[i] if (i is not None and i < len(row)) else None
                row_vals[dst] = float(v) if v is not None else float("nan")

            if extra_fns:
                for dst, fn in extra_fns.items():
                    try:
                        row_vals[dst] = fn(row, col_idx)
                    except Exception:
                        row_vals[dst] = float("nan")

            period_data[dt] = row_vals

        if not period_data:
            return None

        df = pd.DataFrame(period_data)
        df = df[sorted(df.columns, reverse=True)]   # most recent first
        return df if not df.empty else None

    # ------------------------------------------------------------------ #
    # Per-ticker download
    # ------------------------------------------------------------------ #

    def _fund_path(self, ticker: str) -> Path:
        return self.fund_dir / f"{ticker.replace('/', '_')}.pkl"

    def _simfin_ticker(self, ticker: str) -> str:
        """Convert yfinance-style ticker to SimFin style (BRK-B → BRK.B)."""
        return ticker.replace("-", ".")

    def _fetch_ticker(self, ticker: str) -> Optional[dict]:
        """
        Fetch all 3 statements for one ticker (income, balance, cashflow).
        Each statement call returns all periods; quarterly/annual rows are
        split in _unwrap_response.  Returns None if the daily limit is hit
        or all statements are empty.
        """
        sym       = self._simfin_ticker(ticker)
        inc_extra = {"Tax Rate For Calcs": self._tax_rate_fn}
        bal_extra = {"Total Debt":         self._total_debt_fn}

        def fetch(statement: str) -> Optional[object]:
            return self._get(
                "companies/statements/compact",
                {"ticker": sym, "statements": statement},
            )

        inc_raw = fetch("pl")
        if self._limit_hit:
            return None
        bal_raw = fetch("bs")
        if self._limit_hit:
            return None
        cf_raw  = fetch("cf")
        if self._limit_hit:
            return None

        # Split each raw response into quarterly and annual sub-dicts
        q_inc = self._compact_to_df(
            self._unwrap_response(inc_raw, _QUARTERLY_PERIODS) or {},
            _INCOME_FIELD_MAP, inc_extra,
        )
        a_inc = self._compact_to_df(
            self._unwrap_response(inc_raw, _ANNUAL_PERIODS) or {},
            _INCOME_FIELD_MAP, inc_extra,
        )
        q_bal = self._compact_to_df(
            self._unwrap_response(bal_raw, _QUARTERLY_PERIODS) or {},
            _BALANCE_FIELD_MAP, bal_extra,
        )
        a_bal = self._compact_to_df(
            self._unwrap_response(bal_raw, _ANNUAL_PERIODS) or {},
            _BALANCE_FIELD_MAP, bal_extra,
        )
        q_cf = self._compact_to_df(
            self._unwrap_response(cf_raw, _QUARTERLY_PERIODS) or {},
            _CASHFLOW_FIELD_MAP,
        )
        a_cf = self._compact_to_df(
            self._unwrap_response(cf_raw, _ANNUAL_PERIODS) or {},
            _CASHFLOW_FIELD_MAP,
        )

        data = {
            "quarterly_income":   q_inc,
            "quarterly_balance":  q_bal,
            "quarterly_cashflow": q_cf,
            "annual_income":      a_inc,
            "annual_balance":     a_bal,
            "annual_cashflow":    a_cf,
            "info": {"sharesOutstanding": self._fetch_shares_yf(ticker)},
        }

        has_data = any(
            v is not None and isinstance(v, pd.DataFrame)
            for v in data.values()
        )
        return data if has_data else None

    @staticmethod
    def _fetch_shares_yf(ticker: str) -> Optional[float]:
        """
        Fetch shares outstanding from yfinance fast_info (no API key needed).
        Used because SimFin's compact statements API does not include share counts.
        Returns None silently if unavailable.
        """
        try:
            import yfinance as yf
            fi = yf.Ticker(ticker).fast_info
            shares = getattr(fi, "shares", None)
            if shares and float(shares) > 0:
                return float(shares)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Public interface (mirrors DataLoader.download_fundamentals)
    # ------------------------------------------------------------------ #

    def download_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, dict]:
        """
        Download fundamentals for all tickers using SimFin.

        Already-cached tickers are served from disk.  If the daily request
        limit would be exceeded, downloads as many as possible and stops —
        re-running the next day resumes automatically.

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
            logger.info(f"SimFin: all {n_cached} tickers loaded from cache.")
            return result

        remaining_req = self.requests_remaining_today()
        can_download  = remaining_req // self.CALLS_PER_TICKER
        to_download   = need[:can_download]
        deferred      = need[can_download:]

        print(f"\n  SimFin fundamentals:  {n_cached} cached  |  {len(need)} to download")

        if deferred:
            days_left = -(-len(need) // max(can_download, 1))
            print(
                f"  Daily limit:       {self.daily_limit} req/day  "
                f"({remaining_req} remaining, {self.CALLS_PER_TICKER} per ticker)\n"
                f"  Downloading:       {len(to_download)} tickers now\n"
                f"  Deferred:          {len(deferred)} tickers  "
                f"(re-run tomorrow; ~{days_left} day(s) total)\n"
            )
        elif to_download:
            print(f"  Downloading:       {len(to_download)} tickers now")

        if not to_download:
            if deferred:
                print(
                    f"\n  SimFin daily limit reached. "
                    f"Re-run tomorrow to continue.\n"
                    f"  Returning {n_cached} cached tickers."
                )
            return result

        print()
        failed = 0
        for ticker in tqdm(to_download, desc="SimFin fundamentals"):
            if self._limit_hit:
                logger.warning(
                    f"SimFin daily limit hit mid-download. "
                    f"Cached {len(result) - n_cached} new tickers this run."
                )
                break

            data = self._fetch_ticker(ticker)

            if self._limit_hit:
                break

            if data is None:
                failed += 1
                logger.debug(f"SimFin: no data for {ticker}")
                continue

            p = self._fund_path(ticker)
            with open(p, "wb") as f:
                pickle.dump(data, f)
            result[ticker] = data

        newly_downloaded = len(result) - n_cached
        if failed:
            logger.info(f"SimFin: {newly_downloaded} downloaded, {failed} had no data.")

        return result
