"""
Feature engineering layer.

Takes raw price DataFrames and raw fundamental dicts and produces
clean, pre-computed feature DataFrames used by the signal generator.
All features are computed in a look-ahead-safe manner:
  - Price features: available same day (at close).
  - Fundamental features: available only after FUNDAMENTAL_LAG_DAYS from
    the period-end date (quarterly OR annual).

Data-availability fallback strategy
------------------------------------
yfinance returns the last 4-5 quarters of quarterly data and the last
4-5 years of annual data.  For historical simulation dates before ~2024,
quarterly data may not exist.  The accessor automatically falls back to
annual data with the following adaptations:
  - "2 consecutive quarters of growth" → "1 year-over-year comparison"
  - TTM computed from 4 quarters → uses the single annual period value
  - "consistent gross margin" → uses last 2 annual years

This introduces a small approximation (annual ≠ 4×quarterly for some
companies) but is far better than skipping all pre-2024 signals.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers for point-in-time fundamental access
# ------------------------------------------------------------------ #

def _available_cols(df: pd.DataFrame, as_of: pd.Timestamp, lag_days: int) -> pd.Index:
    """
    Return columns (period-end dates) of a fundamental DataFrame whose
    data would be available to an investor on `as_of`, given a filing lag.
    Columns are assumed to be period-end dates (most recent first).
    """
    dates = pd.to_datetime(df.columns)
    avail = dates + pd.Timedelta(days=lag_days)
    return df.columns[avail <= as_of]


def get_row(df: pd.DataFrame, row_candidates: list) -> Optional[pd.Series]:
    """Return the first matching row from a list of possible index names."""
    if df is None:
        return None
    for name in row_candidates:
        if name in df.index:
            return df.loc[name]
    return None


# ------------------------------------------------------------------ #
# Price feature computation (per-ticker, full history)
# ------------------------------------------------------------------ #

def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Given a OHLCV DataFrame for one ticker, return a DataFrame of
    price-derived features indexed by date.
    """
    close = prices["Close"].dropna()
    if len(close) < 60:
        return pd.DataFrame()

    feat = pd.DataFrame(index=close.index)

    # --- EMAs ---
    feat["ema_50"]  = close.ewm(span=50,  adjust=False).mean()
    feat["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # --- 52-week high ---
    feat["high_52w"] = close.rolling(252, min_periods=120).max()

    # --- Price z-score vs 12-month rolling mean ---
    roll_mean = close.rolling(252, min_periods=120).mean()
    roll_std  = close.rolling(252, min_periods=120).std()
    feat["price_zscore_12m"] = (close - roll_mean) / roll_std.replace(0, np.nan)

    # --- Momentum: 12-month return minus last month (12-1 momentum) ---
    feat["ret_12_1"] = close.shift(21) / close.shift(252) - 1

    # --- 6-month return (used by QVM rotation scoring) ---
    feat["ret_6m"] = close / close.shift(126) - 1

    # --- Current close ---
    feat["close"] = close

    return feat.dropna(how="all")


def compute_spy_features(spy_prices: pd.DataFrame) -> pd.DataFrame:
    """SPY-level fear regime features."""
    close = spy_prices["Close"].dropna()
    feat = pd.DataFrame(index=close.index)
    feat["spy_close"]       = close
    feat["spy_ema200"]      = close.ewm(span=200, adjust=False).mean()
    feat["spy_high_rolling"] = close.rolling(252, min_periods=120).max()
    feat["spy_below_200d_pct"] = (feat["spy_ema200"] - close) / feat["spy_ema200"]
    feat["spy_drawdown"]    = (feat["spy_high_rolling"] - close) / feat["spy_high_rolling"]
    return feat.dropna(how="all")


# ------------------------------------------------------------------ #
# Fundamental feature extraction (per ticker, per simulation date)
# ------------------------------------------------------------------ #

class FundamentalAccessor:
    """
    Wraps a raw fundamental dict and provides point-in-time safe
    metric queries.  Automatically uses quarterly data when available
    for a given simulation date, otherwise falls back to annual.
    """

    REVENUE_ROWS    = ["Total Revenue", "Revenue", "TotalRevenue"]
    GROSS_ROWS      = ["Gross Profit", "GrossProfit"]
    OPINCOME_ROWS   = ["Operating Income", "OperatingIncome", "EBIT"]
    NETINCOME_ROWS  = ["Net Income", "NetIncome",
                       "Net Income Common Stockholders",
                       "Net Income Including Noncontrolling Interests"]
    TAXRATE_ROWS    = ["Tax Rate For Calcs", "TaxRateForCalcs", "Effective Tax Rate"]
    OPCASH_ROWS     = ["Operating Cash Flow", "Total Cash From Operating Activities",
                       "OperatingCashFlow", "Cash Flow From Continuing Operating Activities"]
    CAPEX_ROWS      = ["Capital Expenditure", "CapitalExpenditures", "Capex",
                       "Purchase Of Property Plant And Equipment",
                       "Capital Expenditure Reported"]
    TOTALDEBT_ROWS  = ["Total Debt", "TotalDebt",
                       "Long Term Debt And Capital Lease Obligation", "LongTermDebt"]
    EQUITY_ROWS     = ["Stockholders Equity", "Total Stockholders Equity",
                       "StockholdersEquity", "Total Equity Gross Minority Interest",
                       "Common Stock Equity"]
    ASSETS_ROWS     = ["Total Assets", "TotalAssets"]
    CURR_LIAB_ROWS  = ["Current Liabilities", "Total Current Liabilities",
                       "CurrentLiabilities"]
    CASH_ROWS       = ["Cash And Cash Equivalents", "CashAndCashEquivalents",
                       "Cash Cash Equivalents And Short Term Investments",
                       "Cash And Short Term Investments"]

    def __init__(self, raw: dict, lag_days: int = 45):
        self.raw = raw
        self.lag_days = lag_days

    # ---------------------------------------------------------------- #
    # Low-level data accessors                                          #
    # ---------------------------------------------------------------- #

    def _filter_df(self, key: str, as_of: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Return available columns of a statement, or None."""
        df = self.raw.get(key)
        if df is None or df.empty:
            return None
        cols = _available_cols(df, as_of, self.lag_days)
        return df[cols] if len(cols) else None

    def _q_income(self, as_of):
        return self._filter_df("quarterly_income", as_of)

    def _a_income(self, as_of):
        return self._filter_df("annual_income", as_of)

    def _q_balance(self, as_of):
        return self._filter_df("quarterly_balance", as_of)

    def _a_balance(self, as_of):
        return self._filter_df("annual_balance", as_of)

    def _q_cashflow(self, as_of):
        return self._filter_df("quarterly_cashflow", as_of)

    def _a_cashflow(self, as_of):
        return self._filter_df("annual_cashflow", as_of)

    def _best_stmt(self, kind: str, as_of: pd.Timestamp, min_q_cols: int = 1) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Returns (df, frequency) where frequency is 'quarterly' or 'annual'.
        Prefers quarterly if min_q_cols columns are available; falls back to annual.
        kind is one of 'income', 'balance', 'cashflow'.
        """
        q = self._filter_df(f"quarterly_{kind}", as_of)
        if q is not None and q.shape[1] >= min_q_cols:
            return q, "quarterly"
        a = self._filter_df(f"annual_{kind}", as_of)
        if a is not None and a.shape[1] >= 1:
            return a, "annual"
        return None, "none"

    def _best_income(self, as_of: pd.Timestamp, min_q_cols: int = 4) -> Tuple[Optional[pd.DataFrame], str]:
        return self._best_stmt("income", as_of, min_q_cols)

    def _best_balance(self, as_of: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], str]:
        return self._best_stmt("balance", as_of, 1)

    def _best_cashflow(self, as_of: pd.Timestamp, min_q_cols: int = 4) -> Tuple[Optional[pd.DataFrame], str]:
        return self._best_stmt("cashflow", as_of, min_q_cols)

    def _get_scalar(self, df, row_names: list, col_idx: int = 0) -> Optional[float]:
        s = get_row(df, row_names)
        if s is None or len(s) <= col_idx:
            return None
        v = s.iloc[col_idx]
        return float(v) if pd.notna(v) else None

    # ---------------------------------------------------------------- #
    # Public metrics                                                    #
    # ---------------------------------------------------------------- #

    def revenue_growth_yoy(self, as_of: pd.Timestamp) -> Optional[Tuple[float, float]]:
        """
        Returns (growth_period1, growth_period2) – YoY growth.

        Quarterly mode: compares each of the last 2 quarters to same quarter 1yr ago.
          Requires ≥5 available quarterly periods.
        Annual fallback: compares last 2 annual periods YoY.
          Returns (growth_yr1, growth_yr2) or (growth_yr1, None) if only 2 years available.
        """
        # -- Try quarterly first --
        q = self._q_income(as_of)
        if q is not None and q.shape[1] >= 5:
            rev = get_row(q, self.REVENUE_ROWS)
            if rev is not None and len(rev) >= 5:
                def sg(c, p):
                    return float(c / p - 1) if (pd.notna(c) and pd.notna(p) and p != 0) else None
                g1 = sg(rev.iloc[0], rev.iloc[4])
                g2 = sg(rev.iloc[1], rev.iloc[5]) if len(rev) >= 6 else None
                return (g1, g2)

        # -- Annual fallback --
        a = self._a_income(as_of)
        if a is None or a.shape[1] < 2:
            return None
        rev = get_row(a, self.REVENUE_ROWS)
        if rev is None or len(rev) < 2:
            return None

        def sg(c, p):
            return float(c / p - 1) if (pd.notna(c) and pd.notna(p) and p != 0) else None

        g1 = sg(rev.iloc[0], rev.iloc[1])
        g2 = sg(rev.iloc[1], rev.iloc[2]) if len(rev) >= 3 else None
        return (g1, g2)

    def gross_margin(self, as_of: pd.Timestamp) -> Optional[Tuple[float, float]]:
        """Returns (latest_gm, prior_period_gm) as fractions."""
        df, freq = self._best_income(as_of, min_q_cols=2)
        if df is None:
            return None

        rev = get_row(df, self.REVENUE_ROWS)
        gp  = get_row(df, self.GROSS_ROWS)
        if rev is None or gp is None:
            return None

        def sm(g, r):
            return float(g / r) if (pd.notna(g) and pd.notna(r) and r != 0) else None

        gm0 = sm(gp.iloc[0], rev.iloc[0])
        gm1 = sm(gp.iloc[1], rev.iloc[1]) if df.shape[1] >= 2 else None
        return (gm0, gm1)

    def roic(self, as_of: pd.Timestamp) -> Optional[float]:
        """
        ROIC = NOPAT / Invested Capital
        NOPAT = EBIT * (1 - tax_rate)  [using annual or TTM from 4 quarters]
        """
        income, freq = self._best_income(as_of, min_q_cols=4)
        balance, _   = self._best_balance(as_of)
        if income is None or balance is None:
            return None

        ebit_row = get_row(income, self.OPINCOME_ROWS)
        if ebit_row is None:
            return None

        if freq == "quarterly":
            ebit = float(ebit_row.iloc[:4].sum())
        else:
            ebit = float(ebit_row.iloc[0])   # annual value

        tax_row = get_row(income, self.TAXRATE_ROWS)
        if tax_row is not None and len(tax_row) > 0 and pd.notna(tax_row.iloc[0]):
            tax_rate = float(tax_row.iloc[0])
            if tax_rate > 1:
                tax_rate /= 100
            tax_rate = max(0.0, min(tax_rate, 0.6))
        else:
            tax_rate = 0.21

        nopat = ebit * (1 - tax_rate)

        assets   = self._get_scalar(balance, self.ASSETS_ROWS)
        curr_lib = self._get_scalar(balance, self.CURR_LIAB_ROWS)
        cash     = self._get_scalar(balance, self.CASH_ROWS)
        if assets is None or curr_lib is None:
            return None

        excess_cash = (cash or 0) * 0.5
        invested_cap = assets - curr_lib - excess_cash
        if invested_cap <= 0:
            return None

        return nopat / invested_cap

    def de_ratio(self, as_of: pd.Timestamp) -> Optional[float]:
        """Total Debt / Stockholders Equity."""
        balance, _ = self._best_balance(as_of)
        if balance is None:
            return None
        debt   = self._get_scalar(balance, self.TOTALDEBT_ROWS)
        equity = self._get_scalar(balance, self.EQUITY_ROWS)
        if debt is None or equity is None or equity == 0:
            return None
        return debt / abs(equity)

    def fcf(self, as_of: pd.Timestamp) -> Optional[float]:
        """TTM Free Cash Flow = Operating CF - CapEx."""
        cf, freq = self._best_cashflow(as_of, min_q_cols=4)
        if cf is None:
            return None

        opcf_row  = get_row(cf, self.OPCASH_ROWS)
        capex_row = get_row(cf, self.CAPEX_ROWS)
        if opcf_row is None:
            return None

        if freq == "quarterly":
            ttm_opcf  = float(opcf_row.iloc[:4].sum())
            ttm_capex = float(capex_row.iloc[:4].sum()) if capex_row is not None else 0.0
        else:
            ttm_opcf  = float(opcf_row.iloc[0])
            ttm_capex = float(capex_row.iloc[0]) if capex_row is not None else 0.0

        if ttm_capex > 0:
            ttm_capex = -ttm_capex

        return ttm_opcf + ttm_capex

    def cash_runway_months(self, as_of: pd.Timestamp) -> Optional[float]:
        """Months of cash runway for pre-revenue companies."""
        balance, _ = self._best_balance(as_of)
        cf, freq   = self._best_cashflow(as_of, min_q_cols=1)
        if balance is None or cf is None:
            return None

        cash = self._get_scalar(balance, self.CASH_ROWS)
        opcf_row = get_row(cf, self.OPCASH_ROWS)
        if cash is None or opcf_row is None:
            return None

        if freq == "quarterly":
            ttm_opcf = float(opcf_row.iloc[:min(4, len(opcf_row))].sum())
        else:
            ttm_opcf = float(opcf_row.iloc[0])

        if ttm_opcf >= 0:
            return 999

        monthly_burn = -ttm_opcf / 12
        return cash / monthly_burn if monthly_burn > 0 else 999

    def revenue_ttm(self, as_of: pd.Timestamp) -> Optional[float]:
        """TTM revenue."""
        income, freq = self._best_income(as_of, min_q_cols=1)
        if income is None:
            return None
        rev = get_row(income, self.REVENUE_ROWS)
        if rev is None:
            return None
        if freq == "quarterly":
            n = min(4, income.shape[1])
            return float(rev.iloc[:n].sum())
        else:
            return float(rev.iloc[0])

    def net_income_ttm(self, as_of: pd.Timestamp) -> Optional[float]:
        """TTM net income."""
        income, freq = self._best_income(as_of, min_q_cols=1)
        if income is None:
            return None
        ni = get_row(income, self.NETINCOME_ROWS)
        if ni is None:
            return None
        if freq == "quarterly":
            n = min(4, income.shape[1])
            return float(ni.iloc[:n].sum())
        else:
            return float(ni.iloc[0])

    def fcf_yield(
        self,
        as_of: pd.Timestamp,
        market_cap: float,
        total_debt: Optional[float],
        cash: Optional[float],
    ) -> Optional[float]:
        """FCF / Enterprise Value."""
        ttm_fcf = self.fcf(as_of)
        if ttm_fcf is None:
            return None
        ev = market_cap + (total_debt or 0) - (cash or 0)
        if ev <= 0:
            return None
        return ttm_fcf / ev

    def ebit_yield(
        self,
        as_of: pd.Timestamp,
        market_cap: float,
        total_debt: Optional[float],
        cash: Optional[float],
    ) -> Optional[float]:
        """EBIT (TTM) / Enterprise Value."""
        income, freq = self._best_income(as_of, min_q_cols=4)
        if income is None:
            return None
        ebit_row = get_row(income, self.OPINCOME_ROWS)
        if ebit_row is None:
            return None
        if freq == "quarterly":
            ebit = float(ebit_row.iloc[:4].sum())
        else:
            ebit = float(ebit_row.iloc[0])
        ev = market_cap + (total_debt or 0) - (cash or 0)
        if ev <= 0:
            return None
        return ebit / ev

    def gp_assets(self, as_of: pd.Timestamp) -> Optional[float]:
        """Gross Profit (TTM) / Total Assets — Novy-Marx (2013) gross profitability."""
        income, freq = self._best_income(as_of, min_q_cols=4)
        balance, _   = self._best_balance(as_of)
        if income is None or balance is None:
            return None
        gp_row     = get_row(income, self.GROSS_ROWS)
        assets_row = get_row(balance, self.ASSETS_ROWS)
        if gp_row is None or assets_row is None:
            return None
        assets = float(assets_row.iloc[0])
        if assets <= 0:
            return None
        n = min(4, income.shape[1])
        ttm_gp = float(gp_row.iloc[:n].sum()) if freq == "quarterly" else float(gp_row.iloc[0])
        return ttm_gp / assets

    def piotroski_fscore(self, as_of: pd.Timestamp) -> Optional[int]:
        """
        Simplified Piotroski F-Score (0–7 points).

        Criteria scored:
          1. ROA > 0
          2. Operating CF > 0
          3. ROA improving YoY
          4. Cash earnings quality: OCF > net income (accruals < 0)
          5. Leverage (total debt / assets) not increasing YoY
          6. Gross margin improving YoY
          7. Asset turnover (revenue / assets) improving YoY

        Skips current-ratio (#8) and share-dilution (#9) — unreliable in yfinance.
        Returns None when insufficient data for a meaningful score.
        """
        income, freq_i = self._best_income(as_of, min_q_cols=4)
        balance, _     = self._best_balance(as_of)
        cf,      freq_c = self._best_cashflow(as_of, min_q_cols=4)
        if income is None or balance is None or cf is None:
            return None

        # Balance sheet
        assets_row = get_row(balance, self.ASSETS_ROWS)
        if assets_row is None or len(assets_row) < 1:
            return None
        assets_now   = float(assets_row.iloc[0])
        assets_prior = float(assets_row.iloc[1]) if len(assets_row) >= 2 else None
        if assets_now <= 0:
            return None

        debt_row   = get_row(balance, self.TOTALDEBT_ROWS)
        debt_now   = float(debt_row.iloc[0]) if (debt_row is not None and len(debt_row) >= 1) else None
        debt_prior = float(debt_row.iloc[1]) if (debt_row is not None and len(debt_row) >= 2) else None

        # Income — TTM and prior-year TTM
        ni_row  = get_row(income, self.NETINCOME_ROWS)
        rev_row = get_row(income, self.REVENUE_ROWS)
        gp_row  = get_row(income, self.GROSS_ROWS)
        if ni_row is None or rev_row is None:
            return None

        def _ttm(row, start: int) -> Optional[float]:
            if row is None:
                return None
            end = start + (4 if freq_i == "quarterly" else 1)
            sl  = row.iloc[start:end]
            return float(sl.sum()) if len(sl) > 0 else None

        ni_now    = _ttm(ni_row,  0)
        ni_prior  = _ttm(ni_row,  4) if freq_i == "quarterly" else _ttm(ni_row, 1)
        rev_now   = _ttm(rev_row, 0)
        rev_prior = _ttm(rev_row, 4) if freq_i == "quarterly" else _ttm(rev_row, 1)
        gp_now    = _ttm(gp_row,  0)
        gp_prior  = _ttm(gp_row,  4) if freq_i == "quarterly" else _ttm(gp_row, 1)

        if ni_now is None or rev_now is None:
            return None

        # Operating CF
        opcf_row = get_row(cf, self.OPCASH_ROWS)
        if opcf_row is None:
            return None
        ocf_now = float(opcf_row.iloc[:4].sum()) if freq_c == "quarterly" else float(opcf_row.iloc[0])

        roa_now   = ni_now / assets_now
        roa_prior = (ni_prior / assets_prior
                     if ni_prior is not None and assets_prior and assets_prior > 0
                     else None)

        score = 0

        # 1. ROA > 0
        if roa_now > 0:
            score += 1
        # 2. Operating CF > 0
        if ocf_now > 0:
            score += 1
        # 3. ROA improving
        if roa_prior is not None and roa_now > roa_prior:
            score += 1
        # 4. Cash earnings quality (OCF beats net income)
        if ocf_now > ni_now:
            score += 1
        # 5. Leverage not increasing
        if (debt_now is not None and debt_prior is not None
                and assets_prior is not None and assets_prior > 0):
            if debt_now / assets_now <= debt_prior / assets_prior:
                score += 1
        # 6. Gross margin improving
        if (gp_now is not None and gp_prior is not None
                and rev_now > 0 and rev_prior and rev_prior > 0):
            if gp_now / rev_now > gp_prior / rev_prior:
                score += 1
        # 7. Asset turnover improving
        if rev_prior is not None and assets_prior and assets_prior > 0:
            if rev_now / assets_now > rev_prior / assets_prior:
                score += 1

        return score

    def ps_ratio(self, as_of: pd.Timestamp, market_cap: float) -> Optional[float]:
        """Price / Sales (TTM)."""
        rev = self.revenue_ttm(as_of)
        if rev is None or rev <= 0:
            return None
        return market_cap / rev

    def peg_ratio(self, as_of: pd.Timestamp, pe: float) -> Optional[float]:
        """
        PEG = P/E / EPS growth (%).
        EPS growth = YoY net income growth (using annual data if quarterly
        TTM comparison is unavailable).
        """
        if pe is None or pe <= 0:
            return None

        # Try quarterly (need 8 periods for 2-year TTM comparison)
        q = self._q_income(as_of)
        if q is not None and q.shape[1] >= 8:
            ni = get_row(q, self.NETINCOME_ROWS)
            if ni is not None and len(ni) >= 8:
                ttm_now  = float(ni.iloc[:4].sum())
                ttm_prev = float(ni.iloc[4:8].sum())
                if ttm_prev > 0:
                    growth = ttm_now / ttm_prev - 1
                    if growth > 0:
                        return pe / (growth * 100)

        # Annual fallback
        a = self._a_income(as_of)
        if a is not None and a.shape[1] >= 2:
            ni = get_row(a, self.NETINCOME_ROWS)
            if ni is not None and len(ni) >= 2:
                now  = float(ni.iloc[0])
                prev = float(ni.iloc[1])
                if prev > 0 and now > prev:
                    growth = now / prev - 1
                    if growth > 0:
                        return pe / (growth * 100)

        return None

    def is_profitable(self, as_of: pd.Timestamp) -> bool:
        ni = self.net_income_ttm(as_of)
        return ni is not None and ni > 0

    def balance_sheet_snapshot(self, as_of: pd.Timestamp) -> dict:
        balance, _ = self._best_balance(as_of)
        return {
            "total_debt": self._get_scalar(balance, self.TOTALDEBT_ROWS),
            "cash":       self._get_scalar(balance, self.CASH_ROWS),
            "equity":     self._get_scalar(balance, self.EQUITY_ROWS),
        }

    def consecutive_negative_revenue_quarters(self, as_of: pd.Timestamp) -> int:
        """Count most recent consecutive periods with negative YoY revenue growth."""
        # Try quarterly
        q = self._q_income(as_of)
        if q is not None and q.shape[1] >= 5:
            rev = get_row(q, self.REVENUE_ROWS)
            if rev is not None and len(rev) >= 5:
                count = 0
                for i in range(min(4, len(rev) - 4)):
                    curr  = rev.iloc[i]
                    prior = rev.iloc[i + 4]
                    if pd.isna(curr) or pd.isna(prior) or prior == 0:
                        break
                    if curr < prior:
                        count += 1
                    else:
                        break
                return count

        # Annual fallback
        a = self._a_income(as_of)
        if a is not None and a.shape[1] >= 2:
            rev = get_row(a, self.REVENUE_ROWS)
            if rev is not None and len(rev) >= 2:
                count = 0
                for i in range(min(3, len(rev) - 1)):
                    curr  = rev.iloc[i]
                    prior = rev.iloc[i + 1]
                    if pd.isna(curr) or pd.isna(prior) or prior == 0:
                        break
                    if curr < prior:
                        count += 1
                    else:
                        break
                return count

        return 0
