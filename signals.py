"""
Signal generator – applies quality and timing filters to determine entry eligibility.

Two-timeframe design:
  - Quality filters (1-7): evaluated quarterly at each quarter end.
    Stocks that pass all 7 join the "quality watchlist" for the following quarter.
  - Fast filters (8-12): evaluated daily, but only for watchlist stocks.

Entry (crash tier): a watchlist stock that passes all fast filters on a given day.
Exit (crash tier): thesis-break (revenue, margin, leverage) or 3-year max hold.

QVM engine: quality watchlist + value/momentum composite score for rotation.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from features import FundamentalAccessor, compute_price_features, compute_spy_features, get_row

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(
        self,
        config: Config,
        price_features: Dict[str, pd.DataFrame],
        spy_features: pd.DataFrame,
        vix_prices: pd.DataFrame,
        fundamentals: Dict[str, dict],
        sector_map: Dict[str, str],
    ):
        self.cfg = config
        self.price_features = price_features
        self.spy_feat = spy_features
        self.vix = vix_prices["Close"].dropna() if vix_prices is not None else pd.Series(dtype=float)
        self.fundamentals = fundamentals
        self.sector_map = sector_map
        self._sector_momentum: Optional[Dict[str, pd.DataFrame]] = None

    # ================================================================ #
    # Pre-computation
    # ================================================================ #

    def precompute_sector_momentum(self):
        """Build per-sector momentum percentile ranks. Called once before backtest."""
        logger.info("Pre-computing sector momentum ranks...")
        sector_tickers: Dict[str, List[str]] = {}
        for t, s in self.sector_map.items():
            sector_tickers.setdefault(s, []).append(t)

        mom_series: Dict[str, pd.Series] = {}
        for t, feat in self.price_features.items():
            if "ret_12_1" in feat.columns:
                mom_series[t] = feat["ret_12_1"]

        if not mom_series:
            self._sector_momentum = {}
            return

        mom_df = pd.DataFrame(mom_series)
        result: Dict[str, pd.DataFrame] = {}
        for sector, tickers in sector_tickers.items():
            sect_tickers = [t for t in tickers if t in mom_df.columns]
            if not sect_tickers:
                continue
            sub   = mom_df[sect_tickers]
            ranks = sub.rank(axis=1, pct=True)
            result[sector] = ranks

        self._sector_momentum = result
        logger.info("Sector momentum pre-computation complete.")

    def precompute_watchlists(
        self,
        tickers: List[str],
        trading_days: pd.DatetimeIndex,
    ) -> Dict[pd.Timestamp, Tuple[Set[str], int]]:
        """
        Evaluate quality filters (1-7) at each quarter end and build watchlists.
        Returns {first_effective_trading_day: (watchlist_set, n_evaluated)}.
        """
        start = trading_days[0]
        end   = trading_days[-1]

        quarter_ends = pd.date_range(
            start=start - pd.DateOffset(months=4),
            end=end,
            freq="QE",
        )

        result: Dict[pd.Timestamp, Tuple[Set[str], int]] = {}
        logger.info(f"Pre-computing quality watchlists for {len(quarter_ends)} quarters...")

        for qe in tqdm(quarter_ends, desc="Quarterly watchlists"):
            avail     = trading_days[trading_days <= qe]
            eval_date = avail[-1] if len(avail) > 0 else trading_days[0]

            next_days = trading_days[trading_days > qe]
            effective = next_days[0] if len(next_days) > 0 else trading_days[-1]

            passing: Set[str] = set()
            n_eval = 0
            for ticker in tickers:
                if ticker not in self.fundamentals:
                    continue
                n_eval += 1
                passed, _ = self._check_quality_filters(ticker, eval_date)
                if passed:
                    passing.add(ticker)

            result[effective] = (passing, n_eval)
            logger.info(
                f"  Quarter ending {qe.date()} (eval {eval_date.date()}): "
                f"{len(passing)}/{n_eval} on watchlist  [effective {effective.date()}]"
            )

        return result

    def precompute_qvm_watchlists(
        self,
        tickers: List[str],
        trading_days: pd.DatetimeIndex,
        top_n: int = 10,
    ) -> Dict[pd.Timestamp, dict]:
        """
        Evaluate F1-F7 quality filters AND QVM composite scores at each quarter end.

        Returns {effective_date: {
            "watchlist": set,          # tickers passing F1-F7
            "n_eval":    int,
            "top_n":     List[str],    # top_n tickers ranked by QVM score (desc)
            "eval_date": Timestamp,    # quarter-end date used for scoring
        }}.
        """
        start = trading_days[0]
        end   = trading_days[-1]

        quarter_ends = pd.date_range(
            start=start - pd.DateOffset(months=4),
            end=end,
            freq="QE",
        )

        result: Dict[pd.Timestamp, dict] = {}
        logger.info(f"Pre-computing QVM watchlists for {len(quarter_ends)} quarters...")

        for qe in tqdm(quarter_ends, desc="QVM watchlists"):
            avail     = trading_days[trading_days <= qe]
            eval_date = avail[-1] if len(avail) > 0 else trading_days[0]

            next_days = trading_days[trading_days > qe]
            effective = next_days[0] if len(next_days) > 0 else trading_days[-1]

            passing: Set[str] = set()
            n_eval = 0
            for ticker in tickers:
                if ticker not in self.fundamentals:
                    continue
                n_eval += 1
                passed, _ = self._check_quality_filters(ticker, eval_date)
                if passed:
                    passing.add(ticker)

            scores     = self.compute_qvm_scores(passing, eval_date)
            top_n_list = list(scores.head(top_n).index)

            result[effective] = {
                "watchlist": passing,
                "n_eval":    n_eval,
                "top_n":     top_n_list,
                "eval_date": eval_date,
            }
            logger.info(
                f"  Quarter ending {qe.date()} (eval {eval_date.date()}): "
                f"{len(passing)}/{n_eval} on watchlist, "
                f"top {len(top_n_list)} QVM stocks  [effective {effective.date()}]"
            )

        return result

    # ================================================================ #
    # Fast filter helpers (used daily for crash-tier entries)
    # ================================================================ #

    def _macro_not_red(self) -> bool:
        """
        Macro agent gate: returns True if macro regime is GREEN or YELLOW.
        Returns True (allow) if macro_state.json is missing or unreadable —
        fail-open so the model keeps running if the macro agent is not set up.
        Only blocks crash-tier entries when regime is explicitly RED.
        """
        if not getattr(self.cfg, "MACRO_AGENT_ENABLED", False):
            return True
        try:
            import json
            import os
            path = self.cfg.MACRO_STATE_FILE
            if not os.path.exists(path):
                logger.warning("macro_state.json not found — defaulting to GREEN (allow)")
                return True
            with open(path) as f:
                state = json.load(f)
            regime = state.get("regime", "GREEN").upper()
            metastability = state.get("metastability", "LOW").upper()
            if regime == "RED":
                logger.info("Macro agent regime: RED — crash tier entry blocked")
                return False
            logger.info(
                f"Macro agent regime: {regime} "
                f"(metastability: {metastability}) — crash tier entry allowed"
            )
            return True
        except Exception as e:
            logger.warning(f"Macro agent read failed ({e}) — defaulting to GREEN (allow)")
            return True

    def _fear_regime(self, date: pd.Timestamp, min_conditions: int = 2) -> bool:
        """Filter 8: at least min_conditions of 4 fear conditions AND macro not RED."""
        conditions = []

        if len(self.vix) > 0:
            vix_val = self.vix.asof(date)
            conditions.append(bool(pd.notna(vix_val) and vix_val > self.cfg.VIX_FEAR_THRESHOLD))
        else:
            conditions.append(False)

        if len(self.spy_feat) > 0:
            spy_row = self.spy_feat.asof(date)
            if spy_row is not None and pd.notna(spy_row.get("spy_below_200d_pct")):
                conditions.append(float(spy_row["spy_below_200d_pct"]) > self.cfg.SPY_BELOW_200D_PCT)
                conditions.append(float(spy_row["spy_drawdown"]) > self.cfg.MARKET_DRAWDOWN_PCT)
            else:
                conditions.extend([False, False])
        else:
            conditions.extend([False, False])

        today = pd.Timestamp.today().normalize()
        if abs((date.normalize() - today).days) <= 3:
            from fear_greed import fetch_fear_greed_score
            fg_score = fetch_fear_greed_score(self.cfg.CACHE_DIR)
            if fg_score is not None:
                conditions.append(fg_score < self.cfg.CNN_FG_FEAR_THRESHOLD)

        return sum(conditions) >= min_conditions and self._macro_not_red()

    def _price_dislocation(self, ticker: str, date: pd.Timestamp) -> bool:
        """Filter 9: either condition triggers pass."""
        feat = self.price_features.get(ticker)
        if feat is None or len(feat) == 0:
            return False
        feat_row = feat.asof(date)
        if not isinstance(feat_row, pd.Series):
            return False

        zscore   = feat_row.get("price_zscore_12m")
        high_52w = feat_row.get("high_52w")
        close    = feat_row.get("close")

        cond_a = pd.notna(zscore) and float(zscore) < self.cfg.PRICE_ZSCORE_LOWER
        cond_b = (
            pd.notna(high_52w) and pd.notna(close) and high_52w > 0
            and (float(high_52w) - float(close)) / float(high_52w) > self.cfg.PRICE_BELOW_52W_HIGH_PCT
        )
        return cond_a or cond_b

    def _momentum_not_bottom_decile(self, ticker: str, date: pd.Timestamp) -> bool:
        """Filter 10: pass if NOT in bottom decile of sector 12-1 momentum."""
        if self._sector_momentum is None:
            return True

        sector  = self.sector_map.get(ticker)
        if sector is None:
            return True

        sect_df = self._sector_momentum.get(sector)
        if sect_df is None or ticker not in sect_df.columns:
            return True

        try:
            rank = sect_df[ticker].asof(date)
        except Exception:
            return True

        if pd.isna(rank):
            return True

        return float(rank) > self.cfg.MOMENTUM_BOTTOM_DECILE

    def _build_ps_history(
        self,
        ticker: str,
        date: pd.Timestamp,
        fa: FundamentalAccessor,
    ) -> list:
        """
        Build a list of historical P/S values for ticker over the 3-year lookback.
        Shared by _ps_bottom_quartile and _ps_percentile_value.
        """
        price_feat = self.price_features.get(ticker)
        if price_feat is None:
            return []

        info   = fa.raw.get("info") or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        lookback_start = date - pd.Timedelta(days=365 * self.cfg.PS_HISTORY_YEARS)
        lag = pd.Timedelta(days=fa.lag_days)

        def _build(income_df, annualize_factor: float) -> list:
            rev_row = get_row(income_df, FundamentalAccessor.REVENUE_ROWS)
            if rev_row is None:
                return []
            hist = []
            for period_end in rev_row.index:
                avail_date = pd.Timestamp(period_end) + lag
                if avail_date < lookback_start or avail_date > date:
                    continue
                rev = float(rev_row[period_end])
                if pd.isna(rev) or rev <= 0:
                    continue
                try:
                    close_hist = price_feat["close"].asof(avail_date)
                    if shares and pd.notna(close_hist):
                        ps = float(close_hist) * float(shares) / (rev * annualize_factor)
                        hist.append(ps)
                except Exception:
                    pass
            return hist

        qi = fa.raw.get("quarterly_income")
        ps_history = _build(qi, 4) if qi is not None else []

        if len(ps_history) < 4:
            ai = fa.raw.get("annual_income")
            if ai is not None:
                annual_hist = _build(ai, 1)
                if len(annual_hist) > len(ps_history):
                    ps_history = annual_hist

        return ps_history

    def _ps_bottom_quartile(
        self,
        ticker: str,
        date: pd.Timestamp,
        fa: FundamentalAccessor,
        market_cap: float,
        ps_quartile_max: Optional[float] = None,
    ) -> bool:
        """Filter 11: current P/S is in bottom quartile of its own 3-year range."""
        ps_history = self._build_ps_history(ticker, date, fa)

        min_pts = 2 if len(ps_history) < 4 else 4
        if len(ps_history) < min_pts:
            return False

        current_ps = fa.ps_ratio(date, market_cap)
        if current_ps is None:
            return False

        q_max     = ps_quartile_max if ps_quartile_max is not None else self.cfg.PS_QUARTILE_MAX
        threshold = np.percentile(ps_history, q_max * 100)
        return bool(current_ps <= threshold)

    def _above_50ema(self, ticker: str, date: pd.Timestamp) -> bool:
        """Filter 12: price is at or above its 50-day EMA (within 10% tolerance)."""
        feat = self.price_features.get(ticker)
        if feat is None:
            return False
        try:
            row = feat.asof(date)
        except Exception:
            return False
        if row is None or not isinstance(row, pd.Series):
            return False
        close = row.get("close")
        ema50 = row.get("ema_50")
        if pd.isna(close) or pd.isna(ema50):
            return False
        return float(close) >= float(ema50) * 0.90

    # ================================================================ #
    # QVM scoring (used by Engine 2)
    # ================================================================ #

    def _ps_percentile_value(
        self,
        ticker: str,
        date: pd.Timestamp,
        fa: FundamentalAccessor,
        market_cap: float,
    ) -> Optional[float]:
        """
        Return current P/S as a percentile of its own 3-year history.
        0 = cheapest historically, 1 = most expensive.
        """
        ps_history = self._build_ps_history(ticker, date, fa)
        if len(ps_history) < 2:
            return None
        current_ps = fa.ps_ratio(date, market_cap)
        if current_ps is None:
            return None
        return float(np.mean(np.array(ps_history) <= current_ps))

    def _momentum_6m(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """6-month price return from pre-computed price features."""
        feat = self.price_features.get(ticker)
        if feat is None or "ret_6m" not in feat.columns:
            return None
        try:
            val = feat["ret_6m"].asof(date)
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    def _momentum_12_1(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """12-minus-1 month price return (Jegadeesh & Titman) from pre-computed features."""
        feat = self.price_features.get(ticker)
        if feat is None or "ret_12_1" not in feat.columns:
            return None
        try:
            val = feat["ret_12_1"].asof(date)
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    def compute_qvm_scores(
        self,
        watchlist: Set[str],
        date: pd.Timestamp,
    ) -> pd.Series:
        """
        Score and rank watchlist stocks by QVM composite (higher = better).
        Behaviour controlled by cfg.QVM_VARIANT:

        "original"  — Value (50%): P/S percentile vs own 3-year history (lower = cheaper)
                       Momentum (50%): 6-month return
                       Requires both metrics; missing → excluded.

        "composite" — Value (50%): cross-sectional rank average of three sub-metrics:
                         1. EBIT/EV  (higher = cheaper)
                         2. P/S percentile vs own 3-year history (lower = cheaper)
                         3. FCF/EV   (higher = cheaper)
                       One sub-metric missing → average the other two.
                       Two or more missing → excluded that quarter.
                       Momentum (50%): 6-month return.

        Both variants: F3 quality filter = ROIC (applied upstream in watchlist pre-compute).
        Returns a Series sorted descending (best stock first).
        """
        variant = getattr(self.cfg, "QVM_VARIANT", "composite")
        data: Dict[str, dict] = {}

        for ticker in watchlist:
            raw = self.fundamentals.get(ticker)
            if raw is None:
                continue
            mc = self._get_market_cap(ticker, date, raw)
            if mc is None:
                continue
            fa     = FundamentalAccessor(raw, lag_days=self.cfg.FUNDAMENTAL_LAG_DAYS)
            mom_6m = self._momentum_6m(ticker, date)
            if mom_6m is None:
                continue

            if variant == "original":
                ps_pct = self._ps_percentile_value(ticker, date, fa, mc)
                if ps_pct is None:
                    continue
                data[ticker] = {"ps_pct": ps_pct, "mom_6m": mom_6m}

            else:  # "composite"
                bs         = fa.balance_sheet_snapshot(date)
                total_debt = bs.get("total_debt")
                cash       = bs.get("cash")
                ebit_yld   = fa.ebit_yield(date, mc, total_debt, cash)
                ps_pct     = self._ps_percentile_value(ticker, date, fa, mc)
                fcf_yld    = fa.fcf_yield(date, mc, total_debt, cash)
                n_missing  = sum(v is None for v in [ebit_yld, ps_pct, fcf_yld])
                if n_missing >= 2:
                    continue
                data[ticker] = {
                    "ebit_yld": ebit_yld,
                    "ps_pct":   ps_pct,
                    "fcf_yld":  fcf_yld,
                    "mom_6m":   mom_6m,
                }

        if not data:
            return pd.Series(dtype=float)

        df = pd.DataFrame(data).T
        n  = len(df)

        if variant == "original":
            if n > 1:
                # Value: lower P/S percentile = cheaper = higher score
                df["value_score"]    = 1 - df["ps_pct"].rank(ascending=True, pct=True)
                df["momentum_score"] = df["mom_6m"].rank(ascending=True, pct=True)
            else:
                df["value_score"]    = 0.5
                df["momentum_score"] = 0.5

        else:  # "composite"
            if n > 1:
                # Per-metric value scores: higher score = better value (cheaper)
                # EBIT yield: higher = cheaper
                v_ebit = df["ebit_yld"].rank(ascending=True, pct=True)
                # P/S percentile: lower = cheaper → invert
                v_ps   = 1 - df["ps_pct"].rank(ascending=True, pct=True)
                # FCF yield: higher = cheaper
                v_fcf  = df["fcf_yld"].rank(ascending=True, pct=True)
                # Composite value = mean of available sub-metrics (skipna handles 1 missing)
                val_df = pd.concat([v_ebit, v_ps, v_fcf], axis=1)
                val_df.columns = ["v_ebit", "v_ps", "v_fcf"]
                df["value_score"]    = val_df.mean(axis=1)
                df["momentum_score"] = df["mom_6m"].rank(ascending=True, pct=True)
            else:
                df["value_score"]    = 0.5
                df["momentum_score"] = 0.5

        df["composite"] = 0.5 * df["value_score"] + 0.5 * df["momentum_score"]
        return df["composite"].sort_values(ascending=False)

    def rank_watchlist(
        self,
        watchlist: Set[str],
        date: pd.Timestamp,
        top_n: int = 20,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Rank watchlist stocks by composite QVM score.
        Returns list of (ticker, composite, value_score, momentum_score) sorted
        descending by composite, capped at top_n.
        """
        variant = getattr(self.cfg, "QVM_VARIANT", "composite")
        data: Dict[str, dict] = {}

        for ticker in watchlist:
            raw = self.fundamentals.get(ticker)
            if raw is None:
                continue
            mc = self._get_market_cap(ticker, date, raw)
            if mc is None:
                continue
            fa     = FundamentalAccessor(raw, lag_days=self.cfg.FUNDAMENTAL_LAG_DAYS)
            mom_6m = self._momentum_6m(ticker, date)
            if mom_6m is None:
                continue

            if variant == "original":
                ps_pct = self._ps_percentile_value(ticker, date, fa, mc)
                if ps_pct is None:
                    continue
                data[ticker] = {"ps_pct": ps_pct, "mom_6m": mom_6m}
            else:
                bs         = fa.balance_sheet_snapshot(date)
                total_debt = bs.get("total_debt")
                cash       = bs.get("cash")
                ebit_yld   = fa.ebit_yield(date, mc, total_debt, cash)
                ps_pct     = self._ps_percentile_value(ticker, date, fa, mc)
                fcf_yld    = fa.fcf_yield(date, mc, total_debt, cash)
                if sum(v is None for v in [ebit_yld, ps_pct, fcf_yld]) >= 2:
                    continue
                data[ticker] = {
                    "ebit_yld": ebit_yld, "ps_pct": ps_pct,
                    "fcf_yld": fcf_yld, "mom_6m": mom_6m,
                }

        if not data:
            return []

        df = pd.DataFrame(data).T
        n  = len(df)

        if variant == "original":
            if n > 1:
                df["value_score"]    = 1 - df["ps_pct"].rank(ascending=True, pct=True)
                df["momentum_score"] = df["mom_6m"].rank(ascending=True, pct=True)
            else:
                df["value_score"] = df["momentum_score"] = 0.5
        else:
            if n > 1:
                v_ebit = df["ebit_yld"].rank(ascending=True, pct=True)
                v_ps   = 1 - df["ps_pct"].rank(ascending=True, pct=True)
                v_fcf  = df["fcf_yld"].rank(ascending=True, pct=True)
                val_df = pd.concat([v_ebit, v_ps, v_fcf], axis=1)
                val_df.columns = ["v_ebit", "v_ps", "v_fcf"]
                df["value_score"]    = val_df.mean(axis=1)
                df["momentum_score"] = df["mom_6m"].rank(ascending=True, pct=True)
            else:
                df["value_score"] = df["momentum_score"] = 0.5

        df["composite"] = 0.5 * df["value_score"] + 0.5 * df["momentum_score"]
        df = df.sort_values("composite", ascending=False).head(top_n)

        return [
            (ticker, row["composite"], row["value_score"], row["momentum_score"])
            for ticker, row in df.iterrows()
        ]

    # ================================================================ #
    # Quality filters (slow – evaluated quarterly to build watchlist)
    # ================================================================ #

    def _check_quality_filters(
        self,
        ticker: str,
        date: pd.Timestamp,
    ) -> Tuple[bool, Dict[str, Optional[bool]]]:
        """
        Run filters 1-7 (fundamental quality filters).
        Returns (all_pass, {filter_name: result}).
        """
        results: Dict[str, Optional[bool]] = {}

        raw = self.fundamentals.get(ticker)
        if raw is None:
            return False, results

        fa = FundamentalAccessor(raw, lag_days=self.cfg.FUNDAMENTAL_LAG_DAYS)

        # --- Filter 1: Revenue growth ---
        rev_growth = fa.revenue_growth_yoy(date)
        if rev_growth is None:
            results["f1_rev_growth"] = None
        else:
            g1, g2 = rev_growth
            results["f1_rev_growth"] = (
                g1 is not None and g1 > self.cfg.MIN_REVENUE_GROWTH_YOY
                and (g2 is None or g2 > self.cfg.MIN_REVENUE_GROWTH_YOY)
            )

        # --- Filter 2: Gross margin ---
        gm = fa.gross_margin(date)
        if gm is None:
            results["f2_gross_margin"] = None
        else:
            gm0, gm1 = gm
            if gm0 is None:
                results["f2_gross_margin"] = None
            else:
                results["f2_gross_margin"] = (
                    gm0 > self.cfg.MIN_GROSS_MARGIN
                    and (gm1 is None or gm0 >= gm1 - 0.05)
                )

        # --- Filter 3: ROIC ---
        roic = fa.roic(date)
        results["f3_roic"] = None if roic is None else roic > self.cfg.MIN_ROIC

        # --- Filter 4: D/E ratio ---
        de = fa.de_ratio(date)
        results["f4_de_ratio"] = None if de is None else de < self.cfg.MAX_DE_RATIO

        # --- Filter 5: FCF or runway ---
        fcf = fa.fcf(date)
        if fcf is not None and fcf > 0:
            results["f5_fcf"] = True
        else:
            runway = fa.cash_runway_months(date)
            results["f5_fcf"] = None if runway is None else runway >= self.cfg.MIN_CASH_RUNWAY_MONTHS

        # --- Filter 6: PEG or P/S ---
        mc         = self._get_market_cap(ticker, date, raw)
        profitable = fa.is_profitable(date)
        bs         = fa.balance_sheet_snapshot(date)

        if profitable:
            pe  = self._get_pe(ticker, date, fa, mc)
            peg = fa.peg_ratio(date, pe) if pe else None
            if peg is not None:
                results["f6_peg_ps"] = peg < self.cfg.MAX_PEG_RATIO
            else:
                ps = fa.ps_ratio(date, mc) if mc else None
                g1 = rev_growth[0] if rev_growth else None
                if ps is None:
                    results["f6_peg_ps"] = None
                else:
                    results["f6_peg_ps"] = (
                        ps < self.cfg.MAX_PS_PRE_PROFITABLE
                        and g1 is not None
                        and g1 > self.cfg.MIN_GROWTH_PRE_PROFITABLE
                    )
        else:
            ps  = fa.ps_ratio(date, mc) if mc else None
            g1  = rev_growth[0] if rev_growth else None
            if ps is None:
                results["f6_peg_ps"] = None
            else:
                results["f6_peg_ps"] = (
                    ps < self.cfg.MAX_PS_PRE_PROFITABLE
                    and g1 is not None
                    and g1 > self.cfg.MIN_GROWTH_PRE_PROFITABLE
                )

        # --- Filter 7: FCF yield ---
        if mc is not None:
            fy = fa.fcf_yield(date, mc, bs.get("total_debt"), bs.get("cash"))
            results["f7_fcf_yield"] = None if fy is None else fy > self.cfg.MIN_FCF_YIELD
        else:
            results["f7_fcf_yield"] = None

        all_pass = all(v is True for v in results.values())
        return all_pass, results

    # ================================================================ #
    # Entry signal (crash tier – fast filters checked daily)
    # ================================================================ #

    def check_entry(
        self,
        ticker: str,
        date: pd.Timestamp,
        watchlist: Optional[Set[str]] = None,
    ) -> Tuple[bool, Dict[str, Optional[bool]]]:
        """Run fast filters 8-12 (crash-tier entry triggers)."""
        if watchlist is not None and ticker not in watchlist:
            return False, {}

        results: Dict[str, Optional[bool]] = {}

        results["f8_fear_regime"]  = self._fear_regime(date)
        results["f9_price_disloc"] = self._price_dislocation(ticker, date)
        results["f10_momentum"]    = self._momentum_not_bottom_decile(ticker, date)

        raw = self.fundamentals.get(ticker)
        mc  = self._get_market_cap(ticker, date, raw) if raw else None
        if mc is not None:
            fa = FundamentalAccessor(raw, lag_days=self.cfg.FUNDAMENTAL_LAG_DAYS)
            results["f11_ps_quartile"] = self._ps_bottom_quartile(ticker, date, fa, mc)
        else:
            results["f11_ps_quartile"] = False

        results["f12_above_ema50"] = self._above_50ema(ticker, date)

        all_pass = all(v is True for v in results.values())
        return all_pass, results

    def check_all_filters(
        self,
        ticker: str,
        date: pd.Timestamp,
    ) -> Tuple[bool, Dict[str, Optional[bool]]]:
        """Run all 12 filters. Used by diagnose.py."""
        q_pass, q_res = self._check_quality_filters(ticker, date)
        f_pass, f_res = self.check_entry(ticker, date, watchlist=None)
        return q_pass and f_pass, {**q_res, **f_res}

    # ================================================================ #
    # Exit signal (crash tier)
    # ================================================================ #

    def check_exit(
        self,
        ticker: str,
        date: pd.Timestamp,
        entry_date: pd.Timestamp,
    ) -> Tuple[bool, str]:
        """Returns (should_exit, reason). Time-based check handled by caller."""
        raw = self.fundamentals.get(ticker)
        if raw is None:
            return False, ""

        fa = FundamentalAccessor(raw, lag_days=self.cfg.FUNDAMENTAL_LAG_DAYS)

        neg_q = fa.consecutive_negative_revenue_quarters(date)
        if neg_q >= self.cfg.EXIT_NEG_REVENUE_QUARTERS:
            return True, f"neg_revenue_{neg_q}q"

        gm = fa.gross_margin(date)
        if gm is not None and gm[0] is not None:
            if gm[0] < self.cfg.EXIT_MIN_GROSS_MARGIN:
                return True, f"gm_below_{self.cfg.EXIT_MIN_GROSS_MARGIN:.0%}"

        de = fa.de_ratio(date)
        if de is not None and de > self.cfg.EXIT_MAX_DE_RATIO:
            return True, f"de_ratio_{de:.1f}"

        return False, ""

    # ================================================================ #
    # Helpers
    # ================================================================ #

    def _get_market_cap(self, ticker: str, date: pd.Timestamp, raw: dict) -> Optional[float]:
        feat = self.price_features.get(ticker)
        if feat is None:
            return None
        try:
            close = feat["close"].asof(date)
        except Exception:
            return None
        if pd.isna(close):
            return None
        info   = raw.get("info") or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if not shares:
            return None
        return float(close) * float(shares)

    def _get_pe(
        self, ticker: str, date: pd.Timestamp, fa: FundamentalAccessor, market_cap: Optional[float]
    ) -> Optional[float]:
        if market_cap is None:
            return None
        ni = fa.net_income_ttm(date)
        if ni is None or ni <= 0:
            return None
        return market_cap / ni
