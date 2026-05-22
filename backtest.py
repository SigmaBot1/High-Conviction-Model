"""
Portfolio simulation engines.

BacktestEngine   — crash-tier only (original model, unchanged).
QVMBacktestEngine — Engine 2 (QVM rotation) with optional crash-tier overrides.

QVM always-deployed logic:
  - Hold top QVM_TOP_N stocks by composite value+momentum score.
  - Rebalance quarterly: sell stocks that fell out of top N, buy replacements.
  - Combined mode: crash-tier signals override QVM — either upgrade an existing
    QVM position or displace the lowest-scored QVM holding.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from signals import SignalGenerator

logger = logging.getLogger(__name__)


def _compute_atr(
    df: Optional[pd.DataFrame],
    date: pd.Timestamp,
    period: int,
) -> Optional[float]:
    """Average True Range over `period` trading days ending at `date`."""
    if df is None:
        return None
    needed_cols = {"High", "Low", "Close"}
    if not needed_cols.issubset(df.columns):
        return None
    sub = df[df.index <= date].tail(period + 1)
    if len(sub) < period + 1:
        return None
    high  = sub["High"].values
    low   = sub["Low"].values
    close = sub["Close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return float(np.mean(tr[-period:]))


def _highest_close_since_entry(
    df: Optional[pd.DataFrame],
    entry_date: pd.Timestamp,
    date: pd.Timestamp,
) -> Optional[float]:
    """Highest closing price from entry_date through date (inclusive)."""
    if df is None or "Close" not in df.columns:
        return None
    sub = df[(df.index >= entry_date) & (df.index <= date)]
    if sub.empty:
        return None
    return float(sub["Close"].max())


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    capital: float
    tier: str = "crash"   # "crash" or "qvm"
    peak_price: float = 0.0


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp]
    exit_price: Optional[float]
    capital: float
    exit_reason: str = ""
    tier: str = "crash"

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def pnl_dollars(self) -> Optional[float]:
        if self.exit_price is None or self.pnl is None:
            return None
        return self.capital * self.pnl

    @property
    def holding_days(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days


class Portfolio:
    def __init__(self, initial_capital: float, max_positions: int):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: Dict[pd.Timestamp, float] = {}

    @property
    def available_slots(self) -> int:
        return self.max_positions - len(self.positions)

    def position_capital(self) -> float:
        return sum(p.capital for p in self.positions.values())

    def equity(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for t, pos in self.positions.items():
            price = prices.get(t)
            if price and not np.isnan(price):
                total += pos.shares * price
            else:
                total += pos.capital
        return total

    def buy(
        self,
        ticker: str,
        date: pd.Timestamp,
        price: float,
        tier: str = "crash",
        size_pct: Optional[float] = None,
    ):
        """
        Open a new position.
        size_pct: fraction of total equity to deploy.
                  If None, equal-weights across max_positions (1/max_positions).
        """
        if ticker in self.positions or self.available_slots <= 0 or price <= 0:
            return

        total_equity = self.cash + self.position_capital()
        slot_size    = total_equity * size_pct if size_pct is not None else total_equity / self.max_positions
        slot_size    = min(slot_size, self.cash)

        if slot_size <= 0:
            return

        shares = slot_size / price
        self.cash -= slot_size
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_date=date,
            entry_price=price,
            shares=shares,
            capital=slot_size,
            tier=tier,
            peak_price=price,
        )

    def sell(self, ticker: str, date: pd.Timestamp, price: float, reason: str):
        if ticker not in self.positions:
            return
        pos = self.positions.pop(ticker)
        self.cash += pos.shares * price
        self.trades.append(Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=price,
            capital=pos.capital,
            exit_reason=reason,
            tier=pos.tier,
        ))

    def check_trailing_stops(
        self,
        prices: Dict[str, Optional[float]],
        stop_pct: float,
        only_tiers: Optional[set] = None,
        cfg=None,
        price_data: Optional[Dict[str, "pd.DataFrame"]] = None,
        date: Optional["pd.Timestamp"] = None,
    ) -> List[Tuple[str, str]]:
        """Update peak prices and return (ticker, reason) pairs that hit the trailing stop.

        only_tiers: if provided, only positions whose tier is in this set are checked.
                    Peak prices are still updated for all positions.
        Supports cfg.TRAILING_STOP_METHOD == "atr" (Chandelier Exit) or "fixed".
        """
        use_atr = (
            cfg is not None
            and price_data is not None
            and date is not None
            and getattr(cfg, "TRAILING_STOP_METHOD", "fixed") == "atr"
        )
        to_exit = []
        for ticker, pos in self.positions.items():
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue
            if price > pos.peak_price:
                pos.peak_price = price
            if only_tiers is not None and pos.tier not in only_tiers:
                continue
            if use_atr:
                # Grace period: skip stop check for first STOP_GRACE_DAYS
                age_days = (date - pos.entry_date).days
                if age_days < cfg.STOP_GRACE_DAYS:
                    continue
                # Compute ATR over ATR_STOP_PERIOD trading days
                df = price_data.get(ticker)
                atr = _compute_atr(df, date, cfg.ATR_STOP_PERIOD)
                if atr is None:
                    continue
                # Highest close since entry (Chandelier anchors to the peak close)
                peak_close = _highest_close_since_entry(df, pos.entry_date, date)
                if peak_close is None:
                    continue
                stop_level = peak_close - atr * cfg.ATR_STOP_MULTIPLIER
                if price < stop_level:
                    to_exit.append((ticker, "atr_trailing_stop"))
            else:
                if price < pos.peak_price * (1.0 - stop_pct):
                    to_exit.append((ticker, "trailing_stop"))
        return to_exit

    def mark_open_positions(self, date: pd.Timestamp, prices: Dict[str, float]):
        self.equity_history[date] = self.equity(prices)


# ======================================================================== #
#  Engine 1: Crash-Tier Backtest (original model)
# ======================================================================== #

class BacktestEngine:
    """
    Original crash-tier model.  Mostly-cash; only deploys during fear regimes.
    """

    def __init__(
        self,
        config: Config,
        signal_gen: SignalGenerator,
        price_data: Dict[str, pd.DataFrame],
        spy_prices: pd.DataFrame,
        tickers: List[str],
    ):
        self.cfg        = config
        self.sig        = signal_gen
        self.price_data = price_data
        self.spy_prices = spy_prices
        self.tickers    = sorted(tickers)

    def _get_close(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self.price_data.get(ticker)
        if df is None or "Close" not in df.columns:
            return None
        try:
            v = df["Close"].asof(date)
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    def _trading_days(self) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.cfg.START_DATE)
        end   = pd.Timestamp(self.cfg.END_DATE)
        idx   = self.spy_prices.index
        return idx[(idx >= start) & (idx <= end)]

    def run(self) -> dict:
        portfolio    = Portfolio(self.cfg.INITIAL_CAPITAL, self.cfg.MAX_POSITIONS)
        trading_days = self._trading_days()

        logger.info(f"Running simulation over {len(trading_days)} trading days...")
        self.sig.precompute_sector_momentum()
        watchlists      = self.sig.precompute_watchlists(self.tickers, trading_days)
        watchlist_dates = sorted(watchlists.keys())

        current_watchlist: set = set()
        wl_idx = 0
        quarterly_stats: List[dict] = []
        signals_this_quarter = 0
        signal_counts: Dict[str, int] = {}
        for date in tqdm(trading_days, desc="Simulating days"):
            while wl_idx < len(watchlist_dates) and date >= watchlist_dates[wl_idx]:
                effective   = watchlist_dates[wl_idx]
                new_wl, n_eval = watchlists[effective]
                if quarterly_stats:
                    quarterly_stats[-1]["entry_signals"] = signals_this_quarter
                current_watchlist    = new_wl
                signals_this_quarter = 0
                quarterly_stats.append({
                    "quarter_start":  effective,
                    "watchlist_size": len(current_watchlist),
                    "evaluated":      n_eval,
                    "entry_signals":  0,
                })
                logger.info(
                    f"Watchlist updated {effective.date()}: "
                    f"{len(current_watchlist)}/{n_eval} stocks on watchlist"
                )
                wl_idx += 1

            todays_prices = {t: self._get_close(t, date) for t in list(portfolio.positions)}

            to_exit: List[Tuple[str, str]] = []

            max_hold = pd.Timedelta(days=365 * self.cfg.MAX_HOLD_YEARS)
            for ticker, pos in portfolio.positions.items():
                if todays_prices.get(ticker) is None:
                    continue
                if date - pos.entry_date >= max_hold:
                    to_exit.append((ticker, "max_hold_3yr"))
                    continue
                should_exit, reason = self.sig.check_exit(ticker, date, pos.entry_date)
                if should_exit:
                    to_exit.append((ticker, reason))

            for ticker, reason in to_exit:
                price = todays_prices.get(ticker)
                if price:
                    portfolio.sell(ticker, date, price, reason)

            if portfolio.available_slots > 0 and current_watchlist:
                for ticker in self.tickers:
                    if portfolio.available_slots <= 0:
                        break
                    if ticker not in current_watchlist or ticker in portfolio.positions:
                        continue
                    price = self._get_close(ticker, date)
                    if not price:
                        continue
                    try:
                        passed, _ = self.sig.check_entry(ticker, date)
                    except Exception as e:
                        logger.debug(f"Signal error for {ticker} on {date.date()}: {e}")
                        continue
                    if passed:
                        portfolio.buy(ticker, date, price, tier="crash")
                        signal_counts[ticker] = signal_counts.get(ticker, 0) + 1
                        signals_this_quarter += 1

            all_prices = {t: self._get_close(t, date) for t in portfolio.positions}
            portfolio.mark_open_positions(date, all_prices)

        final_date = trading_days[-1]
        for ticker in list(portfolio.positions.keys()):
            price = self._get_close(ticker, final_date)
            if price:
                portfolio.sell(ticker, final_date, price, "backtest_end")

        if quarterly_stats:
            quarterly_stats[-1]["entry_signals"] = signals_this_quarter

        equity_series = pd.Series(portfolio.equity_history).sort_index()
        spy_close     = self.spy_prices["Close"].reindex(trading_days).ffill()
        spy_equity    = (spy_close / spy_close.iloc[0]) * self.cfg.INITIAL_CAPITAL

        return {
            "mode":            "crash",
            "portfolio":       portfolio,
            "equity_series":   equity_series,
            "spy_equity":      spy_equity,
            "spy_close":       spy_close,
            "trades":          portfolio.trades,
            "signal_counts":   signal_counts,
            "trading_days":    trading_days,
            "quarterly_stats": quarterly_stats,
        }


# ======================================================================== #
#  Engine 2: QVM Rotation (always deployed, quarterly rebalance)
# ======================================================================== #

class QVMBacktestEngine:
    """
    Engine 2: Quality-Value-Momentum rotation.

    Always fully deployed in top QVM_TOP_N quality stocks.
    Rebalances at each quarter end.

    Combined mode: crash-tier signals can override QVM positions:
      - If the crash stock is already held as QVM → upgrade tier to "crash".
      - If not held → displace the lowest-scored QVM holding, buy crash pick.
    Crash positions are exempt from quarterly rebalance.
    """

    def __init__(
        self,
        config: Config,
        signal_gen: SignalGenerator,
        price_data: Dict[str, pd.DataFrame],
        spy_prices: pd.DataFrame,
        tickers: List[str],
    ):
        self.cfg        = config
        self.sig        = signal_gen
        self.price_data = price_data
        self.spy_prices = spy_prices
        self.tickers    = sorted(tickers)
        self._qvm_data: Optional[dict] = None          # cached across run() calls
        self._trading_days_cache: Optional[pd.DatetimeIndex] = None

    def _get_close(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        df = self.price_data.get(ticker)
        if df is None or "Close" not in df.columns:
            return None
        try:
            v = df["Close"].asof(date)
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    def _trading_days(self) -> pd.DatetimeIndex:
        if self._trading_days_cache is not None:
            return self._trading_days_cache
        start = pd.Timestamp(self.cfg.START_DATE)
        end   = pd.Timestamp(self.cfg.END_DATE)
        idx   = self.spy_prices.index
        self._trading_days_cache = idx[(idx >= start) & (idx <= end)]
        return self._trading_days_cache

    def run(self, mode: str = "qvm") -> dict:
        """
        mode: 'qvm'      — QVM rotation only
              'combined' — QVM rotation + crash-tier overrides
        """
        top_n        = self.cfg.QVM_TOP_N
        portfolio    = Portfolio(self.cfg.INITIAL_CAPITAL, top_n)
        trading_days = self._trading_days()

        logger.info(f"Running QVM simulation ({mode}) over {len(trading_days)} trading days...")

        if self._qvm_data is None:
            self.sig.precompute_sector_momentum()
            self._qvm_data = self.sig.precompute_qvm_watchlists(
                self.tickers, trading_days, top_n=top_n
            )

        qvm_data    = self._qvm_data
        qvm_dates   = sorted(qvm_data.keys())

        current_watchlist:   set           = set()
        current_qvm_top10:   List[str]     = []
        current_qvm_scores:  pd.Series     = pd.Series(dtype=float)
        wl_idx = 0
        quarterly_stats: List[dict] = []
        signals_this_quarter = 0
        signal_counts: Dict[str, int] = {}
        max_hold_td = pd.Timedelta(days=365 * self.cfg.MAX_HOLD_YEARS)

        # ---- Regime filter pre-computation ----
        regime_enabled = self.cfg.REGIME_FILTER_ENABLED
        if regime_enabled:
            spy_ema_200 = self.spy_prices["Close"].ewm(span=200, adjust=False).mean()
            spy_close_td = self.spy_prices["Close"].reindex(trading_days).ffill()
            spy_ema_td   = spy_ema_200.reindex(trading_days).ffill()
            bear_flags   = spy_close_td < spy_ema_td
        in_bear_regime = False

        for date in tqdm(trading_days, desc=f"QVM ({mode})"):

            # ---- Trailing stop: QVM positions only (fires before quarterly rebalance) ----
            todays_prices = {t: self._get_close(t, date) for t in list(portfolio.positions)}
            trailing_exits = portfolio.check_trailing_stops(
                todays_prices, self.cfg.TRAILING_STOP_PCT, only_tiers={"qvm"},
                cfg=self.cfg, price_data=self.price_data, date=date,
            )
            for ticker, reason in trailing_exits:
                price = todays_prices.get(ticker)
                if price:
                    portfolio.sell(ticker, date, price, reason)
                    logger.debug(f"TRAILING STOP {ticker} on {date.date()} @ {price:.2f}")

            # ---- Regime filter ----
            if regime_enabled:
                is_bear_today = bool(bear_flags.at[date])

                if is_bear_today and not in_bear_regime:
                    # Bear crossover: sell bottom 50% of QVM positions (worst return first)
                    in_bear_regime = True
                    qvm_pos = [(t, p) for t, p in portfolio.positions.items() if p.tier == "qvm"]
                    n_sell = len(qvm_pos) // 2
                    if n_sell > 0:
                        prices_now = {t: self._get_close(t, date) for t, _ in qvm_pos}
                        ranked = sorted(
                            qvm_pos,
                            key=lambda x: (prices_now.get(x[0]) or x[1].entry_price) / x[1].entry_price,
                        )
                        for t, _ in ranked[:n_sell]:
                            p = prices_now.get(t)
                            if p:
                                portfolio.sell(t, date, p, "regime_bear_entry")
                        logger.info(
                            f"BEAR REGIME {date.date()}: sold {n_sell}/{len(qvm_pos)} QVM positions"
                        )

                elif not is_bear_today and in_bear_regime:
                    # Bull crossover: redeploy cash into QVM top list
                    in_bear_regime = False
                    redeployed = 0
                    for ticker in current_qvm_top10:
                        if portfolio.available_slots <= 0:
                            break
                        if ticker in portfolio.positions:
                            continue
                        price = self._get_close(ticker, date)
                        if price:
                            portfolio.buy(ticker, date, price, tier="qvm")
                            signal_counts[ticker] = signal_counts.get(ticker, 0) + 1
                            redeployed += 1
                    logger.info(
                        f"BULL REGIME {date.date()}: redeployed into {redeployed} QVM positions"
                    )

            # ---- Update at quarter boundary + rebalance ----
            while wl_idx < len(qvm_dates) and date >= qvm_dates[wl_idx]:
                effective = qvm_dates[wl_idx]
                entry     = qvm_data[effective]

                if quarterly_stats:
                    quarterly_stats[-1]["entry_signals"] = signals_this_quarter

                current_watchlist  = entry["watchlist"]
                current_qvm_top10  = entry["top_n"]
                # Use eval-date scores for ranking held positions during crash displacement
                current_qvm_scores = self.sig.compute_qvm_scores(
                    current_watchlist, entry["eval_date"]
                )
                signals_this_quarter = 0
                quarterly_stats.append({
                    "quarter_start":  effective,
                    "watchlist_size": len(current_watchlist),
                    "evaluated":      entry["n_eval"],
                    "entry_signals":  0,
                })

                # QVM REBALANCE
                crash_tickers = {t for t, p in portfolio.positions.items() if p.tier == "crash"}
                target_qvm    = [t for t in current_qvm_top10 if t not in crash_tickers]

                # Sell QVM positions that fell out of target list
                for ticker in list(portfolio.positions):
                    pos = portfolio.positions.get(ticker)
                    if pos is None or pos.tier != "qvm":
                        continue
                    if ticker not in target_qvm:
                        price = self._get_close(ticker, date)
                        if price:
                            portfolio.sell(ticker, date, price, "qvm_rebalance")
                            signals_this_quarter += 1

                # Buy target QVM stocks — sector-cap-aware, iterate full ranked list
                # so we can skip sector-capped top picks and take lower-ranked alternatives
                if not in_bear_regime:
                    ranked_candidates = [
                        t for t in current_qvm_scores.index
                        if t not in crash_tickers and t not in portfolio.positions
                    ]
                    max_sect = self.cfg.MAX_SECTOR_PCT
                    sector_map = self.sig.sector_map
                    for ticker in ranked_candidates:
                        if portfolio.available_slots <= 0:
                            break
                        if ticker in portfolio.positions:
                            continue
                        price = self._get_close(ticker, date)
                        if not price:
                            continue
                        # Sector cap check
                        total_eq  = portfolio.cash + portfolio.position_capital()
                        slot_size = total_eq / top_n
                        sector    = sector_map.get(ticker, "Unknown")
                        sect_val  = sum(
                            p.shares * (self._get_close(t, date) or p.entry_price)
                            for t, p in portfolio.positions.items()
                            if sector_map.get(t, "Unknown") == sector
                        )
                        if total_eq > 0 and (sect_val + slot_size) / total_eq > max_sect:
                            continue   # this sector is full — try next candidate
                        portfolio.buy(ticker, date, price, tier="qvm")
                        signal_counts[ticker] = signal_counts.get(ticker, 0) + 1
                        signals_this_quarter += 1

                logger.info(
                    f"QVM rebalance {effective.date()}: "
                    f"{len(target_qvm)} target | "
                    f"{sum(1 for p in portfolio.positions.values() if p.tier == 'qvm')} qvm | "
                    f"{len(crash_tickers)} crash"
                )
                wl_idx += 1

            # ---- Check crash-tier exits (combined mode) ----
            if mode == "combined":
                to_exit: List[Tuple[str, str]] = []
                for ticker, pos in portfolio.positions.items():
                    if pos.tier != "crash":
                        continue
                    price = self._get_close(ticker, date)
                    if price is None:
                        continue
                    if date - pos.entry_date >= max_hold_td:
                        to_exit.append((ticker, "max_hold_3yr"))
                        continue
                    should_exit, reason = self.sig.check_exit(ticker, date, pos.entry_date)
                    if should_exit:
                        to_exit.append((ticker, reason))

                for ticker, reason in to_exit:
                    price = self._get_close(ticker, date)
                    if price:
                        portfolio.sell(ticker, date, price, reason)
                        logger.debug(f"CRASH EXIT {ticker} on {date.date()} – {reason}")

            # ---- Scan for crash entries (combined mode) ----
            if mode == "combined" and current_watchlist:
                for ticker in self.tickers:
                    if ticker not in current_watchlist:
                        continue
                    price = self._get_close(ticker, date)
                    if not price:
                        continue
                    try:
                        passed, _ = self.sig.check_entry(ticker, date)
                    except Exception:
                        continue
                    if not passed:
                        continue

                    if ticker in portfolio.positions:
                        # Already held as QVM → upgrade tier, exempt from rebalance
                        pos = portfolio.positions[ticker]
                        if pos.tier == "qvm":
                            pos.tier = "crash"
                            logger.debug(f"CRASH UPGRADE {ticker} on {date.date()}")
                    else:
                        # Not held — displace lowest-scored QVM holding if portfolio full
                        if portfolio.available_slots <= 0:
                            qvm_holdings = [
                                (t, p) for t, p in portfolio.positions.items()
                                if p.tier == "qvm"
                            ]
                            if not qvm_holdings:
                                continue

                            def _score(t: str) -> float:
                                if t in current_qvm_scores.index:
                                    return float(current_qvm_scores[t])
                                return -1.0   # unscored = worst

                            worst_t = min(qvm_holdings, key=lambda x: _score(x[0]))[0]
                            worst_price = self._get_close(worst_t, date)
                            if worst_price:
                                portfolio.sell(worst_t, date, worst_price, "crash_displaced_qvm")

                        # Sector cap check for crash entry
                        _sector_map  = self.sig.sector_map
                        _max_sect    = self.cfg.MAX_SECTOR_PCT
                        _sector      = _sector_map.get(ticker, "Unknown")
                        _total_eq    = portfolio.cash + portfolio.position_capital()
                        _slot_size   = _total_eq / portfolio.max_positions
                        _sect_val    = sum(
                            p.shares * (self._get_close(t, date) or p.entry_price)
                            for t, p in portfolio.positions.items()
                            if _sector_map.get(t, "Unknown") == _sector
                        )
                        if _total_eq > 0 and (_sect_val + _slot_size) / _total_eq > _max_sect:
                            # Sell weakest same-sector QVM position to make room
                            same_sect_qvm = [
                                (t, p) for t, p in portfolio.positions.items()
                                if _sector_map.get(t, "Unknown") == _sector and p.tier == "qvm"
                            ]
                            if same_sect_qvm:
                                def _ret(item):
                                    t, p = item
                                    cp = self._get_close(t, date) or p.entry_price
                                    return cp / p.entry_price
                                weakest_t, _ = min(same_sect_qvm, key=_ret)
                                wp = self._get_close(weakest_t, date)
                                if wp:
                                    portfolio.sell(weakest_t, date, wp, "sector_cap_displaced")
                                # Recompute after sell
                                _total_eq = portfolio.cash + portfolio.position_capital()
                                _sect_val = sum(
                                    p.shares * (self._get_close(t, date) or p.entry_price)
                                    for t, p in portfolio.positions.items()
                                    if _sector_map.get(t, "Unknown") == _sector
                                )
                            # Reduce size to fit within cap (or skip if no headroom)
                            headroom = _max_sect * _total_eq - _sect_val
                            if headroom <= 0 or _total_eq <= 0:
                                continue
                            _size_pct = min(headroom / _total_eq, 1.0 / portfolio.max_positions)
                            portfolio.buy(ticker, date, price, tier="crash", size_pct=_size_pct)
                        else:
                            portfolio.buy(ticker, date, price, tier="crash")
                        signal_counts[ticker] = signal_counts.get(ticker, 0) + 1
                        signals_this_quarter += 1
                        logger.debug(f"CRASH ENTRY {ticker} on {date.date()} @ {price:.2f}")

            # ---- Record daily equity ----
            all_prices = {t: self._get_close(t, date) for t in portfolio.positions}
            portfolio.mark_open_positions(date, all_prices)

        # ---- Close all positions at backtest end ----
        final_date = trading_days[-1]
        for ticker in list(portfolio.positions.keys()):
            price = self._get_close(ticker, final_date)
            if price:
                portfolio.sell(ticker, final_date, price, "backtest_end")

        if quarterly_stats:
            quarterly_stats[-1]["entry_signals"] = signals_this_quarter

        equity_series = pd.Series(portfolio.equity_history).sort_index()
        spy_close     = self.spy_prices["Close"].reindex(trading_days).ffill()
        spy_equity    = (spy_close / spy_close.iloc[0]) * self.cfg.INITIAL_CAPITAL

        return {
            "mode":            mode,
            "portfolio":       portfolio,
            "equity_series":   equity_series,
            "spy_equity":      spy_equity,
            "spy_close":       spy_close,
            "trades":          portfolio.trades,
            "signal_counts":   signal_counts,
            "trading_days":    trading_days,
            "quarterly_stats": quarterly_stats,
        }
