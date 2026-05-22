"""
Data downloading and caching layer.

Downloads from yfinance and stores as pickle files.
Re-runs use cached data; call with force_refresh=True to re-download.
"""
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles all data I/O.  Price data and fundamental data are separate caches."""

    def __init__(self, config: Config):
        self.config = config
        self.price_dir = Path(config.CACHE_DIR) / "prices"
        self.fund_dir = Path(config.CACHE_DIR) / "fundamentals"
        self.price_dir.mkdir(parents=True, exist_ok=True)
        self.fund_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Price data
    # ------------------------------------------------------------------ #

    def _price_path(self, ticker: str) -> Path:
        return self.price_dir / f"{ticker.replace('/', '_')}.pkl"

    def load_price(self, ticker: str) -> Optional[pd.DataFrame]:
        p = self._price_path(ticker)
        if p.exists():
            try:
                return pd.read_pickle(p)
            except Exception:
                pass
        return None

    def download_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV price data for all tickers.
        Returns {ticker: DataFrame}.
        Missing / empty tickers are omitted from the result.
        """
        need = []
        result: Dict[str, pd.DataFrame] = {}

        req_start = pd.Timestamp(start)
        req_end   = pd.Timestamp(end)

        for t in tickers:
            if not force_refresh:
                cached = self.load_price(t)
                if cached is not None and len(cached) > 50:
                    cached.index = pd.to_datetime(cached.index).tz_localize(None)
                    # Validate cached data covers the requested period
                    covered = (
                        cached.index[0] <= req_start + pd.Timedelta(days=10)
                        and cached.index[-1] >= req_end - pd.Timedelta(days=10)
                    )
                    if covered:
                        result[t] = cached
                        continue
                    # else fall through to re-download
            need.append(t)

        if not need:
            return result

        logger.info(f"Downloading price data for {len(need)} tickers...")

        # Use yf.download in batches for speed
        BATCH = 100
        for i in tqdm(range(0, len(need), BATCH), desc="Price batches"):
            batch = need[i : i + BATCH]
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                for t in batch:
                    try:
                        if len(batch) == 1:
                            df = raw.copy()
                        else:
                            df = raw[t].dropna(how="all")

                        # yfinance batch downloads may return MultiIndex columns
                        # like (ticker, 'Close'); flatten to plain 'Close'
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(-1)

                        df = df.dropna(how="all")
                        if len(df) < 20:
                            continue

                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        df.to_pickle(self._price_path(t))
                        result[t] = df
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Batch download failed ({e}), falling back to per-ticker...")
                for t in batch:
                    df = self._download_single_price(t, start, end)
                    if df is not None:
                        result[t] = df

            time.sleep(self.config.DOWNLOAD_DELAY_SEC)

        return result

    def _download_single_price(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_DOWNLOAD_RETRIES):
            try:
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, auto_adjust=True)
                if len(df) < 20:
                    return None
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_pickle(self._price_path(ticker))
                return df
            except Exception as e:
                if attempt < self.config.MAX_DOWNLOAD_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.debug(f"Price download failed for {ticker}: {e}")
        return None

    # ------------------------------------------------------------------ #
    # Fundamental data
    # ------------------------------------------------------------------ #

    def _fund_path(self, ticker: str) -> Path:
        return self.fund_dir / f"{ticker.replace('/', '_')}.pkl"

    def load_fundamentals(self, ticker: str) -> Optional[dict]:
        p = self._fund_path(ticker)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def download_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, dict]:
        """
        Download quarterly/annual financials, balance sheet, and cash flow.

        Source priority:
          1. SimFin  — when SIMFIN_API_KEY is set (2,000 req/day free, full history)
          2. FMP     — when FMP_API_KEY is set (250 req/day free, top tickers only)
          3. yfinance — fallback (only ~5 recent quarters; limits backtest depth)

        Returns {ticker: data_dict}.
        """
        if self.config.SIMFIN_API_KEY:
            from simfin_loader import SimFinLoader
            return SimFinLoader(self.config, api_key=self.config.SIMFIN_API_KEY).download_fundamentals(
                tickers, force_refresh=force_refresh
            )

        if self.config.FMP_API_KEY:
            from fmp_loader import FMPLoader
            return FMPLoader(self.config).download_fundamentals(
                tickers, force_refresh=force_refresh
            )

        # ---- yfinance fallback ----
        result: Dict[str, dict] = {}
        need = []

        for t in tickers:
            if not force_refresh:
                cached = self.load_fundamentals(t)
                if cached is not None:
                    result[t] = cached
                    continue
            need.append(t)

        if not need:
            return result

        logger.info(f"Downloading fundamental data for {len(need)} tickers (yfinance)...")

        for t in tqdm(need, desc="Fundamentals"):
            data = self._download_single_fundamentals(t)
            if data is not None:
                result[t] = data
            time.sleep(self.config.DOWNLOAD_DELAY_SEC)

        return result

    def _download_single_fundamentals(self, ticker: str) -> Optional[dict]:
        for attempt in range(self.config.MAX_DOWNLOAD_RETRIES):
            try:
                obj = yf.Ticker(ticker)

                def safe(fn):
                    try:
                        r = fn()
                        return r if (r is not None and not (isinstance(r, pd.DataFrame) and r.empty)) else None
                    except Exception:
                        return None

                data = {
                    "quarterly_income": safe(lambda: obj.quarterly_income_stmt),
                    "quarterly_balance": safe(lambda: obj.quarterly_balance_sheet),
                    "quarterly_cashflow": safe(lambda: obj.quarterly_cashflow),
                    "annual_income": safe(lambda: obj.income_stmt),
                    "annual_balance": safe(lambda: obj.balance_sheet),
                    "annual_cashflow": safe(lambda: obj.cashflow),
                    "info": safe(lambda: obj.info),
                }

                # Only cache if we got at least some data
                if any(v is not None for v in data.values()):
                    with open(self._fund_path(ticker), "wb") as f:
                        pickle.dump(data, f)
                    return data

                return None

            except Exception as e:
                if attempt < self.config.MAX_DOWNLOAD_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.debug(f"Fundamentals download failed for {ticker}: {e}")
        return None
