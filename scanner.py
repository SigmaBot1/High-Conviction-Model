"""
High-Conviction Stock Scanner – Phase 2 daily entry point.
==========================================================
Runs the same quality + entry filters from the backtest engine against
live (or cached) data, then dispatches notifications via notify.py.

Usage
-----
  # Dry-run: print what would be sent, using dummy signal data
  python scanner.py --dry-run

  # Live scan against cached price/fundamental data (no re-download)
  python scanner.py

  # Live scan with fresh data
  python scanner.py --refresh-prices --refresh-fundamentals

  # Test notification delivery with dummy data (actually sends)
  python scanner.py --dry-run --send

Environment variables required for notifications
-------------------------------------------------
  GMAIL_ADDRESS        your.email@gmail.com
  GMAIL_APP_PASSWORD   16-character app password from Google
  SLACK_WEBHOOK_URL    (optional) Slack incoming webhook URL
"""
import argparse
import logging
import os
import sys
from datetime import date as dt_date
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# Ensure we can import from the same directory when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from notify import Notifier, Signal, build_email, build_slack_payload


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy data for --dry-run
# ---------------------------------------------------------------------------

_DUMMY_SIGNALS: List[Signal] = [
    Signal(
        ticker="NVDA",
        scan_date=dt_date.today(),
        price=118.42,
        pct_off_52w_high=0.31,       # 31% below 52-week high
        price_zscore=-1.8,
        ps_ratio=18.2,
        roic=0.47,
        gross_margin=0.745,
        vix=31.5,
        filters={
            "f1_rev_growth":   True,
            "f2_gross_margin": True,
            "f3_roic":         True,
            "f4_de_ratio":     True,
            "f5_fcf":          True,
            "f6_peg_ps":       True,
            "f7_fcf_yield":    True,
            "f8_fear_regime":  True,
            "f9_price_disloc": True,
            "f10_momentum":    True,
            "f11_ps_quartile": True,
            "f12_above_ema50": True,
        },
    ),
    Signal(
        ticker="MSFT",
        scan_date=dt_date.today(),
        price=381.20,
        pct_off_52w_high=0.22,
        price_zscore=-1.6,
        ps_ratio=10.8,
        roic=0.38,
        gross_margin=0.698,
        vix=31.5,
        filters={
            "f1_rev_growth":   True,
            "f2_gross_margin": True,
            "f3_roic":         True,
            "f4_de_ratio":     True,
            "f5_fcf":          True,
            "f6_peg_ps":       True,
            "f7_fcf_yield":    True,
            "f8_fear_regime":  True,
            "f9_price_disloc": True,
            "f10_momentum":    True,
            "f11_ps_quartile": True,
            "f12_above_ema50": True,
        },
    ),
]


# ---------------------------------------------------------------------------
# Signal builder – extracts metrics from live data structures
# ---------------------------------------------------------------------------

def build_signal(
    ticker: str,
    scan_date: pd.Timestamp,
    price_features: Dict[str, pd.DataFrame],
    vix_series: pd.Series,
    fundamentals: Dict[str, dict],
    filter_results: Dict[str, Optional[bool]],
) -> Optional[Signal]:
    """
    Construct a Signal from a ticker that has already passed all 12 filters.

    price_features  : {ticker: feature_df} from compute_price_features()
    vix_series      : VIX close series (pd.Series)
    fundamentals    : {ticker: raw_dict} from data loader
    filter_results  : {filter_key: True | False | None} from SignalGenerator
    """
    feat = price_features.get(ticker)
    if feat is None:
        return None

    try:
        row = feat.asof(scan_date)
    except Exception:
        return None

    price    = row.get("close")
    high_52w = row.get("high_52w")
    zscore   = row.get("price_zscore_12m")

    if price is None or pd.isna(price):
        return None

    pct_off = (
        float((high_52w - price) / high_52w)
        if (high_52w and not pd.isna(high_52w) and high_52w > 0)
        else None
    )

    vix_val = None
    if len(vix_series) > 0:
        v = vix_series.asof(scan_date)
        if pd.notna(v):
            vix_val = float(v)

    # Fundamental metrics (best-effort; None if unavailable)
    ps_ratio = roic_val = gross_margin_val = None
    raw = fundamentals.get(ticker)
    if raw is not None:
        from config import Config as _Config
        from features import FundamentalAccessor

        fa = FundamentalAccessor(raw, lag_days=_Config.FUNDAMENTAL_LAG_DAYS)

        info   = raw.get("info") or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        mc     = float(price) * float(shares) if shares else None

        if mc:
            ps_ratio = fa.ps_ratio(scan_date, mc)

        roic_result = fa.roic(scan_date)
        roic_val    = roic_result

        gm = fa.gross_margin(scan_date)
        if gm and gm[0] is not None:
            gross_margin_val = gm[0]

    return Signal(
        ticker=ticker,
        scan_date=scan_date.date(),
        price=float(price),
        pct_off_52w_high=pct_off,
        price_zscore=float(zscore) if zscore is not None and not pd.isna(zscore) else None,
        ps_ratio=ps_ratio,
        roic=roic_val,
        gross_margin=gross_margin_val,
        vix=vix_val,
        filters=filter_results,
    )


# ---------------------------------------------------------------------------
# Live scan
# ---------------------------------------------------------------------------

def run_live_scan(
    cfg: Config,
    refresh_prices: bool = False,
    refresh_fundamentals: bool = False,
) -> List[Signal]:
    """
    Run a full quality + entry filter scan against cached (or freshly downloaded)
    data.  Returns a list of Signal objects for any tickers that pass all 12 filters.
    """
    from universe import get_sp500_universe, get_sector_map
    from data_loader import DataLoader
    from features import compute_price_features, compute_spy_features
    from signals import SignalGenerator

    watchlist_tickers = _read_watchlist()
    if watchlist_tickers:
        tickers    = watchlist_tickers
        sector_map = {t: "Unknown" for t in tickers}
    else:
        universe   = get_sp500_universe()
        tickers    = [t for t, _ in universe]
        sector_map = get_sector_map(universe)

    today_str = dt_date.today().isoformat()
    loader    = DataLoader(cfg)

    # Use start well before today so we have enough history for features
    price_start = "2020-01-01"

    logger.info("Downloading price data...")
    all_tickers = tickers + [cfg.BENCHMARK_TICKER, "^VIX"]
    price_data  = loader.download_prices(
        all_tickers, start=price_start, end=today_str,
        force_refresh=refresh_prices,
    )

    logger.info("Loading fundamental data...")
    fund_data = loader.download_fundamentals(tickers, force_refresh=refresh_fundamentals)

    spy_prices = price_data.get(cfg.BENCHMARK_TICKER)
    vix_prices = price_data.get("^VIX")
    if spy_prices is None:
        logger.error("SPY price data unavailable.")
        return []

    logger.info("Computing features...")
    price_features: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = price_data.get(t)
        if df is not None and len(df) >= 60:
            feat = compute_price_features(df)
            if len(feat) > 0:
                price_features[t] = feat

    spy_feat  = compute_spy_features(spy_prices)
    vix_series = (
        vix_prices["Close"].dropna()
        if vix_prices is not None
        else pd.Series(dtype=float)
    )

    eligible = [t for t in tickers if t in price_features and t in fund_data]
    logger.info(f"{len(eligible)} tickers eligible (price + fundamentals).")

    sig_gen = SignalGenerator(
        config=cfg,
        price_features=price_features,
        spy_features=spy_feat,
        vix_prices=vix_prices,
        fundamentals=fund_data,
        sector_map=sector_map,
    )
    sig_gen.precompute_sector_momentum()

    # Build a same-day watchlist from quality filters
    scan_date = pd.Timestamp(today_str)
    watchlist: Set[str] = set()
    logger.info("Evaluating quality filters (F1-7)...")
    for t in eligible:
        passed, _ = sig_gen._check_quality_filters(t, scan_date)
        if passed:
            watchlist.add(t)
    logger.info(f"Watchlist: {len(watchlist)} tickers passed quality filters.")

    # Check entry filters (F8-12) for watchlist stocks
    signals: List[Signal] = []
    for t in watchlist:
        entry_pass, entry_results = sig_gen.check_entry(t, scan_date, watchlist=None)
        if not entry_pass:
            continue

        # Merge quality + entry filter results for the notification
        _, quality_results = sig_gen._check_quality_filters(t, scan_date)
        all_results = {**quality_results, **entry_results}

        sig = build_signal(
            t, scan_date,
            price_features, vix_series,
            fund_data, all_results,
        )
        if sig is not None:
            signals.append(sig)
            logger.info(f"SIGNAL: {t} @ ${sig.price:.2f}  [{sig.conviction}]")

    logger.info(f"Scan complete. {len(signals)} signal(s) found.")
    return signals


# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

_WATCHLIST_PATH = Path(__file__).parent / "watchlist.txt"


def _read_watchlist() -> List[str]:
    """Return tickers from watchlist.txt, skipping blank lines and comments."""
    if not _WATCHLIST_PATH.exists():
        return []
    lines = _WATCHLIST_PATH.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def run_qvm_rank(cfg: Config, top_n: int = 20) -> None:
    """
    Pull the SP900 quality watchlist (F1-F7 passers), rank by composite QVM
    score, and print the top_n results.
    """
    from universe import get_sp500_universe, get_sector_map
    from data_loader import DataLoader
    from features import compute_price_features, compute_spy_features
    from signals import SignalGenerator

    universe   = get_sp500_universe()
    tickers    = [t for t, _ in universe]
    sector_map = get_sector_map(universe)

    today_str = dt_date.today().isoformat()
    loader    = DataLoader(cfg)

    price_start = "2020-01-01"
    all_tickers = tickers + [cfg.BENCHMARK_TICKER, "^VIX"]
    price_data  = loader.download_prices(all_tickers, start=price_start, end=today_str)
    fund_data   = loader.download_fundamentals(tickers)

    spy_prices = price_data.get(cfg.BENCHMARK_TICKER)
    vix_prices = price_data.get("^VIX")

    price_features: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = price_data.get(t)
        if df is not None and len(df) >= 60:
            feat = compute_price_features(df)
            if len(feat) > 0:
                price_features[t] = feat

    spy_feat = compute_spy_features(spy_prices) if spy_prices is not None else None

    sig_gen = SignalGenerator(
        config=cfg,
        price_features=price_features,
        spy_features=spy_feat,
        vix_prices=vix_prices,
        fundamentals=fund_data,
        sector_map=sector_map,
    )
    sig_gen.precompute_sector_momentum()

    eligible = [t for t in tickers if t in price_features and t in fund_data]
    scan_date = pd.Timestamp(today_str)

    quality_watchlist: Set[str] = set()
    for t in eligible:
        passed, _ = sig_gen._check_quality_filters(t, scan_date)
        if passed:
            quality_watchlist.add(t)

    ranked = sig_gen.rank_watchlist(quality_watchlist, scan_date, top_n=top_n)

    print()
    print(f"  QVM Ranking — {scan_date.date()}  ({len(quality_watchlist)} quality stocks, top {top_n} shown)")
    print(f"  {'Rank':<6}{'Ticker':<10}{'Composite':>10}  {'Value':>8}  {'Momentum':>10}")
    print("  " + "-" * 50)

    for rank, (ticker, composite, value, momentum) in enumerate(ranked, 1):
        print(f"  {rank:<6}{ticker:<10}{composite:>10.3f}  {value:>8.3f}  {momentum:>10.3f}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-Conviction live scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use dummy signal data to test the notification pipeline.",
    )
    p.add_argument(
        "--send",
        action="store_true",
        help=(
            "Actually send notifications even in --dry-run mode. "
            "Useful for testing email/Slack delivery."
        ),
    )
    p.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Force re-download of all price data.",
    )
    p.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Force re-download of all fundamental data.",
    )
    p.add_argument(
        "--notify-on-empty",
        action="store_true",
        help='Send a "no signals today" message when nothing fires.',
    )
    p.add_argument(
        "--qvm-rank",
        action="store_true",
        help="Print top 20 QVM-ranked stocks from the quality watchlist (SP900 universe).",
    )
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def print_dry_run_preview(
    signals: List[Signal],
    notify_on_empty: bool,
) -> None:
    """Print what would be sent without touching SMTP or Slack."""
    print()
    print("=" * 60)
    print("  DRY-RUN PREVIEW  (--send not set; nothing will be sent)")
    print("=" * 60)

    email_result = build_email(signals, notify_on_empty)
    if email_result is None:
        print("\n  EMAIL: would not be sent (no signals, notify_on_empty=False)")
    else:
        subject, text_body, _ = email_result
        print(f"\n  EMAIL SUBJECT:\n    {subject}")
        print("\n  EMAIL BODY (plain text):")
        for line in text_body.splitlines():
            print(f"    {line}")

    slack_payload = build_slack_payload(signals, notify_on_empty)
    if slack_payload is None:
        print("\n  SLACK: would not be sent (no signals, notify_on_empty=False)")
    else:
        print("\n  SLACK PAYLOAD (JSON):")
        import json
        formatted = json.dumps(slack_payload, indent=2)
        for line in formatted.splitlines():
            print(f"    {line}")

    print()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg  = Config()

    if args.qvm_rank:
        run_qvm_rank(cfg)
        return

    notifier = Notifier(notify_on_empty=args.notify_on_empty)

    # ---- Credential check ----
    print()
    print("  Notification channels:")
    if notifier.email_configured:
        print(f"    Email:  {notifier.gmail_address}  [configured]")
    else:
        print("    Email:  NOT configured  (set GMAIL_ADDRESS + GMAIL_APP_PASSWORD)")
    if notifier.slack_configured:
        print("    Slack:  webhook configured")
    else:
        print("    Slack:  NOT configured  (set SLACK_WEBHOOK_URL to enable)")
    print()

    # ---- Signals ----
    if args.dry_run:
        signals = _DUMMY_SIGNALS
        print(f"  [DRY-RUN] Using {len(signals)} dummy signal(s).")
    else:
        signals = run_live_scan(
            cfg,
            refresh_prices=args.refresh_prices,
            refresh_fundamentals=args.refresh_fundamentals,
        )

    # ---- Dispatch ----
    if args.dry_run and not args.send:
        print_dry_run_preview(signals, args.notify_on_empty)
    else:
        if args.dry_run:
            print("  [DRY-RUN + --send] Sending with dummy data...")
        notifier.notify(signals)

    if not args.dry_run:
        n = len(signals)
        if n:
            tickers = ", ".join(s.ticker for s in signals)
            print(f"\n  {n} signal(s) fired today: {tickers}")
        else:
            print("\n  No signals today.")
    print()


if __name__ == "__main__":
    main()
