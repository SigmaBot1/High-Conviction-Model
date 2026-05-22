"""
High-Conviction Stock Model – Phase 1 Backtest
================================================
Entry point.  Run with:

    python main.py

Optional flags:
    --refresh-prices       Force re-download of all price data
    --refresh-fundamentals Force re-download of all fundamental data
    --tickers AAPL MSFT    Test specific tickers only (overrides universe)
    --start 2018-01-01     Override backtest start date
    --end   2024-12-31     Override backtest end date
    --fast                 Use a 50-ticker random sample (quick smoke-test)
    --scope sp500          Override UNIVERSE_SCOPE (sp500 / sp400 / sp500+sp400)
    --no-universe-compare  Skip the sp500 vs sp500+sp400 side-by-side comparison
"""
import argparse
import logging
import sys
import time
from dataclasses import replace as cfg_replace
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from config import Config
from universe import get_sp500_universe, get_universe, get_sector_map
from data_loader import DataLoader
from features import compute_price_features, compute_spy_features
from signals import SignalGenerator
from backtest import BacktestEngine, QVMBacktestEngine
from reporting import Reporter, FinalComparisonReporter


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="High-Conviction Backtest")
    p.add_argument("--refresh-prices",       action="store_true")
    p.add_argument("--refresh-fundamentals", action="store_true")
    p.add_argument("--tickers",              nargs="+", default=None)
    p.add_argument("--start",                default=None)
    p.add_argument("--end",                  default=None)
    p.add_argument("--fast",                 action="store_true",
                   help="Sample 50 random tickers for a quick smoke-test")
    p.add_argument("--scope",                default=None,
                   help="Override UNIVERSE_SCOPE (sp500 / sp400 / sp500+sp400)")
    p.add_argument("--no-universe-compare",  action="store_true",
                   help="Skip the universe expansion comparison run")
    return p.parse_args()


def banner(cfg: Config, n_tickers: int, scope: str):
    print()
    print("=" * 65)
    print("  HIGH-CONVICTION STOCK MODEL  –  PHASE 1 BACKTEST")
    print("=" * 65)
    print(f"  Period:        {cfg.START_DATE} to {cfg.END_DATE}")
    print(f"  Benchmark:     {cfg.BENCHMARK_TICKER}")
    print(f"  Universe:      {n_tickers} tickers  ({scope})")
    print(f"  Max positions: {cfg.MAX_POSITIONS}")
    print(f"  Capital:       ${cfg.INITIAL_CAPITAL:,.0f}")
    print()
    print("  WARNING: SURVIVORSHIP BIAS")
    print("     Using current index constituent lists.  Stocks removed")
    print("     from the index before 2024 are NOT included.  Returns")
    print("     are likely overstated compared to a live implementation.")
    print()
    print("  NOTE: FUNDAMENTAL DATA COVERAGE")
    print("     yfinance provides ~4-8 quarters of historical financials.")
    print("     Effective signal coverage improves from ~2020 onwards.")
    print("=" * 65)
    print()


def download_market_data(cfg, loader, tickers, args):
    all_tickers = tickers + [cfg.BENCHMARK_TICKER, "^VIX"]

    print(f"[1/4] Downloading price data for {len(tickers)} universe tickers + SPY + VIX...")
    price_data = loader.download_prices(
        all_tickers,
        start=cfg.START_DATE,
        end=cfg.END_DATE,
        force_refresh=args.refresh_prices,
    )
    print(f"      Got price data for {len(price_data)} tickers.")

    print(f"\n[2/4] Downloading fundamental data for {len(tickers)} tickers...")
    print(f"      (Cached tickers load instantly; new tickers download now.)")
    fund_data = loader.download_fundamentals(
        tickers,
        force_refresh=args.refresh_fundamentals,
    )
    print(f"      Got fundamental data for {len(fund_data)} tickers.")

    spy_prices = price_data.get(cfg.BENCHMARK_TICKER)
    vix_prices = price_data.get("^VIX")
    if spy_prices is None:
        print("ERROR: Could not download SPY data.  Exiting.")
        sys.exit(1)

    return price_data, fund_data, spy_prices, vix_prices


def build_features(cfg, tickers, price_data, spy_prices):
    print("\n[3/4] Computing price features...")
    price_features = {}
    skipped = 0
    for t in tickers:
        df = price_data.get(t)
        if df is None or len(df) < 60:
            skipped += 1
            continue
        feat = compute_price_features(df)
        if len(feat) > 0:
            price_features[t] = feat
    spy_feat = compute_spy_features(spy_prices)
    print(f"      Features computed for {len(price_features)} tickers "
          f"({skipped} skipped – insufficient history).")
    return price_features, spy_feat


def _run_qvm_variant(
    cfg, variant, tickers, price_features, spy_feat,
    price_data, spy_prices, vix_prices, fund_data, sector_map,
    label="",
):
    """Run QVM-only and Combined for one ticker list + scoring variant."""
    variant_cfg = cfg_replace(cfg, QVM_VARIANT=variant)
    # Build sector_map scoped to this ticker set
    scoped_sector = {t: sector_map[t] for t in tickers if t in sector_map}
    sig = SignalGenerator(
        config=variant_cfg,
        price_features={t: price_features[t] for t in tickers if t in price_features},
        spy_features=spy_feat,
        vix_prices=vix_prices,
        fundamentals={t: fund_data[t] for t in tickers if t in fund_data},
        sector_map=scoped_sector,
    )
    engine = QVMBacktestEngine(
        config=variant_cfg,
        signal_gen=sig,
        price_data=price_data,
        spy_prices=spy_prices,
        tickers=tickers,
    )
    tag = f"[{label}]" if label else f"[{variant}]"
    print(f"  → {tag} Running 'qvm' mode...")
    qvm_r  = engine.run(mode="qvm")
    print(f"  → {tag} Running 'combined' mode...")
    comb_r = engine.run(mode="combined")
    return {"qvm": qvm_r, "combined": comb_r}


def run_all_backtests(
    cfg, tickers, price_features, spy_feat,
    price_data, spy_prices, vix_prices,
    fund_data, sector_map,
    sp500_tickers=None,
    run_universe_compare=True,
):
    """
    Run backtests.  Always runs:
      crash    — crash-tier only (for watchlist funnel report)
      sp900 composite QVM + Combined  (using full expanded universe)

    When run_universe_compare=True AND sp500_tickers is provided, also runs:
      sp500 composite QVM + Combined  (for side-by-side universe comparison)

    Returns dict of all result dicts.
    """
    n_modes = 5 if (run_universe_compare and sp500_tickers) else 3
    print(f"\n[4/4] Running backtest simulation ({n_modes} modes)...")
    print(f"      Universe: {len(tickers)} tickers total"
          + (f"  |  SP500 subset: {len(sp500_tickers)}" if sp500_tickers else ""))
    print()

    # --- Crash-only (full universe, composite config) ---
    composite_cfg = cfg_replace(cfg, QVM_VARIANT="composite")
    scoped_sector = {t: sector_map[t] for t in tickers if t in sector_map}
    sig_crash = SignalGenerator(
        config=composite_cfg,
        price_features={t: price_features[t] for t in tickers if t in price_features},
        spy_features=spy_feat,
        vix_prices=vix_prices,
        fundamentals={t: fund_data[t] for t in tickers if t in fund_data},
        sector_map=scoped_sector,
    )
    crash_engine = BacktestEngine(
        config=composite_cfg,
        signal_gen=sig_crash,
        price_data=price_data,
        spy_prices=spy_prices,
        tickers=tickers,
    )
    print("  → Running 'crash' mode (full universe)...")
    crash_results = crash_engine.run()

    results = {"crash": crash_results}

    # --- Full expanded universe (composite scoring) ---
    full = _run_qvm_variant(
        cfg, "composite", tickers,
        price_features, spy_feat, price_data, spy_prices, vix_prices,
        fund_data, sector_map,
        label=f"SP{len(tickers)} composite",
    )
    uni_key = "sp900" if len(tickers) > 600 else "sp500"
    results[f"{uni_key}_qvm"]      = full["qvm"]
    results[f"{uni_key}_combined"] = full["combined"]

    # --- SP500 subset (composite scoring) for side-by-side comparison ---
    if run_universe_compare and sp500_tickers:
        sp5 = _run_qvm_variant(
            cfg, "composite", sp500_tickers,
            price_features, spy_feat, price_data, spy_prices, vix_prices,
            fund_data, sector_map,
            label="SP500 composite",
        )
        results["sp500_qvm"]      = sp5["qvm"]
        results["sp500_combined"] = sp5["combined"]

    return results


def main():
    setup_logging()
    args = parse_args()
    cfg  = Config()

    if args.start:
        cfg.START_DATE = args.start
    if args.end:
        cfg.END_DATE = args.end
    if args.scope:
        cfg.UNIVERSE_SCOPE = args.scope

    scope = cfg.UNIVERSE_SCOPE

    # ---- Universe ----
    if args.tickers:
        universe     = [(t, "Unknown") for t in args.tickers]
        sp500_subset = None
        run_uni_cmp  = False
    else:
        # Load the full scope universe (may include sp400)
        universe = get_universe(scope)
        # For comparison: also track the sp500-only subset
        if scope == "sp500+sp400" and not args.no_universe_compare:
            sp500_list   = get_sp500_universe()
            sp500_set    = {t for t, _ in sp500_list}
            sp500_subset = [t for t, _ in universe if t in sp500_set]
            run_uni_cmp  = True
        else:
            sp500_subset = None
            run_uni_cmp  = False

    if not universe:
        print("ERROR: Empty universe.  Check network connection.")
        sys.exit(1)

    if args.fast:
        import random
        random.seed(42)
        universe = random.sample(universe, min(50, len(universe)))
        sp500_subset = None
        run_uni_cmp  = False
        print(f"--fast mode: sampled {len(universe)} tickers.")

    tickers    = [t for t, _ in universe]
    sector_map = get_sector_map(universe)

    banner(cfg, len(tickers), scope)

    # ---- API key info ----
    if not cfg.FMP_API_KEY and not cfg.SIMFIN_API_KEY:
        print("=" * 65)
        print("  NOTE: No FMP_API_KEY or SIMFIN_API_KEY set.")
        print("  Using yfinance (~4-8 quarters). Signals sparse before 2022.")
        print("=" * 65)
        print()
    elif cfg.SIMFIN_API_KEY:
        print(f"  SimFin API key found.  (2,000 req/day free tier)")
        print()

    # ---- Data ----
    loader = DataLoader(cfg)
    price_data, fund_data, spy_prices, vix_prices = download_market_data(
        cfg, loader, tickers, args
    )

    # ---- Features ----
    price_features, spy_feat = build_features(cfg, tickers, price_data, spy_prices)

    eligible = [t for t in tickers if t in price_features and t in fund_data]
    print(f"\n      Eligible tickers (price + fundamentals): {len(eligible)}")
    if sp500_subset:
        sp500_eligible = [t for t in sp500_subset if t in price_features and t in fund_data]
        print(f"      Eligible SP500 subset:                  {len(sp500_eligible)}")
    else:
        sp500_eligible = None

    if not eligible:
        print("ERROR: No eligible tickers.  Try --refresh-prices and --refresh-fundamentals.")
        sys.exit(1)

    # ---- Backtest ----
    t0 = time.time()
    all_results = run_all_backtests(
        cfg, eligible, price_features, spy_feat,
        price_data, spy_prices, vix_prices,
        fund_data, sector_map,
        sp500_tickers=sp500_eligible,
        run_universe_compare=run_uni_cmp,
    )
    elapsed = time.time() - t0
    print(f"\n      All simulations complete in {elapsed:.1f}s")

    # ---- Crash-only summary ----
    crash_reporter = Reporter(cfg, all_results["crash"])
    crash_reporter.print_summary()
    crash_reporter.print_watchlist_summary()

    # ---- Universe comparison (SP500 vs expanded) ----
    if run_uni_cmp and "sp500_qvm" in all_results and "sp900_qvm" in all_results:
        uni_results = {
            "sp500_qvm":      all_results["sp500_qvm"],
            "sp500_combined": all_results["sp500_combined"],
            "sp900_qvm":      all_results["sp900_qvm"],
            "sp900_combined": all_results["sp900_combined"],
        }
        uni_labels = {
            "sp500_qvm":      "SP500 QVM",
            "sp500_combined": "SP500 Comb",
            "sp900_qvm":      "SP900 QVM",
            "sp900_combined": "SP900 Comb",
        }
        uni_reporter = FinalComparisonReporter(
            cfg,
            uni_results,
            mode_labels=uni_labels,
            title="UNIVERSE EXPANSION  —  S&P 500 vs S&P 500 + S&P 400  (2014–2024)",
            desc_lines=[
                f"SP500: {len(sp500_eligible)} eligible tickers  |  "
                f"SP900: {len(eligible)} eligible tickers",
                "Both variants: composite value scoring (EBIT yield + P/S pct + FCF yield) · 6m momentum",
            ],
            chart_file="universe_comparison.png",
        )
        uni_reporter.print_comparison()
        uni_reporter.plot_final_chart()
        uni_reporter.save_all()
    else:
        # Single-universe run: just show QVM vs Combined for the composite model
        keys_present = [k for k in ("sp500_qvm", "sp900_qvm") if k in all_results]
        if keys_present:
            base = keys_present[0].replace("_qvm", "")
            single_results = {
                f"{base}_qvm":      all_results[f"{base}_qvm"],
                f"{base}_combined": all_results[f"{base}_combined"],
            }
            # Fall back to 2-column ComparisonReporter behaviour
            from reporting import ComparisonReporter
            comp = ComparisonReporter(cfg, {
                "qvm":      all_results[f"{base}_qvm"],
                "combined": all_results[f"{base}_combined"],
            })
            comp.print_comparison()
            comp.plot_comparison_chart()
            comp.save_all()

    print(f"\nAll outputs written to:  ./{cfg.OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
