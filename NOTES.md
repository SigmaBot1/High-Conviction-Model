# High-Conviction Stock Model — Phase 1 Backtest

## Quick Start

```bash
pip install -r requirements.txt

# Full S&P 500 backtest (2014-2024)
python main.py

# Fast smoke-test with 50 random tickers
python main.py --fast

# Specific tickers + date range
python main.py --tickers NVDA AAPL MSFT --start 2022-01-01 --end 2024-12-31

# Force re-download data
python main.py --refresh-prices --refresh-fundamentals
```

## Diagnostic Tool

```bash
# Show filter breakdown for one ticker on one date
python diagnose.py --ticker NVDA --date 2023-10-01

# Check all 12 default tickers on a specific date
python diagnose.py --date 2022-10-28

# Find all dates when the fear regime was active
python diagnose.py --scan-fear-dates
```

## File Structure

```
high-conviction-model/
├── main.py           Entry point
├── config.py         All model parameters (edit thresholds here)
├── universe.py       S&P 500 constituent list (Wikipedia + fallback)
├── data_loader.py    Download + cache yfinance data
├── features.py       Price & fundamental feature computation
├── signals.py        All 12 filter implementations
├── backtest.py       Portfolio simulation engine
├── reporting.py      Metrics, charts, CSV output
├── diagnose.py       Per-ticker filter breakdown tool
├── data_cache/       Cached price + fundamental data (auto-created)
└── output/           Results: equity_curve.png, trade_log.csv, metrics.csv
```

## The 12 Filters

| # | Filter | Category | Notes |
|---|--------|----------|-------|
| 1 | Revenue growth >15% YoY | Quality | Last 2 periods |
| 2 | Gross margin >40%, not deteriorating | Quality | |
| 3 | ROIC >15% (NOPAT/Invested Capital) | Quality | |
| 4 | D/E ratio <1 | Quality | |
| 5 | FCF positive (or 24+ months runway) | Quality | |
| 6 | PEG <2 (profitable) or P/S <15 + growth >30% | Quality/Valuation | |
| 7 | FCF yield >3% (FCF/EV) | Valuation | |
| 8 | Fear regime: VIX >25 OR SPY >10% below 200d MA OR SPY >10% drawdown | Entry window | 2 of 3 must be true |
| 9 | Price >1.5σ below 12m mean OR >20% below 52w high | Entry window | Either triggers |
| 10 | 12-1 momentum NOT in bottom decile of GICS sector | Entry window | Negative screen |
| 11 | P/S in bottom quartile of own 3-year range | Timing | |
| 12 | Price at or above 50-day EMA | Timing | Stabilisation |

## Known Limitations

### 1. Survivorship Bias (Major)
The universe is the **current** S&P 500 constituent list. Stocks removed before 2024 
(often after poor performance) are not tested. This inflates returns. A production 
system should use point-in-time historical constituent lists (Compustat, Sharadar).

### 2. Fundamental Data Coverage
yfinance provides:
- **Quarterly data**: last ~4-5 quarters (i.e., only ~2024-2025 data)
- **Annual data**: last ~4-5 years (back to ~2020-2022 for most tickers)

The model automatically falls back to annual data when quarterly isn't available.
Effective coverage for quality filters starts around **2022-2023** for most tickers.
Quality filter signals before that date are skipped (treated as FAIL).

**Impact**: The "10-year" backtest effectively tests only the 2022-2024 period for 
fundamental-dependent signals. Price-based signals (filters 8-12) work for the full 
10-year window.

### 3. Shares Outstanding (Minor)
Market cap is estimated as `current_shares_outstanding × historical_price`.
yfinance returns current (not historical) share counts. Split-adjusted prices 
are used so this is mostly correct, but buybacks/issuances cause small errors.

### 4. Look-Ahead in P/S History (Minor)
Filter 11 builds a 3-year P/S history using the current share count. Very slight 
look-ahead for companies that significantly changed their share count.

## Upgrading Data Quality

To fix the fundamental data coverage problem, replace the yfinance fundamental
download with one of these sources:

| Source | Coverage | Cost | Notes |
|--------|----------|------|-------|
| **Sharadar (Quandl)** | 30 years | ~$50/mo | Best for backtesting |
| **SimFin** | 10 years | Free tier | Bulk downloads available |
| **FMP** (Financial Modeling Prep) | 10 years | Free (250 req/day) or ~$15/mo | Needs API key |
| **SEC EDGAR** | 10+ years | Free | Complex to parse |

The `DataLoader` in `data_loader.py` is designed to be swapped out. Implement a 
new `download_fundamentals()` method that returns the same dict format and the 
rest of the engine works unchanged.

## Phase 2: Live Daily Scanner

The codebase is modular for Phase 2. Key changes needed:
1. Replace `BacktestEngine` with a daily scan function
2. Call `signal_gen.check_entry(ticker, today)` for each universe ticker
3. Call `signal_gen.check_exit(ticker, today, entry_date)` for each position
4. Connect to broker API for execution (e.g., Interactive Brokers, Alpaca)

The `SignalGenerator`, `FundamentalAccessor`, and all filter logic are reusable 
as-is in Phase 2.
