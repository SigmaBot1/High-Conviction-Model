"""
Configuration for the High-Conviction Stock Model backtest.
All parameters are centralised here so Phase 2 (live scanner) can import them.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Backtest period
    # ------------------------------------------------------------------ #
    START_DATE: str = "2014-01-01"
    END_DATE: str = "2024-12-31"
    BENCHMARK_TICKER: str = "SPY"
    INITIAL_CAPITAL: float = 1_000_000.0

    # ------------------------------------------------------------------ #
    # Portfolio constraints
    # ------------------------------------------------------------------ #
    MAX_POSITIONS: int = 12

    # ------------------------------------------------------------------ #
    # Quality filters  (Filters 1-7)
    # ------------------------------------------------------------------ #
    # Filter 1 – Revenue growth
    MIN_REVENUE_GROWTH_YOY: float = 0.15        # >15% YoY, last 2 quarters

    # Filter 2 – Gross margin
    MIN_GROSS_MARGIN: float = 0.40              # >40%, not deteriorating

    # Filter 3 – ROIC
    MIN_ROIC: float = 0.15                      # >15% TTM

    # Filter 4 – Debt / Equity
    MAX_DE_RATIO: float = 1.0                   # <1

    # Filter 5 – FCF (positive or 24-month runway for pre-revenue)
    MIN_CASH_RUNWAY_MONTHS: int = 24

    # Filter 6 – PEG / P/S
    MAX_PEG_RATIO: float = 2.0                  # Profitable companies
    MAX_PS_PRE_PROFITABLE: float = 15.0         # Pre-profitable
    MIN_GROWTH_PRE_PROFITABLE: float = 0.30     # >30% rev growth required

    # Filter 7 – FCF Yield
    # Original spec: 3%.  Calibrated value: 1.5%.
    # Rationale: high-growth companies (which pass f1 at 15%+ revenue growth)
    # trade at 30-100x FCF in normal markets.  Even during a 40% market crash
    # they rarely hit 3% FCF yield.  1.5% = ~67x FCF — still selective, and
    # consistent with the intent of buying quality growth at a discount.
    # Set back to 0.03 to restore the original spec.
    MIN_FCF_YIELD: float = 0.015                # >1.5% (FCF / EV)

    # ------------------------------------------------------------------ #
    # Entry window filters  (Filters 8-10)
    # ------------------------------------------------------------------ #
    # Filter 8 – Fear regime (at least 2 of 4 must be True)
    VIX_FEAR_THRESHOLD: float = 25.0
    SPY_BELOW_200D_PCT: float = 0.10            # SPY >10% below 200-day MA
    MARKET_DRAWDOWN_PCT: float = 0.10           # SPY >10% off recent high
    CNN_FG_FEAR_THRESHOLD: float = 35.0         # CNN Fear & Greed score < 35

    # Filter 9 – Price dislocation (either condition triggers pass)
    PRICE_ZSCORE_LOWER: float = -1.5            # >1.5 std devs below 12m mean
    PRICE_BELOW_52W_HIGH_PCT: float = 0.20      # >20% below 52-week high

    # Filter 10 – 12-1 momentum (negative screen)
    MOMENTUM_BOTTOM_DECILE: float = 0.10        # Reject if in bottom 10% of sector

    # ------------------------------------------------------------------ #
    # Timing filters  (Filters 11-12)
    # ------------------------------------------------------------------ #
    PS_HISTORY_YEARS: int = 3                   # P/S range lookback
    PS_QUARTILE_MAX: float = 0.25               # Must be ≤ 25th percentile
    EMA_PERIOD: int = 50                        # 50-day EMA stabilisation check

    # ------------------------------------------------------------------ #
    # Exit thresholds
    # ------------------------------------------------------------------ #
    EXIT_NEG_REVENUE_QUARTERS: int = 2          # Consecutive negative-growth quarters
    EXIT_MIN_GROSS_MARGIN: float = 0.30         # Drop below 30%
    EXIT_MAX_DE_RATIO: float = 2.0              # Exceed D/E of 2
    MAX_HOLD_YEARS: int = 5                     # Time-based fallback

    # ------------------------------------------------------------------ #
    # Data settings
    # ------------------------------------------------------------------ #
    CACHE_DIR: str = "data_cache"
    OUTPUT_DIR: str = "output"
    FUNDAMENTAL_LAG_DAYS: int = 45              # Days after quarter end → filing assumed available
    MAX_DOWNLOAD_RETRIES: int = 3
    DOWNLOAD_DELAY_SEC: float = 0.3             # Polite delay between yfinance calls

    # ------------------------------------------------------------------ #
    # FMP (Financial Modeling Prep) — optional, replaces yfinance fundamentals
    # ------------------------------------------------------------------ #
    # Get a free key at https://site.financialmodelingprep.com/developer/docs
    # Free tier: 250 requests/day, ~5 years of historical data.
    # Set via environment variable:  export FMP_API_KEY=your_key_here
    FMP_API_KEY: str = field(default_factory=lambda: os.environ.get("FMP_API_KEY", ""))
    FMP_DAILY_LIMIT: int = 245                  # Stay 5 under the 250/day hard cap

    # ------------------------------------------------------------------ #
    # SimFin fundamental data (preferred over FMP — 2,000 req/day free tier)
    # Get a free key at https://app.simfin.com/settings
    # Set via environment variable:  export SIMFIN_API_KEY=your_key_here
    SIMFIN_API_KEY: str = field(default_factory=lambda: os.environ.get("SIMFIN_API_KEY", ""))

    # ------------------------------------------------------------------ #
    # QVM rotation engine (Engine 2)
    # ------------------------------------------------------------------ #
    QVM_TOP_N: int = 10                      # Number of stocks to hold at all times

    # ------------------------------------------------------------------ #
    # Risk management
    # ------------------------------------------------------------------ #
    TRAILING_STOP_PCT: float = 0.15          # Fallback stop when method is "fixed"
    TRAILING_STOP_METHOD: str = "atr"        # "fixed" or "atr"
    ATR_STOP_PERIOD: int = 22                # ATR lookback in trading days
    ATR_STOP_MULTIPLIER: float = 3.0         # Stop = peak_close - ATR * multiplier
    STOP_GRACE_DAYS: int = 30                # No stop check for first N days after entry

    # ------------------------------------------------------------------ #
    # Regime filter (QVM exposure reduction in bear markets)
    # ------------------------------------------------------------------ #
    REGIME_FILTER_ENABLED: bool = False      # Disabled: EMA whipsaw degrades performance

    # ------------------------------------------------------------------ #
    # Sector concentration cap
    # ------------------------------------------------------------------ #
    MAX_SECTOR_PCT: float = 0.35             # No single GICS sector > 35% of portfolio

    # ------------------------------------------------------------------ #
    # QVM scoring variant
    # ------------------------------------------------------------------ #
    # "original"  : 50% value (P/S percentile vs own 3yr history)
    #               + 50% momentum (6-month return)
    #               F3 filter = ROIC > 15%
    # "composite" : 33% quality (Piotroski F-Score rank)
    #               + 33% value (EBIT yield + P/S pct + FCF yield composite)
    #               + 33% momentum (12-minus-1-month return)
    #               F3 filter = GP/Assets > 0.33 (Novy-Marx)
    QVM_VARIANT: str = "composite"
    MIN_GP_ASSETS: float = 0.33             # Novy-Marx GP/Assets threshold (composite variant)

    # ------------------------------------------------------------------ #
    # Universe scope
    # ------------------------------------------------------------------ #
    # "sp500"         : S&P 500 large-cap only (~503 tickers)
    # "sp400"         : S&P 400 mid-cap only  (~400 tickers)
    # "sp500+sp400"   : Combined large + mid  (~900 tickers)
    UNIVERSE_SCOPE: str = "sp500+sp400"

    # ------------------------------------------------------------------ #
    # Macro agent integration
    # ------------------------------------------------------------------ #
    # Path to macro_state.json written by the macro agent every Sunday night.
    # When MACRO_AGENT_ENABLED is True, the crash tier (F8) will not activate
    # during RED macro regimes — preventing false activations in sustained
    # bear markets (e.g. 2022 grinding decline where V-shaped recovery never came).
    # Set MACRO_AGENT_ENABLED to False to run the model without this dependency.
    MACRO_STATE_FILE: str = r"C:\Users\mattp\Projects\my-project\Macro Agent\macro_state.json"
    MACRO_AGENT_ENABLED: bool = True
