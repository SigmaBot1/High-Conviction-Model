"""
Performance metrics and output generation.

Produces:
  1. Equity curve chart (model vs SPY)
  2. Summary statistics table
  3. Year-by-year returns table
  4. Full trade log CSV
  5. Console summary
"""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from config import Config
from backtest import Trade

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.03  # 3% annualised, used for Sharpe


class Reporter:
    def __init__(self, config: Config, results: dict):
        self.cfg = config
        self.results = results
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.equity = results["equity_series"]
        self.spy    = results["spy_equity"]
        self.trades: List[Trade] = results["trades"]
        self._metrics: dict = None

    # ================================================================ #
    # Core metrics
    # ================================================================ #

    def _returns(self, series: pd.Series) -> pd.Series:
        return series.pct_change().dropna()

    def _annualised_return(self, series: pd.Series) -> float:
        if len(series) < 2:
            return np.nan
        total = series.iloc[-1] / series.iloc[0] - 1
        years = (series.index[-1] - series.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        return (1 + total) ** (1 / years) - 1

    def _sharpe(self, series: pd.Series) -> float:
        rets = self._returns(series)
        if len(rets) < 20:
            return np.nan
        daily_rf = (1 + RISK_FREE_RATE) ** (1 / 252) - 1
        excess   = rets - daily_rf
        if excess.std() == 0:
            return np.nan
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _max_drawdown(self, series: pd.Series) -> float:
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max
        return float(dd.min())

    def _cagr_by_year(self, series: pd.Series) -> pd.Series:
        """Calendar-year returns."""
        return series.resample("YE").last().pct_change().dropna()

    def compute_metrics(self) -> dict:
        if self._metrics is not None:
            return self._metrics
        eq  = self.equity.reindex(self.spy.index).ffill()
        spy = self.spy

        closed = [t for t in self.trades if t.exit_price is not None and t.pnl is not None and t.exit_reason != "backtest_end"]
        wins   = [t for t in closed if t.pnl > 0]

        ann_model = self._annualised_return(eq)
        ann_spy   = self._annualised_return(spy)

        hold_days = [t.holding_days for t in self.trades if t.holding_days is not None]

        self._metrics = {
            "ann_return_model": ann_model,
            "ann_return_spy":   ann_spy,
            "alpha":            ann_model - ann_spy if not np.isnan(ann_model) and not np.isnan(ann_spy) else np.nan,
            "sharpe_model":     self._sharpe(eq),
            "sharpe_spy":       self._sharpe(spy),
            "max_dd_model":     self._max_drawdown(eq),
            "max_dd_spy":       self._max_drawdown(spy),
            "win_rate":         len(wins) / len(closed) if closed else np.nan,
            "total_trades":     len(self.trades),
            "closed_trades":    len(closed),
            "avg_hold_days":    np.mean(hold_days) if hold_days else np.nan,
        }
        return self._metrics

    # ================================================================ #
    # Console output
    # ================================================================ #

    def print_summary(self):
        m = self.compute_metrics()

        def fmt_pct(v):
            return f"{v*100:.2f}%" if not np.isnan(v) else "N/A"

        def fmt_f(v, decimals=2):
            return f"{v:.{decimals}f}" if not np.isnan(v) else "N/A"

        print()
        print("=" * 60)
        print("  HIGH-CONVICTION MODEL – BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Period:              {self.cfg.START_DATE} → {self.cfg.END_DATE}")
        print(f"  Initial Capital:     ${self.cfg.INITIAL_CAPITAL:,.0f}")
        print()
        print(f"{'METRIC':<35} {'MODEL':>10}  {'SPY':>10}")
        print("-" * 60)
        print(f"{'Annualised Return':<35} {fmt_pct(m['ann_return_model']):>10}  {fmt_pct(m['ann_return_spy']):>10}")
        print(f"{'Sharpe Ratio':<35} {fmt_f(m['sharpe_model']):>10}  {fmt_f(m['sharpe_spy']):>10}")
        print(f"{'Max Drawdown':<35} {fmt_pct(m['max_dd_model']):>10}  {fmt_pct(m['max_dd_spy']):>10}")
        print(f"{'Alpha (ann. excess return)':<35} {fmt_pct(m['alpha']):>10}")
        print()
        print(f"{'Win Rate':<35} {fmt_pct(m['win_rate']):>10}")
        print(f"{'Total Trades (inc. open at end)':<35} {m['total_trades']:>10}")
        print(f"{'Closed Trades (excl. backtest_end)':<35} {m['closed_trades']:>10}")
        print(f"{'Avg Holding Period (days)':<35} {fmt_f(m['avg_hold_days'], 0):>10}")
        print()
        print("WARNING: Survivorship bias - current S&P 500 list used. Results may be overstated.")
        print("NOTE: Fundamental data covers ~4-8 recent quarters for most tickers.")
        print("      Signals before ~2020 may use limited or absent fundamental data.")
        print()

        # Year-by-year table
        self._print_yearly_table()

    def _print_yearly_table(self):
        eq  = self.equity.reindex(self.spy.index).ffill()
        model_yr = self._cagr_by_year(eq)
        spy_yr   = self._cagr_by_year(self.spy)

        all_years = sorted(set(model_yr.index) | set(spy_yr.index))

        print(f"{'YEAR':<8} {'MODEL':>10}  {'SPY':>10}  {'DIFF':>10}")
        print("-" * 44)
        for yr in all_years:
            m_v = model_yr.get(yr, np.nan)
            s_v = spy_yr.get(yr, np.nan)
            diff = m_v - s_v if not (np.isnan(m_v) or np.isnan(s_v)) else np.nan
            yr_label = yr.year if hasattr(yr, "year") else str(yr)[:4]

            def fp(v):
                return f"{v*100:>+.1f}%" if not np.isnan(v) else "  N/A"

            print(f"{yr_label:<8} {fp(m_v):>10}  {fp(s_v):>10}  {fp(diff):>10}")
        print()

    # ================================================================ #
    # Watchlist funnel
    # ================================================================ #

    def print_watchlist_summary(self):
        """Print quarterly watchlist size and entry signal count."""
        quarterly_stats = self.results.get("quarterly_stats", [])
        if not quarterly_stats:
            print("No quarterly watchlist stats available.")
            return

        print()
        print("=" * 65)
        print("  QUARTERLY WATCHLIST FUNNEL")
        print("=" * 65)
        print(f"  {'QUARTER':<12}  {'WATCHLIST':>10}  {'EVALUATED':>10}  {'ENTRIES':>8}")
        print("  " + "-" * 47)

        total_signals = 0
        for row in quarterly_stats:
            qs    = row["quarter_start"]
            q_num = (qs.month - 1) // 3 + 1
            label = f"Q{q_num} {qs.year}"
            wl    = row["watchlist_size"]
            ev    = row["evaluated"]
            sig   = row["entry_signals"]
            total_signals += sig
            print(f"  {label:<12}  {wl:>10}  {ev:>10}  {sig:>8}")

        print("  " + "-" * 47)
        print(f"  {'TOTAL':<12}  {'':>10}  {'':>10}  {total_signals:>8}")
        print()

    # ================================================================ #
    # Charts
    # ================================================================ #

    def plot_equity_curve(self):
        eq  = self.equity.reindex(self.spy.index).ffill()
        spy = self.spy

        fig = plt.figure(figsize=(14, 10))
        gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.4)

        # --- Top: Equity curves ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(eq.index,  eq.values,  label="High-Conviction Model", color="#2196F3", linewidth=1.8)
        ax1.plot(spy.index, spy.values, label="SPY Buy & Hold",          color="#FF9800", linewidth=1.5, alpha=0.85)
        ax1.set_title("High-Conviction Stock Model vs SPY (2014–2024)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # --- Middle: Drawdown ---
        ax2 = fig.add_subplot(gs[1])
        eq_dd  = (eq  - eq.cummax())  / eq.cummax()
        spy_dd = (spy - spy.cummax()) / spy.cummax()
        ax2.fill_between(eq_dd.index,  eq_dd.values  * 100, 0, alpha=0.6, color="#2196F3", label="Model DD")
        ax2.fill_between(spy_dd.index, spy_dd.values * 100, 0, alpha=0.4, color="#FF9800", label="SPY DD")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # --- Bottom: Rolling 1-year alpha ---
        ax3 = fig.add_subplot(gs[2])
        model_ret = eq.pct_change().fillna(0)
        spy_ret   = spy.pct_change().fillna(0)
        rolling_alpha = (model_ret - spy_ret).rolling(252).sum() * 100
        ax3.bar(rolling_alpha.index, rolling_alpha.values,
                color=["#4CAF50" if v >= 0 else "#F44336" for v in rolling_alpha.values],
                width=1, alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("Rolling 1Y Alpha (%)", fontsize=10)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        out = self.output_dir / "equity_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Equity curve saved → {out}")

    # ================================================================ #
    # Trade log
    # ================================================================ #

    def save_trade_log(self):
        if not self.trades:
            print("No trades to save.")
            return

        rows = []
        for t in self.trades:
            rows.append({
                "ticker":       t.ticker,
                "entry_date":   t.entry_date.date() if t.entry_date else None,
                "entry_price":  round(t.entry_price, 4),
                "exit_date":    t.exit_date.date() if t.exit_date else None,
                "exit_price":   round(t.exit_price, 4) if t.exit_price else None,
                "return_pct":   round(t.pnl * 100, 2) if t.pnl is not None else None,
                "pnl_dollars":  round(t.pnl_dollars, 2) if t.pnl_dollars is not None else None,
                "hold_days":    t.holding_days,
                "exit_reason":  t.exit_reason,
            })

        df = pd.DataFrame(rows).sort_values("entry_date")
        out = self.output_dir / "trade_log.csv"
        df.to_csv(out, index=False)
        print(f"Trade log ({len(df)} trades) saved → {out}")

    def save_metrics_csv(self):
        m = self.compute_metrics()
        df = pd.DataFrame([m])
        out = self.output_dir / "metrics.csv"
        df.to_csv(out, index=False)
        print(f"Metrics saved → {out}")


# ======================================================================== #
#  QVM three-way comparison reporter  (SPY / QVM-Only / Combined)
# ======================================================================== #

_MODE_STYLE = {
    "qvm":      {"color": "#2196F3", "label": "QVM Only",       "lw": 1.8},
    "combined": {"color": "#9C27B0", "label": "Combined",       "lw": 1.8},
}
_SPY_STYLE = {"color": "#FF9800", "label": "SPY Buy & Hold", "lw": 1.5, "alpha": 0.85}


class ComparisonReporter:
    """
    Compares SPY buy-and-hold, QVM-only, and Combined (QVM + crash overrides).

    results: {"qvm": result_dict, "combined": result_dict}
    Each result_dict must contain: equity_series, spy_equity, trades.
    """

    def __init__(self, config: Config, results: Dict[str, dict]):
        self.cfg        = config
        self.results    = results          # {"qvm": ..., "combined": ...}
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reporters = {
            mode: Reporter(config, r) for mode, r in results.items()
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _metrics(self, mode: str) -> dict:
        return self._reporters[mode].compute_metrics()

    def _equity(self, mode: str) -> pd.Series:
        r   = self.results[mode]
        spy = r["spy_equity"]
        return r["equity_series"].reindex(spy.index).ffill()

    def _spy_equity(self) -> pd.Series:
        return self.results["qvm"]["spy_equity"]

    def _spy_metrics(self) -> dict:
        return self._metrics("qvm")   # spy_* keys are identical across runs

    # ------------------------------------------------------------------ #
    # Console output
    # ------------------------------------------------------------------ #

    def print_comparison(self):
        modes  = ["qvm", "combined"]
        labels = {"qvm": "QVM Only", "combined": "Combined"}

        def fp(v):
            return f"{v*100:>+.2f}%" if not np.isnan(v) else "   N/A"

        def ff(v, d=2):
            return f"{v:>{d+5}.{d}f}" if not np.isnan(v) else "   N/A"

        width = 14
        header = f"{'METRIC':<35}"
        for m in modes:
            header += f"  {labels[m]:>{width}}"
        header += f"  {'SPY':>{width}}"

        spy_m = self._spy_metrics()

        print()
        print("=" * (35 + (width + 2) * 3))
        print("  QVM vs COMBINED vs SPY  —  BACKTEST COMPARISON  (2014–2024)")
        print("=" * (35 + (width + 2) * 3))
        print(header)
        print("-" * (35 + (width + 2) * 3))

        rows = [
            ("Annualised Return",     [fp(self._metrics(m)["ann_return_model"]) for m in modes],
                                       fp(spy_m["ann_return_spy"])),
            ("Sharpe Ratio",          [ff(self._metrics(m)["sharpe_model"])      for m in modes],
                                       ff(spy_m["sharpe_spy"])),
            ("Max Drawdown",          [fp(self._metrics(m)["max_dd_model"])      for m in modes],
                                       fp(spy_m["max_dd_spy"])),
            ("Alpha (ann.)",          [fp(self._metrics(m)["alpha"])             for m in modes], ""),
            ("Win Rate",              [fp(self._metrics(m)["win_rate"])          for m in modes], ""),
            ("Total Trades",          [f"{self._metrics(m)['total_trades']:>{width}}" for m in modes], ""),
            ("Closed Trades",         [f"{self._metrics(m)['closed_trades']:>{width}}" for m in modes], ""),
            ("Avg Hold (days)",       [ff(self._metrics(m)["avg_hold_days"], 0)  for m in modes], ""),
        ]

        for label, model_vals, spy_val in rows:
            line = f"  {label:<33}"
            for v in model_vals:
                line += f"  {v:>{width}}"
            if spy_val:
                line += f"  {spy_val:>{width}}"
            print(line)

        print()
        self._print_yearly_comparison(modes, labels)

    def _print_yearly_comparison(self, modes: list, labels: dict):
        width = 10

        def fp(v):
            return f"{v*100:>+.1f}%" if not np.isnan(v) else "  N/A"

        yr_data = {}
        for mode in modes:
            eq = self._equity(mode)
            yr_data[mode] = eq.resample("YE").last().pct_change().dropna()

        spy_yr = self._spy_equity().resample("YE").last().pct_change().dropna()
        all_years = sorted(
            set().union(*[set(yr_data[m].index) for m in modes], set(spy_yr.index))
        )

        header = f"  {'YEAR':<6}"
        for m in modes:
            header += f"  {labels[m]:>{width}}"
        header += f"  {'SPY':>{width}}"
        print(header)
        print("  " + "-" * (6 + (width + 2) * (len(modes) + 1)))

        for yr in all_years:
            yr_label = yr.year if hasattr(yr, "year") else str(yr)[:4]
            line = f"  {yr_label:<6}"
            for m in modes:
                v = yr_data[m].get(yr, np.nan)
                line += f"  {fp(v):>{width}}"
            s = spy_yr.get(yr, np.nan)
            line += f"  {fp(s):>{width}}"
            print(line)
        print()

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #

    def plot_comparison_chart(self):
        modes = ["qvm", "combined"]
        spy   = self._spy_equity()

        fig = plt.figure(figsize=(16, 11))
        gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.4)

        # --- Top: Equity curves ---
        ax1 = fig.add_subplot(gs[0])
        for m in modes:
            eq = self._equity(m)
            s  = _MODE_STYLE[m]
            ax1.plot(eq.index, eq.values, label=s["label"], color=s["color"],
                     linewidth=s["lw"])
        ax1.plot(spy.index, spy.values, label=_SPY_STYLE["label"],
                 color=_SPY_STYLE["color"], linewidth=_SPY_STYLE["lw"],
                 alpha=_SPY_STYLE["alpha"])
        ax1.set_title("QVM Only vs Combined vs SPY Buy & Hold  (2014–2024)",
                      fontsize=13, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # --- Middle: Drawdown ---
        ax2 = fig.add_subplot(gs[1])
        for m in modes:
            eq = self._equity(m)
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.fill_between(dd.index, dd.values * 100, 0,
                             alpha=0.45, color=_MODE_STYLE[m]["color"],
                             label=_MODE_STYLE[m]["label"])
        spy_dd = (spy - spy.cummax()) / spy.cummax()
        ax2.fill_between(spy_dd.index, spy_dd.values * 100, 0,
                         alpha=0.25, color=_SPY_STYLE["color"], label="SPY")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.legend(fontsize=9, ncol=3)
        ax2.grid(True, alpha=0.3)

        # --- Bottom: Rolling 1Y alpha vs SPY ---
        ax3 = fig.add_subplot(gs[2])
        spy_ret = spy.pct_change().fillna(0)
        for m in modes:
            eq         = self._equity(m)
            mod_ret    = eq.pct_change().fillna(0)
            roll_alpha = (mod_ret - spy_ret).rolling(252).sum() * 100
            ax3.plot(roll_alpha.index, roll_alpha.values,
                     color=_MODE_STYLE[m]["color"], linewidth=1.2,
                     label=_MODE_STYLE[m]["label"], alpha=0.85)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("Rolling 1Y Alpha (%)", fontsize=10)
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        out = self.output_dir / "comparison_equity_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison chart saved → {out}")

    # ------------------------------------------------------------------ #
    # Per-mode file outputs
    # ------------------------------------------------------------------ #

    def save_all(self):
        for mode, reporter in self._reporters.items():
            trades = reporter.trades
            if trades:
                rows = []
                for t in trades:
                    rows.append({
                        "ticker":      t.ticker,
                        "tier":        t.tier,
                        "entry_date":  t.entry_date.date() if t.entry_date else None,
                        "entry_price": round(t.entry_price, 4),
                        "exit_date":   t.exit_date.date() if t.exit_date else None,
                        "exit_price":  round(t.exit_price, 4) if t.exit_price else None,
                        "return_pct":  round(t.pnl * 100, 2) if t.pnl is not None else None,
                        "pnl_dollars": round(t.pnl_dollars, 2) if t.pnl_dollars is not None else None,
                        "hold_days":   t.holding_days,
                        "exit_reason": t.exit_reason,
                    })
                df = pd.DataFrame(rows).sort_values("entry_date")
                out = self.output_dir / f"trade_log_{mode}.csv"
                df.to_csv(out, index=False)
                print(f"Trade log [{mode}] ({len(df)} trades) saved → {out}")
            else:
                print(f"Trade log [{mode}]: no trades.")

            m         = reporter.compute_metrics()
            m["mode"] = mode
            out = self.output_dir / f"metrics_{mode}.csv"
            pd.DataFrame([m]).to_csv(out, index=False)
            print(f"Metrics [{mode}] saved → {out}")


# ======================================================================== #
#  Configurable multi-version comparison reporter
# ======================================================================== #

# Default styles for the four modes (used when no custom styles supplied)
_DEFAULT_STYLES = [
    {"color": "#90CAF9", "lw": 1.6, "ls": "--"},
    {"color": "#CE93D8", "lw": 1.6, "ls": "--"},
    {"color": "#1565C0", "lw": 2.0, "ls": "-"},
    {"color": "#6A1B9A", "lw": 2.0, "ls": "-"},
]


class FinalComparisonReporter:
    """
    Five-column comparison reporter: four model variants + SPY.

    results      : dict[key → result_dict] — exactly four entries, in display order.
    mode_labels  : dict[key → display_label] — column header for each key.
                   Defaults to the original before/after labels when None.
    title        : main heading line.
    desc_lines   : list of description strings printed below the heading.
    chart_file   : output filename for the equity chart (default "final_comparison.png").
    """

    _DEFAULT_LABELS = {
        "orig_qvm":      "Orig QVM",
        "orig_combined": "Orig Comb",
        "comp_qvm":      "Final QVM",
        "comp_combined": "Final Comb",
    }
    _DEFAULT_TITLE = "BEFORE / AFTER  —  DEFINITIVE COMPARISON  (2014–2024)"
    _DEFAULT_DESC  = [
        "Original: 6m momentum · P/S-only value · ROIC (Tweak-3 baseline)",
        "Final:    6m momentum · composite value (EBIT yield + P/S + FCF yield) · ROIC",
    ]

    def __init__(
        self,
        config: Config,
        results: Dict[str, dict],
        mode_labels: Dict[str, str] = None,
        title: str = None,
        desc_lines: List[str] = None,
        chart_file: str = "final_comparison.png",
    ):
        self.cfg        = config
        self.results    = results
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reporters  = {k: Reporter(config, v) for k, v in results.items()}
        self.mode_labels = mode_labels if mode_labels is not None else self._DEFAULT_LABELS
        self.title       = title       if title       is not None else self._DEFAULT_TITLE
        self.desc_lines  = desc_lines  if desc_lines  is not None else self._DEFAULT_DESC
        self.chart_file  = chart_file
        self._modes      = list(results.keys())   # order preserved (Python 3.7+)

    def _metrics(self, key: str) -> dict:
        return self._reporters[key].compute_metrics()

    def _equity(self, key: str) -> pd.Series:
        r   = self.results[key]
        spy = r["spy_equity"]
        return r["equity_series"].reindex(spy.index).ffill()

    def _spy(self) -> pd.Series:
        return self.results[self._modes[0]]["spy_equity"]

    def _spy_metrics(self) -> dict:
        return self._reporters[self._modes[0]].compute_metrics()

    # ------------------------------------------------------------------ #
    # Console comparison table
    # ------------------------------------------------------------------ #

    def print_comparison(self):
        modes  = self._modes
        labels = self.mode_labels
        W      = 12

        def fp(v):
            return f"{v*100:>+.2f}%" if not np.isnan(v) else "    N/A"

        def ff(v, d=2):
            return f"{v:.{d}f}" if not np.isnan(v) else "  N/A"

        spy_m  = self._spy_metrics()
        border = "=" * (32 + (W + 2) * 5)

        print()
        print(border)
        print(f"  {self.title}")
        for line in self.desc_lines:
            print(f"  {line}")
        print(border)

        header = f"  {'METRIC':<30}"
        for m in modes:
            header += f"  {labels[m]:>{W}}"
        header += f"  {'SPY':>{W}}"
        print(header)
        print("  " + "-" * (30 + (W + 2) * 5))

        def row(label, vals, spy_val=""):
            line = f"  {label:<30}"
            for v in vals:
                line += f"  {v:>{W}}"
            if spy_val:
                line += f"  {spy_val:>{W}}"
            print(line)

        row("Annualised Return",
            [fp(self._metrics(m)["ann_return_model"]) for m in modes],
            fp(spy_m["ann_return_spy"]))
        row("Sharpe Ratio",
            [ff(self._metrics(m)["sharpe_model"]) for m in modes],
            ff(spy_m["sharpe_spy"]))
        row("Max Drawdown",
            [fp(self._metrics(m)["max_dd_model"]) for m in modes],
            fp(spy_m["max_dd_spy"]))
        row("Alpha (ann. vs SPY)",
            [fp(self._metrics(m)["alpha"]) for m in modes])
        row("Win Rate",
            [fp(self._metrics(m)["win_rate"]) for m in modes])
        row("Total Trades",
            [f"{self._metrics(m)['total_trades']:>{W}}" for m in modes])
        row("Avg Hold (days)",
            [f"{self._metrics(m)['avg_hold_days']:>{W}.0f}"
             if not np.isnan(self._metrics(m)["avg_hold_days"]) else f"{'N/A':>{W}}"
             for m in modes])

        print()
        self._print_yearly(modes, labels, W)

    def _print_yearly(self, modes, labels, W):
        def fp(v):
            return f"{v*100:>+.1f}%" if not np.isnan(v) else "  N/A"

        yr_data = {}
        for m in modes:
            eq = self._equity(m)
            yr_data[m] = eq.resample("YE").last().pct_change().dropna()

        spy_yr    = self._spy().resample("YE").last().pct_change().dropna()
        all_years = sorted(
            set().union(*[set(yr_data[m].index) for m in modes], set(spy_yr.index))
        )

        header = f"  {'YEAR':<6}"
        for m in modes:
            header += f"  {labels[m]:>{W}}"
        header += f"  {'SPY':>{W}}"
        print(header)
        print("  " + "-" * (6 + (W + 2) * 5))

        for yr in all_years:
            yr_label = yr.year if hasattr(yr, "year") else str(yr)[:4]
            line = f"  {yr_label:<6}"
            for m in modes:
                v = yr_data[m].get(yr, np.nan)
                line += f"  {fp(v):>{W}}"
            s = spy_yr.get(yr, np.nan)
            line += f"  {fp(s):>{W}}"
            print(line)
        print()

    # ------------------------------------------------------------------ #
    # Five-line equity curve chart
    # ------------------------------------------------------------------ #

    def plot_final_chart(self):
        modes  = self._modes
        labels = self.mode_labels
        spy    = self._spy()

        # Assign a colour/style to each mode from the default palette
        palette = _DEFAULT_STYLES
        mode_styles = {
            m: {**palette[i % len(palette)], "label": labels[m]}
            for i, m in enumerate(modes)
        }

        fig = plt.figure(figsize=(16, 11))
        gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.4)

        # --- Equity curves ---
        ax1 = fig.add_subplot(gs[0])
        for m in modes:
            eq = self._equity(m)
            s  = mode_styles[m]
            ax1.plot(eq.index, eq.values,
                     label=s["label"], color=s["color"],
                     linewidth=s["lw"], linestyle=s["ls"])
        ax1.plot(spy.index, spy.values,
                 label="SPY Buy & Hold", color="#FF9800", linewidth=1.5, alpha=0.85)
        ax1.set_title(self.title, fontsize=12, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=10, ncol=3)
        ax1.grid(True, alpha=0.3)

        # --- Drawdown ---
        ax2 = fig.add_subplot(gs[1])
        for m in modes:
            eq = self._equity(m)
            dd = (eq - eq.cummax()) / eq.cummax()
            s  = mode_styles[m]
            ax2.fill_between(dd.index, dd.values * 100, 0,
                             alpha=0.35, color=s["color"], label=s["label"])
        spy_dd = (spy - spy.cummax()) / spy.cummax()
        ax2.fill_between(spy_dd.index, spy_dd.values * 100, 0,
                         alpha=0.2, color="#FF9800", label="SPY")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.legend(fontsize=8, ncol=5)
        ax2.grid(True, alpha=0.3)

        # --- Rolling 1Y alpha vs SPY (last two modes = the "better" variant) ---
        ax3 = fig.add_subplot(gs[2])
        spy_ret   = spy.pct_change().fillna(0)
        alpha_modes = modes[-2:]   # plot the last two variants (typically the "new" ones)
        for m in alpha_modes:
            eq         = self._equity(m)
            mod_ret    = eq.pct_change().fillna(0)
            roll_alpha = (mod_ret - spy_ret).rolling(252).sum() * 100
            s = mode_styles[m]
            ax3.plot(roll_alpha.index, roll_alpha.values,
                     color=s["color"], linewidth=1.4,
                     label=s["label"], linestyle=s["ls"])
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_ylabel("Rolling 1Y Alpha (%)", fontsize=10)
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        out = self.output_dir / self.chart_file
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison chart saved → {out}")

    # ------------------------------------------------------------------ #
    # Save CSVs
    # ------------------------------------------------------------------ #

    def save_all(self):
        for key, reporter in self._reporters.items():
            if reporter.trades:
                rows = []
                for t in reporter.trades:
                    rows.append({
                        "ticker":      t.ticker,
                        "tier":        t.tier,
                        "entry_date":  t.entry_date.date() if t.entry_date else None,
                        "entry_price": round(t.entry_price, 4),
                        "exit_date":   t.exit_date.date() if t.exit_date else None,
                        "exit_price":  round(t.exit_price, 4) if t.exit_price else None,
                        "return_pct":  round(t.pnl * 100, 2) if t.pnl is not None else None,
                        "pnl_dollars": round(t.pnl_dollars, 2) if t.pnl_dollars is not None else None,
                        "hold_days":   t.holding_days,
                        "exit_reason": t.exit_reason,
                    })
                df  = pd.DataFrame(rows).sort_values("entry_date")
                out = self.output_dir / f"trade_log_{key}.csv"
                df.to_csv(out, index=False)
                print(f"Trade log [{key}] ({len(df)} trades) saved → {out}")
            m         = reporter.compute_metrics()
            m["mode"] = key
            out = self.output_dir / f"metrics_{key}.csv"
            pd.DataFrame([m]).to_csv(out, index=False)
            print(f"Metrics [{key}] saved → {out}")
