"""
Monte Carlo Simulation for the High-Conviction Stock Model
===========================================================
Place this file in your high-conviction-model/ folder alongside main.py.
Run with:  python monte_carlo.py

It reads trade_log.csv from output/ and simulates 10,000 alternative
histories by reshuffling the order and timing of your trades.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NUM_SIMULATIONS = 10000
STARTING_CAPITAL = 1_000_000
OUTPUT_DIR = "output"
TRADE_LOG_PATH = os.path.join(OUTPUT_DIR, "trade_log.csv")
NUM_PATHS_TO_PLOT = 500

# ---------------------------------------------------------------------------
# LOAD TRADE LOG
# ---------------------------------------------------------------------------
def load_trades():
    if not os.path.exists(TRADE_LOG_PATH):
        print(f"ERROR: {TRADE_LOG_PATH} not found.")
        print("Run the backtest first:  python main.py")
        sys.exit(1)

    df = pd.read_csv(TRADE_LOG_PATH)

    return_col = None
    for candidate in ['return_pct', 'return_%', 'Return %', 'return', 'pnl_pct', 'Return_Pct', 'profit_pct']:
        if candidate in df.columns:
            return_col = candidate
            break

    if return_col is None:
        print(f"ERROR: Cannot find a return column in {TRADE_LOG_PATH}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    hold_col = None
    for candidate in ['days_held', 'hold_days', 'holding_period', 'Hold Days', 'avg_hold', 'duration']:
        if candidate in df.columns:
            hold_col = candidate
            break

    returns = df[return_col].dropna().values
    if len(returns) == 0:
        print("ERROR: No valid return data found in trade log.")
        sys.exit(1)

    if np.mean(np.abs(returns)) > 1:
        returns = returns / 100.0

    hold_days = None
    if hold_col is not None:
        hold_days = df[hold_col].dropna().values

    print(f"Loaded {len(returns)} trades from {TRADE_LOG_PATH}")
    print(f"Return column: '{return_col}'")
    print(f"Trade returns: min={returns.min():.2%}, max={returns.max():.2%}, "
          f"mean={returns.mean():.2%}, median={np.median(returns):.2%}")
    if hold_days is not None:
        print(f"Avg holding period: {np.mean(hold_days):.0f} days")
    print(f"Win rate: {(returns > 0).mean():.1%}")
    print()

    return returns, hold_days


# ---------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------
def run_monte_carlo(returns, num_sims=NUM_SIMULATIONS):
    n_trades = len(returns)
    final_values = []
    all_equity_curves = []
    max_drawdowns = []
    sharpe_ratios = []
    win_rates = []
    cagrs = []

    active_years = 5.0
    trades_per_year = n_trades / active_years

    for i in range(num_sims):
        sampled_returns = np.random.choice(returns, size=n_trades, replace=True)

        equity = [STARTING_CAPITAL]
        for r in sampled_returns:
            position_size = 0.10
            portfolio_return = r * position_size
            new_value = equity[-1] * (1 + portfolio_return)
            equity.append(new_value)

        equity = np.array(equity)
        final_val = equity[-1]
        final_values.append(final_val)

        total_return = final_val / STARTING_CAPITAL
        cagr = (total_return ** (1 / active_years)) - 1
        cagrs.append(cagr)

        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = drawdowns.min()
        max_drawdowns.append(max_dd)

        wr = (sampled_returns > 0).mean()
        win_rates.append(wr)

        trade_returns_sized = sampled_returns * 0.10
        if trade_returns_sized.std() > 0:
            sharpe = (trade_returns_sized.mean() / trade_returns_sized.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)

        all_equity_curves.append(equity)

    return {
        'final_values': np.array(final_values),
        'cagrs': np.array(cagrs),
        'max_drawdowns': np.array(max_drawdowns),
        'sharpe_ratios': np.array(sharpe_ratios),
        'win_rates': np.array(win_rates),
        'equity_curves': all_equity_curves,
    }


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------
def print_results(results):
    fv = results['final_values']
    cagrs = results['cagrs']
    mdd = results['max_drawdowns']
    sharpes = results['sharpe_ratios']

    print("=" * 70)
    print(f"  MONTE CARLO SIMULATION RESULTS  ({NUM_SIMULATIONS:,} simulations)")
    print("=" * 70)
    print()

    print("FINAL PORTFOLIO VALUE (starting $1,000,000)")
    print(f"  5th percentile (worst case):   ${np.percentile(fv, 5):>14,.0f}")
    print(f"  25th percentile:               ${np.percentile(fv, 25):>14,.0f}")
    print(f"  50th percentile (median):      ${np.percentile(fv, 50):>14,.0f}")
    print(f"  75th percentile:               ${np.percentile(fv, 75):>14,.0f}")
    print(f"  95th percentile (best case):   ${np.percentile(fv, 95):>14,.0f}")
    print(f"  Mean:                          ${fv.mean():>14,.0f}")
    print()

    print("ANNUALIZED RETURN (CAGR)")
    print(f"  5th percentile:    {np.percentile(cagrs, 5):>8.2%}")
    print(f"  25th percentile:   {np.percentile(cagrs, 25):>8.2%}")
    print(f"  Median:            {np.percentile(cagrs, 50):>8.2%}")
    print(f"  75th percentile:   {np.percentile(cagrs, 75):>8.2%}")
    print(f"  95th percentile:   {np.percentile(cagrs, 95):>8.2%}")
    print(f"  Mean:              {cagrs.mean():>8.2%}")
    print()

    print("MAX DRAWDOWN")
    print(f"  5th percentile (worst):   {np.percentile(mdd, 5):>8.2%}")
    print(f"  25th percentile:          {np.percentile(mdd, 25):>8.2%}")
    print(f"  Median:                   {np.percentile(mdd, 50):>8.2%}")
    print(f"  75th percentile:          {np.percentile(mdd, 75):>8.2%}")
    print(f"  95th percentile (best):   {np.percentile(mdd, 95):>8.2%}")
    print()

    print("SHARPE RATIO (annualized)")
    print(f"  5th percentile:    {np.percentile(sharpes, 5):>8.2f}")
    print(f"  Median:            {np.percentile(sharpes, 50):>8.2f}")
    print(f"  95th percentile:   {np.percentile(sharpes, 95):>8.2f}")
    print()

    print("PROBABILITY ANALYSIS")
    print(f"  Probability of profit:             {(fv > STARTING_CAPITAL).mean():>8.1%}")
    print(f"  Probability of beating SPY (13%):  {(cagrs > 0.13).mean():>8.1%}")
    print(f"  Probability of >10% CAGR:          {(cagrs > 0.10).mean():>8.1%}")
    print(f"  Probability of >5% CAGR:           {(cagrs > 0.05).mean():>8.1%}")
    print(f"  Probability of loss:               {(fv < STARTING_CAPITAL).mean():>8.1%}")
    print(f"  Probability of >30% drawdown:      {(mdd < -0.30).mean():>8.1%}")
    print()


def plot_results(results, trade_returns):
    plt.style.use('dark_background')

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f'Monte Carlo Simulation \u2014 {NUM_SIMULATIONS:,} Simulations\nHigh-Conviction Model (SP900 Combined)',
                 fontsize=18, fontweight='bold', y=0.98, color='white')

    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # --- 1. Equity curves (classic dense MC fan) ---
    ax = axes[0, 0]
    n_plot = min(NUM_PATHS_TO_PLOT, len(results['equity_curves']))
    indices = np.random.choice(len(results['equity_curves']), n_plot, replace=False)

    all_finals_plot = np.array([results['equity_curves'][i][-1] for i in indices])
    vmin = np.percentile(all_finals_plot, 10)
    vmax = np.percentile(all_finals_plot, 90)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    for idx in indices:
        ec = results['equity_curves'][idx]
        color = cmap(norm(ec[-1]))
        ax.plot(ec, color=color, alpha=0.12, linewidth=0.7)

    all_final = results['final_values']
    p5_idx = np.argmin(np.abs(all_final - np.percentile(all_final, 5)))
    p25_idx = np.argmin(np.abs(all_final - np.percentile(all_final, 25)))
    p50_idx = np.argmin(np.abs(all_final - np.percentile(all_final, 50)))
    p75_idx = np.argmin(np.abs(all_final - np.percentile(all_final, 75)))
    p95_idx = np.argmin(np.abs(all_final - np.percentile(all_final, 95)))

    ax.plot(results['equity_curves'][p5_idx], color='#f85149', linewidth=2.5, label='5th pctl', zorder=10)
    ax.plot(results['equity_curves'][p25_idx], color='#d29922', linewidth=2, label='25th pctl', zorder=10)
    ax.plot(results['equity_curves'][p50_idx], color='#58a6ff', linewidth=3, label='Median', zorder=11)
    ax.plot(results['equity_curves'][p75_idx], color='#3fb950', linewidth=2, label='75th pctl', zorder=10)
    ax.plot(results['equity_curves'][p95_idx], color='#39d353', linewidth=2.5, label='95th pctl', zorder=10)
    ax.axhline(y=STARTING_CAPITAL, color='white', linestyle='--', alpha=0.2, linewidth=1)

    ax.set_title('Equity Curves', fontsize=13, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_xlabel('Trade Number')
    ax.legend(fontsize=8, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # --- 2. Final value distribution ---
    ax = axes[0, 1]
    fv = results['final_values']
    ax.hist(fv, bins=100, color='#1f6feb', edgecolor='none', alpha=0.85)
    ax.axvline(x=STARTING_CAPITAL, color='#f85149', linestyle='--', linewidth=2, label=f'Start ${STARTING_CAPITAL/1e6:.0f}M')
    ax.axvline(x=np.median(fv), color='#58a6ff', linestyle='-', linewidth=2.5, label=f'Median ${np.median(fv)/1e6:.1f}M')
    ax.axvline(x=np.percentile(fv, 5), color='#d29922', linestyle='--', linewidth=1.5, label=f'5th ${np.percentile(fv, 5)/1e6:.1f}M')
    ax.axvline(x=np.percentile(fv, 95), color='#39d353', linestyle='--', linewidth=1.5, label=f'95th ${np.percentile(fv, 95)/1e6:.1f}M')

    ax.set_title('Final Portfolio Value', fontsize=13, fontweight='bold')
    ax.set_xlabel('Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # --- 3. CAGR distribution ---
    ax = axes[0, 2]
    cagrs = results['cagrs'] * 100
    ax.hist(cagrs, bins=100, color='#3fb950', edgecolor='none', alpha=0.85)
    ax.axvline(x=13.19, color='#f85149', linestyle='--', linewidth=2, label='SPY (13.2%)')
    ax.axvline(x=np.median(cagrs), color='#58a6ff', linestyle='-', linewidth=2.5, label=f'Median ({np.median(cagrs):.1f}%)')
    ax.axvline(x=0, color='white', linestyle='-', linewidth=0.5, alpha=0.2)

    ax.set_title('Annualized Return (CAGR)', fontsize=13, fontweight='bold')
    ax.set_xlabel('CAGR (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # --- 4. Max drawdown distribution ---
    ax = axes[1, 0]
    mdd = results['max_drawdowns'] * 100
    ax.hist(mdd, bins=100, color='#f85149', edgecolor='none', alpha=0.85)
    ax.axvline(x=-33.72, color='#d29922', linestyle='--', linewidth=2, label='SPY DD (-33.7%)')
    ax.axvline(x=np.median(mdd), color='#58a6ff', linestyle='-', linewidth=2.5, label=f'Median ({np.median(mdd):.1f}%)')

    ax.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # --- 5. Sharpe ratio distribution ---
    ax = axes[1, 1]
    sharpes = results['sharpe_ratios']
    ax.hist(sharpes, bins=100, color='#a371f7', edgecolor='none', alpha=0.85)
    ax.axvline(x=0.64, color='#f85149', linestyle='--', linewidth=2, label='SPY (0.64)')
    ax.axvline(x=np.median(sharpes), color='#58a6ff', linestyle='-', linewidth=2.5, label=f'Median ({np.median(sharpes):.2f})')

    ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # --- 6. Summary stats table ---
    ax = axes[1, 2]
    ax.axis('off')

    fv_raw = results['final_values']
    cagrs_raw = results['cagrs']
    mdd_raw = results['max_drawdowns']
    sharpes_raw = results['sharpe_ratios']

    table_data = [
        ['Metric', '5th Pctl', 'Median', '95th Pctl', 'SPY'],
        ['CAGR', f'{np.percentile(cagrs_raw, 5):.1%}', f'{np.median(cagrs_raw):.1%}', f'{np.percentile(cagrs_raw, 95):.1%}', '13.2%'],
        ['Final Value', f'${np.percentile(fv_raw, 5):,.0f}', f'${np.median(fv_raw):,.0f}', f'${np.percentile(fv_raw, 95):,.0f}', '$3.45M'],
        ['Max DD', f'{np.percentile(mdd_raw, 5):.1%}', f'{np.median(mdd_raw):.1%}', f'{np.percentile(mdd_raw, 95):.1%}', '-33.7%'],
        ['Sharpe', f'{np.percentile(sharpes_raw, 5):.2f}', f'{np.median(sharpes_raw):.2f}', f'{np.percentile(sharpes_raw, 95):.2f}', '0.64'],
        ['', '', '', '', ''],
        ['Prob Profit', f'{(fv_raw > STARTING_CAPITAL).mean():.1%}', '', '', ''],
        ['Prob > SPY', f'{(cagrs_raw > 0.13).mean():.1%}', '', '', ''],
        ['Prob > 10%', f'{(cagrs_raw > 0.10).mean():.1%}', '', '', ''],
        ['Prob Loss', f'{(fv_raw < STARTING_CAPITAL).mean():.1%}', '', '', ''],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(5):
        table[0, j].set_facecolor('#1f6feb')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[i, j]
            cell.set_facecolor('#21262d' if i % 2 == 0 else '#161b22')
            cell.set_text_props(color='#c9d1d9', fontsize=10)
            cell.set_edgecolor('#30363d')

    ax.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = os.path.join(OUTPUT_DIR, 'monte_carlo_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    plt.style.use('default')
    print(f"Chart saved to {output_path}")

    stats_df = pd.DataFrame({
        'final_value': results['final_values'],
        'cagr': results['cagrs'],
        'max_drawdown': results['max_drawdowns'],
        'sharpe': results['sharpe_ratios'],
        'win_rate': results['win_rates'],
    })
    stats_path = os.path.join(OUTPUT_DIR, 'monte_carlo_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved to {stats_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    print("=" * 70)
    print("  HIGH-CONVICTION MODEL \u2014 MONTE CARLO SIMULATION")
    print("=" * 70)
    print()

    trade_returns, hold_days = load_trades()

    print(f"Running {NUM_SIMULATIONS:,} simulations...")
    print()

    results = run_monte_carlo(trade_returns, NUM_SIMULATIONS)

    print_results(results)
    plot_results(results, trade_returns)

    print()
    print("Done. Check output/ folder for monte_carlo_results.png and monte_carlo_stats.csv")
