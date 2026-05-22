"""
WALK-FORWARD ANALYSIS — High-Conviction Stock Model
=====================================================
Run with:  python walk_forward.py

Analyzes the model's backtest trades across rolling time windows
to detect overfitting, period dependency, and consistency.

This script reads your existing trade_log CSV (no need to re-run backtests).
For a fixed-parameter model like ours (thresholds set by logic, not optimization),
walk-forward validates that the edge is structural, not period-specific.

Place in your high-conviction-model/ folder alongside main.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
TRADE_LOG_PATHS = [
    os.path.join(OUTPUT_DIR, "trade_log_sp900_combined.csv"),
    os.path.join(OUTPUT_DIR, "trade_log_sp500_combined.csv"),
    os.path.join(OUTPUT_DIR, "trade_log.csv"),
]

# Walk-forward windows
# Train window: used to establish "would you trust this model?"
# Test window: out-of-sample validation
TRAIN_YEARS = 2
TEST_YEARS = 1
STEP_YEARS = 1  # Slide forward by this much each fold

# SPY annual returns for comparison (from backtest results)
SPY_ANNUAL = {
    2015: 1.2, 2016: 12.0, 2017: 21.7, 2018: -4.6, 2019: 31.2,
    2020: 18.3, 2021: 28.7, 2022: -18.2, 2023: 26.2, 2024: 25.3,
}

STARTING_CAPITAL = 1_000_000
POSITION_SIZE = 0.10


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
def load_trades():
    """Load trade log."""
    for path in TRADE_LOG_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
            
            # Convert return_pct to decimal if needed
            if 'return_pct' in df.columns:
                if df['return_pct'].abs().mean() > 1:
                    df['return_dec'] = df['return_pct'] / 100.0
                else:
                    df['return_dec'] = df['return_pct']
                    df['return_pct'] = df['return_pct'] * 100
            
            print(f"Loaded {len(df)} trades from {path}")
            print(f"Date range: {df['entry_date'].min().date()} to {df['entry_date'].max().date()}")
            return df
    
    print("ERROR: No trade log found in output/")
    print("Run the backtest first: python main.py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------
def compute_period_metrics(trades, period_name, start_date, end_date):
    """Compute performance metrics for trades within a date range."""
    # Filter trades that ENTERED during this period
    mask = (trades['entry_date'] >= start_date) & (trades['entry_date'] < end_date)
    period_trades = trades[mask].copy()
    
    if len(period_trades) == 0:
        return None
    
    returns = period_trades['return_pct'].values
    returns_dec = period_trades['return_dec'].values
    
    # Basic stats
    n_trades = len(period_trades)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0
    
    # Simulated equity curve
    equity = [STARTING_CAPITAL]
    for r in returns_dec:
        port_return = r * POSITION_SIZE
        equity.append(equity[-1] * (1 + port_return))
    equity = np.array(equity)
    
    # CAGR
    years = max((end_date - start_date).days / 365.25, 0.5)
    total_return = equity[-1] / STARTING_CAPITAL
    cagr = (total_return ** (1 / years)) - 1
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = drawdowns.min()
    
    # Sharpe (annualized from trade returns)
    trade_returns = returns_dec * POSITION_SIZE
    trades_per_year = n_trades / years if years > 0 else n_trades
    if trade_returns.std() > 0 and trades_per_year > 0:
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    # Profit factor
    gross_wins = winners.sum() if len(winners) > 0 else 0
    gross_losses = abs(losers.sum()) if len(losers) > 0 else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # By tier
    qvm_trades = period_trades[period_trades['tier'] == 'qvm'] if 'tier' in period_trades.columns else period_trades
    crash_trades = period_trades[period_trades['tier'] == 'crash'] if 'tier' in period_trades.columns else pd.DataFrame()
    
    # Average hold
    avg_hold = period_trades['hold_days'].mean() if 'hold_days' in period_trades.columns else 0
    
    # SPY return for comparison
    start_year = start_date.year
    end_year = end_date.year - 1 if end_date.month <= 3 else end_date.year
    spy_years = [y for y in range(start_year, end_year + 1) if y in SPY_ANNUAL]
    spy_cumulative = 1.0
    for y in spy_years:
        spy_cumulative *= (1 + SPY_ANNUAL[y] / 100)
    spy_cagr = (spy_cumulative ** (1 / max(len(spy_years), 1))) - 1 if spy_years else 0
    
    return {
        'period': period_name,
        'start': start_date.strftime('%Y-%m'),
        'end': end_date.strftime('%Y-%m'),
        'n_trades': n_trades,
        'n_qvm': len(qvm_trades),
        'n_crash': len(crash_trades),
        'win_rate': win_rate,
        'avg_return': returns.mean(),
        'median_return': np.median(returns),
        'best_trade': returns.max(),
        'worst_trade': returns.min(),
        'avg_hold': avg_hold,
        'cagr': cagr * 100,
        'spy_cagr': spy_cagr * 100,
        'alpha': (cagr - spy_cagr) * 100,
        'max_drawdown': max_dd * 100,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'equity_curve': equity,
    }


def run_walk_forward(trades):
    """Run rolling walk-forward analysis."""
    # Determine date range with actual trades
    min_date = trades['entry_date'].min()
    max_date = trades['entry_date'].max()
    
    # Only use years with meaningful trade activity
    trades_by_year = trades.groupby(trades['entry_date'].dt.year).size()
    active_years = trades_by_year[trades_by_year >= 3].index.tolist()
    
    if len(active_years) < 3:
        print(f"WARNING: Only {len(active_years)} years with 3+ trades. Walk-forward needs at least 3.")
        print(f"Active years: {active_years}")
        print("Results will be limited.\n")
    
    print(f"Active trading years: {active_years}")
    print(f"Total trades: {len(trades)}")
    print()
    
    # --- ANALYSIS 1: Year-by-Year Consistency ---
    print("=" * 70)
    print("  ANALYSIS 1: YEAR-BY-YEAR PERFORMANCE CONSISTENCY")
    print("=" * 70)
    print()
    
    yearly_results = []
    for year in sorted(active_years):
        start = pd.Timestamp(f'{year}-01-01')
        end = pd.Timestamp(f'{year+1}-01-01')
        result = compute_period_metrics(trades, str(year), start, end)
        if result:
            yearly_results.append(result)
    
    print(f"{'YEAR':<6} {'TRADES':>7} {'WIN%':>6} {'AVG':>7} {'CAGR':>7} {'SPY':>7} {'ALPHA':>7} {'SHARPE':>7} {'MAX DD':>8} {'PF':>6}")
    print("-" * 78)
    for r in yearly_results:
        pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else "inf"
        print(f"{r['period']:<6} {r['n_trades']:>7} {r['win_rate']:>5.0f}% {r['avg_return']:>+6.1f}% "
              f"{r['cagr']:>+6.1f}% {r['spy_cagr']:>+6.1f}% {r['alpha']:>+6.1f}% "
              f"{r['sharpe']:>6.2f} {r['max_drawdown']:>+7.1f}% {pf_str:>6}")
    
    # Consistency metrics
    if len(yearly_results) >= 3:
        cagrs = [r['cagr'] for r in yearly_results]
        alphas = [r['alpha'] for r in yearly_results]
        win_rates = [r['win_rate'] for r in yearly_results]
        sharpes = [r['sharpe'] for r in yearly_results]
        
        print()
        print("CONSISTENCY METRICS:")
        print(f"  CAGR std dev:     {np.std(cagrs):.1f}%  (lower = more consistent)")
        print(f"  CAGR range:       {min(cagrs):.1f}% to {max(cagrs):.1f}%")
        print(f"  Positive CAGR:    {sum(1 for c in cagrs if c > 0)}/{len(cagrs)} years")
        print(f"  Positive alpha:   {sum(1 for a in alphas if a > 0)}/{len(alphas)} years")
        print(f"  Win rate range:   {min(win_rates):.0f}% to {max(win_rates):.0f}%")
        print(f"  Sharpe range:     {min(sharpes):.2f} to {max(sharpes):.2f}")
    
    # --- ANALYSIS 2: Rolling Train/Test Windows ---
    print()
    print("=" * 70)
    print(f"  ANALYSIS 2: ROLLING WALK-FORWARD ({TRAIN_YEARS}yr TRAIN / {TEST_YEARS}yr TEST)")
    print("=" * 70)
    print()
    
    wf_results = []
    
    if len(active_years) >= TRAIN_YEARS + TEST_YEARS:
        start_year = min(active_years)
        end_year = max(active_years)
        
        current_start = start_year
        fold = 1
        
        while current_start + TRAIN_YEARS + TEST_YEARS <= end_year + 1:
            train_start = pd.Timestamp(f'{current_start}-01-01')
            train_end = pd.Timestamp(f'{current_start + TRAIN_YEARS}-01-01')
            test_start = train_end
            test_end = pd.Timestamp(f'{current_start + TRAIN_YEARS + TEST_YEARS}-01-01')
            
            train_result = compute_period_metrics(
                trades, f"Train {current_start}-{current_start+TRAIN_YEARS-1}",
                train_start, train_end
            )
            test_result = compute_period_metrics(
                trades, f"Test {current_start+TRAIN_YEARS}",
                test_start, test_end
            )
            
            if train_result and test_result:
                wf_results.append({
                    'fold': fold,
                    'train': train_result,
                    'test': test_result,
                })
                
                print(f"Fold {fold}: Train {current_start}-{current_start+TRAIN_YEARS-1} → Test {current_start+TRAIN_YEARS}")
                print(f"  {'':12} {'TRAIN':>10} {'TEST':>10} {'DIFF':>10}")
                print(f"  {'Trades':12} {train_result['n_trades']:>10} {test_result['n_trades']:>10}")
                print(f"  {'Win Rate':12} {train_result['win_rate']:>9.0f}% {test_result['win_rate']:>9.0f}% {test_result['win_rate']-train_result['win_rate']:>+9.1f}%")
                print(f"  {'Avg Return':12} {train_result['avg_return']:>+9.1f}% {test_result['avg_return']:>+9.1f}% {test_result['avg_return']-train_result['avg_return']:>+9.1f}%")
                print(f"  {'Sharpe':12} {train_result['sharpe']:>10.2f} {test_result['sharpe']:>10.2f} {test_result['sharpe']-train_result['sharpe']:>+10.2f}")
                print(f"  {'Max DD':12} {train_result['max_drawdown']:>+9.1f}% {test_result['max_drawdown']:>+9.1f}%")
                print(f"  {'Profit Fct':12} {train_result['profit_factor']:>10.2f} {test_result['profit_factor']:>10.2f}")
                print()
            
            current_start += STEP_YEARS
            fold += 1
    else:
        print(f"Not enough active years for {TRAIN_YEARS}yr train + {TEST_YEARS}yr test windows.")
        print(f"Have {len(active_years)} active years, need at least {TRAIN_YEARS + TEST_YEARS}.")
    
    # --- ANALYSIS 3: First Half vs Second Half ---
    print("=" * 70)
    print("  ANALYSIS 3: FIRST HALF vs SECOND HALF (STRUCTURAL STABILITY)")
    print("=" * 70)
    print()
    
    if len(active_years) >= 4:
        mid = len(active_years) // 2
        first_half_years = active_years[:mid]
        second_half_years = active_years[mid:]
        
        h1_start = pd.Timestamp(f'{first_half_years[0]}-01-01')
        h1_end = pd.Timestamp(f'{first_half_years[-1]+1}-01-01')
        h2_start = pd.Timestamp(f'{second_half_years[0]}-01-01')
        h2_end = pd.Timestamp(f'{second_half_years[-1]+1}-01-01')
        
        h1 = compute_period_metrics(trades, f"H1 ({first_half_years[0]}-{first_half_years[-1]})", h1_start, h1_end)
        h2 = compute_period_metrics(trades, f"H2 ({second_half_years[0]}-{second_half_years[-1]})", h2_start, h2_end)
        
        if h1 and h2:
            print(f"  {'METRIC':<20} {'FIRST HALF':>12} {'SECOND HALF':>12} {'DEGRADATION':>12}")
            print(f"  {'-'*56}")
            print(f"  {'Period':<20} {h1['start']+' → '+h1['end']:>12} {h2['start']+' → '+h2['end']:>12}")
            print(f"  {'Trades':<20} {h1['n_trades']:>12} {h2['n_trades']:>12}")
            print(f"  {'Win Rate':<20} {h1['win_rate']:>11.0f}% {h2['win_rate']:>11.0f}% {h2['win_rate']-h1['win_rate']:>+11.1f}%")
            print(f"  {'Avg Return':<20} {h1['avg_return']:>+11.1f}% {h2['avg_return']:>+11.1f}% {h2['avg_return']-h1['avg_return']:>+11.1f}%")
            print(f"  {'Sharpe':<20} {h1['sharpe']:>12.2f} {h2['sharpe']:>12.2f} {h2['sharpe']-h1['sharpe']:>+12.2f}")
            print(f"  {'Max Drawdown':<20} {h1['max_drawdown']:>+11.1f}% {h2['max_drawdown']:>+11.1f}%")
            print(f"  {'Profit Factor':<20} {h1['profit_factor']:>12.2f} {h2['profit_factor']:>12.2f}")
            
            print()
            
            # Overfitting assessment
            degradation_score = 0
            checks = []
            
            # Win rate degradation
            wr_diff = h2['win_rate'] - h1['win_rate']
            if wr_diff < -10:
                degradation_score += 2
                checks.append(f"Win rate dropped {abs(wr_diff):.0f}pp (significant)")
            elif wr_diff < -5:
                degradation_score += 1
                checks.append(f"Win rate dropped {abs(wr_diff):.0f}pp (moderate)")
            else:
                checks.append(f"Win rate stable ({wr_diff:+.0f}pp)")
            
            # Sharpe degradation
            sharpe_diff = h2['sharpe'] - h1['sharpe']
            if sharpe_diff < -0.5:
                degradation_score += 2
                checks.append(f"Sharpe dropped {abs(sharpe_diff):.2f} (significant)")
            elif sharpe_diff < -0.2:
                degradation_score += 1
                checks.append(f"Sharpe dropped {abs(sharpe_diff):.2f} (moderate)")
            else:
                checks.append(f"Sharpe stable ({sharpe_diff:+.2f})")
            
            # Average return degradation
            ret_diff = h2['avg_return'] - h1['avg_return']
            if ret_diff < -5:
                degradation_score += 2
                checks.append(f"Avg return dropped {abs(ret_diff):.1f}pp (significant)")
            elif ret_diff < -2:
                degradation_score += 1
                checks.append(f"Avg return dropped {abs(ret_diff):.1f}pp (moderate)")
            else:
                checks.append(f"Avg return stable ({ret_diff:+.1f}pp)")
            
            # Profit factor
            if h2['profit_factor'] < 1.0 and h1['profit_factor'] > 1.0:
                degradation_score += 3
                checks.append("Profit factor went below 1.0 (losing money in H2)")
            elif h2['profit_factor'] < h1['profit_factor'] * 0.5:
                degradation_score += 2
                checks.append(f"Profit factor halved ({h1['profit_factor']:.1f} → {h2['profit_factor']:.1f})")
            
            print("OVERFITTING ASSESSMENT:")
            for check in checks:
                print(f"  {check}")
            
            print()
            if degradation_score >= 4:
                print("  VERDICT: HIGH OVERFITTING RISK")
                print("  The model performs significantly worse in the second half.")
                print("  The backtest results may be misleading.")
            elif degradation_score >= 2:
                print("  VERDICT: MODERATE CONCERN")
                print("  Some performance degradation in the second half.")
                print("  Forward testing is important before deploying capital.")
            else:
                print("  VERDICT: LOW OVERFITTING RISK")
                print("  Performance is consistent across both halves.")
                print("  The edge appears structural, not period-dependent.")
    
    # --- ANALYSIS 4: QVM vs Crash Tier Stability ---
    print()
    print("=" * 70)
    print("  ANALYSIS 4: TIER-LEVEL CONSISTENCY")
    print("=" * 70)
    print()
    
    if 'tier' in trades.columns:
        for tier_name in ['qvm', 'crash']:
            tier_trades = trades[trades['tier'] == tier_name]
            if len(tier_trades) < 5:
                print(f"  {tier_name.upper()}: Only {len(tier_trades)} trades, insufficient for analysis.")
                continue
            
            print(f"  {tier_name.upper()} TIER ({len(tier_trades)} trades):")
            tier_yearly = []
            for year in active_years:
                year_trades = tier_trades[tier_trades['entry_date'].dt.year == year]
                if len(year_trades) > 0:
                    wr = (year_trades['return_pct'] > 0).mean() * 100
                    avg_ret = year_trades['return_pct'].mean()
                    tier_yearly.append({
                        'year': year,
                        'trades': len(year_trades),
                        'win_rate': wr,
                        'avg_return': avg_ret,
                    })
            
            if tier_yearly:
                print(f"    {'YEAR':<6} {'TRADES':>7} {'WIN%':>6} {'AVG RET':>8}")
                print(f"    {'-'*30}")
                for ty in tier_yearly:
                    print(f"    {ty['year']:<6} {ty['trades']:>7} {ty['win_rate']:>5.0f}% {ty['avg_return']:>+7.1f}%")
                
                avg_wr = np.mean([t['win_rate'] for t in tier_yearly])
                std_wr = np.std([t['win_rate'] for t in tier_yearly])
                print(f"    Avg win rate: {avg_wr:.0f}% +/- {std_wr:.0f}%")
                print()
    
    return yearly_results, wf_results


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_results(yearly_results, wf_results):
    """Generate walk-forward charts."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Walk-Forward Analysis — High-Conviction Model',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # --- 1. Year-by-year returns vs SPY ---
    ax = axes[0, 0]
    years = [r['period'] for r in yearly_results]
    model_cagrs = [r['cagr'] for r in yearly_results]
    spy_cagrs = [r['spy_cagr'] for r in yearly_results]
    
    x = np.arange(len(years))
    width = 0.35
    ax.bar(x - width/2, model_cagrs, width, color='#1f6feb', label='Model', alpha=0.85)
    ax.bar(x + width/2, spy_cagrs, width, color='#d29922', label='SPY', alpha=0.85)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_title('Annual Returns: Model vs SPY', fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.legend()
    
    # --- 2. Rolling win rate ---
    ax = axes[0, 1]
    win_rates = [r['win_rate'] for r in yearly_results]
    ax.plot(years, win_rates, 'o-', color='#3fb950', linewidth=2, markersize=8)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% (breakeven)')
    ax.fill_between(years, 50, win_rates, alpha=0.2,
                    where=[w >= 50 for w in win_rates], color='#3fb950')
    ax.fill_between(years, 50, win_rates, alpha=0.2,
                    where=[w < 50 for w in win_rates], color='#f85149')
    ax.set_title('Win Rate by Year', fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.legend()
    ax.set_ylim(0, 100)
    
    # --- 3. Walk-forward train vs test ---
    ax = axes[1, 0]
    if wf_results:
        fold_labels = [f"Fold {r['fold']}" for r in wf_results]
        train_sharpes = [r['train']['sharpe'] for r in wf_results]
        test_sharpes = [r['test']['sharpe'] for r in wf_results]
        
        x = np.arange(len(fold_labels))
        width = 0.35
        ax.bar(x - width/2, train_sharpes, width, color='#58a6ff', label='Train (in-sample)', alpha=0.85)
        ax.bar(x + width/2, test_sharpes, width, color='#a371f7', label='Test (out-of-sample)', alpha=0.85)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels)
        ax.set_title('Walk-Forward: Train vs Test Sharpe', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Insufficient data for\nrolling walk-forward',
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Walk-Forward: Train vs Test Sharpe', fontweight='bold')
    
    # --- 4. Consistency summary ---
    ax = axes[1, 1]
    ax.axis('off')
    
    if len(yearly_results) >= 3:
        cagrs_arr = [r['cagr'] for r in yearly_results]
        alphas = [r['alpha'] for r in yearly_results]
        sharpes_arr = [r['sharpe'] for r in yearly_results]
        
        summary = [
            ['Metric', 'Value', 'Assessment'],
            ['Years analyzed', str(len(yearly_results)), ''],
            ['Positive return years', f"{sum(1 for c in cagrs_arr if c > 0)}/{len(cagrs_arr)}", 
             'Good' if sum(1 for c in cagrs_arr if c > 0) > len(cagrs_arr)*0.6 else 'Concern'],
            ['Positive alpha years', f"{sum(1 for a in alphas if a > 0)}/{len(alphas)}",
             'Good' if sum(1 for a in alphas if a > 0) > len(alphas)*0.4 else 'Concern'],
            ['CAGR consistency', f"{np.std(cagrs_arr):.1f}% std dev",
             'Good' if np.std(cagrs_arr) < 20 else 'High variance'],
            ['Win rate consistency', f"{np.std([r['win_rate'] for r in yearly_results]):.1f}% std dev",
             'Good' if np.std([r['win_rate'] for r in yearly_results]) < 15 else 'High variance'],
            ['Avg Sharpe', f"{np.mean(sharpes_arr):.2f}",
             'Good' if np.mean(sharpes_arr) > 0.3 else 'Weak'],
        ]
        
        table = ax.table(cellText=summary, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        for j in range(3):
            table[0, j].set_facecolor('#1B3A4B')
            table[0, j].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(summary)):
            for j in range(3):
                if j == 2 and summary[i][2] == 'Good':
                    table[i, j].set_text_props(color='green')
                elif j == 2 and summary[i][2] in ['Concern', 'High variance', 'Weak']:
                    table[i, j].set_text_props(color='red')
    
    ax.set_title('Consistency Summary', fontweight='bold', pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = os.path.join(OUTPUT_DIR, 'walk_forward_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nChart saved to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print()
    print("=" * 70)
    print("  HIGH-CONVICTION MODEL — WALK-FORWARD ANALYSIS")
    print("=" * 70)
    print()
    
    trades = load_trades()
    
    # Filter to only completed trades (not backtest_end)
    if 'exit_reason' in trades.columns:
        completed = trades[trades['exit_reason'] != 'backtest_end']
        print(f"Using {len(completed)} completed trades (excluded {len(trades)-len(completed)} open at backtest end)")
    else:
        completed = trades
    
    print()
    yearly_results, wf_results = run_walk_forward(completed)
    
    print()
    print("=" * 70)
    print("  GENERATING CHARTS")
    print("=" * 70)
    plot_results(yearly_results, wf_results)
    
    print()
    print("=" * 70)
    print("  WHAT THESE RESULTS MEAN")
    print("=" * 70)
    print()
    print("  Walk-forward analysis splits your backtest into time windows and")
    print("  checks if the model works consistently across all of them.")
    print()
    print("  GOOD signs (low overfitting risk):")
    print("    - Win rate stays above 50% in most years")
    print("    - Test-period Sharpe is similar to train-period Sharpe")
    print("    - No single year dominates total returns")
    print("    - Second half performance matches first half")
    print()
    print("  BAD signs (high overfitting risk):")
    print("    - One monster year inflates everything (remove it, model breaks)")
    print("    - Test Sharpe consistently lower than train Sharpe")
    print("    - Win rate swings wildly year to year (>20pp range)")
    print("    - Second half materially worse than first half")
    print()
    print("Done. Check output/walk_forward_results.png")
