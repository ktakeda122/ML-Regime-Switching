"""
Analyze and Plot OOS Performance
Reads 'oos_predictions.csv' and generates performance charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Loading data
df = pd.read_csv('oos_predictions.csv')
df['date'] = pd.to_datetime(df['date'])

actual = df['Actual'].values
bench = df['Benchmark'].values
glob = df['Global_RBF'].values
infl = df['Regime_Infl_6m'].values

# ============================================================
# A. Goyal-Welch CSPE (Cumulative Squared Prediction Error) Diff
# ============================================================
# Calculate the square of the error
err_bench = (actual - bench) ** 2
err_glob = (actual - glob) ** 2
err_infl = (actual - infl) ** 2

# Accumulation of "reductions" in error relative to the benchmark (Historical Mean)
# If the graph is rising to the right, the benchmark is being beaten (the prediction is correct).
cspe_diff_glob = np.cumsum(err_bench - err_glob)
cspe_diff_infl = np.cumsum(err_bench - err_infl)

# ============================================================
# B. Simple Trading Strategy (Market Timing)
# ============================================================

ret_bh = actual  # Buy and hold (always invest in the market)

# Method1: If the predicted value is > 0, invest in the stock market (earn excess returns)
# If the predicted value is <= 0, retreat to the risk-free asset (excess return 0)
ret_glob = np.where(glob > 0, actual, 0)
ret_infl = np.where(infl > 0, actual, 0)

# Method2: Invest only when forecasts exceed historical benchmarks
# ret_glob = np.where(glob > bench, actual, 0)
# ret_infl = np.where(infl > bench, actual, 0)

# Cumulative return (simple sum; np.cumprod(1+ret) for compound return)
cum_ret_bh = np.cumsum(ret_bh)
cum_ret_glob = np.cumsum(ret_glob)
cum_ret_infl = np.cumsum(ret_infl)

wealth_bh = np.cumprod(1 + ret_bh)
wealth_glob = np.cumprod(1 + ret_glob)
wealth_infl = np.cumprod(1 + ret_infl)

# ============================================================
# drawing
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Graph 1: Difference in cumulative prediction error (academic evaluation)
axes[0].plot(df['date'], cspe_diff_glob, label='Global RBF vs Benchmark', linestyle='--', color='gray')
axes[0].plot(df['date'], cspe_diff_infl, label='Regime_Infl_6m vs Benchmark', linewidth=2, color='blue')
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_title('Cumulative OOS Performance (Goyal-Welch $\Delta$CSPE)\n*Upward slope indicates outperformance vs Historical Mean')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_ylabel('$\Delta$ CSPE')

# Graph 2: Cumulative returns of market timing strategies (practical evaluation)
axes[1].plot(df['date'], cum_ret_bh, label='Buy & Hold (Market)', color='black', alpha=0.3)
axes[1].plot(df['date'], cum_ret_glob, label='Timing: Global RBF', linestyle='--', color='gray')
axes[1].plot(df['date'], cum_ret_infl, label='Timing: Regime_Infl_6m', linewidth=2, color='green')
axes[1].set_title('Cumulative Excess Returns (Simple Timing Strategy)')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)
axes[1].set_ylabel('Cumulative Excess Return')

plt.tight_layout()
plt.savefig('performance_analysis.png')
print("-> Saved performance charts to 'performance_analysis.png'")


# compound return
plt.figure(figsize=(12, 6))

plt.plot(df['date'], wealth_bh, label='Buy & Hold (Market)', color='black', alpha=0.3)
plt.plot(df['date'], wealth_glob, label='Timing: Global RBF', linestyle='--', color='gray')
plt.plot(df['date'], wealth_infl, label='Timing: Regime_Infl_6m', linewidth=2, color='green')

# log scale: the same "% change" will be plotted at the same angle for every decade.
plt.yscale('log') 

plt.title('Compound Wealth Index (Log Scale)\nValue of $1 invested at OOS start (1965)')
plt.legend(loc='upper left')
plt.grid(alpha=0.3, which='both') # Fine grid for logarithmic scale
plt.ylabel('Wealth Index ($)')
plt.xlabel('Year')

plt.tight_layout()
plt.savefig('compound_performance.png')
print("-> Saved compound performance chart to 'compound_performance.png'")