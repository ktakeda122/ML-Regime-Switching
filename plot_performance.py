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
err_bench = (actual - bench) ** 2
err_glob = (actual - glob) ** 2
err_infl = (actual - infl) ** 2

cspe_diff_glob = np.cumsum(err_bench - err_glob)
cspe_diff_infl = np.cumsum(err_bench - err_infl)

# ============================================================
# B. Compound Trading Strategy (Market Timing)
# ============================================================

ret_bh = actual  # Buy and hold (always invest in the market)

# Method 1: Invest if > 0 
ret_glob_0 = np.where(glob > 0, actual, 0)
ret_infl_0 = np.where(infl > 0, actual, 0)

# Method 2: Invest if > bench
ret_glob_b = np.where(glob > bench, actual, 0)
ret_infl_b = np.where(infl > bench, actual, 0)

# Compound returns (Wealth Index: when investing $1)
wealth_bh = np.cumprod(1 + ret_bh)

# > 0 Rule Asset Trend
wealth_glob_0 = np.cumprod(1 + ret_glob_0)
wealth_infl_0 = np.cumprod(1 + ret_infl_0)

# > Bench rule asset trend
wealth_glob_b = np.cumprod(1 + ret_glob_b)
wealth_infl_b = np.cumprod(1 + ret_infl_b)

# ============================================================
# drawing
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

# Graph 1: CSPE (Academic evaluation of prediction errors)
axes[0].plot(df['date'], cspe_diff_glob, label='Global RBF vs Benchmark', linestyle='--', color='gray')
axes[0].plot(df['date'], cspe_diff_infl, label='Regime_Infl_6m vs Benchmark', linewidth=2, color='blue')
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_title('Cumulative OOS Performance (Goyal-Welch $\Delta$CSPE)\n*Upward slope indicates outperformance vs Historical Mean')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_ylabel('$\Delta$ CSPE')

# Graph 2: Compound Asset Trends (Normal/Linear Scale)
axes[1].plot(df['date'], wealth_bh, label='Buy & Hold (Market)', color='black', alpha=0.3)
axes[1].plot(df['date'], wealth_glob_0, label='Global RBF (>0)', linestyle='--', color='lightgray')
axes[1].plot(df['date'], wealth_infl_0, label='Regime_Infl_6m (>0)', linestyle='--', color='lightgreen')
axes[1].plot(df['date'], wealth_glob_b, label='Global RBF (>bench)', linestyle='-', color='gray')
axes[1].plot(df['date'], wealth_infl_b, label='Regime_Infl_6m (>bench)', linewidth=2, color='green')
axes[1].set_title('Compound Wealth Index (Linear Scale)')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)
axes[1].set_ylabel('Wealth Index ($)')

# Graph 3: Compound Asset Trends (Log Scale)
axes[2].plot(df['date'], wealth_bh, label='Buy & Hold (Market)', color='black', alpha=0.3)
axes[2].plot(df['date'], wealth_glob_0, label='Global RBF (>0)', linestyle='--', color='lightgray')
axes[2].plot(df['date'], wealth_infl_0, label='Regime_Infl_6m (>0)', linestyle='--', color='lightgreen')
axes[2].plot(df['date'], wealth_glob_b, label='Global RBF (>bench)', linestyle='-', color='gray')
axes[2].plot(df['date'], wealth_infl_b, label='Regime_Infl_6m (>bench)', linewidth=2, color='green')
axes[2].set_title('Compound Wealth Index (Log Scale)')
axes[2].legend(loc='upper left')
axes[2].grid(alpha=0.3, which='both')
axes[2].set_yscale('log') # Set the Y axis to a log scale here
axes[2].set_ylabel('Wealth Index (Log $)')
axes[2].set_xlabel('Year')

plt.tight_layout()
plt.savefig('combined_performance.png')
print("-> Saved combined performance chart to 'combined_performance.png'")