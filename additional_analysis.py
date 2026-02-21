import pandas as pd
import numpy as np

# ============================================================
# 1. Loading data and restoring regime
# ============================================================
# Loading prediction results
df = pd.read_csv('oos_predictions.csv')
df['date'] = pd.to_datetime(df['date'])

# Restore and merge the flags for "Regime_Infl_6m" from gw.csv
gw = pd.read_csv('gw.csv')
gw['date'] = pd.to_datetime(gw['yyyymm'], format='%Y%m')
rolling_med_infl = gw['infl_lag1'].rolling(window=60, min_periods=12).median().shift(1)
short_ma_infl = gw['infl_lag1'].rolling(window=6, min_periods=1).mean()
gw['regime_infl_6m'] = (short_ma_infl > rolling_med_infl).astype(int)

# Combine regime flags with OOS period data
df = pd.merge(df, gw[['date', 'regime_infl_6m']], on='date', how='left')

actual = df['Actual'].values
bench = df['Benchmark'].values
glob = df['Global_RBF'].values
infl = df['Regime_Infl_6m'].values
regime = df['regime_infl_6m'].values

# ============================================================
# 2. Defining the evaluation function
# ============================================================
def compute_oos_r2(act, prd, bnc):
    if len(act) == 0: return np.nan
    mse_model = np.mean((act - prd)**2)
    mse_bench = np.mean((act - bnc)**2)
    return 1 - (mse_model / mse_bench)

def compute_hit_ratio(act, prd):
    """Sign agreement rate (proportion of people guessing plus/minus)"""
    return np.mean((act > 0) == (prd > 0)) * 100

def compute_portfolio_metrics(ret):
    """Annualized return, volatility, Sharpe ratio, maximum decline"""
    ann_ret = np.mean(ret) * 12
    ann_vol = np.std(ret) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    wealth = np.cumprod(1 + ret)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / peaks
    max_dd = np.min(drawdowns) * 100
    
    return ann_ret * 100, ann_vol * 100, sharpe, max_dd

# ============================================================
# 3. Running and outputting the analysis
# ============================================================
print("="*65)
print(" SUPPLEMENTARY ANALYSIS FOR REPORT ")
print("="*65)

# --- A. Hit Ratio ---
print("\n[A] Directional Accuracy (Hit Ratio: %)")
print("-" * 45)
print(f"Historical Mean (Always > 0) : {compute_hit_ratio(actual, bench):.2f}%")
print(f"Global_RBF                   : {compute_hit_ratio(actual, glob):.2f}%")
print(f"Regime_Infl_6m               : {compute_hit_ratio(actual, infl):.2f}%")
print("-" * 45)

# --- B. Regime Decomposition ---
print("\n[B] Performance Decomposition by Regime (OOS R2)")
print("-" * 65)
print(f"{'Regime State':<20} | {'N (Months)':<10} | {'Global_RBF':<12} | {'Regime_Infl_6m':<12}")
print("-" * 65)

for r_val, r_name in [(0, "Low Inflation (0)"), (1, "High Inflation (1)")]:
    mask = (regime == r_val)
    n_obs = np.sum(mask)
    r2_glob = compute_oos_r2(actual[mask], glob[mask], bench[mask])
    r2_infl = compute_oos_r2(actual[mask], infl[mask], bench[mask])
    print(f"{r_name:<20} | {n_obs:<10} | {r2_glob:>10.5f} | {r2_infl:>10.5f}")
print("-" * 65)

# --- C. Risk-Adjusted Performance  ---
# 1. Naive Rule (Predict > 0)
ret_bh = actual
ret_glob_0 = np.where(glob > 0, actual, 0)
ret_infl_0 = np.where(infl > 0, actual, 0)

# 2. Active Rule (Predict > Benchmark)
ret_glob_b = np.where(glob > bench, actual, 0)
ret_infl_b = np.where(infl > bench, actual, 0)

metrics_bh = compute_portfolio_metrics(ret_bh)
metrics_glob_0 = compute_portfolio_metrics(ret_glob_0)
metrics_infl_0 = compute_portfolio_metrics(ret_infl_0)
metrics_glob_b = compute_portfolio_metrics(ret_glob_b)
metrics_infl_b = compute_portfolio_metrics(ret_infl_b)

print("\n[C] Portfolio Performance Comparison (1965-2020)")
print("-" * 85)
print(f"{'Strategy':<28} | {'Ann. Ret(%)':>11} | {'Ann. Vol(%)':>11} | {'Sharpe':>8} | {'Max DD(%)':>10}")
print("-" * 85)
print(f"{'Buy & Hold (Market)':<28} | {metrics_bh[0]:>11.2f} | {metrics_bh[1]:>11.2f} | {metrics_bh[2]:>8.2f} | {metrics_bh[3]:>10.2f}")
print("-" * 85)
print("Panel A: Naive Rule (Predict > 0)")
print(f"{'  Global_RBF (> 0)':<28} | {metrics_glob_0[0]:>11.2f} | {metrics_glob_0[1]:>11.2f} | {metrics_glob_0[2]:>8.2f} | {metrics_glob_0[3]:>10.2f}")
print(f"{'  Regime_Infl_6m (> 0)':<28} | {metrics_infl_0[0]:>11.2f} | {metrics_infl_0[1]:>11.2f} | {metrics_infl_0[2]:>8.2f} | {metrics_infl_0[3]:>10.2f}")
print("-" * 85)
print("Panel B: Active Rule (Predict > Benchmark)")
print(f"{'  Global_RBF (> bench)':<28} | {metrics_glob_b[0]:>11.2f} | {metrics_glob_b[1]:>11.2f} | {metrics_glob_b[2]:>8.2f} | {metrics_glob_b[3]:>10.2f}")
print(f"{'  Regime_Infl_6m (> bench)':<28} | {metrics_infl_b[0]:>11.2f} | {metrics_infl_b[1]:>11.2f} | {metrics_infl_b[2]:>8.2f} | {metrics_infl_b[3]:>10.2f}")
print("-" * 85)