"""
Research Experiment: Regime-Switching RBF Kernel Models
=======================================================
Stage 1: Observable Regimes (Short-term MAs vs 60m Median) + Visualization
Stage 2: Predicted Latent Regimes (GMM probabilities) & Soft Switching
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
OOS_START = '1965-01-01'
REGIME_ROLLING_WINDOW = 60
SHORT_WINDOWS = [1, 3, 6, 12]  # Stage 1: Short-term MAs
RBF_N_COMPONENTS = 1000
REGIME_MIN_SAMPLES = 50

# ============================================================
# 1. Data Loading & Regime Indicators
# ============================================================

df = pd.read_csv('gw.csv')
df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m')

target_col = 'CRSP_SPvw_minus_Rfree'
predictors = [col for col in df.columns if col.endswith('_lag1')]

# Merge USREC
usrec = pd.read_csv('USREC.csv')
usrec['date'] = pd.to_datetime(usrec['observation_date'])
usrec = usrec[['date', 'USREC']]
df = pd.merge(df, usrec, on='date', how='left')

# (A) NBER Regime
df['regime_nber'] = df['USREC'].fillna(0).astype(int)

# (B) Stage 1: Inflation & Volatility Regimes (Short-term MA vs 60m Median)
rolling_med_infl = df['infl_lag1'].rolling(window=REGIME_ROLLING_WINDOW, min_periods=12).median().shift(1)
rolling_med_vol = df['svar_lag1'].rolling(window=REGIME_ROLLING_WINDOW, min_periods=12).median().shift(1)

regime_cols_infl = []
regime_cols_vol = []

for w in SHORT_WINDOWS:
    # Short-term MA (shifted by 1 is already handled by 'infl_lag1', so we just take rolling mean of the lag)
    short_ma_infl = df['infl_lag1'].rolling(window=w, min_periods=1).mean()
    col_infl = f'regime_infl_{w}m'
    df[col_infl] = (short_ma_infl > rolling_med_infl).astype(int)
    regime_cols_infl.append(col_infl)

    short_ma_vol = df['svar_lag1'].rolling(window=w, min_periods=1).mean()
    col_vol = f'regime_vol_{w}m'
    df[col_vol] = (short_ma_vol > rolling_med_vol).astype(int)
    regime_cols_vol.append(col_vol)

df.fillna({col: 0 for col in regime_cols_infl + regime_cols_vol}, inplace=True)

oos_start_date = pd.Timestamp(OOS_START)
oos_start_idx = df[df['date'] == oos_start_date].index[0]

# ============================================================
# 1.5 Visualization of Regimes
# ============================================================
def plot_regimes(df_plot, start_date):
    plot_df = df_plot[df_plot['date'] >= start_date].set_index('date')
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # NBER
    axes[0].plot(plot_df.index, plot_df['regime_nber'], color='black', drawstyle='steps-post')
    axes[0].set_title('NBER Recessions (Given)')
    axes[0].set_yticks([0, 1])

    # Inflation (Compare 1m and 12m)
    axes[1].plot(plot_df.index, plot_df['regime_infl_1m'], label='1m vs 60m Median', alpha=0.7, drawstyle='steps-post')
    axes[1].plot(plot_df.index, plot_df['regime_infl_12m'], label='12m vs 60m Median', alpha=0.7, drawstyle='steps-post', linestyle='--')
    axes[1].set_title('Stage 1: Inflation Regimes')
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc='upper right')

    # Volatility (Compare 1m and 12m)
    axes[2].plot(plot_df.index, plot_df['regime_vol_1m'], label='1m vs 60m Median', alpha=0.7, drawstyle='steps-post')
    axes[2].plot(plot_df.index, plot_df['regime_vol_12m'], label='12m vs 60m Median', alpha=0.7, drawstyle='steps-post', linestyle='--')
    axes[2].set_title('Stage 1: Volatility Regimes')
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('regime_transitions.png')
    plt.close()
    print("-> Saved regime visualization to 'regime_transitions.png'")

plot_regimes(df, OOS_START)


# ============================================================
# 2. Model Definitions
# ============================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

def fit_rbf_ridge(X_train_sc, y_train, X_test_sc):
    """Global RBF"""
    rbf = RBFSampler(gamma=1.0, n_components=RBF_N_COMPONENTS, random_state=42)
    X_tr_rbf = rbf.fit_transform(X_train_sc)
    X_te_rbf = rbf.transform(X_test_sc)
    model = RidgeCV(cv=5, alphas=RIDGE_ALPHAS).fit(X_tr_rbf, y_train)
    return model.predict(X_te_rbf)[0]

def fit_regime_rbf(df_full, predictors, target_col, i, regime_col, scaler, X_test_sc, global_pred):
    """Stage 1: Hard Switch Regime RBF"""
    current_regime = df_full.iloc[i][regime_col]
    train_data = df_full.iloc[:i]
    regime_train = train_data[train_data[regime_col] == current_regime]

    if len(regime_train) < REGIME_MIN_SAMPLES:
        return global_pred

    X_tr_sc = scaler.transform(regime_train[predictors])
    y_tr = regime_train[target_col].values
    rbf = RBFSampler(gamma=1.0, n_components=RBF_N_COMPONENTS, random_state=42)
    X_tr_rbf = rbf.fit_transform(X_tr_sc)
    X_te_rbf = rbf.transform(X_test_sc)
    model = RidgeCV(cv=5, alphas=RIDGE_ALPHAS).fit(X_tr_rbf, y_tr)
    return model.predict(X_te_rbf)[0]

def fit_stage2_gmm_soft(train_data, test_row, predictors, target_col, scaler, X_test_sc, global_pred):
    """Stage 2: GMM Latent Regime Inference & Soft Switching"""
    # Use Macro indicators to deduce latent structure
    macro_cols = ['infl_lag1', 'svar_lag1']
    
    macro_scaler = StandardScaler()
    X_macro_tr = macro_scaler.fit_transform(train_data[macro_cols])
    X_macro_te = macro_scaler.transform(test_row[macro_cols])

    # Infer 2 latent regimes
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_macro_tr)

    # Probabilities & Assignments
    train_labels = gmm.predict(X_macro_tr)
    test_probs = gmm.predict_proba(X_macro_te)[0]  # [Prob(Regime=0), Prob(Regime=1)]

    preds = []
    # Train RBF for Regime 0 and Regime 1
    for regime_id in [0, 1]:
        mask = (train_labels == regime_id)
        regime_train = train_data[mask]
        
        if len(regime_train) < REGIME_MIN_SAMPLES:
            preds.append(global_pred)
        else:
            X_tr_sc = scaler.transform(regime_train[predictors])
            y_tr = regime_train[target_col].values
            rbf = RBFSampler(gamma=1.0, n_components=RBF_N_COMPONENTS, random_state=42)
            X_tr_rbf = rbf.fit_transform(X_tr_sc)
            X_te_rbf = rbf.transform(X_test_sc)
            model = RidgeCV(cv=5, alphas=RIDGE_ALPHAS).fit(X_tr_rbf, y_tr)
            preds.append(model.predict(X_te_rbf)[0])
            
    # Soft Switch: Probability-weighted combination
    final_pred = (test_probs[0] * preds[0]) + (test_probs[1] * preds[1])
    return final_pred

# ============================================================
# 3. OOS Loop
# ============================================================

model_names = ['Global_RBF', 'Regime_NBER'] 
model_names += [f'Regime_Infl_{w}m' for w in SHORT_WINDOWS]
model_names += [f'Regime_Vol_{w}m' for w in SHORT_WINDOWS]
model_names += ['Stage2_GMM_Soft']

predictions = {name: [] for name in model_names}
actuals = []
benchmarks = []
fallback_counts = {name: 0 for name in model_names[1:]}

n_oos = len(df) - oos_start_idx
print(f"\nRunning OOS loop ({n_oos} steps)...")

for i in range(oos_start_idx, len(df)):
    step = i - oos_start_idx

    train_data = df.iloc[:i]
    test_row = df.iloc[[i]]

    y_train = train_data[target_col]
    X_train = train_data[predictors]
    X_test = test_row[predictors]

    actuals.append(test_row[target_col].values[0])
    benchmarks.append(y_train.mean())

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 1. Global Baseline
    global_pred = fit_rbf_ridge(X_train_sc, y_train, X_test_sc)
    predictions['Global_RBF'].append(global_pred)

    # 2. Stage 1 Models (Hard Switch)
    stage1_variants = [('regime_nber', 'Regime_NBER')]
    for w in SHORT_WINDOWS:
        stage1_variants.append((f'regime_infl_{w}m', f'Regime_Infl_{w}m'))
        stage1_variants.append((f'regime_vol_{w}m', f'Regime_Vol_{w}m'))

    for regime_col, name in stage1_variants:
        pred = fit_regime_rbf(df, predictors, target_col, i, regime_col, scaler, X_test_sc, global_pred)
        predictions[name].append(pred)
        if pred == global_pred:
            fallback_counts[name] += 1

    # 3. Stage 2 Model (GMM Soft Switch)
    stage2_pred = fit_stage2_gmm_soft(train_data, test_row, predictors, target_col, scaler, X_test_sc, global_pred)
    predictions['Stage2_GMM_Soft'].append(stage2_pred)
    # Track fallback roughly for Stage 2 if it matches global exactly (rare due to soft mix)
    if stage2_pred == global_pred:
        fallback_counts['Stage2_GMM_Soft'] += 1

    if step % 100 == 0:
        print(f"  Step {step}/{n_oos}...")

print("OOS loop complete.\n")

# ============================================================
# 4. Results
# ============================================================

def compute_oos_r2(preds, actuals, benchmarks):
    preds, actuals, benchmarks = np.array(preds), np.array(actuals), np.array(benchmarks)
    mse_model = np.mean((actuals - preds) ** 2)
    mse_bench = np.mean((actuals - benchmarks) ** 2)
    return 1 - (mse_model / mse_bench)

def compute_msfe(preds, actuals):
    return np.mean((np.array(actuals) - np.array(preds)) ** 2)

COL_W = 60
header = f"{'Model':<25} | {'OOS R2':>10} | {'MSFE':>12} | {'Fallbacks':>9}"
sep = "-" * COL_W

print("=" * COL_W)
print("RBF REGIME EXPERIMENT: OOS Comparison")
print("=" * COL_W)
print(header)
print(sep)

msfe_bench = compute_msfe(benchmarks, actuals)
print(f"{'Historical Mean (bench.)':<25} | {'0.00000':>10} | {msfe_bench:>12.8f} |       ---")
print(sep)

for name in model_names:
    r2 = compute_oos_r2(predictions[name], actuals, benchmarks)
    msfe = compute_msfe(predictions[name], actuals)
    fb = fallback_counts.get(name, '---')
    fb_str = f"{fb:>9}" if isinstance(fb, int) else f"{'---':>9}"
    print(f"{name:<25} | {r2:>10.5f} | {msfe:>12.8f} | {fb_str}")

print(sep)


# ============================================================
# 5. Export Predictions for Plotting
# ============================================================
# Extract OOS period for date column
dates_oos = df['date'].iloc[oos_start_idx:].values

results_df = pd.DataFrame({
    'date': dates_oos,
    'Actual': actuals,
    'Benchmark': benchmarks,
    'Global_RBF': predictions['Global_RBF'],
    'Regime_Infl_6m': predictions['Regime_Infl_6m']
})

results_df.to_csv('oos_predictions.csv', index=False)
print("-> Saved OOS predictions to 'oos_predictions.csv'")