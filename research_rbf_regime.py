"""
Research Experiment: Regime-Switching RBF Kernel Models
=======================================================
Can combining regime switching with nonlinear RBF kernels
outperform the Global RBF baseline?

Data: gw.csv only (full 1927-2020 history, no FRED-MD).
Models: Global RBF vs 3 Regime-RBF variants.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
OOS_START = '1965-01-01'
REGIME_ROLLING_WINDOW = 60
RBF_N_COMPONENTS = 1000
REGIME_MIN_SAMPLES = 50  # Fall back to Global RBF if regime has fewer obs

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

# Regime indicators
df['regime_nber'] = df['USREC'].fillna(0).astype(int)

rolling_med_infl = (
    df['infl_lag1']
    .rolling(window=REGIME_ROLLING_WINDOW, min_periods=12)
    .median()
    .shift(1)
)
df['regime_inflation'] = (df['infl_lag1'] > rolling_med_infl).astype(int)
df['regime_inflation'] = df['regime_inflation'].fillna(0).astype(int)

rolling_med_vol = (
    df['svar_lag1']
    .rolling(window=REGIME_ROLLING_WINDOW, min_periods=12)
    .median()
    .shift(1)
)
df['regime_volatility'] = (df['svar_lag1'] > rolling_med_vol).astype(int)
df['regime_volatility'] = df['regime_volatility'].fillna(0).astype(int)

oos_start_date = pd.Timestamp(OOS_START)
oos_start_idx = df[df['date'] == oos_start_date].index[0]

print(f"Dataset: {df.shape[0]} months, {len(predictors)} predictors")
print(f"OOS period: {OOS_START} onwards ({df.shape[0] - oos_start_idx} months)")
print(f"RBF n_components={RBF_N_COMPONENTS}, regime min_samples={REGIME_MIN_SAMPLES}")


# ============================================================
# 2. Model Definitions
# ============================================================

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def fit_rbf_ridge(X_train_sc, y_train, X_test_sc):
    """Global RBF: RBFSampler + RidgeCV on full expanding window."""
    rbf = RBFSampler(gamma=1.0, n_components=RBF_N_COMPONENTS, random_state=42)
    X_tr_rbf = rbf.fit_transform(X_train_sc)
    X_te_rbf = rbf.transform(X_test_sc)
    model = RidgeCV(cv=5, alphas=RIDGE_ALPHAS).fit(X_tr_rbf, y_train)
    return model.predict(X_te_rbf)[0]


def fit_regime_rbf(df_full, predictors, target_col, i, regime_col,
                   scaler, X_test_sc, global_pred):
    """Regime RBF: train RBF only on regime-matching data.
    Falls back to global_pred if regime sample < REGIME_MIN_SAMPLES."""
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


# ============================================================
# 3. OOS Loop
# ============================================================

model_names = [
    'Global_RBF',
    'Regime_RBF_NBER',
    'Regime_RBF_Inflation',
    'Regime_RBF_Volatility',
]

predictions = {name: [] for name in model_names}
actuals = []
benchmarks = []
fallback_counts = {name: 0 for name in model_names[1:]}  # track fallbacks

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

    # Scale on full expanding window
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Global RBF (always computed — also used as fallback)
    global_pred = fit_rbf_ridge(X_train_sc, y_train, X_test_sc)
    predictions['Global_RBF'].append(global_pred)

    # Regime RBF variants
    for regime_col, name in [
        ('regime_nber', 'Regime_RBF_NBER'),
        ('regime_inflation', 'Regime_RBF_Inflation'),
        ('regime_volatility', 'Regime_RBF_Volatility'),
    ]:
        pred = fit_regime_rbf(df, predictors, target_col, i, regime_col,
                              scaler, X_test_sc, global_pred)
        predictions[name].append(pred)
        if pred == global_pred:
            fallback_counts[name] += 1

    if step % 100 == 0:
        print(f"  Step {step}/{n_oos}...")

print("OOS loop complete.\n")


# ============================================================
# 4. Results
# ============================================================

def compute_oos_r2(preds, actuals, benchmarks):
    preds = np.array(preds)
    actuals = np.array(actuals)
    benchmarks = np.array(benchmarks)
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
