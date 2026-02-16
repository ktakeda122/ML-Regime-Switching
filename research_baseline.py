"""
Research Benchmark: Horse Race Comparison of Forecasting Models
==============================================================
Unified OOS pipeline comparing multiple model families on
S&P 500 excess return prediction.

Research question: Can Regime Switching beat the best nonlinear models?

Data pipelines:
  df_gw     — Full GW history (1927-2020, 1128 months) for standard models.
  df_merged — GW + FRED-MD (~1959-2020, ~743 months) for NN_FRED only.
"""

import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

# ============================================================
# Configuration Flags
# ============================================================
RUN_LINEAR_BASELINES = True    # OLS Kitchen Sink, Ridge, Lasso
RUN_ROLLING_WINDOW = True      # OLS on last 60 months
RUN_REGIME_SWITCHING = True    # Regime-conditional models (NBER, Inflation, Volatility)
RUN_NONLINEAR_COMPLEX = True   # RBF Kernel + Neural Nets

OOS_START = '1965-01-01'
ROLLING_WINDOW_SIZE = 60
REGIME_ROLLING_WINDOW = 60
RBF_N_COMPONENTS = 1000
NN_ENSEMBLE_SIZE = 5

# ============================================================
# 1. Data Loading & Regime Feature Engineering
# ============================================================

# --- df_gw: Full GW dataset (1927-2020) ---
df_gw = pd.read_csv('gw.csv')
df_gw['date'] = pd.to_datetime(df_gw['yyyymm'], format='%Y%m')

target_col = 'CRSP_SPvw_minus_Rfree'
predictors_gw = [col for col in df_gw.columns if col.endswith('_lag1')]

# Merge USREC (NBER recession indicator)
usrec = pd.read_csv('USREC.csv')
usrec['date'] = pd.to_datetime(usrec['observation_date'])
usrec = usrec[['date', 'USREC']]
df_gw = pd.merge(df_gw, usrec, on='date', how='left')

# Create Regime Indicators on the full GW dataset
df_gw['regime_nber'] = df_gw['USREC'].fillna(0).astype(int)

rolling_median_infl = (
    df_gw['infl_lag1']
    .rolling(window=REGIME_ROLLING_WINDOW, min_periods=12)
    .median()
    .shift(1)
)
df_gw['regime_inflation'] = (df_gw['infl_lag1'] > rolling_median_infl).astype(int)
df_gw['regime_inflation'] = df_gw['regime_inflation'].fillna(0).astype(int)

rolling_median_vol = (
    df_gw['svar_lag1']
    .rolling(window=REGIME_ROLLING_WINDOW, min_periods=12)
    .median()
    .shift(1)
)
df_gw['regime_volatility'] = (df_gw['svar_lag1'] > rolling_median_vol).astype(int)
df_gw['regime_volatility'] = df_gw['regime_volatility'].fillna(0).astype(int)

# --- df_merged: GW + FRED-MD (~1959-2020) for NN_FRED only ---
fred_df = pd.read_csv('FREDMD.csv')
fred_df['date'] = pd.to_datetime(fred_df['date'])
fred_df = fred_df.sort_values('date')
fred_df = fred_df[fred_df['date'] >= '1959-01-01']

missing_pct = fred_df.drop(columns='date').isna().mean()
drop_cols = missing_pct[missing_pct > 0.20].index.tolist()
fred_df = fred_df.drop(columns=drop_cols)
fred_df = fred_df.ffill().bfill()

fred_features = fred_df.set_index('date')
fred_features = fred_features.shift(1)
fred_features.columns = [col + '_fred' for col in fred_features.columns]
fred_features = fred_features.reset_index()

df_merged = pd.merge(df_gw, fred_features, on='date', how='inner')
df_merged = df_merged.reset_index(drop=True)

predictors_macro = [col for col in df_merged.columns if col.endswith('_fred')]
predictors_all = predictors_gw + predictors_macro

df_merged = df_merged.dropna(subset=predictors_all + [target_col]).reset_index(drop=True)

# Build a date -> row-index lookup for df_merged (used by NN_FRED)
merged_date_to_idx = dict(zip(df_merged['date'], df_merged.index))

# --- OOS start indices ---
oos_start_date = pd.Timestamp(OOS_START)
oos_start_gw = df_gw[df_gw['date'] == oos_start_date].index[0]

print(f"df_gw:     {df_gw.shape[0]} months (full history 1927-2020)")
print(f"df_merged: {df_merged.shape[0]} months (FRED-MD overlap ~1959-2020)")
print(f"  GW predictors: {len(predictors_gw)}, FRED-MD predictors: {len(predictors_macro)}")
print(f"OOS period: {OOS_START} onwards ({df_gw.shape[0] - oos_start_gw} months)")
print(f"Active flags: Linear={RUN_LINEAR_BASELINES}, Rolling={RUN_ROLLING_WINDOW}, "
      f"Regime={RUN_REGIME_SWITCHING}, Nonlinear={RUN_NONLINEAR_COMPLEX}")


# ============================================================
# 2. Model Definitions
# ============================================================

def fit_ols_kitchen_sink(X_train, y_train, X_test):
    """OLS with all GW predictors (expanding window)."""
    X_tr = sm.add_constant(X_train)
    X_te = sm.add_constant(X_test, has_constant='add')
    model = sm.OLS(y_train, X_tr).fit()
    return model.predict(X_te).values[0]


def fit_ridge(X_train_scaled, y_train, X_test_scaled):
    """Ridge regression with 5-fold CV alpha selection."""
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    model = RidgeCV(cv=5, alphas=alphas).fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)[0]


def fit_lasso(X_train_scaled, y_train, X_test_scaled):
    """Lasso regression with 5-fold CV alpha selection."""
    model = LassoCV(cv=5, random_state=42, n_alphas=100).fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)[0]


def fit_rolling_ols(df_full, preds_cols, target_col, i, window=60):
    """OLS using only the most recent `window` months of data."""
    start_idx = max(0, i - window)
    train_data = df_full.iloc[start_idx:i]
    test_row = df_full.iloc[[i]]

    if len(train_data) < 12:
        return train_data[target_col].mean()

    X_tr = sm.add_constant(train_data[preds_cols])
    X_te = sm.add_constant(test_row[preds_cols], has_constant='add')
    model = sm.OLS(train_data[target_col], X_tr).fit()
    return model.predict(X_te).values[0]


def fit_regime_ridge(df_full, preds_cols, target_col, i, regime_col,
                     scaler, X_test_scaled):
    """Ridge trained only on past observations matching the current regime."""
    current_regime = df_full.iloc[i][regime_col]
    train_data = df_full.iloc[:i]

    regime_train = train_data[train_data[regime_col] == current_regime]

    if len(regime_train) < 12:
        return train_data[target_col].mean()

    # Scale regime subset using the already-fitted global scaler
    X_tr_sc = scaler.transform(regime_train[preds_cols])
    y_tr = regime_train[target_col]
    model = Ridge(alpha=10.0).fit(X_tr_sc, y_tr)
    return model.predict(X_test_scaled)[0]


def fit_rbf_ridge(X_train_scaled, y_train, X_test_scaled, n_components=1000):
    """RBF kernel approximation + RidgeCV (standard GW predictors only)."""
    rbf = RBFSampler(gamma=1.0, n_components=n_components, random_state=42)
    X_train_rbf = rbf.fit_transform(X_train_scaled)
    X_test_rbf = rbf.transform(X_test_scaled)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    model = RidgeCV(cv=5, alphas=alphas).fit(X_train_rbf, y_train)
    return model.predict(X_test_rbf)[0]


def fit_neural_net(X_train_scaled, y_train, X_test_scaled, n_ensemble=5):
    """Ensemble of MLPRegressors."""
    preds = []
    for seed in range(n_ensemble):
        nn = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16, 8, 4),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.005,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            max_iter=500,
            random_state=42 + seed,
        )
        try:
            nn.fit(X_train_scaled, y_train)
            preds.append(nn.predict(X_test_scaled)[0])
        except Exception:
            preds.append(y_train.mean())
    return np.mean(preds)


# ============================================================
# 3. Unified OOS Loop
# ============================================================

# Build model registry based on active flags
model_names = []
if RUN_LINEAR_BASELINES:
    model_names += ['OLS_KitchenSink', 'Ridge', 'Lasso']
if RUN_ROLLING_WINDOW:
    model_names += ['Rolling_OLS_60m']
if RUN_REGIME_SWITCHING:
    model_names += ['Regime_NBER', 'Regime_Inflation', 'Regime_Volatility']
if RUN_NONLINEAR_COMPLEX:
    model_names += ['RBF_Kernel', 'NN_GW', 'NN_FRED']

predictions = {name: [] for name in model_names}
actuals = []
benchmarks = []

n_oos = len(df_gw) - oos_start_gw
print(f"\nRunning unified OOS loop ({n_oos} steps)...")

for i in range(oos_start_gw, len(df_gw)):
    step = i - oos_start_gw

    # === Standard models use df_gw (full 1927-2020 history) ===
    train_gw = df_gw.iloc[:i]
    test_gw = df_gw.iloc[[i]]

    y_train = train_gw[target_col]
    X_train_gw = train_gw[predictors_gw]
    X_test_gw = test_gw[predictors_gw]

    actuals.append(test_gw[target_col].values[0])
    benchmarks.append(y_train.mean())

    # Scaled GW features (for Ridge, Lasso, RBF, NN_GW)
    scaler_gw = StandardScaler()
    X_train_gw_sc = scaler_gw.fit_transform(X_train_gw)
    X_test_gw_sc = scaler_gw.transform(X_test_gw)

    # --- Linear Baselines (df_gw) ---
    if RUN_LINEAR_BASELINES:
        predictions['OLS_KitchenSink'].append(
            fit_ols_kitchen_sink(X_train_gw, y_train, X_test_gw))
        predictions['Ridge'].append(
            fit_ridge(X_train_gw_sc, y_train, X_test_gw_sc))
        predictions['Lasso'].append(
            fit_lasso(X_train_gw_sc, y_train, X_test_gw_sc))

    # --- Rolling Window (df_gw) ---
    if RUN_ROLLING_WINDOW:
        predictions['Rolling_OLS_60m'].append(
            fit_rolling_ols(df_gw, predictors_gw, target_col, i,
                            window=ROLLING_WINDOW_SIZE))

    # --- Regime Switching (df_gw, Ridge base model) ---
    if RUN_REGIME_SWITCHING:
        predictions['Regime_NBER'].append(
            fit_regime_ridge(df_gw, predictors_gw, target_col, i,
                             'regime_nber', scaler_gw, X_test_gw_sc))
        predictions['Regime_Inflation'].append(
            fit_regime_ridge(df_gw, predictors_gw, target_col, i,
                             'regime_inflation', scaler_gw, X_test_gw_sc))
        predictions['Regime_Volatility'].append(
            fit_regime_ridge(df_gw, predictors_gw, target_col, i,
                             'regime_volatility', scaler_gw, X_test_gw_sc))

    # --- Nonlinear / Complex ---
    if RUN_NONLINEAR_COMPLEX:
        # RBF Kernel (df_gw, GW predictors, n=1000)
        predictions['RBF_Kernel'].append(
            fit_rbf_ridge(X_train_gw_sc, y_train, X_test_gw_sc,
                          n_components=RBF_N_COMPONENTS))

        # NN_GW: Neural Net on GW predictors only (df_gw)
        predictions['NN_GW'].append(
            fit_neural_net(X_train_gw_sc, y_train, X_test_gw_sc,
                           n_ensemble=NN_ENSEMBLE_SIZE))

        # NN_FRED: Neural Net on GW + FRED-MD (df_merged, shorter history)
        current_date = df_gw.iloc[i]['date']
        m_idx = merged_date_to_idx.get(current_date)
        if m_idx is not None and m_idx >= 12:
            train_m = df_merged.iloc[:m_idx]
            test_m = df_merged.iloc[[m_idx]]
            X_train_all = train_m[predictors_all]
            X_test_all = test_m[predictors_all]
            scaler_all = StandardScaler()
            X_train_all_sc = scaler_all.fit_transform(X_train_all)
            X_test_all_sc = scaler_all.transform(X_test_all)
            predictions['NN_FRED'].append(
                fit_neural_net(X_train_all_sc, train_m[target_col],
                               X_test_all_sc, n_ensemble=NN_ENSEMBLE_SIZE))
        else:
            # Date not in FRED range — fall back to historical mean
            predictions['NN_FRED'].append(y_train.mean())

    if step % 100 == 0:
        print(f"  Step {step}/{n_oos}...")

print("OOS loop complete.\n")


# ============================================================
# 4. Results: OOS R2 and MSFE Comparison Table
# ============================================================

def compute_oos_r2(preds, actuals, benchmarks):
    """OOS R2 = 1 - MSE_model / MSE_benchmark."""
    preds = np.array(preds)
    actuals = np.array(actuals)
    benchmarks = np.array(benchmarks)
    mse_model = np.mean((actuals - preds) ** 2)
    mse_bench = np.mean((actuals - benchmarks) ** 2)
    return 1 - (mse_model / mse_bench)


def compute_msfe(preds, actuals):
    """Mean Squared Forecast Error."""
    return np.mean((np.array(actuals) - np.array(preds)) ** 2)


COL_W = 55
header = f"{'Model':<25} | {'OOS R2':>10} | {'MSFE':>12}"
sep = "-" * COL_W

print("=" * COL_W)
print("HORSE RACE: OOS Forecast Comparison")
print("=" * COL_W)
print(header)
print(sep)

msfe_bench = compute_msfe(benchmarks, actuals)
print(f"{'Historical Mean (bench.)':<25} | {'0.00000':>10} | {msfe_bench:>12.8f}")
print(sep)

for name in model_names:
    r2 = compute_oos_r2(predictions[name], actuals, benchmarks)
    msfe = compute_msfe(predictions[name], actuals)
    print(f"{name:<25} | {r2:>10.5f} | {msfe:>12.8f}")

print(sep)
