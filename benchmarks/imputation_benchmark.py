"""
GeoXGB Imputation Benchmark
============================

Evaluates whether GeoXGB can serve as a high-quality imputer for missing data,
compared to standard approaches.

Two-part evaluation
-------------------
Part 1 -- Imputation quality (RMSE on masked values)
    How accurately does each method recover the true feature values?
    Methods: Mean, k-NN (k=5), Ridge regression, GeoXGB

Part 2 -- Downstream prediction quality (AUC / R2)
    After imputing, how well does the final model predict the outcome?
    Imputers:    Mean, k-NN, GeoXGB
    Final model: GeoXGBClassifier / GeoXGBRegressor (default params)
    Baseline:    XGBoost with native NaN routing (no imputation needed)

Design
------
- Datasets: classification (n=1500, 10 features, 6 informative, correlated)
            regression   (n=1500, Friedman #1)
- Missingness: MNAR -- high values of 3 features are preferentially masked
  (30% missing rate). The masked features are correlated with neighbours so
  model-based imputation can genuinely recover signal.
- Train/test split: 80/20; imputers fitted on train only (no leakage).
- Seeds: 5 seeds; mean results reported.

GeoXGBImputer
-------------
Single-pass per-feature regression imputer. For each column with missing
values, a GeoXGBRegressor is trained on all training rows where BOTH the
target column and all predictor columns are observed. Missing values at
inference are filled column-by-column; any still-missing predictors are
filled with the training-set column mean.
"""
import warnings
import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

RANDOM_SEEDS   = [0, 1, 2, 3, 4]
N_SAMPLES      = 1500
MISSING_COLS   = [0, 2, 4]
MISSING_FRAC   = 0.30          # 30 % of each target column masked
IMP_ROUNDS     = 300           # GeoXGBImputer: rounds per imputation model
FINAL_ROUNDS   = 1000          # final GeoXGB / XGBoost rounds
COL_W          = 24


# ---------------------------------------------------------------------------
# Missingness helpers
# ---------------------------------------------------------------------------

def introduce_mnar(X, cols, frac, rng):
    """Mask the top-frac high values of each col (MNAR)."""
    X = X.copy()
    for col in cols:
        thresh = np.percentile(X[:, col], 100 * (1 - frac))
        high  = X[:, col] > thresh
        noise = rng.random(len(X)) < 0.20   # 20% chance low values also masked
        X[high | noise, col] = np.nan
    return X


# ---------------------------------------------------------------------------
# GeoXGBImputer
# ---------------------------------------------------------------------------

class GeoXGBImputer:
    """
    Single-pass per-feature GeoXGB imputer.

    For each column with missing values, fits a GeoXGBRegressor on
    fully-observed training rows, then predicts missing values at
    transform time.  Any still-missing predictor values are filled
    with the training-set column mean before prediction.
    """

    def __init__(self, n_rounds=IMP_ROUNDS, random_state=0):
        self.n_rounds     = n_rounds
        self.random_state = random_state
        self._models      = {}
        self._means       = {}
        self._missing_cols = None

    def fit(self, X_train):
        X = np.array(X_train, dtype=np.float64)
        self._missing_cols = np.where(np.isnan(X).any(axis=0))[0]
        self._means        = {c: np.nanmean(X[:, c]) for c in range(X.shape[1])}

        for col in self._missing_cols:
            other = [c for c in range(X.shape[1]) if c != col]
            # Only rows where this col AND all predictors are observed
            complete = ~np.isnan(X[:, col])
            for oc in other:
                complete &= ~np.isnan(X[:, oc])

            if complete.sum() < 20:
                continue  # fall back to mean

            model = GeoXGBRegressor(
                n_rounds=self.n_rounds,
                auto_expand=False,
                random_state=self.random_state,
            )
            model.fit(X[complete][:, other], X[complete, col])
            self._models[col] = model

        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64).copy()
        for col in self._missing_cols:
            other       = [c for c in range(X.shape[1]) if c != col]
            missing_idx = np.where(np.isnan(X[:, col]))[0]
            if len(missing_idx) == 0:
                continue
            if col not in self._models:
                X[missing_idx, col] = self._means[col]
                continue

            X_pred = X[missing_idx][:, other].copy()
            for j, oc in enumerate(other):
                nan_j = np.isnan(X_pred[:, j])
                if nan_j.any():
                    X_pred[nan_j, j] = self._means[oc]

            X[missing_idx, col] = self._models[col].predict(X_pred)
        return X

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)


# ---------------------------------------------------------------------------
# Helpers: build imputed datasets
# ---------------------------------------------------------------------------

def _impute_mean(X_tr, X_te):
    imp = SimpleImputer(strategy="mean")
    return imp.fit_transform(X_tr), imp.transform(X_te)


def _impute_knn(X_tr, X_te, k=5):
    imp = KNNImputer(n_neighbors=k)
    return imp.fit_transform(X_tr), imp.transform(X_te)


def _impute_geoxgb(X_tr, X_te, seed):
    imp = GeoXGBImputer(random_state=seed)
    return imp.fit_transform(X_tr), imp.transform(X_te)


# ---------------------------------------------------------------------------
# Imputation quality: RMSE on known-truth masked values
# ---------------------------------------------------------------------------

def imputation_rmse(X_imputed, X_true, nan_mask):
    """RMSE of imputed values vs true values at masked positions."""
    errs = []
    for col in range(X_true.shape[1]):
        mask = nan_mask[:, col]
        if mask.any():
            errs.append(np.sqrt(np.mean((X_imputed[mask, col] - X_true[mask, col]) ** 2)))
    return np.mean(errs) if errs else np.nan


# ---------------------------------------------------------------------------
# Single-seed evaluation for ONE dataset / task
# ---------------------------------------------------------------------------

def evaluate_seed_clf(seed):
    rng  = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=10, n_informative=6,
        n_redundant=2, random_state=seed,
    )
    X = X.astype(np.float64)
    X_miss = introduce_mnar(X, MISSING_COLS, MISSING_FRAC, rng)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_miss, y, test_size=0.20, stratify=y, random_state=seed,
    )
    # true test values (before masking) for imputation quality eval
    X_te_true = X[X_miss.shape[0] - len(X_te):]   # not perfectly aligned — compute properly
    # Rebuild: split the full X the same way for ground truth
    _, X_te_full, _, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed,
    )
    te_nan_mask = np.isnan(X_te)

    # ---- impute ----
    X_tr_mean, X_te_mean   = _impute_mean(X_tr, X_te)
    X_tr_knn,  X_te_knn    = _impute_knn(X_tr, X_te)
    X_tr_geo,  X_te_geo    = _impute_geoxgb(X_tr, X_te, seed)

    # ---- imputation quality (RMSE on test set masked values) ----
    rmse = {
        "mean":   imputation_rmse(X_te_mean, X_te_full, te_nan_mask),
        "knn":    imputation_rmse(X_te_knn,  X_te_full, te_nan_mask),
        "geoxgb": imputation_rmse(X_te_geo,  X_te_full, te_nan_mask),
    }

    # ---- downstream AUC ----
    def auc_geo(X_t, X_e):
        m = GeoXGBClassifier(n_rounds=FINAL_ROUNDS, auto_expand=False, random_state=seed)
        m.fit(X_t, y_tr)
        return roc_auc_score(y_te, m.predict_proba(X_e)[:, 1])

    def auc_xgb(X_t, X_e):
        m = XGBClassifier(n_estimators=FINAL_ROUNDS, learning_rate=0.1, max_depth=4,
                          tree_method="hist", eval_metric="auc",
                          verbosity=0, random_state=seed)
        m.fit(X_t, y_tr)
        return roc_auc_score(y_te, m.predict_proba(X_e)[:, 1])

    downstream = {
        "GeoXGB(GeoXGB-imp)":  auc_geo(X_tr_geo,  X_te_geo),
        "GeoXGB(kNN-imp)":     auc_geo(X_tr_knn,  X_te_knn),
        "GeoXGB(mean-imp)":    auc_geo(X_tr_mean, X_te_mean),
        "XGBoost(GeoXGB-imp)": auc_xgb(X_tr_geo,  X_te_geo),
        "XGBoost(kNN-imp)":    auc_xgb(X_tr_knn,  X_te_knn),
        "XGBoost(mean-imp)":   auc_xgb(X_tr_mean, X_te_mean),
        "XGBoost(native-NaN)": auc_xgb(X_tr,      X_te),
    }
    return rmse, downstream


def evaluate_seed_reg(seed):
    rng  = np.random.default_rng(seed)
    X, y = make_friedman1(n_samples=N_SAMPLES, n_features=10, noise=1.0,
                          random_state=seed)
    X = X.astype(np.float64)
    X_miss = introduce_mnar(X, MISSING_COLS, MISSING_FRAC, rng)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_miss, y, test_size=0.20, random_state=seed,
    )
    _, X_te_full, _, _ = train_test_split(X, y, test_size=0.20, random_state=seed)
    te_nan_mask = np.isnan(X_te)

    X_tr_mean, X_te_mean = _impute_mean(X_tr, X_te)
    X_tr_knn,  X_te_knn  = _impute_knn(X_tr, X_te)
    X_tr_geo,  X_te_geo  = _impute_geoxgb(X_tr, X_te, seed)

    rmse = {
        "mean":   imputation_rmse(X_te_mean, X_te_full, te_nan_mask),
        "knn":    imputation_rmse(X_te_knn,  X_te_full, te_nan_mask),
        "geoxgb": imputation_rmse(X_te_geo,  X_te_full, te_nan_mask),
    }

    def r2_geo(X_t, X_e):
        m = GeoXGBRegressor(n_rounds=FINAL_ROUNDS, auto_expand=False, random_state=seed)
        m.fit(X_t, y_tr)
        return r2_score(y_te, m.predict(X_e))

    def r2_xgb(X_t, X_e):
        m = XGBRegressor(n_estimators=FINAL_ROUNDS, learning_rate=0.1, max_depth=4,
                         tree_method="hist", verbosity=0, random_state=seed)
        m.fit(X_t, y_tr)
        return r2_score(y_te, m.predict(X_e))

    downstream = {
        "GeoXGB(GeoXGB-imp)":  r2_geo(X_tr_geo,  X_te_geo),
        "GeoXGB(kNN-imp)":     r2_geo(X_tr_knn,  X_te_knn),
        "GeoXGB(mean-imp)":    r2_geo(X_tr_mean, X_te_mean),
        "XGBoost(GeoXGB-imp)": r2_xgb(X_tr_geo,  X_te_geo),
        "XGBoost(kNN-imp)":    r2_xgb(X_tr_knn,  X_te_knn),
        "XGBoost(mean-imp)":   r2_xgb(X_tr_mean, X_te_mean),
        "XGBoost(native-NaN)": r2_xgb(X_tr,      X_te),
    }
    return rmse, downstream


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "GeoXGB(GeoXGB-imp)", "GeoXGB(kNN-imp)", "GeoXGB(mean-imp)",
    "XGBoost(GeoXGB-imp)", "XGBoost(kNN-imp)", "XGBoost(mean-imp)",
    "XGBoost(native-NaN)",
]

print(f"GeoXGB Imputation Benchmark  |  n={N_SAMPLES}, MNAR {int(MISSING_FRAC*100)}%,"
      f" {len(RANDOM_SEEDS)} seeds")
print(f"Imputer rounds={IMP_ROUNDS}, Final model rounds={FINAL_ROUNDS}")
print()

for task_name, evaluate_fn, metric_label in [
    ("Classification (AUC)", evaluate_seed_clf, "AUC"),
    ("Regression / Friedman1 (R2)",  evaluate_seed_reg, "R2 "),
]:
    print(f"{'='*70}")
    print(f"  TASK: {task_name}")
    print(f"{'='*70}")

    all_rmse   = {k: [] for k in ["mean", "knn", "geoxgb"]}
    all_ds     = {m: [] for m in MODEL_ORDER}

    for seed in RANDOM_SEEDS:
        print(f"  seed {seed}...", end="", flush=True)
        rmse, ds = evaluate_fn(seed)
        for k in all_rmse:
            all_rmse[k].append(rmse[k])
        for m in MODEL_ORDER:
            all_ds[m].append(ds[m])
        print(" done")

    # Part 1 — imputation quality
    print()
    print(f"  Part 1: Imputation quality  (RMSE on masked test values, lower=better)")
    print(f"  {'Method':<14}  {'Mean RMSE':>10}  {'Std':>6}")
    print("  " + "-" * 34)
    for k in ["mean", "knn", "geoxgb"]:
        vals = all_rmse[k]
        marker = " *" if np.mean(vals) == min(np.mean(all_rmse[v]) for v in all_rmse) else "  "
        print(f"  {k:<14}  {np.mean(vals):>10.4f}  {np.std(vals):>6.4f}{marker}")

    # Part 2 — downstream performance
    print()
    print(f"  Part 2: Downstream {metric_label} (higher=better)")
    means = {m: np.mean(all_ds[m]) for m in MODEL_ORDER}
    stds  = {m: np.std(all_ds[m])  for m in MODEL_ORDER}
    best  = max(means.values())
    geo_geo = means["GeoXGB(GeoXGB-imp)"]

    print(f"  {'Model':<{COL_W}}  {f'Mean {metric_label}':>10}  {'Std':>6}  {'vs XGB-native':>14}")
    print("  " + "-" * (COL_W + 38))
    xgb_nat = means["XGBoost(native-NaN)"]
    for m in MODEL_ORDER:
        marker  = " *" if means[m] == best else "  "
        delta   = means[m] - xgb_nat if m != "XGBoost(native-NaN)" else 0.0
        delta_s = f"{delta:+.4f}" if m != "XGBoost(native-NaN)" else "baseline"
        print(f"  {m:<{COL_W}}  {means[m]:>10.4f}  {stds[m]:>6.4f}  {delta_s:>14}{marker}")

    print()

print("=== DONE ===")
