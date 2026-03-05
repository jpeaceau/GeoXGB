"""
GeoXGB vs XGBoost head-to-head on large datasets (n >= 5000).
Tests the benefit of sample_block_n='auto' (activated when n > 5000).

Regression datasets:
  friedman1_10k  n=10000
  reg_5k         n=5000
  reg_20k        n=20000

Classification datasets:
  clf_10k        n=10000, binary
  clf_20k        n=20000, multiclass (5 classes)

Metric: R2 for regression; AUC-ROC (OvR macro) for classification.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
from sklearn.datasets import make_friedman1, make_regression, make_classification
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor, GeoXGBClassifier

N_FOLDS = 3
N_SEEDS = 3

# ── Datasets ──────────────────────────────────────────────────────────────────

reg_datasets = {
    "friedman1_10k": make_friedman1(n_samples=10_000, n_features=10, noise=1.0, random_state=0),
    "reg_5k":        make_regression(n_samples=5_000,  n_features=20, n_informative=15, noise=1.0, random_state=0),
    "reg_20k":       make_regression(n_samples=20_000, n_features=20, n_informative=15, noise=1.0, random_state=0),
}

# Binary classification: n=10k, 15 informative features
X_clf_bin, y_clf_bin = make_classification(
    n_samples=10_000, n_features=20, n_informative=15, n_redundant=2,
    n_classes=2, class_sep=0.8, random_state=0
)
# Multiclass: n=20k, 5 classes
X_clf_mc, y_clf_mc = make_classification(
    n_samples=20_000, n_features=20, n_informative=15, n_redundant=2,
    n_classes=5, n_clusters_per_class=1, class_sep=0.8, random_state=0
)
clf_datasets = {
    "clf_binary_10k":    (X_clf_bin, y_clf_bin, 2),
    "clf_5class_20k":    (X_clf_mc,  y_clf_mc,  5),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def auc(y_true, proba):
    if proba.shape[1] == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


def run_reg_kfold(model_fn, X, y):
    scores, times = [], []
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, val in kf.split(X):
            m = model_fn(seed)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            times.append(time.perf_counter() - t0)
            scores.append(r2_score(y[val], m.predict(X[val])))
    return np.mean(scores), np.std(scores), np.mean(times)


def run_clf_kfold(model_fn, X, y):
    scores, times = [], []
    for seed in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, val in skf.split(X, y):
            m = model_fn(seed)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            times.append(time.perf_counter() - t0)
            scores.append(auc(y[val], m.predict_proba(X[val])))
    return np.mean(scores), np.std(scores), np.mean(times)


# ── Model factories ───────────────────────────────────────────────────────────

def geo_reg(seed):
    return GeoXGBRegressor(n_rounds=1000, learning_rate=0.02, max_depth=3,
                           sample_block_n="auto", random_state=seed)

def xgb_tuned_reg(seed):
    return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=3,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbosity=0)

def xgb_default_reg(seed):
    return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6,
                             random_state=seed, n_jobs=-1, verbosity=0)

def geo_clf(seed):
    return GeoXGBClassifier(n_rounds=1000, learning_rate=0.02, max_depth=3,
                             sample_block_n="auto", random_state=seed, n_jobs=-1)

def xgb_tuned_clf(seed):
    return xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3,
                               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False,
                               random_state=seed, n_jobs=-1, verbosity=0, eval_metric="logloss")

def xgb_default_clf(seed):
    return xgb.XGBClassifier(n_estimators=1000, random_state=seed,
                               n_jobs=-1, verbosity=0, eval_metric="logloss")


# ── Main ──────────────────────────────────────────────────────────────────────

print()
print("GeoXGB vs XGBoost -- large-n datasets (sample_block_n='auto')")
print("=" * 72)

# Regression
print()
print("  REGRESSION (metric: R2)")
print(f"  {'Model':<18}  {'Dataset':<18}  {'R2':>8}  {'std':>6}  {'t/fold':>7}")
print("  " + "-" * 64)

reg_results = {}
for ds_name, (X, y) in reg_datasets.items():
    n, d = X.shape
    for label, fn in [("GeoXGB_auto", geo_reg), ("XGB_tuned", xgb_tuned_reg), ("XGB_default", xgb_default_reg)]:
        mean_r2, std_r2, mean_t = run_reg_kfold(fn, X, y)
        reg_results[(label, ds_name)] = (mean_r2, std_r2)
        print(f"  {label:<18}  {ds_name:<18}  {mean_r2:.4f}  {std_r2:.4f}  {mean_t:.2f}s")

# Classification
print()
print("  CLASSIFICATION (metric: AUC-ROC)")
print(f"  {'Model':<18}  {'Dataset':<20}  {'AUC':>8}  {'std':>6}  {'t/fold':>7}")
print("  " + "-" * 66)

clf_results = {}
for ds_name, (X, y, n_cls) in clf_datasets.items():
    n, d = X.shape
    for label, fn in [("GeoXGB_auto", geo_clf), ("XGB_tuned", xgb_tuned_clf), ("XGB_default", xgb_default_clf)]:
        mean_auc, std_auc, mean_t = run_clf_kfold(fn, X, y)
        clf_results[(label, ds_name)] = (mean_auc, std_auc)
        print(f"  {label:<18}  {ds_name:<20}  {mean_auc:.4f}  {std_auc:.4f}  {mean_t:.2f}s")

# Summaries
print()
print("  REGRESSION SUMMARY -- GeoXGB_auto vs XGB_tuned:")
for ds in reg_datasets:
    g = reg_results[("GeoXGB_auto", ds)][0]
    x = reg_results[("XGB_tuned", ds)][0]
    winner = "GeoXGB" if g > x else "XGBoost"
    print(f"    {ds:<18}  GeoXGB={g:.4f}  XGB={x:.4f}  diff={g-x:+.4f}  {winner}")

print()
print("  CLASSIFICATION SUMMARY -- GeoXGB_auto vs XGB_tuned:")
for ds in clf_datasets:
    g = clf_results[("GeoXGB_auto", ds)][0]
    x = clf_results[("XGB_tuned", ds)][0]
    winner = "GeoXGB" if g > x else "XGBoost"
    print(f"    {ds:<20}  GeoXGB={g:.4f}  XGB={x:.4f}  diff={g-x:+.4f}  {winner}")
print()
