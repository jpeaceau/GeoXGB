"""Benchmark the kernel optimizations: colsample, GOSS, predict stride."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from geoxgb import GeoXGBClassifier
import xgboost as xgb

n_total = 400_000
X, y = make_classification(n_samples=n_total, n_features=19, n_informative=10,
                           n_redundant=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

common = dict(
    n_rounds=2000, fast_refit=True, y_weight=0, learning_rate=0.02,
    max_depth=3, refit_interval=50, convergence_tol=None
)

# Run each config twice and take the faster time to reduce noise
configs = {
    "baseline":                  {},
    "col=0.7":                   {"colsample_bytree": 0.7},
    "GOSS 20/20":                {"goss_alpha": 0.2, "goss_beta": 0.2},
    "GOSS 30/20":                {"goss_alpha": 0.3, "goss_beta": 0.2},
    "stride=10":                 {"predict_stride": 10},
    "stride=5":                  {"predict_stride": 5},
    "col + GOSS20":              {"colsample_bytree": 0.7, "goss_alpha": 0.2, "goss_beta": 0.2},
    "col + GOSS20 + stride5":    {"colsample_bytree": 0.7, "goss_alpha": 0.2, "goss_beta": 0.2,
                                  "predict_stride": 5},
}

print(f"=== n={len(X_tr)}, d=19, R=2000 (best of 2 runs) ===")
print(f"{'Config':<28s}  {'Time':>7s}  {'AUC':>8s}  {'vs base':>8s}")
baseline_time = None

for name, extra in configs.items():
    params = {**common, **extra}
    best_time = float('inf')
    best_auc = 0
    for _ in range(2):
        clf = GeoXGBClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        proba = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        if elapsed < best_time:
            best_time = elapsed
            best_auc = auc
    if baseline_time is None:
        baseline_time = best_time
    speedup = baseline_time / best_time
    print(f"{name:<28s}  {best_time:>6.1f}s  {best_auc:>7.4f}  {speedup:>7.2f}x")

# XGBoost
xclf = xgb.XGBClassifier(n_estimators=2000, max_depth=3, learning_rate=0.02,
                          tree_method='hist', verbosity=0, random_state=42)
t0 = time.perf_counter()
xclf.fit(X_tr, y_tr)
t_xgb = time.perf_counter() - t0
proba = xclf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
print(f"{'XGBoost':<28s}  {t_xgb:>6.1f}s  {auc:>7.4f}")
