"""Benchmark: gradient amplification + reduce_ratio tuning for classification."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from geoxgb import GeoXGBClassifier
import xgboost as xgb

n_total = 10_000
X, y = make_classification(
    n_samples=n_total, n_features=20, n_informative=12,
    n_redundant=4, n_clusters_per_class=3, flip_y=0.05,
    random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

common = dict(
    n_rounds=1000, learning_rate=0.02, max_depth=3,
    refit_interval=50, y_weight=0, convergence_tol=None,
)

configs = {
    # Baselines
    "baseline (rr=0.8)":           {"reduce_ratio": 0.8},
    "GOSS 20/20":                  {"goss_alpha": 0.2, "goss_beta": 0.2},
    # Gradient amplification alone
    "gbw=0.7 rr=0.8":             {"grad_budget_weight": 0.7, "reduce_ratio": 0.8},
    "gbw=1.0 rr=0.8":             {"grad_budget_weight": 1.0, "reduce_ratio": 0.8},
    # Lower reduce_ratio (HVRT regularization)
    "rr=0.5":                      {"reduce_ratio": 0.5},
    "rr=0.3":                      {"reduce_ratio": 0.3},
    # Gradient amplification + aggressive reduction (deterministic GOSS via HVRT)
    "gbw=0.7 rr=0.5":             {"grad_budget_weight": 0.7, "reduce_ratio": 0.5},
    "gbw=1.0 rr=0.5":             {"grad_budget_weight": 1.0, "reduce_ratio": 0.5},
    "gbw=0.7 rr=0.3":             {"grad_budget_weight": 0.7, "reduce_ratio": 0.3},
    "gbw=1.0 rr=0.3":             {"grad_budget_weight": 1.0, "reduce_ratio": 0.3},
    # Combinations with other opts
    "gbw=0.7 rr=0.5 stride5":     {"grad_budget_weight": 0.7, "reduce_ratio": 0.5,
                                    "predict_stride": 5},
    "gbw=1.0 rr=0.3 col=0.7":     {"grad_budget_weight": 1.0, "reduce_ratio": 0.3,
                                    "colsample_bytree": 0.7},
}

print(f"=== Classification: n={len(X_tr)}, d=20, R=1000 ===")
print(f"{'Config':<30s}  {'Time':>7s}  {'AUC':>8s}")
print("-" * 50)

for name, extra in configs.items():
    params = {**common, **extra}
    clf = GeoXGBClassifier(**params)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"{name:<30s}  {elapsed:>6.1f}s  {auc:>7.4f}")

# XGBoost reference
xclf = xgb.XGBClassifier(
    n_estimators=1000, max_depth=3, learning_rate=0.02,
    tree_method='hist', verbosity=0, random_state=42,
)
t0 = time.perf_counter()
xclf.fit(X_tr, y_tr)
t_xgb = time.perf_counter() - t0
proba = xclf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
print(f"{'XGBoost':<30s}  {t_xgb:>6.1f}s  {auc:>7.4f}")
