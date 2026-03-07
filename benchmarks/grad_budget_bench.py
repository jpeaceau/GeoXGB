"""Benchmark gradient-aware budget allocation vs baseline on classification."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from geoxgb import GeoXGBClassifier
import xgboost as xgb

# ---------------------------------------------------------------------------
# Dataset: moderately hard binary classification
# ---------------------------------------------------------------------------
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
    "baseline":           {},
    "gbw=0.3":            {"grad_budget_weight": 0.3},
    "gbw=0.5":            {"grad_budget_weight": 0.5},
    "gbw=0.7":            {"grad_budget_weight": 0.7},
    "gbw=1.0":            {"grad_budget_weight": 1.0},
    "GOSS 20/20":         {"goss_alpha": 0.2, "goss_beta": 0.2},
    "gbw=0.5+stride5":   {"grad_budget_weight": 0.5, "predict_stride": 5},
}

print(f"=== Classification: n={len(X_tr)}, d=20, R=1000 ===")
print(f"{'Config':<24s}  {'Time':>7s}  {'AUC':>8s}")
print("-" * 44)

for name, extra in configs.items():
    params = {**common, **extra}
    clf = GeoXGBClassifier(**params)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"{name:<24s}  {elapsed:>6.1f}s  {auc:>7.4f}")

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
print(f"{'XGBoost':<24s}  {t_xgb:>6.1f}s  {auc:>7.4f}")

# ---------------------------------------------------------------------------
# Second dataset: breast cancer (small, standard)
# ---------------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X2, y2 = data.data, data.target
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)

print(f"\n=== Breast Cancer: n={len(X2_tr)}, d=30, R=500 ===")
print(f"{'Config':<24s}  {'AUC':>8s}")
print("-" * 34)

common2 = dict(n_rounds=500, learning_rate=0.02, max_depth=3,
               refit_interval=50, y_weight=0, convergence_tol=None)

for name, gbw in [("baseline", 0.0), ("gbw=0.3", 0.3), ("gbw=0.5", 0.5),
                   ("gbw=0.7", 0.7), ("gbw=1.0", 1.0)]:
    clf = GeoXGBClassifier(**common2, grad_budget_weight=gbw)
    clf.fit(X2_tr, y2_tr)
    proba = clf.predict_proba(X2_te)[:, 1]
    auc = roc_auc_score(y2_te, proba)
    print(f"{name:<24s}  {auc:>7.4f}")

xclf2 = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.02,
                           tree_method='hist', verbosity=0, random_state=42)
xclf2.fit(X2_tr, y2_tr)
auc2 = roc_auc_score(y2_te, xclf2.predict_proba(X2_te)[:, 1])
print(f"{'XGBoost':<24s}  {auc2:>7.4f}")
