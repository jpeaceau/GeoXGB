"""Benchmark: measure timing improvement from data structure optimizations."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from geoxgb import GeoXGBClassifier, GeoXGBRegressor

# --- Classification with GOSS (exercises the optimized path) ---
print("=== Classification (GOSS active) ===")
X, y = make_classification(n_samples=10_000, n_features=20, n_informative=12,
                           n_redundant=4, n_clusters_per_class=3, flip_y=0.05,
                           random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

common_cls = dict(n_rounds=1000, learning_rate=0.02, max_depth=3,
                  refit_interval=50, y_weight=0, convergence_tol=None)

configs_cls = {
    "baseline":      {},
    "GOSS 20/20":    {"goss_alpha": 0.2, "goss_beta": 0.2},
    "GOSS 30/10":    {"goss_alpha": 0.3, "goss_beta": 0.1},
}

print(f"{'Config':<20s}  {'Time':>7s}  {'AUC':>8s}")
print("-" * 40)
for name, extra in configs_cls.items():
    clf = GeoXGBClassifier(**common_cls, **extra)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"{name:<20s}  {elapsed:>6.2f}s  {auc:>7.4f}")

# --- Regression (exercises expand path) ---
print("\n=== Regression (expand active) ===")
X_r, y_r = make_regression(n_samples=10_000, n_features=10, n_informative=8,
                           noise=10, random_state=42)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

common_reg = dict(n_rounds=1000, learning_rate=0.02, max_depth=3,
                  refit_interval=50, convergence_tol=None)

configs_reg = {
    "expand=0.0":  {"expand_ratio": 0.0},
    "expand=0.1":  {"expand_ratio": 0.1},
    "expand=0.3":  {"expand_ratio": 0.3},
}

print(f"{'Config':<20s}  {'Time':>7s}  {'R2':>8s}")
print("-" * 40)
for name, extra in configs_reg.items():
    reg = GeoXGBRegressor(**common_reg, **extra)
    t0 = time.perf_counter()
    reg.fit(Xr_tr, yr_tr)
    elapsed = time.perf_counter() - t0
    r2 = r2_score(yr_te, reg.predict(Xr_te))
    print(f"{name:<20s}  {elapsed:>6.2f}s  {r2:>7.4f}")
