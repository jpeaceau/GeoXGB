"""Benchmark n_bins=16,32,64 with current fast_refit + binned predict pipeline."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from geoxgb import GeoXGBClassifier, GeoXGBRegressor

# --- Classification (churn-like, 50k) ---
print("=== Classification (n=50k, d=19, n_rounds=1000) ===")
print(f"{'bins':>6s}  {'time':>7s}  {'AUC':>8s}  {'ms/rnd':>7s}")
X, y = make_classification(n_samples=50_000, n_features=19, n_informative=10,
                           n_redundant=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

for bins in [16, 32, 64, 128]:
    clf = GeoXGBClassifier(n_rounds=1000, fast_refit=True,
                           y_weight=0, learning_rate=0.02, max_depth=3,
                           refit_interval=50, n_bins=bins,
                           convergence_tol=None)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"{bins:>6d}  {elapsed:>6.1f}s  {auc:>7.4f}  {elapsed/1000*1000:>6.1f}")

# --- Regression (friedman-like, 50k) ---
print(f"\n=== Regression (n=50k, d=10, n_rounds=1000) ===")
print(f"{'bins':>6s}  {'time':>7s}  {'R2':>8s}  {'ms/rnd':>7s}")
X, y = make_regression(n_samples=50_000, n_features=10, n_informative=8,
                       noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

for bins in [16, 32, 64, 128]:
    reg = GeoXGBRegressor(n_rounds=1000, fast_refit=True,
                          y_weight=0.25, learning_rate=0.02, max_depth=3,
                          refit_interval=50, n_bins=bins,
                          convergence_tol=None)
    t0 = time.perf_counter()
    reg.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    preds = reg.predict(X_te)
    r2 = r2_score(y_te, preds)
    print(f"{bins:>6d}  {elapsed:>6.1f}s  {r2:>7.4f}  {elapsed/1000*1000:>6.1f}")

# --- Large-n classification (200k) ---
print(f"\n=== Classification (n=200k, d=19, n_rounds=1000) ===")
print(f"{'bins':>6s}  {'time':>7s}  {'AUC':>8s}  {'ms/rnd':>7s}")
X, y = make_classification(n_samples=200_000, n_features=19, n_informative=10,
                           n_redundant=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

for bins in [16, 32, 64, 128]:
    clf = GeoXGBClassifier(n_rounds=1000, fast_refit=True,
                           y_weight=0, learning_rate=0.02, max_depth=3,
                           refit_interval=50, n_bins=bins,
                           convergence_tol=None)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    print(f"{bins:>6d}  {elapsed:>6.1f}s  {auc:>7.4f}  {elapsed/1000*1000:>6.1f}")
