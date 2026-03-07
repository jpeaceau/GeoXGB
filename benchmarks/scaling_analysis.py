"""Scaling analysis: GeoXGB vs XGBoost cost at various n and n_rounds."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from geoxgb import GeoXGBClassifier
import xgboost as xgb

# --- Scaling with n (fixed n_rounds=500) ---
print("=== Cost vs n (n_rounds=500) ===")
print(f"{'n':>8s}  {'GeoXGB':>8s}  {'ms/rnd':>7s}  {'XGB':>8s}  {'ratio':>6s}")
n_rounds_fixed = 500
for n in [10_000, 25_000, 50_000, 100_000, 200_000, 400_000]:
    X, y = make_classification(n_samples=n, n_features=19, n_informative=10,
                               n_redundant=5, random_state=42)
    # GeoXGB
    clf = GeoXGBClassifier(n_rounds=n_rounds_fixed, fast_refit=True,
                           y_weight=0, learning_rate=0.02, max_depth=3,
                           refit_interval=50, convergence_tol=None)
    t0 = time.perf_counter()
    clf.fit(X, y)
    t_geo = time.perf_counter() - t0

    # XGBoost
    xclf = xgb.XGBClassifier(n_estimators=n_rounds_fixed, max_depth=3,
                              learning_rate=0.02, tree_method='hist',
                              verbosity=0, random_state=42)
    t0 = time.perf_counter()
    xclf.fit(X, y)
    t_xgb = time.perf_counter() - t0

    ratio = t_geo / t_xgb if t_xgb > 0 else float('inf')
    print(f"{n:>8,d}  {t_geo:>7.1f}s  {t_geo/n_rounds_fixed*1000:>6.1f}  {t_xgb:>7.1f}s  {ratio:>5.1f}x")

# --- Scaling with n_rounds (fixed n=50k) ---
print(f"\n=== Cost vs n_rounds (n=50k) ===")
print(f"{'rounds':>8s}  {'GeoXGB':>8s}  {'ms/rnd':>7s}  {'XGB':>8s}  {'ratio':>6s}")
X, y = make_classification(n_samples=50_000, n_features=19, n_informative=10,
                           n_redundant=5, random_state=42)
for nr in [100, 250, 500, 1000, 2000, 4000]:
    clf = GeoXGBClassifier(n_rounds=nr, fast_refit=True,
                           y_weight=0, learning_rate=0.02, max_depth=3,
                           refit_interval=50, convergence_tol=None)
    t0 = time.perf_counter()
    clf.fit(X, y)
    t_geo = time.perf_counter() - t0

    xclf = xgb.XGBClassifier(n_estimators=nr, max_depth=3,
                              learning_rate=0.02, tree_method='hist',
                              verbosity=0, random_state=42)
    t0 = time.perf_counter()
    xclf.fit(X, y)
    t_xgb = time.perf_counter() - t0

    ratio = t_geo / t_xgb if t_xgb > 0 else float('inf')
    print(f"{nr:>8,d}  {t_geo:>7.1f}s  {t_geo/nr*1000:>6.1f}  {t_xgb:>7.1f}s  {ratio:>5.1f}x")
