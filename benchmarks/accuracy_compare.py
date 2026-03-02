"""
Accuracy comparison: GeoXGB (any version) vs XGBoost.
Outputs a compact JSON block for easy side-by-side comparison.

Usage:
  /tmp/geoxgb_pypi_env/Scripts/python benchmarks/accuracy_compare.py   # PyPI 0.1.7
  python benchmarks/accuracy_compare.py                                  # local 0.2.0
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

# ── Version banner ────────────────────────────────────────────────────────────
import geoxgb
geoxgb_version = getattr(geoxgb, "__version__", "?")
print(f"Python {sys.version.split()[0]}  |  geoxgb {geoxgb_version}  |  xgboost {xgb.__version__}")

# ── Datasets ──────────────────────────────────────────────────────────────────
datasets = {}

X_f, y_f = make_friedman1(n_samples=5000, n_features=10, random_state=0)
datasets["friedman1 (n=5000,d=10)"] = (X_f, y_f)

X_c, y_c = fetch_california_housing(return_X_y=True)
rng = np.random.RandomState(0)
idx = rng.choice(len(X_c), min(10000, len(X_c)), replace=False)
datasets["california (n=10k,d=8)"] = (X_c[idx], y_c[idx])

X_s = rng.randn(3000, 20)
y_s = (X_s[:, 0]**2 + np.sin(X_s[:, 1]) +
       X_s[:, 2] * X_s[:, 3] + 0.5 * rng.randn(3000))
datasets["synthetic nonlin (n=3k,d=20)"] = (X_s, y_s)

# ── Model factories ───────────────────────────────────────────────────────────
def make_geoxgb():
    try:
        # 0.2.0 C++ path
        from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config
        cfg = make_cpp_config(
            n_rounds=200, learning_rate=0.1, max_depth=3,
            min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.5,
            refit_interval=5, auto_expand=True, expand_ratio=0.1,
            min_train_samples=100, n_bins=64, random_state=0,
        )
        return CppGeoXGBRegressor(cfg), "GeoXGB 0.2.0 (C++)"
    except Exception:
        pass
    try:
        # 0.1.x Python path — uses n_rounds, not n_estimators
        from geoxgb import GeoXGBRegressor
        return GeoXGBRegressor(
            n_rounds=200, learning_rate=0.1, max_depth=3,
            min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.5,
            refit_interval=5, auto_expand=True, expand_ratio=0.1,
            random_state=0,
        ), f"GeoXGB {geoxgb_version} (Python)"
    except Exception as e:
        print(f"  [WARN] GeoXGB init failed: {e}")
        return None, "GeoXGB (unavailable)"

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=0, verbosity=0,
    ), "XGBoost 3.x"

# ── CV runner ─────────────────────────────────────────────────────────────────
def cv_r2(model_fn, X, y, n_splits=5):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(X):
        m, _ = model_fn()
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[va], m.predict(X[va])))
    return float(np.mean(scores)), float(np.std(scores))

# ── Run ───────────────────────────────────────────────────────────────────────
results = {}
print()

for ds_name, (X, y) in datasets.items():
    print(f"Dataset: {ds_name}")
    row = {}

    t0 = time.perf_counter()
    geo_mean, geo_std = cv_r2(make_geoxgb, X, y)
    geo_time = time.perf_counter() - t0
    _, geo_label = make_geoxgb()
    row["geoxgb"] = {"r2": round(geo_mean, 4), "std": round(geo_std, 4),
                     "time_s": round(geo_time, 1), "label": geo_label}
    print(f"  {geo_label:<36} R2={geo_mean:.4f} +/- {geo_std:.4f}  ({geo_time:.1f}s)")

    t0 = time.perf_counter()
    xgb_mean, xgb_std = cv_r2(make_xgb, X, y)
    xgb_time = time.perf_counter() - t0
    _, xgb_label = make_xgb()
    row["xgboost"] = {"r2": round(xgb_mean, 4), "std": round(xgb_std, 4),
                      "time_s": round(xgb_time, 1), "label": xgb_label}
    print(f"  {xgb_label:<36} R2={xgb_mean:.4f} +/- {xgb_std:.4f}  ({xgb_time:.1f}s)")

    delta = geo_mean - xgb_mean
    print(f"  Delta (GeoXGB - XGB):                  {delta:+.4f}")
    row["delta_r2"] = round(delta, 4)
    results[ds_name] = row
    print()

# ── JSON summary ──────────────────────────────────────────────────────────────
print("=== JSON ===")
print(json.dumps({"geoxgb_version": geoxgb_version, "results": results}, indent=2))
