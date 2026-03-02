"""
Parameter sweep: expand_ratio × y_weight on real-world regression datasets
===========================================================================
Motivation: GeoXGB loses on all 4 real-world regression datasets vs XGBoost
with the current defaults (expand_ratio=0.1, y_weight=0.5).  The HPO meta-
analysis was calibrated on synthetic data; this script checks whether
adjusting these two parameters closes the gap on real-world data.

Sweep:
  expand_ratio : [0.0, 0.1]
  y_weight     : [0.2, 0.3, 0.5, 0.7]
  → 8 GeoXGB configs + 1 XGBoost baseline per dataset

Datasets (same as realworld_compare.py, regression only):
  california_housing, concrete_compressive, abalone, wine_quality

5-fold CV, no X normalisation, NaN imputed.
"""
import sys, io, warnings, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config

RNG      = 42
N_SPLITS = 5

EXPAND_RATIOS = [0.0, 0.1]
Y_WEIGHTS     = [0.2, 0.3, 0.5, 0.7]

BASE_GEO = dict(
    n_rounds=500, learning_rate=0.1, max_depth=3,
    min_samples_leaf=5, reduce_ratio=0.7,
    refit_interval=5, auto_expand=True,
    min_train_samples=100, n_bins=64, random_state=RNG,
)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RNG, verbosity=0, n_jobs=-1,
)


def clean_X(df):
    if not isinstance(df, pd.DataFrame):
        arr = np.asarray(df, dtype=np.float64)
        if np.isnan(arr).any():
            arr = SimpleImputer(strategy="median").fit_transform(arr)
        return arr
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if num_cols and df[num_cols].isnull().any().any():
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
        df[cat_cols] = OrdinalEncoder().fit_transform(df[cat_cols])
    return df.values.astype(np.float64)


def load_datasets():
    datasets = {}
    print("Loading datasets...")

    print("  california_housing...", end=" ", flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    idx = np.random.RandomState(RNG).choice(len(X), 8000, replace=False)
    datasets["california_housing"] = (X[idx].astype(np.float64), y[idx])
    print(f"n={len(idx)}, d={X.shape[1]}")

    print("  concrete_compressive...", end=" ", flush=True)
    try:
        data = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc, yc = clean_X(data.data), np.asarray(data.target, dtype=np.float64)
        datasets["concrete_compressive"] = (Xc, yc)
        print(f"n={len(yc)}, d={Xc.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  abalone...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=183, as_frame=True, parser="auto")
        Xa, ya = clean_X(data.data), np.asarray(data.target, dtype=np.float64)
        datasets["abalone"] = (Xa, ya)
        print(f"n={len(ya)}, d={Xa.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  wine_quality...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=40691, as_frame=True, parser="auto")
        Xw, yw = clean_X(data.data), np.asarray(data.target, dtype=np.float64)
        datasets["wine_quality"] = (Xw, yw)
        print(f"n={len(yw)}, d={Xw.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print()
    return datasets


def cv_r2(model_fn, X, y):
    """Returns (mean_r2, std_r2) over N_SPLITS folds."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
    scores = []
    for tr, va in kf.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[va], m.predict(X[va])))
    return float(np.mean(scores)), float(np.std(scores))


def geo_factory(er, yw):
    def _make():
        cfg = make_cpp_config(**BASE_GEO, expand_ratio=er, y_weight=yw)
        return CppGeoXGBRegressor(cfg)
    return _make


def xgb_factory():
    return xgb.XGBRegressor(**XGB_PARAMS)


def main():
    print("=" * 72)
    print("  GeoXGB param sweep: expand_ratio × y_weight — Regression")
    print("=" * 72)
    print(f"  expand_ratio : {EXPAND_RATIOS}")
    print(f"  y_weight     : {Y_WEIGHTS}")
    print(f"  n_rounds=500, depth=3, lr=0.1, {N_SPLITS}-fold CV")
    print()

    datasets = load_datasets()

    # Cache all results: {ds_name: {"xgb": (mean,std), (er,yw): (mean,std), ...}}
    all_results = {}
    configs = [(er, yw) for er in EXPAND_RATIOS for yw in Y_WEIGHTS]

    for ds_name, (X, y) in datasets.items():
        print(f"── {ds_name}  (n={len(y)}, d={X.shape[1]}) " + "─" * 28)
        row = {}

        # XGBoost baseline
        t0 = time.perf_counter()
        xm, xs = cv_r2(xgb_factory, X, y)
        row["xgb"] = (xm, xs)
        print(f"  {'XGBoost (baseline)':<26}  R²={xm:.4f}±{xs:.4f}  ({time.perf_counter()-t0:.0f}s)")

        for er, yw in configs:
            label = f"er={er} yw={yw}"
            t0 = time.perf_counter()
            gm, gs = cv_r2(geo_factory(er, yw), X, y)
            delta = gm - xm
            sign = "+" if delta >= 0 else ""
            row[(er, yw)] = (gm, gs)
            print(f"  GeoXGB {label:<19}  R²={gm:.4f}±{gs:.4f}  Δ={sign}{delta:.4f}  ({time.perf_counter()-t0:.0f}s)")

        best_key = max(configs, key=lambda k: row[k][0])
        best_r2 = row[best_key][0]
        best_delta = best_r2 - xm
        sign = "+" if best_delta >= 0 else ""
        print(f"  → Best: er={best_key[0]} yw={best_key[1]}  "
              f"R²={best_r2:.4f}  Δ={sign}{best_delta:.4f} vs XGBoost")
        print()
        all_results[ds_name] = row

    # ── Summary across all datasets ───────────────────────────────────────────
    ds_names = list(all_results.keys())
    print("=" * 72)
    print("  Summary: mean Δ(GeoXGB - XGBoost) across all datasets")
    print("=" * 72)
    print(f"\n  {'Config':<20}  {'Mean Δ':>8}  {'Win%':>6}  per-dataset deltas")
    print("  " + "-" * 70)

    ranked = []
    for er, yw in configs:
        deltas = [all_results[ds][(er, yw)][0] - all_results[ds]["xgb"][0]
                  for ds in ds_names]
        ranked.append((er, yw, float(np.mean(deltas)), deltas))

    ranked.sort(key=lambda x: -x[2])
    for er, yw, mean_d, deltas in ranked:
        sign = "+" if mean_d >= 0 else ""
        detail = "  ".join(f"{v:+.4f}" for v in deltas)
        print(f"  er={er} yw={yw:<16}  {sign}{mean_d:.4f}   "
              f"{sum(1 for d in deltas if d>0)/len(deltas)*100:.0f}%   {detail}")

    # Reference: default config (er=0.1, yw=0.5)
    if (0.1, 0.5) in dict([(k[:2], k) for er, yw, *k in ranked]):
        default_row = next((r for r in ranked if r[0] == 0.1 and r[1] == 0.5), None)
        if default_row:
            print(f"\n  Default config (er=0.1, yw=0.5): mean Δ = {default_row[2]:+.4f}")

    best = ranked[0]
    print(f"  Best config  (er={best[0]}  yw={best[1]}): mean Δ = {best[2]:+.4f}")
    improvement = best[2] - next(r[2] for r in ranked if r[0] == 0.1 and r[1] == 0.5)
    sign = "+" if improvement >= 0 else ""
    print(f"  Improvement over default: {sign}{improvement:.4f}")
    print()


if __name__ == "__main__":
    main()
