"""
mae_gap_diagnostic.py
=====================
Isolates why GeoXGB trails XGBoost on MAE for california and friedman1.

Hypotheses tested:
  A. Tree depth mismatch  — XGBoost uses max_depth=4, GeoXGB uses max_depth=2
  B. Sample reduction     — FPS reduces training set to 80%; does full-set help?
  C. Column subsampling   — XGBoost colsample_bytree=0.8 gives implicit regularization
  D. Second-order gradients — XGBoost uses Newton step; GeoXGB uses first-order only

Controls:
  XGBoost depth=4       — current benchmark baseline
  XGBoost depth=2       — matched to GeoXGB (tests hypothesis A)
  XGBoost depth=2 nosub — no column subsampling (tests hypothesis C)
  GeoXGB depth=2        — current benchmark baseline
  GeoXGB depth=4        — matched to XGBoost (tests hypothesis A)
  GeoXGB no-reduce      — reduce_ratio=1.0, expand_ratio=0.0 (tests hypothesis B)
  GeoXGB depth=4 no-red — combined (A+B)
  HART+simplex depth=2  — best HART variant from prior bench
  HART+simplex depth=4  — HART with matched depth

Usage:
  python benchmarks/mae_gap_diagnostic.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_friedman1
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

N_SEEDS  = 1
N_FOLDS  = 3
N_CAP    = 2000
N_ROUNDS = 1000

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed; XGBoost configs skipped.")

from geoxgb import GeoXGBRegressor

# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------

def load_california():
    d = fetch_california_housing()
    X, y = d.data, d.target
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=N_CAP, replace=False)
    return X[idx], y[idx], f"california(n={N_CAP})"


def load_friedman1():
    X, y = make_friedman1(n_samples=min(1000, N_CAP), noise=1.0, random_state=0)
    return X, y, f"friedman1(n={len(X)})"


# ------------------------------------------------------------------
# Config factory
# ------------------------------------------------------------------

_GEO_BASE = dict(
    n_rounds=N_ROUNDS, learning_rate=0.02, min_samples_leaf=5,
    reduce_ratio=0.8, expand_ratio=0.1, y_weight=0.5,
    variance_weighted=False, refit_interval=5, auto_noise=False,
    noise_guard=False, random_state=42,
)

CONFIGS = []

# XGBoost configs (hypothesis A, C)
if HAS_XGB:
    CONFIGS += [
        ("XGB depth=4 sub=0.8",  "xgb", dict(max_depth=4, subsample=0.8,  colsample_bytree=0.8)),
        ("XGB depth=2 sub=0.8",  "xgb", dict(max_depth=2, subsample=0.8,  colsample_bytree=0.8)),
        ("XGB depth=2 sub=1.0",  "xgb", dict(max_depth=2, subsample=1.0,  colsample_bytree=1.0)),
        ("XGB depth=4 sub=1.0",  "xgb", dict(max_depth=4, subsample=1.0,  colsample_bytree=1.0)),
    ]

# GeoXGB configs (hypothesis A, B)
CONFIGS += [
    ("GeoXGB depth=2 r=0.8",   "geo", dict(max_depth=2, reduce_ratio=0.8, expand_ratio=0.1,  partitioner='hvrt', method='variance_ordered', generation_strategy='epanechnikov', adaptive_reduce_ratio=False)),
    ("GeoXGB depth=4 r=0.8",   "geo", dict(max_depth=4, reduce_ratio=0.8, expand_ratio=0.1,  partitioner='hvrt', method='variance_ordered', generation_strategy='epanechnikov', adaptive_reduce_ratio=False)),
    ("GeoXGB depth=2 no-red",  "geo", dict(max_depth=2, reduce_ratio=1.0, expand_ratio=0.0,  partitioner='hvrt', method='variance_ordered', generation_strategy='epanechnikov', adaptive_reduce_ratio=False)),
    ("GeoXGB depth=4 no-red",  "geo", dict(max_depth=4, reduce_ratio=1.0, expand_ratio=0.0,  partitioner='hvrt', method='variance_ordered', generation_strategy='epanechnikov', adaptive_reduce_ratio=False)),
    # HART+simplex (best from prior bench)
    ("HART+simplex depth=2",   "geo", dict(max_depth=2, reduce_ratio=0.8, expand_ratio=0.1,  partitioner='hart', method='orthant_stratified', generation_strategy='simplex_mixup', adaptive_reduce_ratio=True)),
    ("HART+simplex depth=4",   "geo", dict(max_depth=4, reduce_ratio=0.8, expand_ratio=0.1,  partitioner='hart', method='orthant_stratified', generation_strategy='simplex_mixup', adaptive_reduce_ratio=True)),
]


def make_model(kind, kwargs):
    if kind == "xgb":
        return xgb.XGBRegressor(
            n_estimators=N_ROUNDS, learning_rate=0.02,
            tree_method="hist", random_state=42, verbosity=0,
            **kwargs,
        )
    else:
        kw = dict(_GEO_BASE)
        kw.update(kwargs)
        return GeoXGBRegressor(**kw)


def cv_evaluate(kind, kwargs, X, y):
    maes, y_range = [], y.max() - y.min() + 1e-12
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = make_model(kind, kwargs)
            m.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[te], m.predict(X[te])))
    return float(np.mean(maes)), float(np.mean(maes) / y_range)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print(f"MAE gap diagnostic  |  seeds={N_SEEDS} folds={N_FOLDS} n_cap={N_CAP} n_rounds={N_ROUNDS}")

    datasets = [load_california(), load_friedman1()]
    all_rows = []

    for X, y, ds_name in datasets:
        print(f"\n{'='*72}")
        print(f"Dataset: {ds_name}  d={X.shape[1]}")
        print(f"{'='*72}")
        print(f"  {'Config':<30}  {'MAE':>8}  {'NMAE':>8}  {'vs XGB4':>9}  {'time':>7}")

        # reference MAE (XGB depth=4 sub=0.8)
        ref_mae, ref_nmae = None, None
        xgb4_key = "XGB depth=4 sub=0.8"

        results = {}
        for label, kind, kwargs in CONFIGS:
            t0 = time.time()
            mae, nmae = cv_evaluate(kind, kwargs, X, y)
            elapsed = time.time() - t0
            results[label] = (mae, nmae, elapsed)
            if label == xgb4_key:
                ref_mae, ref_nmae = mae, nmae

        for label, kind, kwargs in CONFIGS:
            mae, nmae, elapsed = results[label]
            vs = (nmae - ref_nmae) if ref_nmae is not None else float("nan")
            tag = " <-- better" if vs < -0.001 else ("" if abs(vs) <= 0.001 else " <-- worse")
            sign = "+" if vs >= 0 else ""
            print(f"  {label:<30}  {mae:>8.4f}  {nmae:>8.4f}  {sign}{vs:>+8.4f}  {elapsed:>6.1f}s{tag}")
            all_rows.append(dict(dataset=ds_name, config=label, MAE=mae, NMAE=nmae, vs_xgb4=vs, time=elapsed))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("HYPOTHESIS SUMMARY")
    print(f"{'='*72}")
    df = pd.DataFrame(all_rows)

    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name].set_index("config")
        print(f"\n{ds_name}:")
        ref_nmae = sub.loc[xgb4_key, "NMAE"] if xgb4_key in sub.index else None

        checks = [
            ("Hyp A (depth): XGB4 vs XGB2",        "XGB depth=4 sub=0.8", "XGB depth=2 sub=0.8"),
            ("Hyp C (colsub): XGB2_sub vs XGB2",   "XGB depth=2 sub=0.8", "XGB depth=2 sub=1.0"),
            ("Hyp A (depth): Geo4 vs Geo2",         "GeoXGB depth=2 r=0.8", "GeoXGB depth=4 r=0.8"),
            ("Hyp B (reduce): Geo2_red vs Geo2",    "GeoXGB depth=2 r=0.8", "GeoXGB depth=2 no-red"),
            ("Depth=4 closes gap?",                  "XGB depth=4 sub=0.8", "GeoXGB depth=4 r=0.8"),
            ("No-reduce closes gap?",                "XGB depth=4 sub=0.8", "GeoXGB depth=2 no-red"),
            ("HART2 vs XGB4",                        "XGB depth=4 sub=0.8", "HART+simplex depth=2"),
            ("HART4 vs XGB4",                        "XGB depth=4 sub=0.8", "HART+simplex depth=4"),
        ]

        for label, key_a, key_b in checks:
            if key_a not in sub.index or key_b not in sub.index:
                continue
            nmae_a = sub.loc[key_a, "NMAE"]
            nmae_b = sub.loc[key_b, "NMAE"]
            delta = nmae_b - nmae_a   # negative = B is better
            effect = "B better" if delta < -0.001 else ("A better" if delta > 0.001 else "tied")
            print(f"  {label:<45} delta={delta:+.4f}  ({effect})")

    return df


if __name__ == "__main__":
    df = main()
