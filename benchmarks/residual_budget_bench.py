"""
residual_budget_bench.py
========================
Compares 7 FPS/partitioner configurations vs XGBoost baseline on
california housing (subsampled), concrete strength, and friedman1 datasets.

Metrics: R2, MAE, NMAE (normalised MAE), net_rel_adv vs XGBoost.

Configurations
--------------
0. XGBoost baseline
1. GeoXGB baseline          hvrt / variance_ordered / adaptive=False / 0.8
2. reduce_ratio=0.95        hvrt / variance_ordered / adaptive=False / 0.95
3. MAD budget               hvrt / kde_stratified   / adaptive=False / 0.8
4. Residual-stratified      hvrt / residual_stratified / adaptive=False / 0.8
5. Adaptive ratio           hvrt / variance_ordered / adaptive=True  / 0.8
6. HART                     hart / variance_ordered / adaptive=False / 0.8
7. HART + all               hart / residual_stratified / adaptive=True / 0.8

Usage
-----
  python benchmarks/residual_budget_bench.py          # full: 3 seeds x 5 folds
  python benchmarks/residual_budget_bench.py --fast   # quick: 1 seed x 3 folds, n<=2000
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

FAST = "--fast" in sys.argv
N_SEEDS  = 1 if FAST else 3
N_FOLDS  = 3 if FAST else 5
N_CAP    = 2000  # always cap — n=20k × 7 configs × 15 folds is prohibitive
N_ROUNDS = 300 if FAST else 1000

# ------------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------------
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed; XGBoost baseline skipped.")

# ------------------------------------------------------------------
# GeoXGB
# ------------------------------------------------------------------
from geoxgb import GeoXGBRegressor

# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------

def load_california(n_cap=N_CAP):
    d = fetch_california_housing()
    X, y = d.data, d.target
    if n_cap and len(X) > n_cap:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), size=n_cap, replace=False)
        X, y = X[idx], y[idx]
    return X, y, f"california(n={len(X)})"


def load_concrete(n_cap=N_CAP):
    try:
        df = pd.read_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "concrete/compressive/Concrete_Data.xls",
            engine="openpyxl"
        )
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
    except Exception:
        # Fallback: synthetic concrete-like dataset
        rng = np.random.RandomState(0)
        X = rng.randn(1030, 8)
        y = (X[:, 0] * 2 + X[:, 1] ** 2 + rng.randn(1030) * 0.5).astype(float)
    if n_cap and len(X) > n_cap:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), size=n_cap, replace=False)
        X, y = X[idx], y[idx]
    return X, y, f"concrete(n={len(X)})"


def load_friedman1(n_cap=N_CAP):
    from sklearn.datasets import make_friedman1
    n = min(1000, n_cap) if n_cap else 1000
    X, y = make_friedman1(n_samples=n, noise=1.0, random_state=0)
    return X, y, f"friedman1(n={len(X)})"


# ------------------------------------------------------------------
# Config definitions
# ------------------------------------------------------------------

_REG_DEFAULTS = dict(
    n_rounds=N_ROUNDS, learning_rate=0.02, max_depth=2, min_samples_leaf=5,
    reduce_ratio=0.8, expand_ratio=0.1, y_weight=0.5,
    variance_weighted=False, refit_interval=5, auto_noise=False,
    noise_guard=False, random_state=42,
)

CONFIGS = [
    # (label, partitioner, method, adaptive_reduce_ratio, reduce_ratio, generation_strategy)
    ("GeoXGB baseline",        "hvrt",         "variance_ordered",    False, 0.8,  "epanechnikov"),
    ("HART + simplex",         "hart",         "orthant_stratified",  True,  0.8,  "simplex_mixup"),
    ("PyramidHART + simplex",  "pyramid_hart", "orthant_stratified",  True,  0.8,  "simplex_mixup"),
    ("PyramidHART + laplace",  "pyramid_hart", "orthant_stratified",  True,  0.8,  "laplace"),
    ("PyramidHART + epan",     "pyramid_hart", "orthant_stratified",  True,  0.8,  "epanechnikov"),
    ("PyramidHART no-adapt",   "pyramid_hart", "orthant_stratified",  False, 0.8,  "simplex_mixup"),
    ("PyramidHART var-ordered","pyramid_hart", "variance_ordered",    True,  0.8,  "simplex_mixup"),
]


def make_geoxgb(partitioner, method, adaptive_reduce_ratio, reduce_ratio, generation_strategy):
    kw = dict(_REG_DEFAULTS)
    kw["partitioner"] = partitioner
    kw["method"] = method
    kw["adaptive_reduce_ratio"] = adaptive_reduce_ratio
    kw["reduce_ratio"] = reduce_ratio
    kw["generation_strategy"] = generation_strategy
    return GeoXGBRegressor(**kw)


def make_xgb():
    if not HAS_XGB:
        return None
    return xgb.XGBRegressor(
        n_estimators=N_ROUNDS, learning_rate=0.02, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42,
        verbosity=0,
    )


# ------------------------------------------------------------------
# CV evaluation
# ------------------------------------------------------------------

def cv_evaluate(model_factory, X, y, n_splits=N_FOLDS, n_seeds=N_SEEDS):
    r2s, maes = [], []
    y_range = y.max() - y.min() + 1e-12
    for seed in range(n_seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = model_factory()
            if m is None:
                return None, None
            m.fit(X[tr], y[tr])
            preds = m.predict(X[te])
            r2s.append(r2_score(y[te], preds))
            maes.append(mean_absolute_error(y[te], preds))
    return float(np.mean(r2s)), float(np.mean(maes) / y_range)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    mode = "FAST" if FAST else "FULL"
    print(f"Mode: {mode}  |  seeds={N_SEEDS}  folds={N_FOLDS}  n_cap={N_CAP}  n_rounds={N_ROUNDS}")

    datasets = [load_california(), load_friedman1(), load_concrete()]
    all_rows = []

    for X, y, ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}  d={X.shape[1]}")
        print(f"{'='*70}")

        # XGBoost baseline
        t0 = time.time()
        xgb_r2, xgb_nmae = cv_evaluate(make_xgb, X, y)
        xgb_time = time.time() - t0
        xgb_mae_abs = xgb_nmae * (y.max() - y.min()) if xgb_nmae is not None else None
        if xgb_r2 is not None:
            print(f"  {'XGBoost':<25}  R2={xgb_r2:.4f}  NMAE={xgb_nmae:.4f}  t={xgb_time:.1f}s")
        all_rows.append({
            "dataset": ds_name, "config": "XGBoost",
            "R2": xgb_r2, "NMAE": xgb_nmae,
            "vs_xgb_R2": 0.0, "vs_xgb_NMAE": 0.0,
        })

        for label, partitioner, method, adaptive, rr, gen in CONFIGS:
            t0 = time.time()
            r2, nmae = cv_evaluate(
                lambda p=partitioner, m=method, a=adaptive, r=rr, g=gen: make_geoxgb(p, m, a, r, g),
                X, y,
            )
            elapsed = time.time() - t0
            vs_r2   = (r2   - xgb_r2)   if xgb_r2   is not None else float("nan")
            vs_nmae = (nmae - xgb_nmae)  if xgb_nmae is not None else float("nan")
            sr2   = "+" if vs_r2   >= 0 else ""
            snmae = "+" if vs_nmae >= 0 else ""
            # negative vs_nmae = lower MAE = better
            mae_tag = " <-- better MAE" if vs_nmae < -0.001 else ""
            print(
                f"  {label:<25}  R2={r2:.4f}  NMAE={nmae:.4f}  "
                f"dR2={sr2}{vs_r2:.4f}  dNMAE={snmae}{vs_nmae:.4f}  "
                f"t={elapsed:.1f}s{mae_tag}"
            )
            all_rows.append({
                "dataset": ds_name, "config": label,
                "R2": r2, "NMAE": nmae,
                "vs_xgb_R2": vs_r2, "vs_xgb_NMAE": vs_nmae,
            })

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY — mean delta vs XGBoost across all datasets")
    print(f"{'='*70}")
    df = pd.DataFrame(all_rows)
    geo = df[df["config"] != "XGBoost"].copy()

    n_ds = len(datasets)
    summary = geo.groupby("config").agg(
        mean_dR2=("vs_xgb_R2",   "mean"),
        mean_dNMAE=("vs_xgb_NMAE", "mean"),
        R2_wins=("vs_xgb_R2",   lambda x: int((x > 0).sum())),
        MAE_wins=("vs_xgb_NMAE", lambda x: int((x < 0).sum())),
    )
    # Reindex to match CONFIGS order
    ordered = [c[0] for c in CONFIGS]
    summary = summary.reindex(ordered)
    summary.columns = [f"mean_dR2/{n_ds}ds", f"mean_dNMAE/{n_ds}ds",
                       f"R2_wins/{n_ds}", f"MAE_wins/{n_ds}"]
    print(summary.to_string(float_format="{:+.4f}".format))

    # vs GeoXGB baseline
    print(f"\n--- vs GeoXGB baseline ---")
    base_r2   = geo[geo["config"] == "GeoXGB baseline"]["vs_xgb_R2"].mean()
    base_nmae = geo[geo["config"] == "GeoXGB baseline"]["vs_xgb_NMAE"].mean()
    for label, _, _, _, _, _ in CONFIGS[1:]:
        row = geo[geo["config"] == label]
        dr2   = row["vs_xgb_R2"].mean()   - base_r2
        dnmae = row["vs_xgb_NMAE"].mean() - base_nmae
        sr2   = "+" if dr2   >= 0 else ""
        snmae = "+" if dnmae >= 0 else ""
        mae_tag = " <-- MAE gain" if dnmae < -0.001 else ""
        print(f"  {label:<25}  dR2_vs_base={sr2}{dr2:.4f}  dNMAE_vs_base={snmae}{dnmae:.4f}{mae_tag}")

    return df


if __name__ == "__main__":
    df = main()
