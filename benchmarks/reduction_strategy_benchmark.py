"""
Reduction Strategy Benchmark
=============================

Compares four HVRT-based sample reduction strategies for GeoXGB:

  fps              -- Farthest Point Sampling (default, O(budget^2/P) per partition)
  kde_stratified   -- Centroid-distance density proxy, quantile selection (O(n*d))
  variance_ordered -- Highest local k-NN variance first (HVRT built-in, O(n log n))
  stratified       -- Fully-vectorised random stratified (HVRT built-in, O(n))

  kde_stratified replaces per-partition k-NN with centroid distance as density
  proxy (O(n*d) vs O(n*k*log(n/P))).  medoid_fps was dropped: HVRT 2.6.0
  produces bit-for-bit identical selections to fps on convex partitions.

Sections
--------
  1. Quality comparison   -- 5-fold CV across standard datasets (n<=5000)
  2. Scaling comparison   -- single-split fit time at n=1k,5k,10k,20k
                             (fps skipped for n_train>6000 due to O(n^2) cost)

Parallelism
-----------
  joblib.Parallel dispatches all (dataset, method, fold) jobs across
  available cores. n_jobs defaults to all physical cores.

Usage
-----
  python benchmarks/reduction_strategy_benchmark.py
  python benchmarks/reduction_strategy_benchmark.py --n-jobs 4
"""
from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import (
    load_diabetes,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_regression,
)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

_SEP  = "=" * 72
_SEP2 = "-" * 60

RANDOM_STATE = 42

# Methods to compare
_METHODS = ["fps", "kde_stratified", "variance_ordered", "stratified"]
# Skip fps at large n to avoid O(n^2) hang; threshold in n_train
_FPS_MAX_N = 6000


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

def _build_datasets_quality():
    """Standard-size datasets for quality (CV) comparison."""
    rng = np.random.RandomState(RANDOM_STATE)
    datasets = []

    # --- Regression ---
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0,
                           random_state=RANDOM_STATE)
    datasets.append(dict(name="friedman1_n1k", X=X, y=y, task="reg"))

    X, y = make_friedman2(n_samples=1000, noise=0.1, random_state=RANDOM_STATE)
    datasets.append(dict(name="friedman2_n1k", X=X, y=y, task="reg"))

    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10,
                            noise=10.0, random_state=RANDOM_STATE)
    datasets.append(dict(name="regression_n1k_p20", X=X, y=y, task="reg"))

    X, y = make_friedman1(n_samples=3000, n_features=10, noise=1.0,
                           random_state=RANDOM_STATE)
    datasets.append(dict(name="friedman1_n3k", X=X, y=y, task="reg"))

    X, y = make_friedman1(n_samples=5000, n_features=10, noise=1.0,
                           random_state=RANDOM_STATE)
    datasets.append(dict(name="friedman1_n5k", X=X, y=y, task="reg"))

    X_diab, y_diab = load_diabetes(return_X_y=True)
    datasets.append(dict(name="diabetes_n442", X=X_diab, y=y_diab, task="reg"))

    # --- Classification ---
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                                random_state=RANDOM_STATE)
    datasets.append(dict(name="classif_n1k", X=X, y=y, task="clf"))

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                                flip_y=0.25, random_state=RANDOM_STATE)
    datasets.append(dict(name="noisy_clf_n1k", X=X, y=y, task="clf"))

    X, y = make_classification(n_samples=1000, n_features=40, n_informative=8,
                                n_redundant=10, random_state=RANDOM_STATE)
    datasets.append(dict(name="sparse_highdim_n1k", X=X, y=y, task="clf"))

    X, y = make_classification(n_samples=3000, n_features=10, n_informative=6,
                                random_state=RANDOM_STATE)
    datasets.append(dict(name="classif_n3k", X=X, y=y, task="clf"))

    X, y = make_classification(n_samples=5000, n_features=10, n_informative=6,
                                random_state=RANDOM_STATE)
    datasets.append(dict(name="classif_n5k", X=X, y=y, task="clf"))

    # Imbalanced
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                                weights=[0.85, 0.15], random_state=RANDOM_STATE)
    datasets.append(dict(name="imbalanced_n1k", X=X, y=y, task="clf"))

    return datasets


def _build_datasets_scaling():
    """Large-n datasets for scaling / fit-time comparison."""
    datasets = []
    for n in [1000, 5000, 10000, 20000]:
        X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0,
                               random_state=RANDOM_STATE)
        datasets.append(dict(name=f"friedman1_n{n//1000}k", X=X, y=y,
                             task="reg", n=n))
    for n in [1000, 5000, 10000, 20000]:
        X, y = make_classification(n_samples=n, n_features=10,
                                   n_informative=6, random_state=RANDOM_STATE)
        datasets.append(dict(name=f"classif_n{n//1000}k", X=X, y=y,
                             task="clf", n=n))
    return datasets


# ---------------------------------------------------------------------------
# Single-fold evaluation worker
# ---------------------------------------------------------------------------

def _eval_fold(ds_name: str, task: str, method: str,
               X_tr, y_tr, X_te, y_te) -> dict[str, Any]:
    """Train one GeoXGB model and return score."""
    try:
        if task == "reg":
            model = GeoXGBRegressor(
                n_rounds=500, method=method, random_state=RANDOM_STATE,
            )
            model.fit(X_tr, y_tr)
            score = float(r2_score(y_te, model.predict(X_te)))
        else:
            model = GeoXGBClassifier(
                n_rounds=500, method=method, random_state=RANDOM_STATE,
            )
            model.fit(X_tr, y_tr)
            score = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        return dict(ds=ds_name, method=method, score=score, error=None)
    except Exception as e:
        return dict(ds=ds_name, method=method, score=float("nan"),
                    error=str(e))


# ---------------------------------------------------------------------------
# Section 1: Quality comparison
# ---------------------------------------------------------------------------

def _section1(n_jobs: int):
    print(f"\n{_SEP}")
    print("  1. Quality Comparison (5-fold CV, n_rounds=500)")
    print(f"{_SEP}")
    print("  Datasets: 12  |  Methods: fps, kde_stratified, variance_ordered, stratified")
    print(f"  n_jobs={n_jobs}\n")

    datasets = _build_datasets_quality()

    jobs = []
    meta = []
    for ds in datasets:
        X, y, task, name = ds["X"], ds["y"], ds["task"], ds["name"]
        n_train = int(len(X) * 0.8)
        if task == "reg":
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            splits = list(cv.split(X))
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_STATE)
            splits = list(cv.split(X, y))

        for method in _METHODS:
            # Skip fps on large datasets
            if method == "fps" and n_train > _FPS_MAX_N:
                continue
            for fold_idx, (tr, te) in enumerate(splits):
                jobs.append(delayed(_eval_fold)(
                    name, task, method,
                    X[tr], y[tr], X[te], y[te],
                ))
                meta.append((name, task, method))

    print(f"  Dispatching {len(jobs)} jobs...", flush=True)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(jobs)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # Aggregate
    from collections import defaultdict
    scores: dict[tuple, list] = defaultdict(list)
    errors: dict[tuple, list] = defaultdict(list)
    for r in results:
        k = (r["ds"], r["method"])
        if r["error"]:
            errors[k].append(r["error"])
        else:
            scores[k].append(r["score"])

    # Report: one row per dataset, columns per method
    metric_label = {"reg": "R2", "clf": "AUC"}
    ds_names = [d["name"] for d in datasets]
    ds_tasks = {d["name"]: d["task"] for d in datasets}

    header = (f"  {'Dataset':<26}  {'Metric':>5}  {'fps':>8}  {'kde_strat':>9}"
              f"  {'var_ord':>7}  {'strat':>6}  {'winner':>15}")
    print(header)
    print(f"  {'-'*26}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*15}")

    wins = {m: 0 for m in _METHODS}
    for name in ds_names:
        task = ds_tasks[name]
        metric = metric_label[task]
        row_scores = {}
        for method in _METHODS:
            k = (name, method)
            if scores[k]:
                row_scores[method] = float(np.mean(scores[k]))
            elif errors[k]:
                row_scores[method] = float("nan")
            else:
                row_scores[method] = float("nan")  # skipped

        fps_s = row_scores.get("fps",              float("nan"))
        kde_s = row_scores.get("kde_stratified",   float("nan"))
        var_s = row_scores.get("variance_ordered", float("nan"))
        str_s = row_scores.get("stratified",       float("nan"))

        valid = {m: s for m, s in row_scores.items()
                 if not np.isnan(s)}
        winner = max(valid, key=valid.get) if valid else "?"
        if winner in wins:
            wins[winner] += 1

        def _fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "  skip"

        print(f"  {name:<26}  {metric:>5}  "
              f"{_fmt(fps_s):>8}  {_fmt(kde_s):>9}  "
              f"{_fmt(var_s):>7}  {_fmt(str_s):>6}  {winner:>15}")

    print()
    print("  Win counts (best mean CV score per dataset):")
    for m, w in wins.items():
        print(f"    {m:<20}: {w}")

    # Delta vs fps baseline
    print()
    print("  Mean delta vs fps:")
    kde_deltas, var_deltas, str_deltas = [], [], []
    for name in ds_names:
        fps_s = scores.get((name, "fps"), [])
        kde_s = scores.get((name, "kde_stratified"), [])
        var_s = scores.get((name, "variance_ordered"), [])
        str_s = scores.get((name, "stratified"), [])
        if fps_s and kde_s:
            kde_deltas.append(np.mean(kde_s) - np.mean(fps_s))
        if fps_s and var_s:
            var_deltas.append(np.mean(var_s) - np.mean(fps_s))
        if fps_s and str_s:
            str_deltas.append(np.mean(str_s) - np.mean(fps_s))
    if kde_deltas:
        print(f"    kde_stratified  : {np.mean(kde_deltas):+.4f} "
              f"(range [{min(kde_deltas):+.4f}, {max(kde_deltas):+.4f}])")
    if var_deltas:
        print(f"    variance_ordered: {np.mean(var_deltas):+.4f} "
              f"(range [{min(var_deltas):+.4f}, {max(var_deltas):+.4f}])")
    if str_deltas:
        print(f"    stratified      : {np.mean(str_deltas):+.4f} "
              f"(range [{min(str_deltas):+.4f}, {max(str_deltas):+.4f}])")


# ---------------------------------------------------------------------------
# Scaling worker
# ---------------------------------------------------------------------------

def _time_fit(ds_name: str, task: str, method: str,
              X_tr, y_tr) -> dict[str, Any]:
    """Time a single GeoXGB fit."""
    try:
        if task == "reg":
            model = GeoXGBRegressor(
                n_rounds=500, method=method, random_state=RANDOM_STATE,
            )
        else:
            model = GeoXGBClassifier(
                n_rounds=500, method=method, random_state=RANDOM_STATE,
            )
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        return dict(ds=ds_name, method=method, fit_s=elapsed, error=None)
    except Exception as e:
        return dict(ds=ds_name, method=method, fit_s=float("nan"),
                    error=str(e))


# ---------------------------------------------------------------------------
# Section 2: Scaling comparison
# ---------------------------------------------------------------------------

def _section2(n_jobs: int):
    print(f"\n{_SEP}")
    print("  2. Scaling Comparison (fit time, n_rounds=500, single split)")
    print(f"{_SEP}")
    print("  fps skipped for n_train > 6000 (O(n^2) cost)\n")

    datasets = _build_datasets_scaling()

    jobs = []
    for ds in datasets:
        X, y, task, name = ds["X"], ds["y"], ds["task"], ds["name"]
        n = ds["n"]
        n_tr = int(n * 0.8)
        X_tr, y_tr = X[:n_tr], y[:n_tr]

        for method in _METHODS:
            if method == "fps" and n_tr > _FPS_MAX_N:
                continue
            jobs.append(delayed(_time_fit)(name, task, method, X_tr, y_tr))

    print(f"  Dispatching {len(jobs)} jobs...", flush=True)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(jobs)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    # Build result dict
    times: dict[tuple, float] = {}
    for r in results:
        times[(r["ds"], r["method"])] = r["fit_s"]

    header = (f"  {'Dataset':<22}  {'n_train':>8}  {'fps':>8}  "
              f"{'kde_strat':>9}  {'var_ord':>7}  {'strat':>6}")
    print(header)
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}")

    ds_order = [d["name"] for d in datasets]
    ns = {d["name"]: int(d["n"] * 0.8) for d in datasets}

    def _fs(v):
        return f"{v:.2f}s" if not np.isnan(v) else "  skip"

    for name in ds_order:
        n_tr = ns[name]
        fps_t = times.get((name, "fps"),              float("nan"))
        kde_t = times.get((name, "kde_stratified"),   float("nan"))
        var_t = times.get((name, "variance_ordered"), float("nan"))
        str_t = times.get((name, "stratified"),       float("nan"))
        print(f"  {name:<22}  {n_tr:>8}  "
              f"{_fs(fps_t):>8}  {_fs(kde_t):>9}  "
              f"{_fs(var_t):>7}  {_fs(str_t):>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()

    import geoxgb, hvrt as _hvrt
    print(f"\n{_SEP}")
    print("  REDUCTION STRATEGY BENCHMARK")
    print(f"{_SEP}")
    print(f"  geoxgb {geoxgb.__version__}  |  hvrt {_hvrt.__version__}")
    print(f"  methods: {_METHODS}")
    print(f"  fps skipped for n_train > {_FPS_MAX_N}")

    _section1(args.n_jobs)
    _section2(args.n_jobs)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
