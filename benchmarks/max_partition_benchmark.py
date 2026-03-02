"""
Max Partition Size Benchmark
==============================

Tests the effect of ``hvrt_max_samples_leaf`` on fit speed and prediction
quality.  Setting a maximum partition size forces HVRT to create more,
smaller partitions (n_partitions = ceil(n / max_samples_leaf)), capping
the per-partition work for reduction strategies.

Hypothesis
----------
  - Regression data auto-tunes to few, large partitions -> slow FPS/k-NN
  - Capping partition size forces more partitions -> each partition cheaper
  - min_samples_leaf still constrains from below (minimum takes priority)

Sections
--------
  1. Scaling  -- fit time at n=1k,5k,10k,20k across max_samples_leaf values
  2. Quality  -- 5-fold CV at n=1k,3k,5k to check accuracy regression

Method: variance_ordered (new default) throughout.

Usage
-----
  python benchmarks/max_partition_benchmark.py
  python benchmarks/max_partition_benchmark.py --n-jobs 4
"""
from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import (
    make_classification,
    make_friedman1,
    load_diabetes,
)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

_SEP = "=" * 72
RANDOM_STATE = 42
METHOD = "variance_ordered"

# max_samples_leaf values to sweep; None = HVRT auto-tune (baseline)
_MAX_SIZES = [None, 100, 200, 500, 1000]
_LABEL = {None: "auto"}
_LABEL.update({v: str(v) for v in _MAX_SIZES if v is not None})


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _time_fit(ds_name, task, max_leaf, n, X_tr, y_tr):
    """Time a single GeoXGB fit."""
    kwargs = dict(n_rounds=500, method=METHOD, random_state=RANDOM_STATE)
    if max_leaf is not None:
        kwargs["hvrt_max_samples_leaf"] = max_leaf
    try:
        model = GeoXGBRegressor(**kwargs) if task == "reg" else GeoXGBClassifier(**kwargs)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        return dict(ds=ds_name, max_leaf=max_leaf, n=n,
                    fit_s=time.perf_counter() - t0, error=None)
    except Exception as e:
        return dict(ds=ds_name, max_leaf=max_leaf, n=n,
                    fit_s=float("nan"), error=str(e))


def _eval_fold(ds_name, task, max_leaf, X_tr, y_tr, X_te, y_te):
    """Train one fold and return CV score."""
    kwargs = dict(n_rounds=500, method=METHOD, random_state=RANDOM_STATE)
    if max_leaf is not None:
        kwargs["hvrt_max_samples_leaf"] = max_leaf
    try:
        if task == "reg":
            model = GeoXGBRegressor(**kwargs)
            model.fit(X_tr, y_tr)
            score = float(r2_score(y_te, model.predict(X_te)))
        else:
            model = GeoXGBClassifier(**kwargs)
            model.fit(X_tr, y_tr)
            score = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        return dict(ds=ds_name, max_leaf=max_leaf, score=score, error=None)
    except Exception as e:
        return dict(ds=ds_name, max_leaf=max_leaf, score=float("nan"), error=str(e))


# ---------------------------------------------------------------------------
# Section 1: Scaling
# ---------------------------------------------------------------------------

def _section1(n_jobs):
    print(f"\n{_SEP}")
    print("  1. Scaling (fit time, n_rounds=500, single split)")
    print(f"{_SEP}")
    print(f"  Method: {METHOD}  |  max_leaf values: {_MAX_SIZES}\n")

    scaling_ns = [1000, 5000, 10000, 20000]
    datasets = []
    for n in scaling_ns:
        X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0,
                              random_state=RANDOM_STATE)
        datasets.append(dict(name=f"friedman1_n{n//1000}k", X=X, y=y,
                             task="reg", n=n))
    for n in scaling_ns:
        X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                                   random_state=RANDOM_STATE)
        datasets.append(dict(name=f"classif_n{n//1000}k", X=X, y=y,
                             task="clf", n=n))

    jobs = []
    for ds in datasets:
        n_tr = int(ds["n"] * 0.8)
        X_tr, y_tr = ds["X"][:n_tr], ds["y"][:n_tr]
        for ml in _MAX_SIZES:
            jobs.append(delayed(_time_fit)(
                ds["name"], ds["task"], ml, ds["n"], X_tr, y_tr
            ))

    print(f"  Dispatching {len(jobs)} jobs...", flush=True)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(jobs)
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    # Index results
    times: dict[tuple, float] = {}
    for r in results:
        times[(r["ds"], r["max_leaf"])] = r["fit_s"]

    # Print per task type
    for task_label, task_key in [("Regression", "reg"), ("Classification", "clf")]:
        print(f"  --- {task_label} ---")
        col_w = 8
        header = f"  {'Dataset':<22}  {'n_train':>8}"
        for ml in _MAX_SIZES:
            header += f"  {_LABEL[ml]:>{col_w}}"
        print(header)
        sep = f"  {'-'*22}  {'-'*8}" + f"  {'-'*col_w}" * len(_MAX_SIZES)
        print(sep)

        task_ds = [d for d in datasets if d["task"] == task_key]
        for ds in task_ds:
            n_tr = int(ds["n"] * 0.8)
            row = f"  {ds['name']:<22}  {n_tr:>8}"
            auto_t = times.get((ds["name"], None), float("nan"))
            for ml in _MAX_SIZES:
                t = times.get((ds["name"], ml), float("nan"))
                if np.isnan(t):
                    row += f"  {'  err':>{col_w}}"
                else:
                    row += f"  {t:.2f}s".rjust(col_w + 2)
            print(row)
        print()

    # Speedup vs auto at large n
    print("  Speedup vs auto (auto / max_leaf) at large n:")
    for task_label, task_key in [("Regression", "reg"), ("Classification", "clf")]:
        print(f"    {task_label}:")
        large_ds = [d for d in datasets if d["task"] == task_key and d["n"] >= 10000]
        for ds in large_ds:
            auto_t = times.get((ds["name"], None), float("nan"))
            parts = [f"    {ds['name']}:"]
            for ml in _MAX_SIZES:
                if ml is None:
                    continue
                t = times.get((ds["name"], ml), float("nan"))
                if not np.isnan(auto_t) and not np.isnan(t) and t > 1e-9:
                    parts.append(f"max={ml} {auto_t/t:.2f}x")
                else:
                    parts.append(f"max={ml} --")
            print("      " + "  ".join(parts[1:]) + f"  (auto={auto_t:.1f}s, ds={ds['name']})")


# ---------------------------------------------------------------------------
# Section 2: Quality
# ---------------------------------------------------------------------------

def _section2(n_jobs):
    print(f"\n{_SEP}")
    print("  2. Quality (5-fold CV, n_rounds=500)")
    print(f"{_SEP}")
    print(f"  Method: {METHOD}  |  max_leaf values: {_MAX_SIZES}\n")

    datasets = []
    for n in [1000, 3000, 5000]:
        X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0,
                              random_state=RANDOM_STATE)
        datasets.append(dict(name=f"friedman1_n{n//1000}k", X=X, y=y, task="reg"))

    X, y = load_diabetes(return_X_y=True)
    datasets.append(dict(name="diabetes_n442", X=X, y=y, task="reg"))

    for n in [1000, 3000, 5000]:
        X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                                   random_state=RANDOM_STATE)
        datasets.append(dict(name=f"classif_n{n//1000}k", X=X, y=y, task="clf"))

    jobs = []
    for ds in datasets:
        X, y, task = ds["X"], ds["y"], ds["task"]
        if task == "reg":
            splits = list(KFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE).split(X))
        else:
            splits = list(StratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=RANDOM_STATE).split(X, y))
        for ml in _MAX_SIZES:
            for tr, te in splits:
                jobs.append(delayed(_eval_fold)(
                    ds["name"], task, ml, X[tr], y[tr], X[te], y[te]
                ))

    print(f"  Dispatching {len(jobs)} jobs...", flush=True)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(jobs)
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    from collections import defaultdict
    scores: dict[tuple, list] = defaultdict(list)
    for r in results:
        if not r["error"] and not np.isnan(r["score"]):
            scores[(r["ds"], r["max_leaf"])].append(r["score"])

    col_w = 8
    header = f"  {'Dataset':<26}  {'Metric':>5}"
    for ml in _MAX_SIZES:
        header += f"  {_LABEL[ml]:>{col_w}}"
    header += f"  {'best_max':>8}"
    print(header)
    print(f"  {'-'*26}  {'-'*5}" + f"  {'-'*col_w}" * len(_MAX_SIZES) + f"  {'-'*8}")

    metric_label = {"reg": "R2", "clf": "AUC"}
    for ds in datasets:
        name, task = ds["name"], ds["task"]
        metric = metric_label[task]
        row_means = {}
        for ml in _MAX_SIZES:
            sv = scores.get((name, ml), [])
            row_means[ml] = float(np.mean(sv)) if sv else float("nan")

        auto_mean = row_means[None]
        row = f"  {name:<26}  {metric:>5}"
        for ml in _MAX_SIZES:
            v = row_means[ml]
            row += f"  {v:.4f}".rjust(col_w + 2) if not np.isnan(v) else f"  {'  --':>{col_w}}"
        # Best non-auto max_leaf
        non_auto = {ml: s for ml, s in row_means.items()
                    if ml is not None and not np.isnan(s)}
        best_ml = max(non_auto, key=non_auto.get) if non_auto else None
        row += f"  {str(best_ml):>8}"
        print(row)

    print()
    print("  Mean delta vs auto (across all datasets):")
    for ml in _MAX_SIZES:
        if ml is None:
            continue
        deltas = []
        for ds in datasets:
            auto_s = scores.get((ds["name"], None), [])
            ml_s = scores.get((ds["name"], ml), [])
            if auto_s and ml_s:
                deltas.append(np.mean(ml_s) - np.mean(auto_s))
        if deltas:
            print(f"    max={str(ml):<6}: {np.mean(deltas):+.4f} "
                  f"(range [{min(deltas):+.4f}, {max(deltas):+.4f}])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    import geoxgb, hvrt as _hvrt
    print(f"\n{_SEP}")
    print("  MAX PARTITION SIZE BENCHMARK")
    print(f"{_SEP}")
    print(f"  geoxgb {geoxgb.__version__}  |  hvrt {_hvrt.__version__}")
    print(f"  method: {METHOD}  |  max_sizes: {_MAX_SIZES}")

    _section1(args.n_jobs)
    _section2(args.n_jobs)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
