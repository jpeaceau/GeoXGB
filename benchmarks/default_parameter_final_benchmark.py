"""
GeoXGB -- Default Parameter Final Determination Benchmark
==========================================================

Determines the optimal default values for bandwidth and generation_strategy
across five diverse datasets, using the same dataset suite as the refit
interval and max-depth benchmarks for consistency.

Background
----------
Section 12 (kde_bandwidth_benchmark.py) identified the following top candidates
on two datasets (Friedman #1 regression + synthetic binary classification):

  bandwidth=0.75            combined z=+1.28  (regression winner)
  generation_strategy='epanechnikov'  combined z=+0.97  (consistent on both)
  bandwidth=0.50            combined z=+0.76
  bandwidth='scott'         combined z=+0.58
  bandwidth=0.10+adaptive   combined z=+0.52
  bandwidth='auto' (h=0.10) combined z=-0.45  (current default -- 10th of 13)

This benchmark tests the top 7 candidates across all 5 standard datasets
at n_rounds=1000 (full training budget) to make a robust recommendation.

Conditions
----------
  1. auto / None              bandwidth='auto',  strategy=None,           adaptive=False
  2. bw=0.75 / None           bandwidth=0.75,    strategy=None,           adaptive=False
  3. auto / epanechnikov       bandwidth='auto',  strategy='epanechnikov', adaptive=False
  4. bw=0.50 / None           bandwidth=0.50,    strategy=None,           adaptive=False
  5. scott / None             bandwidth='scott', strategy=None,           adaptive=False
  6. bw=0.10+adaptive         bandwidth=0.10,    strategy=None,           adaptive=True
  7. bw=0.75 / epanechnikov   bandwidth=0.75,    strategy='epanechnikov', adaptive=False

Datasets (same as refit_interval_benchmark and max_depth_benchmark)
--------
  friedman1      1000 samples, 10 features, 5 signal, noise=1.0     regression  R^2
  friedman2      1000 samples,  4 features, 4 signal, noise=0.0     regression  R^2
  classification 1000 samples, 10 features, 5 informative, sep=1.0  binary AUC
  sparse_highdim 1000 samples, 40 features, 8 informative, noise=20 regression  R^2
  noisy_clf      1000 samples, 20 features, 5 informative, flip=0.1  binary AUC

Fixed GeoXGB params (matching established baselines):
  n_rounds=1000, learning_rate=0.2, max_depth=4, min_samples_leaf=5,
  reduce_ratio=0.7, refit_interval=20, auto_noise=True, auto_expand=True,
  cache_geometry=False

Usage
-----
    python benchmarks/default_parameter_final_benchmark.py

Requirements: geoxgb, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import (
    make_classification, make_friedman1, make_friedman2, make_regression,
)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ROUNDS      = 1000
LEARNING_RATE = 0.2
N_FOLDS       = 5
RANDOM_STATE  = 42
N_JOBS        = -1

GEO_FIXED = dict(
    n_rounds=N_ROUNDS,
    learning_rate=LEARNING_RATE,
    max_depth=4,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    refit_interval=20,
    auto_noise=True,
    auto_expand=True,
    cache_geometry=False,
    random_state=RANDOM_STATE,
)

# (label, bandwidth, generation_strategy, adaptive_bandwidth)
CONDITIONS = [
    ("auto / None  (default)",   "auto",   None,             False),
    ("bw=0.75 / None",            0.75,    None,             False),
    ("auto / epanechnikov",       "auto",  "epanechnikov",   False),
    ("bw=0.50 / None",            0.50,    None,             False),
    ("scott / None",              "scott", None,             False),
    ("bw=0.10+adaptive",          0.10,    None,             True),
    ("bw=0.75 / epanechnikov",    0.75,    "epanechnikov",   False),
]

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _make_datasets():
    X1, y1 = make_friedman1(
        n_samples=1_000, n_features=10, noise=1.0, random_state=RANDOM_STATE
    )
    X2, y2 = make_friedman2(
        n_samples=1_000, noise=0.0, random_state=RANDOM_STATE
    )
    X3, y3 = make_classification(
        n_samples=1_000, n_features=10, n_informative=5,
        n_redundant=0, n_repeated=0, n_clusters_per_class=2,
        class_sep=1.0, random_state=RANDOM_STATE,
    )
    X4, y4 = make_regression(
        n_samples=1_000, n_features=40, n_informative=8,
        noise=20.0, random_state=RANDOM_STATE,
    )
    X5, y5 = make_classification(
        n_samples=1_000, n_features=20, n_informative=5,
        n_redundant=5, n_repeated=0, n_clusters_per_class=1,
        class_sep=0.5, flip_y=0.10, random_state=RANDOM_STATE,
    )
    return {
        "friedman1":      (X1, y1, "regression",     "R^2"),
        "friedman2":      (X2, y2, "regression",     "R^2"),
        "classification": (X3, y3, "classification", "AUC"),
        "sparse_highdim": (X4, y4, "regression",     "R^2"),
        "noisy_clf":      (X5, y5, "classification", "AUC"),
    }


# ---------------------------------------------------------------------------
# CV worker -- module-level for Windows pickling
# ---------------------------------------------------------------------------

def _cv_worker(
    cond_idx, task, X_tr, y_tr, X_val, y_val
):
    import warnings as _w
    _w.filterwarnings("ignore")

    lbl, bw, strategy, adaptive = CONDITIONS[cond_idx]
    try:
        if task == "regression":
            from geoxgb import GeoXGBRegressor as M
        else:
            from geoxgb import GeoXGBClassifier as M
        model = M(
            **GEO_FIXED,
            bandwidth=bw,
            generation_strategy=strategy,
            adaptive_bandwidth=adaptive,
        )
        model.fit(X_tr, y_tr)
        if task == "regression":
            return float(r2_score(y_val, model.predict(X_val)))
        else:
            return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _section("DEFAULT PARAMETER FINAL DETERMINATION BENCHMARK")
    print(
        f"\n  {len(CONDITIONS)} conditions x {N_FOLDS}-fold CV x 5 datasets"
        f"\n  Fixed: n_rounds={N_ROUNDS}, lr={LEARNING_RATE}, max_depth=4,"
        f"\n         reduce_ratio=0.7, refit_interval=20, auto_expand=True"
        f"\n  Baseline: condition 0 (bandwidth='auto', strategy=None)"
    )

    datasets = _make_datasets()

    # -----------------------------------------------------------------------
    # Build and dispatch all jobs
    # -----------------------------------------------------------------------

    jobs = []
    job_meta = []   # (ds_name, cond_idx, fold_idx)

    for ds_name, (X, y, task, metric) in datasets.items():
        kf_cls = StratifiedKFold if task == "classification" else KFold
        kf = kf_cls(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X, y if task == "classification" else None))
        for cond_idx in range(len(CONDITIONS)):
            for fi, (tr, val) in enumerate(splits):
                jobs.append(
                    delayed(_cv_worker)(
                        cond_idx, task,
                        X[tr], y[tr], X[val], y[val],
                    )
                )
                job_meta.append((ds_name, cond_idx, fi))

    print(f"\n  Total jobs: {len(jobs)}  ({len(CONDITIONS)} conds x {N_FOLDS} folds x {len(datasets)} datasets)")
    print("  Running...")
    t0 = time.perf_counter()
    flat = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(jobs)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Reshape into scores[ds_name][cond_idx, fold_idx]
    # -----------------------------------------------------------------------

    scores = {ds: np.full((len(CONDITIONS), N_FOLDS), np.nan)
              for ds in datasets}
    for (ds_name, cond_idx, fi), val in zip(job_meta, flat):
        scores[ds_name][cond_idx, fi] = val

    # -----------------------------------------------------------------------
    # Per-dataset results
    # -----------------------------------------------------------------------

    _section("PER-DATASET RESULTS")
    means = {}    # means[ds][cond_idx]

    for ds_name, (X, y, task, metric) in datasets.items():
        _subsection(f"{ds_name}  ({task}, metric={metric})")
        ds_means = np.nanmean(scores[ds_name], axis=1)
        ds_stds  = np.nanstd(scores[ds_name],  axis=1)
        baseline = float(ds_means[0])
        means[ds_name] = ds_means

        order = np.argsort(-ds_means)
        print(f"\n  {'Condition':<30s}  {metric:>8s}  {'Std':>6s}  {'vs default':>11s}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*6}  {'-'*11}")
        for i in order:
            lbl = CONDITIONS[i][0]
            m   = float(ds_means[i])
            s   = float(ds_stds[i])
            d   = m - baseline
            marker = " *" if i == 0 else "  "
            print(f"{marker} {lbl:<30s}  {m:>8.4f}  {s:>6.4f}  {d:>+11.4f}")
        print(f"\n  * = current default (bandwidth='auto')")

    # -----------------------------------------------------------------------
    # Win count per condition
    # -----------------------------------------------------------------------

    _section("WIN COUNT AND Z-SCORE RANKING")

    wins = np.zeros(len(CONDITIONS), dtype=int)
    for ds_name in datasets:
        best_i = int(np.nanargmax(means[ds_name]))
        wins[best_i] += 1

    # Cross-dataset z-score: normalise each dataset's scores then average
    z_all = np.zeros(len(CONDITIONS))
    for ds_name in datasets:
        m = means[ds_name]
        mu = np.nanmean(m)
        sd = np.nanstd(m) + 1e-12
        z_all += (m - mu) / sd
    z_all /= len(datasets)

    order_z = np.argsort(-z_all)

    print(f"\n  {'Rank':<5s}  {'Condition':<30s}  {'Wins':>5s}  {'Mean z':>8s}  ", end="")
    for ds_name in datasets:
        print(f"  {ds_name[:10]:>10s}", end="")
    print()
    print(f"  {'-'*5}  {'-'*30}  {'-'*5}  {'-'*8}  ", end="")
    for _ in datasets:
        print(f"  {'-'*10}", end="")
    print()

    for rank, i in enumerate(order_z, 1):
        lbl = CONDITIONS[i][0]
        marker = " *" if i == 0 else "  "
        print(f"{marker} {rank:<4d}  {lbl:<30s}  {wins[i]:>5d}  {z_all[i]:>8.4f}  ", end="")
        for ds_name, (_, _, _, metric) in datasets.items():
            d = float(means[ds_name][i]) - float(means[ds_name][0])
            print(f"  {d:>+10.4f}", end="")
        print()

    print(f"\n  * = current default  |  Deltas are vs default (condition 0)")
    print(f"  Columns show per-dataset delta vs default")

    # -----------------------------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------------------------

    _section("FINAL RECOMMENDATION")

    best_i    = int(order_z[0])
    best_lbl  = CONDITIONS[best_i][0]
    best_bw   = CONDITIONS[best_i][1]
    best_st   = CONDITIONS[best_i][2]
    best_ad   = CONDITIONS[best_i][3]
    best_wins = int(wins[best_i])

    print(f"\n  Overall winner (by combined z-score): {best_lbl}")
    print(f"    bandwidth={best_bw}")
    print(f"    generation_strategy={best_st}")
    print(f"    adaptive_bandwidth={best_ad}")
    print(f"    Dataset wins: {best_wins}/5")
    print(f"    Combined z:   {z_all[best_i]:.4f}")

    # Per-task analysis
    reg_ds  = [d for d, (_, _, t, _) in datasets.items() if t == "regression"]
    clf_ds  = [d for d, (_, _, t, _) in datasets.items() if t == "classification"]

    reg_z  = np.zeros(len(CONDITIONS))
    clf_z  = np.zeros(len(CONDITIONS))
    for ds in reg_ds:
        m = means[ds]; mu = np.nanmean(m); sd = np.nanstd(m) + 1e-12
        reg_z += (m - mu) / sd
    for ds in clf_ds:
        m = means[ds]; mu = np.nanmean(m); sd = np.nanstd(m) + 1e-12
        clf_z += (m - mu) / sd
    reg_z /= len(reg_ds)
    clf_z /= len(clf_ds)

    print(f"\n  Per-task z-score leaders:")
    print(f"    Regression    (3 datasets): {CONDITIONS[int(np.argmax(reg_z))][0]}"
          f"  (z={reg_z[int(np.argmax(reg_z))]:.4f})")
    print(f"    Classification (2 datasets): {CONDITIONS[int(np.argmax(clf_z))][0]}"
          f"  (z={clf_z[int(np.argmax(clf_z))]:.4f})")

    # Decision logic
    default_wins_neither = (wins[0] == 0)
    winner_consistent    = best_wins >= 3

    print(f"\n  Decision:")
    if best_i == 0:
        print(f"  Current default 'auto' is the optimal choice across all 5 datasets.")
        print(f"  Recommendation: keep bandwidth='auto', generation_strategy=None.")
    else:
        print(f"  Recommended new defaults:")
        print(f"    bandwidth          = {best_bw!r}")
        print(f"    generation_strategy = {best_st!r}")
        print(f"    adaptive_bandwidth = {best_ad!r}")
        if winner_consistent:
            print(f"  Confidence: HIGH ({best_wins}/5 dataset wins, consistent z-score leader)")
        else:
            print(f"  Confidence: MODERATE (only {best_wins}/5 dataset wins -- HPO recommended)")

        # Check if epanechnikov is a strong contender regardless of winner
        epan_i = 2  # index of 'auto / epanechnikov' condition
        if best_i != epan_i and z_all[epan_i] > z_all[0]:
            print(f"\n  Note: 'auto / epanechnikov' (z={z_all[epan_i]:.4f}) also beats the current")
            print(f"  default and requires no bandwidth tuning -- a principled alternative")
            print(f"  if the winner requires manual bandwidth selection.")

    print()


if __name__ == "__main__":
    main()
