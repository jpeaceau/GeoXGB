"""
GeoXGB vs XGBoost -- Parallel Cross-Validation Benchmark
=========================================================

Demonstrates fold-level parallelism for GeoXGB and contrasts it with
XGBoost's intra-fold (OpenMP) parallelism strategy.

Dataset: synthetic binary classification, 1,000 samples, 10 features
  (same seed and structure as classification_benchmark.py)

Parallelism strategies
----------------------
  GeoXGB  : fold-level -- 3 folds run concurrently as independent processes
             via joblib (n_jobs=3).  GeoXGB's boosting loop is
             single-threaded; parallelising across folds uses all available
             cores without over-subscribing any single fold.

  XGBoost : intra-fold -- n_jobs=-1 inside XGBClassifier activates OpenMP
             histogram construction across all physical cores.  Folds are
             run sequentially so XGBoost has the full core budget per fold.

Usage
-----
    python benchmarks/parallel_cv_benchmark.py

Requirements: geoxgb, xgboost, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import multiprocessing
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from geoxgb import GeoXGBClassifier

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 1_000
N_SIGNAL     = 5
N_NOISE      = 5
N_FEATURES   = N_SIGNAL + N_NOISE

# ---------------------------------------------------------------------------
# Hyperparameters
# HPO-tuned via random search (9 configs, 3-fold CV) on the same dataset:
#   GeoXGB best:  n_rounds=150, learning_rate=0.1, max_depth=6  (AUC=0.9679)
#   XGBoost best: n_estimators=150, learning_rate=0.2, max_depth=4 (AUC=0.9704)
# ---------------------------------------------------------------------------

GEO_PARAMS: dict = dict(
    n_rounds=150,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    random_state=RANDOM_STATE,
)

XGB_PARAMS: dict = dict(
    n_estimators=150,
    learning_rate=0.2,
    max_depth=4,
    n_jobs=-1,        # OpenMP parallelism -- all cores per fold
    random_state=RANDOM_STATE,
    verbosity=0,
)

N_FOLDS = 3

# ---------------------------------------------------------------------------
# Per-fold evaluation helpers
# (module-level so they are picklable on Windows multiprocessing)
# ---------------------------------------------------------------------------

def _run_geo_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> tuple[int, float, float]:
    t0  = time.perf_counter()
    clf = GeoXGBClassifier(**params)
    clf.fit(X[train_idx], y[train_idx])
    proba = clf.predict_proba(X[val_idx])[:, 1]
    auc   = roc_auc_score(y[val_idx], proba)
    return fold_idx, auc, time.perf_counter() - t0


def _run_xgb_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> tuple[int, float, float]:
    from xgboost import XGBClassifier as _XGB
    t0  = time.perf_counter()
    clf = _XGB(**params)
    clf.fit(X[train_idx], y[train_idx])
    proba = clf.predict_proba(X[val_idx])[:, 1]
    auc   = roc_auc_score(y[val_idx], proba)
    return fold_idx, auc, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SEP = "=" * 68


def _hdr(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _fold_rows(results: list[tuple[int, float, float]]) -> None:
    for fold_idx, auc, elapsed in sorted(results, key=lambda r: r[0]):
        print(f"    Fold {fold_idx + 1}   AUC = {auc:.6f}   ({elapsed:.1f}s)")


def _summary_table(
    geo_results: list,
    xgb_results: list | None,
    t_geo: float,
    t_xgb: float | None,
) -> None:
    geo_aucs = [r[1] for r in sorted(geo_results, key=lambda r: r[0])]

    _hdr("RESULTS -- GeoXGB vs XGBoost  (3-fold stratified AUC)")
    W = 12
    if xgb_results:
        xgb_aucs = [r[1] for r in sorted(xgb_results, key=lambda r: r[0])]
        print(f"\n    {'':30s} {'GeoXGB':>{W}s} {'XGBoost':>{W}s}")
        print(f"    {'-' * 56}")
        for i, (g, x) in enumerate(zip(geo_aucs, xgb_aucs)):
            print(f"    Fold {i + 1} AUC{'':25s} {g:>{W}.6f} {x:>{W}.6f}")
        print(f"    {'-' * 56}")
        geo_mean, xgb_mean = np.mean(geo_aucs), np.mean(xgb_aucs)
        geo_std,  xgb_std  = np.std(geo_aucs),  np.std(xgb_aucs)
        print(f"    Mean AUC{'':29s} {geo_mean:>{W}.6f} {xgb_mean:>{W}.6f}")
        print(f"    Std  AUC{'':29s} {geo_std:>{W}.6f}  {xgb_std:>{W}.6f}")
        print(f"    Wall time (s){'':24s} {t_geo:>{W}.1f} {t_xgb:>{W}.1f}")
        print(f"    {'-' * 56}")
        delta = geo_mean - xgb_mean
        if abs(delta) < 0.001:
            verdict = "Tie  (diff < 0.001)"
        elif delta > 0:
            verdict = f"GeoXGB  (+{delta:.6f})"
        else:
            verdict = f"XGBoost  ({delta:+.6f})"
        print(f"\n    AUC diff (GeoXGB - XGBoost) = {delta:+.6f}")
        print(f"    Winner : {verdict}")
    else:
        geo_mean = np.mean(geo_aucs)
        geo_std  = np.std(geo_aucs)
        print(f"\n    {'':30s} {'GeoXGB':>{W}s}")
        print(f"    {'-' * 44}")
        for i, g in enumerate(geo_aucs):
            print(f"    Fold {i + 1} AUC{'':25s} {g:>{W}.6f}")
        print(f"    {'-' * 44}")
        print(f"    Mean AUC{'':29s} {geo_mean:>{W}.6f}")
        print(f"    Std  AUC{'':29s} {geo_std:>{W}.6f}")
        print(f"    Wall time (s){'':24s} {t_geo:>{W}.1f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    _hdr("Dataset")
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_SIGNAL,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"\n    Samples  : {len(X):,}")
    print(f"    Features : {N_FEATURES}  ({N_SIGNAL} signal + {N_NOISE} noise)")
    print(f"    Classes  : {counts}")
    print(f"    CPUs     : {cpu_count}")
    print(f"\n    Parallelism strategy")
    print(f"      GeoXGB  : fold-level (n_jobs={N_FOLDS}, one process per fold)")
    print(f"      XGBoost : intra-fold (n_jobs=-1, all {cpu_count} cores per fold)")

    # -----------------------------------------------------------------------
    # Cross-validation splits
    # -----------------------------------------------------------------------
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = [(i, tr, val) for i, (tr, val) in enumerate(skf.split(X, y))]
    print(f"\n    Fold sizes (train / val):")
    for i, tr, val in folds:
        print(f"      Fold {i + 1}: {len(tr):,} / {len(val):,}")

    # -----------------------------------------------------------------------
    # GeoXGB: 3 folds in parallel
    # -----------------------------------------------------------------------
    _hdr(f"GeoXGBClassifier  [parallel folds, n_jobs={N_FOLDS} x {cpu_count} CPUs available]")
    print(f"    Parameters: {GEO_PARAMS}\n")

    t_geo_wall = time.perf_counter()
    try:
        geo_raw = Parallel(n_jobs=N_FOLDS, prefer="processes", verbose=5)(
            delayed(_run_geo_fold)(i, tr, val, X, y, GEO_PARAMS)
            for i, tr, val in folds
        )
    except Exception as exc:
        print(f"  [parallel failed: {exc!r}]  -- running sequentially")
        geo_raw = [
            _run_geo_fold(i, tr, val, X, y, GEO_PARAMS)
            for i, tr, val in folds
        ]
    t_geo_wall = time.perf_counter() - t_geo_wall

    geo_results = sorted(geo_raw, key=lambda r: r[0])
    _fold_rows(geo_results)
    print(f"\n    Mean AUC  : {np.mean([r[1] for r in geo_results]):.6f}")
    print(f"    Std  AUC  : {np.std([r[1] for r in geo_results]):.6f}")
    print(f"    Wall time : {t_geo_wall:.1f}s")

    # -----------------------------------------------------------------------
    # XGBoost: n_jobs=-1 per fold, sequential folds
    # -----------------------------------------------------------------------
    xgb_results = None
    t_xgb_wall  = None

    if _XGB_AVAILABLE:
        _hdr(f"XGBClassifier  [n_jobs=-1 x {cpu_count} cores,  sequential folds]")
        print(f"    Parameters: {XGB_PARAMS}\n")

        t_xgb_wall = time.perf_counter()
        xgb_raw = [
            _run_xgb_fold(i, tr, val, X, y, XGB_PARAMS)
            for i, tr, val in folds
        ]
        t_xgb_wall = time.perf_counter() - t_xgb_wall

        xgb_results = sorted(xgb_raw, key=lambda r: r[0])
        _fold_rows(xgb_results)
        print(f"\n    Mean AUC  : {np.mean([r[1] for r in xgb_results]):.6f}")
        print(f"    Std  AUC  : {np.std([r[1] for r in xgb_results]):.6f}")
        print(f"    Wall time : {t_xgb_wall:.1f}s")
    else:
        print("\n  XGBoost not installed -- skipping baseline.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _summary_table(geo_results, xgb_results, t_geo_wall, t_xgb_wall)


if __name__ == "__main__":
    main()
