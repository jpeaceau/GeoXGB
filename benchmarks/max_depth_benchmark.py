"""
GeoXGB -- Max Depth Benchmark (Multiprocessing)
===============================================

Sweeps max_depth across {3, 4, 5} with all other settings fixed.

Settings: n_rounds=1000, learning_rate=0.2, refit_interval=20,
          tree_criterion=squared_error (default)

Datasets
--------
  friedman1      : 10 feat, moderate noise  (R^2)
  friedman2      : 10 feat, zero noise      (R^2)
  classification : 10 feat, clean binary    (AUC)
  sparse_highdim : 40 feat, 8 informative, high noise  (R^2)
  noisy_clf      : 20 feat, 10% label flip, low sep    (AUC)

CV: 5-fold

Usage
-----
    python benchmarks/max_depth_benchmark.py
"""

from __future__ import annotations

import multiprocessing
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import (
    make_classification,
    make_friedman1,
    make_friedman2,
    make_regression,
)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ROUNDS       = 1000
LEARNING_RATE  = 0.2
REFIT_INTERVAL = 20
N_FOLDS        = 5
RANDOM_STATE   = 42

MAX_DEPTHS = [3, 4, 5]
DEFAULT_DEPTH = 4

GEO_FIXED = dict(
    n_rounds=N_ROUNDS,
    learning_rate=LEARNING_RATE,
    refit_interval=REFIT_INTERVAL,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    auto_noise=True,
    cache_geometry=False,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _make_datasets() -> dict:
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
# Worker
# ---------------------------------------------------------------------------

def _eval_fold(
    depth: int,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple:
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(**GEO_FIXED, max_depth=depth)
    t0 = time.perf_counter()

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(r2_score(y[val_idx], m.predict(X[val_idx])))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(roc_auc_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))

    return depth, score, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title: str) -> None:
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


def _bar(val: float, ref: float, width: int = 28) -> str:
    filled = int(round(width * max(val, 0.0) / max(ref, 1e-12)))
    return "#" * filled + "." * (width - filled)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    datasets  = _make_datasets()
    n_jobs    = len(MAX_DEPTHS) * len(datasets) * N_FOLDS

    _section("GeoXGB -- Max Depth Benchmark")
    print(
        f"\n  n_rounds        : {N_ROUNDS}"
        f"\n  learning_rate   : {LEARNING_RATE}"
        f"\n  refit_interval  : {REFIT_INTERVAL}"
        f"\n  max_depth sweep : {MAX_DEPTHS}  (current default: {DEFAULT_DEPTH})"
        f"\n  CV folds        : {N_FOLDS}"
        f"\n  Total jobs      : {n_jobs}"
        f"\n  CPUs            : {cpu_count}"
    )

    # -----------------------------------------------------------------------
    # CV splits
    # -----------------------------------------------------------------------
    splits: dict[str, list] = {}
    for ds_name, (X, y, task, _) in datasets.items():
        kf = (
            StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if task == "classification"
            else KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )
        splits[ds_name] = list(kf.split(X, y))

    # -----------------------------------------------------------------------
    # Jobs
    # -----------------------------------------------------------------------
    jobs: list[tuple] = []
    job_keys: list[tuple] = []

    for ds_name, (X, y, task, _) in datasets.items():
        for depth in MAX_DEPTHS:
            for fi, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((depth, task, X, y, tr, val))
                job_keys.append((ds_name, depth, fi))

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    print(f"\n  Launching {n_jobs} jobs across {cpu_count} workers...")
    t_wall = time.perf_counter()
    try:
        raw = Parallel(n_jobs=cpu_count, prefer="processes", verbose=2)(
            delayed(_eval_fold)(*job) for job in jobs
        )
    except Exception as exc:
        print(f"\n  [parallel failed: {exc!r}] -- running sequentially")
        raw = [_eval_fold(*job) for job in jobs]
    t_wall = time.perf_counter() - t_wall
    print(f"\n  Done. Wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)")

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    from collections import defaultdict
    scores: dict[tuple, list[float]] = defaultdict(list)
    times:  dict[tuple, list[float]] = defaultdict(list)

    for (ds_name, depth, _fi), (_, score, elapsed) in zip(job_keys, raw):
        scores[(ds_name, depth)].append(score)
        times[ (ds_name, depth)].append(elapsed)

    # -----------------------------------------------------------------------
    # Per-dataset tables
    # -----------------------------------------------------------------------
    _section("Per-Dataset Results")

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name})")

        rows = []
        for d in MAX_DEPTHS:
            sc = scores[(ds_name, d)]
            rows.append({
                "depth": d,
                "mean":  float(np.mean(sc)),
                "std":   float(np.std(sc)),
                "time":  float(np.mean(times[(ds_name, d)])),
            })
        rows.sort(key=lambda r: -r["mean"])
        best_mean = rows[0]["mean"]
        base_mean = next(r["mean"] for r in rows if r["depth"] == DEFAULT_DEPTH)

        print(
            f"\n  {'depth':>5s}  {'mean':>9s}  {'std':>6s}"
            f"  {'vs depth=4':>10s}  {'avg time':>8s}  score"
        )
        print(
            f"  {'-'*5}  {'-'*9}  {'-'*6}"
            f"  {'-'*10}  {'-'*8}  {'-'*28}"
        )
        for r in rows:
            delta = r["mean"] - base_mean
            tag   = " [best]" if abs(r["mean"] - best_mean) < 1e-9 else (
                    " [default]" if r["depth"] == DEFAULT_DEPTH else "")
            print(
                f"  {r['depth']:>5d}  {r['mean']:>9.5f}  {r['std']:>6.4f}"
                f"  {delta:>+10.5f}  {r['time']:>7.1f}s  {_bar(r['mean'], best_mean)}{tag}"
            )

    # -----------------------------------------------------------------------
    # Cross-dataset z-score ranking
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Z-Score Ranking")

    z_by_depth: dict[int, list[float]] = {d: [] for d in MAX_DEPTHS}
    for ds_name in datasets:
        ds_scores = {d: float(np.mean(scores[(ds_name, d)])) for d in MAX_DEPTHS}
        mu  = float(np.mean(list(ds_scores.values())))
        sig = float(np.std(list(ds_scores.values()))) or 1e-9
        for d, v in ds_scores.items():
            z_by_depth[d].append((v - mu) / sig)

    ranked = sorted(MAX_DEPTHS, key=lambda d: -float(np.mean(z_by_depth[d])))
    base_z = float(np.mean(z_by_depth[DEFAULT_DEPTH]))
    max_z  = float(np.mean(z_by_depth[ranked[0]]))

    print(f"\n  {'rank':>4s}  {'depth':>5s}  {'mean z':>8s}  {'vs depth=4':>10s}  bar")
    print(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*28}")
    for rank, d in enumerate(ranked, 1):
        z   = float(np.mean(z_by_depth[d]))
        dz  = z - base_z
        bar = _bar(max(z, 0), max(max_z, 1e-9))
        tag = " [best]" if rank == 1 else (" [default]" if d == DEFAULT_DEPTH else "")
        print(f"  {rank:>4d}  {d:>5d}  {z:>8.4f}  {dz:>+10.4f}  {bar}{tag}")

    # -----------------------------------------------------------------------
    # Head-to-head summary
    # -----------------------------------------------------------------------
    _section("Head-to-Head vs depth=4 (default)")
    print(
        f"\n  {'dataset':<18s}  {'depth=3':>12s}  {'depth=4':>12s}  {'depth=5':>12s}  winner"
    )
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*7}")
    for ds_name, (_, _, _, metric_name) in datasets.items():
        sc = {d: float(np.mean(scores[(ds_name, d)])) for d in MAX_DEPTHS}
        winner = max(sc, key=sc.__getitem__)
        row = f"  {ds_name:<18s}"
        for d in MAX_DEPTHS:
            flag = " *" if d == winner else "  "
            row += f"  {sc[d]:>9.5f}{flag}"
        row += f"  d={winner}"
        print(row)

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")
    best_d   = ranked[0]
    best_z_v = float(np.mean(z_by_depth[best_d]))
    print(f"\n  Best max_depth overall : {best_d}  (mean z={best_z_v:.4f})")
    print(f"  Current default (d=4)  : z={base_z:.4f}")
    print(f"  Improvement            : {best_z_v - base_z:>+.4f} z-score units")
    print(
        f"\n  Wall time : {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used : {cpu_count}"
        f"\n"
    )


if __name__ == "__main__":
    main()
