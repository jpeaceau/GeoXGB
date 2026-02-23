"""
GeoXGB -- Weak Learner Criterion Benchmark (Multiprocessing)
============================================================

Compares DecisionTreeRegressor split criteria:
  squared_error  : minimise MSE at each split (sklearn default)
  friedman_mse   : Friedman improvement score -- penalises unbalanced
                   splits and can be faster / more accurate in practice

Note: GeoXGB uses DecisionTreeRegressor for ALL tasks (including
classification) because gradient boosting fits regression trees to
continuous pseudo-residuals (log-loss gradients), not class labels.
Gini / entropy are irrelevant here.

Settings: n_rounds=1000, learning_rate=0.2, refit_interval=20
          (refit_interval=20 is the recommended value from the
          refit interval sensitivity benchmark)

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
    python benchmarks/criterion_benchmark.py

Requirements: geoxgb, scikit-learn, numpy, joblib
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

N_ROUNDS      = 1000
LEARNING_RATE = 0.2
REFIT_INTERVAL = 20
N_FOLDS       = 5
RANDOM_STATE  = 42

CRITERIA = ["squared_error", "friedman_mse"]

GEO_FIXED = dict(
    n_rounds=N_ROUNDS,
    learning_rate=LEARNING_RATE,
    refit_interval=REFIT_INTERVAL,
    max_depth=4,
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
# Worker -- module-level for Windows pickling
# ---------------------------------------------------------------------------

def _eval_fold(
    criterion: str,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple:
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(**GEO_FIXED, tree_criterion=criterion)
    t0 = time.perf_counter()

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(r2_score(y[val_idx], m.predict(X[val_idx])))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(roc_auc_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))

    return criterion, score, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Formatting helpers
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
    n_jobs    = len(CRITERIA) * len(datasets) * N_FOLDS

    _section("GeoXGB -- Weak Learner Criterion Benchmark")
    print(
        f"\n  n_rounds        : {N_ROUNDS}"
        f"\n  learning_rate   : {LEARNING_RATE}"
        f"\n  refit_interval  : {REFIT_INTERVAL}"
        f"\n  Criteria tested : {CRITERIA}"
        f"\n  CV folds        : {N_FOLDS}"
        f"\n  Total jobs      : {n_jobs}"
        f"\n  CPUs            : {cpu_count}"
    )
    print(f"\n  Datasets:")
    ds_notes = {
        "friedman1":      "10 feat, noise=1.0, regression",
        "friedman2":      "10 feat, noise=0.0, regression",
        "classification": "10 feat, 5 informative, clean binary",
        "sparse_highdim": "40 feat, 8 informative, noise=20, regression",
        "noisy_clf":      "20 feat, flip_y=0.10, sep=0.5, binary",
    }
    for ds, note in ds_notes.items():
        print(f"    {ds:<18s}  {note}")
    print(
        f"\n  Note: GeoXGB fits DecisionTreeRegressor to pseudo-residuals"
        f"\n        for all tasks. Gini/entropy do not apply."
        f"\n        squared_error = sklearn default (currently unspecified)."
    )

    # -----------------------------------------------------------------------
    # CV splits
    # -----------------------------------------------------------------------
    splits: dict[str, list] = {}
    for ds_name, (X, y, task, _) in datasets.items():
        if task == "classification":
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
        else:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits[ds_name] = list(kf.split(X, y))

    # -----------------------------------------------------------------------
    # Job list
    # -----------------------------------------------------------------------
    jobs: list[tuple] = []
    job_keys: list[tuple] = []

    for ds_name, (X, y, task, _) in datasets.items():
        for criterion in CRITERIA:
            for fi, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((criterion, task, X, y, tr, val))
                job_keys.append((ds_name, criterion, fi))

    # -----------------------------------------------------------------------
    # Parallel execution
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

    for (ds_name, criterion, _fi), (_, score, elapsed) in zip(job_keys, raw):
        scores[(ds_name, criterion)].append(score)
        times[ (ds_name, criterion)].append(elapsed)

    # -----------------------------------------------------------------------
    # Per-dataset tables
    # -----------------------------------------------------------------------
    _section("Per-Dataset Results")

    baseline = "squared_error"

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name})")

        rows = []
        for c in CRITERIA:
            sc = scores[(ds_name, c)]
            rows.append({
                "criterion": c,
                "mean":  float(np.mean(sc)),
                "std":   float(np.std(sc)),
                "folds": sc,
                "time":  float(np.mean(times[(ds_name, c)])),
            })
        rows.sort(key=lambda r: -r["mean"])
        best_mean = rows[0]["mean"]
        base_mean = next(r["mean"] for r in rows if r["criterion"] == baseline)

        print(
            f"\n  {'criterion':<16s}  {'mean':>9s}  {'std':>6s}"
            f"  {'vs squared_err':>14s}  {'avg time':>8s}  score"
        )
        print(
            f"  {'-'*16}  {'-'*9}  {'-'*6}"
            f"  {'-'*14}  {'-'*8}  {'-'*28}"
        )
        for r in rows:
            delta   = r["mean"] - base_mean
            delta_s = f"{delta:>+.6f}"
            bar     = _bar(r["mean"], best_mean)
            tag     = " [best]" if r is rows[0] else (" [base]" if r["criterion"] == baseline else "")
            print(
                f"  {r['criterion']:<16s}  {r['mean']:>9.5f}  {r['std']:>6.4f}"
                f"  {delta_s:>14s}  {r['time']:>7.1f}s  {bar}{tag}"
            )

    # -----------------------------------------------------------------------
    # Cross-dataset z-score summary
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Z-Score Summary")
    print(
        "\n  Z-scores normalise each dataset's scores before averaging,"
        "\n  removing scale differences between R^2 and AUC."
    )

    z_by_criterion: dict[str, list[float]] = {c: [] for c in CRITERIA}
    for ds_name in datasets:
        ds_scores = {c: float(np.mean(scores[(ds_name, c)])) for c in CRITERIA}
        mu  = float(np.mean(list(ds_scores.values())))
        sig = float(np.std(list(ds_scores.values()))) or 1e-9
        for c, v in ds_scores.items():
            z_by_criterion[c].append((v - mu) / sig)

    ranked = sorted(CRITERIA, key=lambda c: -float(np.mean(z_by_criterion[c])))
    base_z = float(np.mean(z_by_criterion[baseline]))

    print(f"\n  {'criterion':<16s}  {'mean z':>8s}  {'vs squared_err':>14s}  bar")
    print(f"  {'-'*16}  {'-'*8}  {'-'*14}  {'-'*28}")
    max_z = float(np.mean(z_by_criterion[ranked[0]]))
    for c in ranked:
        z   = float(np.mean(z_by_criterion[c]))
        dz  = z - base_z
        bar = _bar(max(z, 0), max(max_z, 1e-9))
        tag = " [best]" if c == ranked[0] else (" [base]" if c == baseline else "")
        print(
            f"  {c:<16s}  {z:>8.4f}  {dz:>+14.4f}  {bar}{tag}"
        )

    # -----------------------------------------------------------------------
    # Head-to-head per dataset
    # -----------------------------------------------------------------------
    _section("Head-to-Head Summary")
    print(
        f"\n  {'dataset':<18s}  {'winner':<16s}  {'delta':>10s}  {'faster?':>10s}"
    )
    print(
        f"  {'-'*18}  {'-'*16}  {'-'*10}  {'-'*10}"
    )
    for ds_name in datasets:
        sc = {c: float(np.mean(scores[(ds_name, c)])) for c in CRITERIA}
        tm = {c: float(np.mean(times[(ds_name, c)])) for c in CRITERIA}
        winner  = max(sc, key=sc.__getitem__)
        loser   = [c for c in CRITERIA if c != winner][0]
        delta   = sc[winner] - sc[loser]
        faster  = min(tm, key=tm.__getitem__)
        time_delta = abs(tm[CRITERIA[0]] - tm[CRITERIA[1]])
        faster_s = f"{faster} ({time_delta:.1f}s)" if time_delta > 0.5 else "tied"
        print(
            f"  {ds_name:<18s}  {winner:<16s}  {delta:>+10.6f}  {faster_s:>10s}"
        )

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")
    best_c   = ranked[0]
    best_z_v = float(np.mean(z_by_criterion[best_c]))

    print(f"\n  Best criterion overall : {best_c}  (mean z={best_z_v:.4f})")
    print(f"  Baseline (squared_err) : z={base_z:.4f}")
    print(f"  Improvement            : {best_z_v - base_z:>+.4f} z-score units")
    print(
        f"\n  Wall time : {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used : {cpu_count}"
        f"\n"
    )


if __name__ == "__main__":
    main()
