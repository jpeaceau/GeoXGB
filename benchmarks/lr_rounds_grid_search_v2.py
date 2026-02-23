"""
GeoXGB -- High-Round Sweep (Multiprocessing)
============================================

Follow-up to lr_rounds_grid_search.py.  Learning rate is fixed at the
previously identified optimum (0.2) and n_rounds is swept from 1,000
upward to find where performance plateaus relative to runtime cost.

Datasets
--------
  friedman1      : Friedman #1 regression (1,000 samples, 10 features)
                   y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
  friedman2      : Friedman #2 regression (1,000 samples, 4 features)
                   y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2)  -- strong interactions
  classification : Synthetic binary classification (1,000 samples, 10 features)
                   5 informative + 5 noise, 2 clusters per class

Fixed parameters
----------------
  learning_rate = 0.2  (best from v1 grid search)
  max_depth=4, min_samples_leaf=5, reduce_ratio=0.7
  refit_interval=10, auto_noise=True, cache_geometry=False

Round sweep
-----------
  [1000, 1500, 2000, 3000, 5000, 7500, 10000]

Metric: R^2 (regression), ROC-AUC (classification)
CV   : 3-fold

Output
------
  Per-dataset table: rounds vs mean CV score + per-fold scores + wall time
  Score gain over 1000-round baseline
  Runtime cost per extra gain unit
  Recommendation: best round count given a configurable time budget

Usage
-----
    python benchmarks/lr_rounds_grid_search_v2.py

    # Limit to fewer round values for a quick run:
    python benchmarks/lr_rounds_grid_search_v2.py --rounds 1000 2000 5000

Requirements: geoxgb, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification, make_friedman1, make_friedman2
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.2
ROUNDS_SWEEP  = [1000, 1500, 2000, 3000, 5000, 7500, 10000]
N_FOLDS       = 3
RANDOM_STATE  = 42

GEO_FIXED = dict(
    learning_rate=LEARNING_RATE,
    max_depth=4,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    refit_interval=10,
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
    return {
        "friedman1":      (X1, y1, "regression",     "R^2"),
        "friedman2":      (X2, y2, "regression",     "R^2"),
        "classification": (X3, y3, "classification", "AUC"),
    }

# ---------------------------------------------------------------------------
# Worker -- module-level for Windows pickling
# ---------------------------------------------------------------------------

def _eval_fold(
    n_rounds: int,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[float, float]:
    """Fit one model on one fold. Returns (score, elapsed_seconds)."""
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(n_rounds=n_rounds, **GEO_FIXED)
    t0 = time.perf_counter()

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        pred  = m.predict(X[val_idx])
        score = float(r2_score(y[val_idx], pred))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        proba = m.predict_proba(X[val_idx])[:, 1]
        score = float(roc_auc_score(y[val_idx], proba))

    return score, time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title: str) -> None:
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


def _bar(val: float, ref: float, width: int = 32) -> str:
    """ASCII bar scaled to ref."""
    filled = int(round(width * max(val, 0.0) / max(ref, 1e-12)))
    return "#" * filled + "." * (width - filled)


def _delta_str(delta: float) -> str:
    if abs(delta) < 5e-5:
        return "  (no change)"
    sign = "+" if delta > 0 else "-"
    return f"  ({sign}{abs(delta):.6f})"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rounds_sweep: list[int]) -> None:
    cpu_count = multiprocessing.cpu_count()
    datasets  = _make_datasets()
    n_jobs    = len(rounds_sweep) * len(datasets) * N_FOLDS

    _section("GeoXGB -- High-Round Sweep  (lr=0.2 fixed)")
    print(
        f"\n  learning_rate : {LEARNING_RATE}  (fixed -- best from v1 grid search)"
        f"\n  n_rounds sweep: {rounds_sweep}"
        f"\n  Datasets      : {', '.join(datasets.keys())}"
        f"\n  CV folds      : {N_FOLDS}"
        f"\n  Total jobs    : {n_jobs}"
        f"\n  CPUs          : {cpu_count}"
        f"\n"
        f"\n  Fixed params  : max_depth=4  min_samples_leaf=5  reduce_ratio=0.7"
        f"\n                  refit_interval=10  auto_noise=True  cache_geometry=False"
    )

    # -----------------------------------------------------------------------
    # Build CV splits
    # -----------------------------------------------------------------------
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ds_name, (X, y, task, _m) in datasets.items():
        if task == "classification":
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
            splits[ds_name] = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            splits[ds_name] = list(kf.split(X))

    # -----------------------------------------------------------------------
    # Build job list
    # -----------------------------------------------------------------------
    jobs: list[tuple] = []
    job_keys: list[tuple[str, int, int]] = []  # (ds, n_rounds, fold_idx)

    for ds_name, (X, y, task, _m) in datasets.items():
        for n_rounds in rounds_sweep:
            for fold_idx, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((n_rounds, task, X, y, tr, val))
                job_keys.append((ds_name, n_rounds, fold_idx))

    # -----------------------------------------------------------------------
    # Parallel execution
    # -----------------------------------------------------------------------
    print(f"\n  Launching {n_jobs} jobs...")
    t_wall = time.perf_counter()
    try:
        raw_results = Parallel(
            n_jobs=cpu_count,
            prefer="processes",
            verbose=2,
        )(
            delayed(_eval_fold)(*job) for job in jobs
        )
    except Exception as exc:
        print(f"\n  [parallel failed: {exc!r}]  -- running sequentially")
        raw_results = [_eval_fold(*job) for job in jobs]
    t_wall = time.perf_counter() - t_wall
    print(f"\n  Done. Wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)")

    # -----------------------------------------------------------------------
    # Aggregate: results[ds_name][n_rounds] = {"scores": [...], "times": [...]}
    # -----------------------------------------------------------------------
    results: dict[str, dict[int, dict]] = {
        ds: {r: {"scores": [], "times": []} for r in rounds_sweep}
        for ds in datasets
    }
    for (ds_name, n_rounds, _fold_idx), (score, elapsed) in zip(job_keys, raw_results):
        results[ds_name][n_rounds]["scores"].append(score)
        results[ds_name][n_rounds]["times"].append(elapsed)

    # -----------------------------------------------------------------------
    # Per-dataset tables
    # -----------------------------------------------------------------------
    _section("Per-Dataset Results")

    # Collect per-dataset best scores for scaling bars
    ds_summary: dict[str, list[dict]] = {}

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name}  task: {task})")

        rows = []
        for n_rounds in rounds_sweep:
            scores = results[ds_name][n_rounds]["scores"]
            times  = results[ds_name][n_rounds]["times"]
            rows.append({
                "rounds":     n_rounds,
                "mean_score": float(np.mean(scores)),
                "std_score":  float(np.std(scores)),
                "fold_scores": scores,
                "mean_time":  float(np.mean(times)),
                "total_time": float(np.sum(times)),
            })
        ds_summary[ds_name] = rows

        baseline_score = rows[0]["mean_score"]  # score at lowest rounds
        best_score     = max(r["mean_score"] for r in rows)
        bar_ref        = best_score

        # Header
        print(
            f"\n  {'rounds':>8s}  {'mean ' + metric_name:>10s}  {'std':>6s}  "
            f"{'delta vs 1k':>11s}  {'avg fold time':>13s}  score bar"
        )
        print(f"  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*11}  {'-'*13}  {'-'*32}")

        for row in rows:
            delta   = row["mean_score"] - baseline_score
            d_str   = f"{delta:+.6f}"
            bar     = _bar(row["mean_score"], bar_ref)
            best_tag = " [best]" if abs(row["mean_score"] - best_score) < 1e-9 else "       "
            print(
                f"  {row['rounds']:>8d}  {row['mean_score']:>10.6f}  "
                f"{row['std_score']:>6.4f}  {d_str:>11s}  "
                f"{row['mean_time']:>11.1f}s  {bar}{best_tag}"
            )

        # Per-fold breakdown
        print(f"\n  Per-fold scores:")
        print(f"  {'rounds':>8s}", end="")
        for i in range(N_FOLDS):
            print(f"   fold {i+1}", end="")
        print()
        print(f"  {'-'*8}", end="")
        for _ in range(N_FOLDS):
            print(f"  {'-'*7}", end="")
        print()
        for row in rows:
            print(f"  {row['rounds']:>8d}", end="")
            for s in row["fold_scores"]:
                print(f"  {s:>7.5f}", end="")
            print()

        # Gain-vs-cost analysis
        print(f"\n  Gain vs cost (relative to {rounds_sweep[0]:,} rounds):")
        print(
            f"  {'rounds':>8s}  {'score gain':>10s}  "
            f"{'time/fold (s)':>13s}  {'gain per 100s':>13s}"
        )
        print(f"  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*13}")
        base_time = rows[0]["mean_time"]
        for row in rows:
            gain      = row["mean_score"] - baseline_score
            time_diff = row["mean_time"] - base_time
            eff       = (gain / time_diff * 100) if time_diff > 0.1 else float("inf")
            eff_str   = f"{eff:>13.4f}" if eff != float("inf") else "      baseline"
            print(
                f"  {row['rounds']:>8d}  {gain:>+10.6f}  "
                f"{row['mean_time']:>13.1f}  {eff_str}"
            )

    # -----------------------------------------------------------------------
    # Cross-dataset summary table
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Summary")
    print(
        "\n  Mean z-score across all datasets (higher = better relative to peers)."
        "\n  Z-scores remove scale differences between R^2 and AUC."
    )

    # Compute z-scores per dataset then average across datasets
    z_by_rounds: dict[int, list[float]] = {r: [] for r in rounds_sweep}
    for ds_name, rows in ds_summary.items():
        scores_arr = [row["mean_score"] for row in rows]
        mu  = float(np.mean(scores_arr))
        sig = float(np.std(scores_arr)) or 1e-9
        for row in rows:
            z_by_rounds[row["rounds"]].append((row["mean_score"] - mu) / sig)

    print(
        f"\n  {'rounds':>8s}  {'mean z-score':>12s}  "
        f"{'delta z vs 1k':>14s}  bar"
    )
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*32}")

    baseline_z  = float(np.mean(z_by_rounds[rounds_sweep[0]]))
    max_z       = max(float(np.mean(v)) for v in z_by_rounds.values())
    best_rounds = max(rounds_sweep, key=lambda r: float(np.mean(z_by_rounds[r])))

    for r in rounds_sweep:
        z    = float(np.mean(z_by_rounds[r]))
        dz   = z - baseline_z
        bar  = _bar(max(z, 0), max(max_z, 1e-9))
        flag = " [best]" if r == best_rounds else "       "
        print(f"  {r:>8d}  {z:>12.4f}  {dz:>+14.4f}  {bar}{flag}")

    # -----------------------------------------------------------------------
    # Diminishing returns analysis
    # -----------------------------------------------------------------------
    _section("Diminishing Returns Analysis")
    print(
        "\n  Marginal z-score gain per 1,000 additional rounds."
        "\n  When this drops below 0.05 the extra compute rarely justifies itself."
    )
    print(f"\n  {'interval':>18s}  {'delta z':>8s}  {'delta z / 1k rounds':>20s}  note")
    print(f"  {'-'*18}  {'-'*8}  {'-'*20}  {'-'*30}")

    z_vals = [(r, float(np.mean(z_by_rounds[r]))) for r in rounds_sweep]
    for i in range(1, len(z_vals)):
        r_prev, z_prev = z_vals[i - 1]
        r_cur,  z_cur  = z_vals[i]
        dz       = z_cur - z_prev
        dr       = r_cur - r_prev
        dz_per1k = dz / (dr / 1000.0)
        note     = "plateau" if dz_per1k < 0.05 else ("good gain" if dz_per1k > 0.2 else "")
        label    = f"{r_prev:,} -> {r_cur:,}"
        print(f"  {label:>18s}  {dz:>+8.4f}  {dz_per1k:>20.4f}  {note}")

    # -----------------------------------------------------------------------
    # Wall-time breakdown
    # -----------------------------------------------------------------------
    _section("Wall-Time Summary")
    print(
        f"\n  Total benchmark wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used: {cpu_count}"
        f"\n"
        f"\n  Estimated single-fold training time per dataset:"
    )
    print(f"\n  {'dataset':<18s}", end="")
    for r in rounds_sweep:
        print(f"  {r:>7d}", end="")
    print()
    print(f"  {'-'*18}", end="")
    for _ in rounds_sweep:
        print(f"  {'-'*7}", end="")
    print()
    for ds_name, rows in ds_summary.items():
        print(f"  {ds_name:<18s}", end="")
        for row in rows:
            print(f"  {row['mean_time']:>6.1f}s", end="")
        print()

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")

    # Find the smallest round count within 0.02 z of best
    threshold = max_z - 0.02
    efficient = next(
        (r for r in rounds_sweep if float(np.mean(z_by_rounds[r])) >= threshold),
        best_rounds,
    )

    print(
        f"\n  Fixed:  learning_rate = {LEARNING_RATE}"
        f"\n"
        f"\n  Best round count overall  : {best_rounds:,}"
        f"\n  Efficient round count     : {efficient:,}"
        f"  (first count within 0.02 z-score of best)"
        f"\n"
        f"\n  Per-dataset best scores at {best_rounds:,} rounds:"
    )
    for ds_name, (X, y, task, metric_name) in datasets.items():
        row = next(r for r in ds_summary[ds_name] if r["rounds"] == best_rounds)
        print(f"    {ds_name:<18s}  {metric_name} = {row['mean_score']:.6f}"
              f"  (avg fold time = {row['mean_time']:.1f}s)")

    print(
        f"\n  Notes"
        f"\n  -----"
        f"\n  - learning_rate=0.2 was chosen from v1 grid search."
        f"\n  - Scores are averaged over {N_FOLDS} CV folds; use std to gauge stability."
        f"\n  - refit_interval=10 is fixed: at 10,000 rounds this means 1,000 HVRT"
        f"\n    refits per fold.  If wall time is a bottleneck, increase"
        f"\n    refit_interval (e.g. 50) or set cache_geometry=True."
        f"\n  - If the best round count is the last in the sweep, re-run with higher"
        f"\n    values to confirm the plateau."
        f"\n"
        f"\n  Wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used: {cpu_count}"
        f"\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoXGB high-round sweep (lr=0.2 fixed)"
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=ROUNDS_SWEEP,
        metavar="N",
        help=(
            "Round counts to evaluate (default: "
            + " ".join(str(r) for r in ROUNDS_SWEEP)
            + ")"
        ),
    )
    args = parser.parse_args()
    main(sorted(set(args.rounds)))
