"""
GeoXGB -- Rounds & Learning Rate Grid Search (Multiprocessing)
==============================================================

Sweeps n_rounds x learning_rate across three datasets using all available
CPU cores.  Every (config, dataset, fold) triple is an independent job so
the search saturates the machine efficiently.

Datasets
--------
  friedman1      : Friedman #1 regression (1,000 samples, 10 features)
                   y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
  friedman2      : Friedman #2 regression (1,000 samples, 4 features)
                   y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2)  -- strong interactions
  classification : Synthetic binary classification (1,000 samples, 10 features)
                   5 informative + 5 noise, 2 clusters per class

Search space
------------
  n_rounds     : [50, 100, 200, 400, 700, 1000]
  learning_rate: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

Fixed parameters (same for all configs)
----------------------------------------
  max_depth=4, min_samples_leaf=5, reduce_ratio=0.7
  refit_interval=10, auto_noise=True, cache_geometry=False

Metric: R^2 (regression), ROC-AUC (classification)
CV   : 3-fold stratified (classification) / 3-fold (regression)

Output
------
  Per-dataset ASCII heatmap (rows = rounds, cols = lr)
  Best cell highlighted with [*]
  Cross-dataset ranking of top configurations
  Wall-time summary

Usage
-----
    python benchmarks/lr_rounds_grid_search.py

Requirements: geoxgb, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import multiprocessing
import time
import warnings
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification, make_friedman1, make_friedman2
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

ROUNDS_GRID = [50, 100, 200, 400, 700, 1000]
LR_GRID     = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

N_FOLDS      = 3
RANDOM_STATE = 42

# Fixed non-tuned parameters applied to every config
GEO_FIXED = dict(
    max_depth=4,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    cache_geometry=False,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Dataset definitions (generated at module level -- small so pickling is fast)
# ---------------------------------------------------------------------------

def _make_datasets() -> dict:
    """Return a dict of {name: (X, y, task, metric_name)}."""
    X1, y1 = make_friedman1(
        n_samples=1_000, n_features=10, noise=1.0, random_state=RANDOM_STATE
    )
    X2, y2 = make_friedman2(n_samples=1_000, noise=0.0, random_state=RANDOM_STATE)
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
# Worker function -- must be at module level for Windows pickling
# ---------------------------------------------------------------------------

def _eval_fold(
    n_rounds: int,
    learning_rate: float,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> float:
    """Fit one model on one fold, return the validation metric."""
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(n_rounds=n_rounds, learning_rate=learning_rate, **GEO_FIXED)

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[val_idx])
        return float(r2_score(y[val_idx], pred))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        proba = m.predict_proba(X[val_idx])[:, 1]
        return float(roc_auc_score(y[val_idx], proba))

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 76
_SEP2 = "-" * 76


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title: str) -> None:
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


def _heatmap(
    grid: dict[tuple[int, float], list[float]],
    rounds_list: list[int],
    lr_list: list[float],
    metric: str,
) -> None:
    """Print a rounds x lr heatmap of mean CV scores."""
    # Compute means
    means: dict[tuple[int, float], float] = {}
    for key, scores in grid.items():
        means[key] = float(np.mean(scores))

    best_key  = max(means, key=means.__getitem__)
    best_val  = means[best_key]

    # Column header
    col_w = 9
    lr_header = "  ".join(f"lr={lr:<5.2f}" for lr in lr_list)
    print(f"\n  {'rounds':>10s}   {lr_header}")
    print(f"  {'-'*10}   {'-' * (col_w * len(lr_list) + 2 * (len(lr_list) - 1))}")

    for r in rounds_list:
        row_vals = []
        for lr in lr_list:
            val = means.get((r, lr), float("nan"))
            tag = "[*]" if (r, lr) == best_key else "   "
            row_vals.append(f"{tag}{val:+.4f}")
        row_str = "  ".join(f"{v:>9s}" for v in row_vals)
        print(f"  {r:>10d}   {row_str}")

    print(f"\n  Best: rounds={best_key[0]}  lr={best_key[1]}  "
          f"mean {metric}={best_val:.6f}")


def _top_configs(
    all_means: dict[str, dict[tuple[int, float], float]],
    metric_names: dict[str, str],
    top_n: int = 10,
) -> None:
    """Rank configs by mean score across all datasets (z-scored per dataset)."""
    # Collect all keys
    all_keys: set[tuple[int, float]] = set()
    for d in all_means.values():
        all_keys.update(d.keys())

    # Z-score normalise per dataset then average
    ranked: list[tuple[float, int, float]] = []
    for key in all_keys:
        z_scores = []
        for ds_name, ds_means in all_means.items():
            vals = list(ds_means.values())
            mu  = float(np.mean(vals))
            sig = float(np.std(vals)) or 1e-9
            z   = (ds_means.get(key, mu) - mu) / sig
            z_scores.append(z)
        ranked.append((float(np.mean(z_scores)), key[0], key[1]))

    ranked.sort(reverse=True)
    print(f"\n  {'Rank':>4s}  {'n_rounds':>8s}  {'lr':>6s}  {'mean z-score':>12s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*12}")
    for rank, (z, r, lr) in enumerate(ranked[:top_n], 1):
        print(f"  {rank:>4d}  {r:>8d}  {lr:>6.2f}  {z:>12.4f}")

    best_z, best_r, best_lr = ranked[0]
    print(f"\n  Overall best: n_rounds={best_r}  learning_rate={best_lr}  "
          f"(z={best_z:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    n_configs = len(ROUNDS_GRID) * len(LR_GRID)
    datasets  = _make_datasets()

    _section("GeoXGB -- Rounds x Learning Rate Grid Search")
    print(
        f"\n  Datasets      : {', '.join(datasets.keys())}"
        f"\n  n_rounds      : {ROUNDS_GRID}"
        f"\n  learning_rate : {LR_GRID}"
        f"\n  Configs       : {n_configs}  (rounds x lr)"
        f"\n  CV folds      : {N_FOLDS}"
        f"\n  Total jobs    : {n_configs * len(datasets) * N_FOLDS}"
        f"\n  CPUs          : {cpu_count}"
        f"\n"
        f"\n  Fixed params  : {GEO_FIXED}"
    )

    # -----------------------------------------------------------------------
    # Build fold splits for each dataset
    # -----------------------------------------------------------------------
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ds_name, (X, y, task, _metric) in datasets.items():
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
    job_keys: list[tuple[str, int, float, int]] = []  # (ds, rounds, lr, fold)

    for ds_name, (X, y, task, _metric) in datasets.items():
        for n_rounds, lr in product(ROUNDS_GRID, LR_GRID):
            for fold_idx, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((n_rounds, lr, task, X, y, tr, val))
                job_keys.append((ds_name, n_rounds, lr, fold_idx))

    total_jobs = len(jobs)
    print(f"\n  Launching {total_jobs} jobs across {cpu_count} workers...")

    # -----------------------------------------------------------------------
    # Parallel execution
    # -----------------------------------------------------------------------
    t_wall = time.perf_counter()
    try:
        scores_flat = Parallel(
            n_jobs=cpu_count,
            prefer="processes",
            verbose=2,
        )(
            delayed(_eval_fold)(*job) for job in jobs
        )
    except Exception as exc:
        print(f"\n  [parallel failed: {exc!r}]  -- falling back to sequential")
        scores_flat = [_eval_fold(*job) for job in jobs]
    t_wall = time.perf_counter() - t_wall

    print(f"\n  Done. Wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)")

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------
    # grid[ds_name][(n_rounds, lr)] = [fold_scores...]
    grid: dict[str, dict[tuple[int, float], list[float]]] = {
        ds: {} for ds in datasets
    }
    for (ds_name, n_rounds, lr, _fold_idx), score in zip(job_keys, scores_flat):
        key = (n_rounds, lr)
        grid[ds_name].setdefault(key, []).append(score)

    all_means: dict[str, dict[tuple[int, float], float]] = {}
    for ds_name, ds_grid in grid.items():
        all_means[ds_name] = {k: float(np.mean(v)) for k, v in ds_grid.items()}

    # -----------------------------------------------------------------------
    # Per-dataset heatmaps
    # -----------------------------------------------------------------------
    _section("Per-Dataset Heatmaps")

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name})")
        _heatmap(grid[ds_name], ROUNDS_GRID, LR_GRID, metric_name)

    # -----------------------------------------------------------------------
    # Cross-dataset ranking
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Ranking  (z-scored, top 10)")
    print(
        "\n  Each config is z-scored within each dataset then averaged across"
        "\n  datasets to give a dataset-agnostic quality measure."
    )
    metric_names = {ds: info[3] for ds, info in datasets.items()}
    _top_configs(all_means, metric_names, top_n=10)

    # -----------------------------------------------------------------------
    # Learning rate slice -- best rounds per lr
    # -----------------------------------------------------------------------
    _section("Learning Rate Analysis  (aggregated across datasets & rounds)")
    print(
        "\n  For each learning rate, show the best score achieved across"
        "\n  all round counts, averaged over datasets (z-scored)."
    )
    lr_best_z: dict[float, float] = {}
    for lr in LR_GRID:
        z_per_ds = []
        for ds_name, ds_means in all_means.items():
            vals = list(ds_means.values())
            mu   = float(np.mean(vals))
            sig  = float(np.std(vals)) or 1e-9
            lr_scores = [ds_means[(r, lr)] for r in ROUNDS_GRID if (r, lr) in ds_means]
            if lr_scores:
                z_per_ds.append((max(lr_scores) - mu) / sig)
        lr_best_z[lr] = float(np.mean(z_per_ds)) if z_per_ds else float("nan")

    print(f"\n  {'lr':>6s}  {'best z-score':>12s}  bar")
    print(f"  {'-'*6}  {'-'*12}  {'-'*30}")
    max_z = max(lr_best_z.values())
    for lr, z in sorted(lr_best_z.items()):
        bar = "#" * int(round(30 * max(z, 0) / max(max_z, 1e-9)))
        flag = " <-- prev best" if abs(lr - 0.5) < 1e-9 else ""
        print(f"  {lr:>6.2f}  {z:>12.4f}  {bar}{flag}")

    best_lr = max(lr_best_z, key=lr_best_z.__getitem__)
    print(f"\n  Best learning rate: {best_lr}")

    # -----------------------------------------------------------------------
    # Rounds slice -- effect of round count (best lr fixed)
    # -----------------------------------------------------------------------
    _section("Round Count Analysis  (aggregated across datasets & best lr)")
    print(
        f"\n  For each round count, show the mean z-score using the best lr"
        f"\n  found above ({best_lr}) and the overall best lr per dataset."
    )
    print(f"\n  {'rounds':>8s}  {'z @ best_lr':>11s}  {'z @ per-ds best':>15s}  bar")
    print(f"  {'-'*8}  {'-'*11}  {'-'*15}  {'-'*30}")

    for r in ROUNDS_GRID:
        # z at overall best lr
        z_fixed = []
        for ds_name, ds_means in all_means.items():
            vals = list(ds_means.values())
            mu   = float(np.mean(vals))
            sig  = float(np.std(vals)) or 1e-9
            v    = ds_means.get((r, best_lr), mu)
            z_fixed.append((v - mu) / sig)
        z_fixed_mean = float(np.mean(z_fixed))

        # z at per-dataset best lr for this round count
        z_best = []
        for ds_name, ds_means in all_means.items():
            vals = list(ds_means.values())
            mu   = float(np.mean(vals))
            sig  = float(np.std(vals)) or 1e-9
            lr_scores_r = {lr: ds_means[(r, lr)] for lr in LR_GRID if (r, lr) in ds_means}
            if lr_scores_r:
                best_v = max(lr_scores_r.values())
                z_best.append((best_v - mu) / sig)
        z_best_mean = float(np.mean(z_best)) if z_best else float("nan")

        bar = "#" * int(round(30 * max(z_fixed_mean, 0) / max(max_z, 1e-9)))
        print(f"  {r:>8d}  {z_fixed_mean:>11.4f}  {z_best_mean:>15.4f}  {bar}")

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")

    # Find overall best config
    overall_best: tuple[float, int, float] = max(
        (
            (
                float(np.mean([
                    (all_means[ds].get((r, lr), 0) - np.mean(list(all_means[ds].values())))
                    / (np.std(list(all_means[ds].values())) or 1e-9)
                    for ds in datasets
                ])),
                r,
                lr,
            )
            for r, lr in product(ROUNDS_GRID, LR_GRID)
        ),
        key=lambda t: t[0],
    )
    best_z_val, best_r_final, best_lr_final = overall_best

    print(
        f"\n  Best overall config (z-score across all datasets):"
        f"\n    n_rounds     = {best_r_final}"
        f"\n    learning_rate = {best_lr_final}"
        f"\n    mean z-score  = {best_z_val:.4f}"
        f"\n"
        f"\n  Per-dataset best scores at this config:"
    )
    for ds_name, (X, y, task, metric_name) in datasets.items():
        v = all_means[ds_name].get((best_r_final, best_lr_final), float("nan"))
        print(f"    {ds_name:<18s}  {metric_name} = {v:.6f}")

    print(
        f"\n  Notes"
        f"\n  -----"
        f"\n  - Higher rounds never hurts but runtime scales linearly."
        f"\n  - If wall time is a concern, pick the smallest rounds value"
        f"\n    whose z-score is within 0.1 of the best."
        f"\n  - Learning rate and rounds interact: a lower lr may benefit"
        f"\n    more from additional rounds than a higher lr."
        f"\n  - These results are for max_depth=4. Deeper trees may shift"
        f"\n    the optimal lr downward."
        f"\n"
        f"\n  Wall time: {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used: {cpu_count}"
        f"\n"
    )


if __name__ == "__main__":
    main()
