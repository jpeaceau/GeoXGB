"""
GeoXGB vs XGBoost Comparison Benchmark
=======================================

Compares four configurations on the four meta-analysis datasets using
identical 5-fold CV splits and z-score-normalised targets:

  GeoXGB (old defaults)  -- baseline values from meta_analysis.py
  GeoXGB (new defaults)  -- values recommended by OAT + pairwise analysis
  XGBoost (sklearn defaults)
  XGBoost (matched)      -- n_estimators=1000, lr=0.05, matched budget to GeoXGB

All datasets, seeds, and fold splits are identical to meta_analysis.py so
results are directly comparable with the OAT/pairwise baseline figures.

Output
------
  benchmarks/results/xgb_comparison.csv   (per-fold raw results)

Usage
-----
    python benchmarks/xgb_comparison.py
    python benchmarks/xgb_comparison.py --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_friedman1, make_friedman2, make_regression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "xgb_comparison.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {
    "GeoXGB_old": dict(
        _model="geoxgb",
        n_rounds          = 1000,
        learning_rate     = 0.2,
        max_depth         = 4,
        auto_noise        = True,
        refit_noise_floor = 0.05,
        noise_guard       = True,
        expand_ratio      = 0.0,
        refit_interval    = 20,
        y_weight          = 0.5,
        # unchanged params
        min_samples_leaf  = 5,
        reduce_ratio      = 0.7,
        method            = "variance_ordered",
        variance_weighted = True,
        assignment_strategy = "auto",
        min_train_samples = 5000,
        cache_geometry    = False,
        auto_expand       = True,
        bandwidth         = "auto",
        generation_strategy = "epanechnikov",
        tree_splitter     = "random",
        random_state      = 42,
    ),
    "GeoXGB_new": dict(
        _model="geoxgb",
        n_rounds          = 1000,
        learning_rate     = 0.05,   # 0.2  -> 0.05
        max_depth         = 2,      # 4    -> 2
        auto_noise        = False,  # True -> False
        refit_noise_floor = 0.0,    # 0.05 -> 0.0
        noise_guard       = False,  # True -> False
        expand_ratio      = 0.1,    # 0.0  -> 0.1
        refit_interval    = 10,     # 20   -> 10
        y_weight          = 0.9,    # 0.5  -> 0.9
        # unchanged params
        min_samples_leaf  = 5,
        reduce_ratio      = 0.7,
        method            = "variance_ordered",
        variance_weighted = True,
        assignment_strategy = "auto",
        min_train_samples = 5000,
        cache_geometry    = False,
        auto_expand       = True,
        bandwidth         = "auto",
        generation_strategy = "epanechnikov",
        tree_splitter     = "random",
        random_state      = 42,
    ),
    "GeoXGB_new_3k": dict(
        _model="geoxgb",
        n_rounds          = 3000,   # higher ceiling; expected to converge well before
        learning_rate     = 0.05,
        max_depth         = 2,
        auto_noise        = False,
        refit_noise_floor = 0.0,
        noise_guard       = False,
        expand_ratio      = 0.1,
        refit_interval    = 10,
        y_weight          = 0.9,
        cone_guard        = False,
        min_samples_leaf  = 5,
        reduce_ratio      = 0.7,
        method            = "variance_ordered",
        variance_weighted = True,
        assignment_strategy = "auto",
        min_train_samples = 5000,
        cache_geometry    = False,
        auto_expand       = True,
        bandwidth         = "auto",
        generation_strategy = "epanechnikov",
        tree_splitter     = "random",
        random_state      = 42,
    ),
    "GeoXGB_cone_3k": dict(
        _model="geoxgb",
        n_rounds          = 3000,   # ceiling; cone guard will stop early if residuals exhaust
        learning_rate     = 0.05,
        max_depth         = 2,
        auto_noise        = False,
        refit_noise_floor = 0.0,
        noise_guard       = False,
        expand_ratio      = 0.1,
        refit_interval    = 10,
        y_weight          = 0.9,
        cone_guard        = True,   # geometry-informed early stopping
        min_samples_leaf  = 5,
        reduce_ratio      = 0.7,
        method            = "variance_ordered",
        variance_weighted = True,
        assignment_strategy = "auto",
        min_train_samples = 5000,
        cache_geometry    = False,
        auto_expand       = True,
        bandwidth         = "auto",
        generation_strategy = "epanechnikov",
        tree_splitter     = "random",
        random_state      = 42,
    ),
    "XGBoost_default": dict(
        _model="xgboost",
        # sklearn API defaults (XGBoost 1.7+)
        n_estimators      = 100,
        max_depth         = 6,
        learning_rate     = 0.3,
        subsample         = 1.0,
        colsample_bytree  = 1.0,
        min_child_weight  = 1,
        reg_alpha         = 0.0,
        reg_lambda        = 1.0,
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = 1,
    ),
    "XGBoost_matched": dict(
        _model="xgboost",
        # Matched learning budget: same n_estimators and lr as GeoXGB_new
        n_estimators      = 1000,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        reg_alpha         = 0.0,
        reg_lambda        = 1.0,
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = 1,
    ),
}

FIELDNAMES = [
    "model", "dataset", "n_samples", "n_features",
    "fold", "val_rmse", "val_mae", "val_r2", "train_time_s",
    "convergence_round", "convergence_reason", "n_refit_history", "status",
]

N_FOLDS      = 5
RANDOM_STATE = 42

DATASET_PROPERTIES: dict[str, dict] = {
    "friedman1":    dict(n_samples=1_000, n_features=10),
    "friedman2":    dict(n_samples=1_000, n_features=4),
    "sparse_noisy": dict(n_samples=1_000, n_features=40),
    "dense_clean":  dict(n_samples=1_000, n_features=20),
}

# ---------------------------------------------------------------------------
# Datasets  (identical seeds to meta_analysis.py)
# ---------------------------------------------------------------------------

def _make_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    X1, y1 = make_friedman1(n_samples=1_000, n_features=10, noise=1.0,
                            random_state=RANDOM_STATE)
    X2, y2 = make_friedman2(n_samples=1_000, noise=0.0,
                            random_state=RANDOM_STATE)
    X3, y3 = make_regression(n_samples=1_000, n_features=40,
                             n_informative=8, noise=20.0,
                             random_state=RANDOM_STATE)
    X4, y4 = make_regression(n_samples=1_000, n_features=20,
                             n_informative=10, noise=5.0,
                             random_state=RANDOM_STATE)
    raw = {
        "friedman1":    (X1, y1),
        "friedman2":    (X2, y2),
        "sparse_noisy": (X3, y3),
        "dense_clean":  (X4, y4),
    }
    return {
        name: (X, (y - y.mean()) / (y.std() + 1e-10))
        for name, (X, y) in raw.items()
    }

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _eval_config(
    model_name:  str,
    params:      dict,
    X:           np.ndarray,
    y:           np.ndarray,
    train_idx:   np.ndarray,
    val_idx:     np.ndarray,
    dataset_name: str,
    fold:        int,
) -> dict:
    import warnings as _w
    _w.filterwarnings("ignore")

    model_type = params.pop("_model")
    t0 = time.perf_counter()
    try:
        if model_type == "geoxgb":
            from geoxgb import GeoXGBRegressor
            model = GeoXGBRegressor(**params)
        else:
            from xgboost import XGBRegressor
            model = XGBRegressor(verbosity=0, **params)

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        y_val  = y[val_idx]
        rmse   = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        mae    = float(mean_absolute_error(y_val, y_pred))
        r2     = float(r2_score(y_val, y_pred))
        conv_round  = getattr(model, "convergence_round_",  "")
        conv_reason = getattr(model, "convergence_reason_", "")
        n_hist      = len(getattr(model, "refit_history_",  []))
        status = "ok"
    except Exception as exc:
        rmse = mae = r2 = float("nan")
        conv_round = conv_reason = n_hist = ""
        status = repr(exc)[:120]

    props = DATASET_PROPERTIES.get(dataset_name, {})
    return dict(
        model              = model_name,
        dataset            = dataset_name,
        n_samples          = props.get("n_samples", X.shape[0]),
        n_features         = props.get("n_features", X.shape[1]),
        fold               = fold,
        val_rmse           = rmse,
        val_mae            = mae,
        val_r2             = r2,
        train_time_s       = time.perf_counter() - t0,
        convergence_round  = conv_round  if conv_round  is not None else "",
        convergence_reason = conv_reason if conv_reason is not None else "",
        n_refit_history    = n_hist,
        status             = status,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_jobs: int = -1) -> None:
    datasets = _make_datasets()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Build all (model, dataset, fold) jobs, skip already-saved rows
    done: set[tuple] = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["dataset"], row["fold"]))

    jobs = []
    for model_name, cfg in CONFIGS.items():
        for ds_name, (X, y) in datasets.items():
            for fold, (tr, va) in enumerate(kf.split(X)):
                key = (model_name, ds_name, str(fold))
                if key in done:
                    continue
                jobs.append((model_name, dict(cfg), X, y, tr, va, ds_name, fold))

    if not jobs:
        print("All jobs already complete -- reading saved results.")
    else:
        n_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        n_workers = min(n_workers, len(jobs))
        print(f"Running {len(jobs)} jobs on {n_workers} workers "
              f"({len(done)} already saved) ...")

        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_eval_config)(*j) for j in jobs
        )

        write_header = not os.path.exists(OUT_CSV) or len(done) == 0
        with open(OUT_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in FIELDNAMES})

        errors = [r for r in results if r["status"] != "ok"]
        if errors:
            print(f"  {len(errors)} errors:")
            for e in errors[:5]:
                print(f"    {e['model']} / {e['dataset']} / fold {e['fold']}: {e['status']}")

    print_summary()


def print_summary() -> None:
    import statistics

    with open(OUT_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    from collections import defaultdict
    data: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["model"]][r["dataset"]].append(float(r["val_r2"]))

    DS     = ["friedman1", "friedman2", "sparse_noisy", "dense_clean"]
    MODELS = list(CONFIGS.keys())

    # header
    col_w = 22
    print()
    print("=" * (20 + col_w * 4 + 8))
    print("  Comparison Summary -- mean R2 across 5 folds (std in parentheses)")
    print("=" * (20 + col_w * 4 + 8))
    print(f"  {'model':<20}" + "".join(f"{d:>{col_w}}" for d in
          ["friedman1", "friedman2", "sparse_noisy", "dense_clean"]) + f"{'avg':>8}")
    print("  " + "-" * (18 + col_w * 4 + 8))

    for m in MODELS:
        parts = []
        avgs  = []
        for ds in DS:
            scores = data[m].get(ds, [])
            if scores:
                mu  = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                parts.append(f"{mu:.4f} ({std:.4f})")
                avgs.append(mu)
            else:
                parts.append("---")
        avg_str = f"{statistics.mean(avgs):.4f}" if avgs else "---"
        print(f"  {m:<20}" + "".join(f"{p:>{col_w}}" for p in parts) + f"{avg_str:>8}")

    # delta rows vs each XGBoost config
    print()
    print("  -- Delta vs XGBoost_default --")
    for m in ["GeoXGB_old", "GeoXGB_new", "XGBoost_matched"]:
        parts = []
        avgs  = []
        for ds in DS:
            s_m   = data[m].get(ds, [])
            s_ref = data["XGBoost_default"].get(ds, [])
            if s_m and s_ref:
                delta = statistics.mean(s_m) - statistics.mean(s_ref)
                parts.append(f"{delta:>+.4f}")
                avgs.append(delta)
            else:
                parts.append("---")
        avg_str = f"{statistics.mean(avgs):>+.4f}" if avgs else "---"
        print(f"  {m:<20}" + "".join(f"{p:>{col_w}}" for p in parts) + f"{avg_str:>8}")

    print()
    print("  -- Delta vs XGBoost_matched --")
    for m in ["GeoXGB_old", "GeoXGB_new"]:
        parts = []
        avgs  = []
        for ds in DS:
            s_m   = data[m].get(ds, [])
            s_ref = data["XGBoost_matched"].get(ds, [])
            if s_m and s_ref:
                delta = statistics.mean(s_m) - statistics.mean(s_ref)
                parts.append(f"{delta:>+.4f}")
                avgs.append(delta)
            else:
                parts.append("---")
        avg_str = f"{statistics.mean(avgs):>+.4f}" if avgs else "---"
        print(f"  {m:<20}" + "".join(f"{p:>{col_w}}" for p in parts) + f"{avg_str:>8}")

    # Train time summary
    print()
    print("  -- Mean train time (seconds per fold) --")
    time_data: dict = defaultdict(lambda: defaultdict(list))
    conv_data: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        time_data[r["model"]][r["dataset"]].append(float(r["train_time_s"]))
        if r.get("convergence_round") not in ("", None):
            conv_data[r["model"]][r["dataset"]].append(int(r["convergence_round"]))
    for m in MODELS:
        times = [statistics.mean(time_data[m][ds]) for ds in DS if time_data[m].get(ds)]
        avg_t = statistics.mean(times) if times else float("nan")
        conv_rounds = [v for ds in DS for v in conv_data[m].get(ds, [])]
        conv_str = f"  early-stop @ round {statistics.mean(conv_rounds):.0f} avg" if conv_rounds else ""
        print(f"  {m:<22}  avg={avg_t:.2f}s{conv_str}")

    print(f"\n  Results saved --> {OUT_CSV}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all CPUs)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary from saved CSV without re-running")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    else:
        run(n_jobs=args.jobs)
