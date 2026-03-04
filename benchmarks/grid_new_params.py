"""
Grid search over the 4 newly-wired C++ parameters:
  partitioner × reduce_method × generation_strategy × adaptive_reduce_ratio

n_rounds fixed at 5000 to approach the performance ceiling (no overfitting in GeoXGB).
Other params held at OAT-optimal defaults from meta_reg sweep.

Runs 4 datasets in parallel (one process each); within each dataset the 36 combos
are sequential.  Results written to benchmarks/results/grid_new_params_<dataset>.csv.
"""

import itertools
import multiprocessing
import os
import csv
import time
import numpy as np
from sklearn.datasets import (make_friedman1, make_friedman2,
                               make_classification, load_diabetes)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

# ── Grid axes ──────────────────────────────────────────────────────────────────
PARTITIONERS   = ["hvrt", "pyramid_hart", "hart"]
METHODS        = ["variance_ordered", "orthant_stratified"]
STRATEGIES     = ["epanechnikov", "simplex_mixup", "laplace"]
# adaptive_reduce_ratio is C++ only — silently ignored by Python path.
# Include False only; True would give duplicate results.
ADAPTIVE_RATIO = [False]

GRID = list(itertools.product(PARTITIONERS, METHODS, STRATEGIES, ADAPTIVE_RATIO))

# ── Fixed boosting params (OAT-optimal from meta_reg sweep) ───────────────────
FIXED = dict(
    n_rounds        = 5000,
    learning_rate   = 0.1,
    max_depth       = 4,
    refit_interval  = 50,
    auto_noise      = True,
    noise_guard     = True,
    auto_expand     = True,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset definitions ────────────────────────────────────────────────────────
def get_dataset(name):
    if name == "diabetes":
        d = load_diabetes()
        return d.data, d.target, "regression"
    if name == "friedman1":
        X, y = make_friedman1(n_samples=1000, noise=1.0, random_state=42)
        return X, y, "regression"
    if name == "friedman2":
        X, y = make_friedman2(n_samples=1000, noise=0.1, random_state=42)
        return X, y, "regression"
    if name == "classification":
        X, y = make_classification(n_samples=1000, n_features=20,
                                   n_informative=10, random_state=42)
        return X, y.astype(float), "classification"
    raise ValueError(name)


def cv_score(X, y, task, params, n_splits, seed=42):
    """Run cross-validated score for given params."""
    from geoxgb import GeoXGBRegressor, GeoXGBClassifier
    if task == "regression":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in splitter.split(X):
            m = GeoXGBRegressor(**params)
            m.fit(X[train_idx], y[train_idx])
            scores.append(r2_score(y[val_idx], m.predict(X[val_idx])))
        return float(np.mean(scores)), float(np.std(scores))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        yi = y.astype(int)
        for train_idx, val_idx in splitter.split(X, yi):
            m = GeoXGBClassifier(**params)
            m.fit(X[train_idx], y[train_idx])
            prob = m.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(yi[val_idx], prob))
        return float(np.mean(scores)), float(np.std(scores))


def run_dataset(dataset_name):
    X, y, task = get_dataset(dataset_name)
    n_splits = 5 if dataset_name == "diabetes" else 3
    metric   = "R2" if task == "regression" else "AUC"

    out_path = os.path.join(RESULTS_DIR, f"grid_new_params_{dataset_name}.csv")
    fieldnames = ["partitioner", "method", "generation_strategy",
                  "adaptive_reduce_ratio", f"cv_{metric.lower()}", "cv_std", "elapsed_s"]

    print(f"[{dataset_name}] Starting {len(GRID)} combos x {n_splits}-fold CV  "
          f"(n={len(X)}, task={task})", flush=True)
    t_start = time.perf_counter()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (part, method, strat, adapt) in enumerate(GRID):
            params = dict(
                **FIXED,
                partitioner           = part,
                method                = method,
                generation_strategy   = strat,
                adaptive_reduce_ratio = adapt,
            )
            t0 = time.perf_counter()
            try:
                mean_s, std_s = cv_score(X, y, task, params, n_splits)
                elapsed = time.perf_counter() - t0
                row = dict(partitioner=part, method=method,
                           generation_strategy=strat,
                           adaptive_reduce_ratio=adapt,
                           **{f"cv_{metric.lower()}": round(mean_s, 6),
                              "cv_std": round(std_s, 6),
                              "elapsed_s": round(elapsed, 2)})
                writer.writerow(row)
                f.flush()
                pct = (i + 1) / len(GRID) * 100
                elapsed_total = time.perf_counter() - t_start
                print(f"  [{dataset_name}] {pct:5.1f}% ({i+1:2d}/{len(GRID)})  "
                      f"{part:12s} {method:20s} {strat:13s} adapt={str(adapt):5s}  "
                      f"{metric}={mean_s:.4f}±{std_s:.4f}  {elapsed:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  [{dataset_name}] ERROR combo {i}: {e}", flush=True)
                writer.writerow(dict(partitioner=part, method=method,
                                     generation_strategy=strat,
                                     adaptive_reduce_ratio=adapt,
                                     **{f"cv_{metric.lower()}": float("nan"),
                                        "cv_std": float("nan"),
                                        "elapsed_s": -1}))

    total = time.perf_counter() - t_start
    print(f"[{dataset_name}] Done in {total/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    datasets = ["diabetes", "friedman1", "friedman2", "classification"]
    print(f"Launching {len(datasets)} datasets in parallel, "
          f"{len(GRID)} combos each, n_rounds=5000")

    with multiprocessing.Pool(processes=len(datasets)) as pool:
        pool.map(run_dataset, datasets)

    # ── Summary: print top-5 per dataset ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 5 PER DATASET")
    print("=" * 70)
    for ds in datasets:
        path = os.path.join(RESULTS_DIR, f"grid_new_params_{ds}.csv")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    key = "cv_r2" if "r2" in row else "cv_auc"
                    rows.append((float(row[key]), row))
                except (ValueError, KeyError):
                    pass
        rows.sort(key=lambda x: x[0], reverse=True)
        metric = "R2" if "friedman" in ds or ds == "diabetes" else "AUC"
        print(f"\n{ds.upper()} (top 5 by CV {metric}):")
        print(f"  {'Partitioner':<14} {'Method':<22} {'Strategy':<15} {'Adapt':>6}  {metric}")
        print(f"  {'-'*72}")
        for score, row in rows[:5]:
            k = "cv_r2" if "r2" in row else "cv_auc"
            print(f"  {row['partitioner']:<14} {row['method']:<22} "
                  f"{row['generation_strategy']:<15} {row['adaptive_reduce_ratio']:>6}  "
                  f"{float(row[k]):.4f}±{float(row['cv_std']):.4f}")
