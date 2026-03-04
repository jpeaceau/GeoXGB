"""
Optuna HPO sweep — fixed discrete params from grid search winner:
  partitioner='pyramid_hart', method='variance_ordered', generation_strategy='simplex_mixup'

Search space (12 params):
  n_rounds, learning_rate, max_depth, refit_interval, expand_ratio,
  reduce_ratio, y_weight, min_samples_leaf, auto_expand,
  hvrt_min_samples_leaf, auto_noise, noise_guard

4 datasets run in parallel (one process each).
Each study persisted to benchmarks/results/optuna_<dataset>.db — resumable on interrupt.
Wall-clock timeout: 10 h per dataset.

Usage:
  python benchmarks/optuna_fixed_discrete.py           # run full study
  python benchmarks/optuna_fixed_discrete.py --summary # print best params from existing DBs
"""

import multiprocessing
import os
import csv
import sys
import time
import numpy as np
import optuna
from sklearn.datasets import (make_friedman1, make_friedman2,
                               make_classification, load_diabetes)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMEOUT_S = 10 * 3600   # 10 hours per dataset

# ── Fixed discrete params (grid search winner) ─────────────────────────────
FIXED = dict(
    partitioner         = "pyramid_hart",
    method              = "variance_ordered",
    generation_strategy = "simplex_mixup",
)

# ── Warm-start: best known params (previous Optuna run + grid context) ─────
WARM_START = dict(
    n_rounds              = 500,
    learning_rate         = 0.05,
    max_depth             = 5,
    refit_interval        = 50,
    expand_ratio          = 0.1,
    reduce_ratio          = 0.8,
    y_weight              = 0.3,
    min_samples_leaf      = 15,
    auto_expand           = True,
    hvrt_min_samples_leaf = 20,
    auto_noise            = True,
    noise_guard           = True,
)


# ── Datasets ───────────────────────────────────────────────────────────────
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
    from geoxgb import GeoXGBRegressor, GeoXGBClassifier
    if task == "regression":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = [
            r2_score(y[vi],
                     GeoXGBRegressor(**params).fit(X[ti], y[ti]).predict(X[vi]))
            for ti, vi in kf.split(X)
        ]
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yi = y.astype(int)
        scores = [
            roc_auc_score(yi[vi],
                          GeoXGBClassifier(**params).fit(X[ti], y[ti])
                          .predict_proba(X[vi])[:, 1])
            for ti, vi in kf.split(X, yi)
        ]
    return float(np.mean(scores))


# ── Per-dataset Optuna worker ──────────────────────────────────────────────
def run_dataset(dataset_name):
    X, y, task = get_dataset(dataset_name)
    n_splits  = 5 if dataset_name == "diabetes" else 3
    metric    = "R2" if task == "regression" else "AUC"

    db_path  = os.path.join(RESULTS_DIR, f"optuna_{dataset_name}.db")
    csv_path = os.path.join(RESULTS_DIR, f"optuna_{dataset_name}.csv")
    storage  = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name  = f"geoxgb_{dataset_name}",
        storage     = storage,
        direction   = "maximize",
        load_if_exists = True,
        sampler     = optuna.samplers.TPESampler(seed=42, n_startup_trials=25),
    )

    # Warm-start on fresh study only
    if len(study.trials) == 0:
        study.enqueue_trial(WARM_START)

    def objective(trial):
        params = dict(
            **FIXED,
            n_rounds              = trial.suggest_categorical(
                                        "n_rounds", [100, 200, 500, 1000, 2000, 3000, 5000]),
            learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth             = trial.suggest_int("max_depth", 2, 8),
            refit_interval        = trial.suggest_categorical(
                                        "refit_interval", [10, 25, 50, 100, 200]),
            expand_ratio          = trial.suggest_float("expand_ratio", 0.0, 0.5),
            reduce_ratio          = trial.suggest_float("reduce_ratio", 0.3, 0.9),
            y_weight              = trial.suggest_float("y_weight", 0.1, 0.9),
            min_samples_leaf      = trial.suggest_int("min_samples_leaf", 1, 30),
            auto_expand           = trial.suggest_categorical("auto_expand", [True, False]),
            hvrt_min_samples_leaf = trial.suggest_int("hvrt_min_samples_leaf", 5, 40),
            auto_noise            = trial.suggest_categorical("auto_noise", [True, False]),
            noise_guard           = trial.suggest_categorical("noise_guard", [True, False]),
        )
        return cv_score(X, y, task, params, n_splits)

    # ── CSV incremental logging ────────────────────────────────────────────
    header_written = [False]
    existing_trials = len(study.trials)
    if existing_trials > 0:
        header_written[0] = True   # resumed study — don't re-write header

    def progress_callback(study, trial):
        n    = trial.number
        val  = trial.value if trial.value is not None else float("nan")
        best = study.best_value
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
        print(f"  [{dataset_name}] #{n:4d}  {metric}={val:.4f}  "
              f"best={best:.4f}  {elapsed}", flush=True)
        try:
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if not header_written[0]:
                    w.writerow(["trial", metric.lower(), "best_so_far"]
                               + list(trial.params.keys()))
                    header_written[0] = True
                w.writerow([n, round(val, 6), round(best, 6)]
                           + list(trial.params.values()))
        except Exception:
            pass

    t_start = time.time()
    print(f"[{dataset_name}] Starting  n={len(X)}  task={task}  "
          f"existing_trials={existing_trials}  timeout={TIMEOUT_S//3600}h",
          flush=True)

    study.optimize(objective, timeout=TIMEOUT_S,
                   callbacks=[progress_callback], gc_after_trial=True)

    best = study.best_trial
    print(f"\n[{dataset_name}] DONE — best {metric}={best.value:.4f}  "
          f"total_trials={len(study.trials)}", flush=True)
    print(f"  params: {best.params}", flush=True)


# ── Summary helper ─────────────────────────────────────────────────────────
def print_summary(datasets):
    print("\n" + "=" * 72)
    print("BEST PARAMS PER DATASET")
    print("=" * 72)
    for ds in datasets:
        db_path = os.path.join(RESULTS_DIR, f"optuna_{ds}.db")
        if not os.path.exists(db_path):
            print(f"\n{ds.upper()}: no DB found")
            continue
        try:
            study = optuna.load_study(
                study_name=f"geoxgb_{ds}",
                storage=f"sqlite:///{db_path}",
            )
            best  = study.best_trial
            metric = "AUC" if ds == "classification" else "R2"
            n_done = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"\n{ds.upper()}  best {metric}={best.value:.4f}  "
                  f"({n_done} completed trials)")
            for k, v in best.params.items():
                print(f"  {k:25s} = {v}")
        except Exception as e:
            print(f"\n{ds.upper()}: error — {e}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = ["diabetes", "friedman1", "friedman2", "classification"]

    if "--summary" in sys.argv:
        print_summary(datasets)
        sys.exit(0)

    print(f"Launching {len(datasets)} Optuna studies in parallel")
    print(f"Fixed: {FIXED}")
    print(f"Timeout: {TIMEOUT_S // 3600} h per dataset\n")

    with multiprocessing.Pool(processes=len(datasets)) as pool:
        pool.map(run_dataset, datasets)

    print_summary(datasets)
