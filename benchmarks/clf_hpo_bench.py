"""
1-hour classification HPO across 3 diverse sklearn datasets.

Fixed discrete (from grid search):
  partitioner='pyramid_hart', method='variance_ordered',
  generation_strategy='simplex_mixup'

Datasets (run in parallel, one process each):
  breast_cancer  n=569,   30 features, binary
  wine           n=178,   13 features, 3-class
  digits         n=1797,  64 features, 10-class

Results:
  benchmarks/results/clf_hpo_<dataset>.db   (Optuna SQLite, resumable)
  benchmarks/results/clf_hpo_<dataset>.csv  (incremental trial log)
  benchmarks/results/clf_hpo.log            (stdout)

Usage:
  python benchmarks/clf_hpo_bench.py            # run study
  python benchmarks/clf_hpo_bench.py --summary  # print best params
"""

import multiprocessing
import os
import csv
import sys
import time
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMEOUT_S = 3600  # 1 hour per dataset

FIXED = dict(
    partitioner         = "pyramid_hart",
    method              = "variance_ordered",
    generation_strategy = "simplex_mixup",
)

# Warm-start: best known classification params from previous Optuna run
WARM_START = dict(
    n_rounds              = 3000,
    learning_rate         = 0.05,
    max_depth             = 4,
    refit_interval        = 50,
    expand_ratio          = 0.1,
    reduce_ratio          = 0.8,
    y_weight              = 0.5,
    min_samples_leaf      = 5,
    auto_expand           = True,
    hvrt_min_samples_leaf = 15,
    auto_noise            = True,
    noise_guard           = False,
)


def get_dataset(name):
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data, d.target.astype(float), 2
    if name == "wine":
        d = load_wine()
        return d.data, d.target.astype(float), 3
    if name == "digits":
        d = load_digits()
        return d.data, d.target.astype(float), 10
    raise ValueError(name)


def cv_auc(X, y, n_classes, params, n_splits=5, seed=42):
    from geoxgb import GeoXGBClassifier
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    scores = []
    for ti, vi in kf.split(X, yi):
        m = GeoXGBClassifier(**params)
        m.fit(X[ti], y[ti])
        if n_classes == 2:
            prob = m.predict_proba(X[vi])[:, 1]
            scores.append(roc_auc_score(yi[vi], prob))
        else:
            prob = m.predict_proba(X[vi])
            y_bin = label_binarize(yi[vi], classes=list(range(n_classes)))
            scores.append(roc_auc_score(y_bin, prob, multi_class="ovr", average="macro"))
    return float(np.mean(scores))


def run_dataset(dataset_name):
    X, y, n_classes = get_dataset(dataset_name)
    n_splits = 5 if len(X) < 400 else 3

    db_path  = os.path.join(RESULTS_DIR, f"clf_hpo_{dataset_name}.db")
    csv_path = os.path.join(RESULTS_DIR, f"clf_hpo_{dataset_name}.csv")
    storage  = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name     = f"clf_{dataset_name}",
        storage        = storage,
        direction      = "maximize",
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(WARM_START)

    def objective(trial):
        params = dict(
            **FIXED,
            n_rounds              = trial.suggest_categorical(
                                        "n_rounds", [200, 500, 1000, 2000, 3000, 5000]),
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
        return cv_auc(X, y, n_classes, params, n_splits)

    header_written = [len(study.trials) > 0]

    def progress_callback(study, trial):
        val  = trial.value if trial.value is not None else float("nan")
        best = study.best_value
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
        print(f"  [{dataset_name}] #{trial.number:4d}  AUC={val:.4f}  "
              f"best={best:.4f}  {elapsed}", flush=True)
        try:
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if not header_written[0]:
                    w.writerow(["trial", "auc", "best_so_far"] + list(trial.params.keys()))
                    header_written[0] = True
                w.writerow([trial.number, round(val, 6), round(best, 6)]
                           + list(trial.params.values()))
        except Exception:
            pass

    t_start = time.time()
    print(f"[{dataset_name}] Starting  n={len(X)}  classes={n_classes}  "
          f"splits={n_splits}  timeout={TIMEOUT_S}s", flush=True)

    study.optimize(objective, timeout=TIMEOUT_S,
                   callbacks=[progress_callback], gc_after_trial=True)

    best = study.best_trial
    print(f"\n[{dataset_name}] DONE — best AUC={best.value:.4f}  "
          f"total_trials={len(study.trials)}", flush=True)
    print(f"  params: {best.params}", flush=True)


def print_summary():
    datasets = ["breast_cancer", "wine", "digits"]
    print("\n" + "=" * 72)
    print("BEST CLASSIFICATION PARAMS")
    print("=" * 72)
    for ds in datasets:
        db_path = os.path.join(RESULTS_DIR, f"clf_hpo_{ds}.db")
        if not os.path.exists(db_path):
            print(f"\n{ds.upper()}: no DB found")
            continue
        try:
            study = optuna.load_study(
                study_name=f"clf_{ds}",
                storage=f"sqlite:///{db_path}",
            )
            best  = study.best_trial
            n_done = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"\n{ds.upper()}  best AUC={best.value:.4f}  "
                  f"({n_done} completed trials)")
            for k, v in best.params.items():
                print(f"  {k:25s} = {v}")
        except Exception as e:
            print(f"\n{ds.upper()}: error — {e}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        print_summary()
        sys.exit(0)

    datasets = ["breast_cancer", "wine", "digits"]
    print(f"Launching {len(datasets)} classification HPO studies in parallel")
    print(f"Fixed: {FIXED}")
    print(f"Timeout: {TIMEOUT_S}s ({TIMEOUT_S//60} min) per dataset\n")

    with multiprocessing.Pool(processes=len(datasets)) as pool:
        pool.map(run_dataset, datasets)

    print_summary()
