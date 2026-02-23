"""
GeoXGB -- Regressor vs Classifier on Binary AUC  (Multiprocessing)
===================================================================

Compares GeoXGBRegressor (MSE on {0,1} targets) against GeoXGBClassifier
(log-loss) on binary classification measured by AUC.

For AUC, only ranking matters -- calibration is irrelevant. The hypothesis
is that log-loss gradients compress toward zero as the model gets confident,
degrading the signal HVRT sees at each refit. MSE gradients stay larger.

Diagnostics:
  - AUC at round checkpoints {100, 300, 500, 1000}
  - Mean noise_modulation across refits (proxy for gradient signal quality)
  - Delta = reg_auc - clf_auc  (positive means regressor wins)

Usage
-----
    python benchmarks/regressor_vs_classifier_binary.py
"""

from __future__ import annotations

import multiprocessing
import warnings
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE      = 42
LEARNING_RATE     = 0.2
REFIT_INTERVAL    = 20
MAX_DEPTH         = 4
N_FOLDS           = 5
ROUND_CHECKPOINTS = [100, 300, 500, 1000]

GEO_FIXED = dict(
    learning_rate    = LEARNING_RATE,
    refit_interval   = REFIT_INTERVAL,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = 5,
    reduce_ratio     = 0.7,
    auto_noise       = True,
    cache_geometry   = False,
    random_state     = RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _make_datasets():
    X1, y1 = make_classification(
        n_samples=1_000, n_features=10, n_informative=5,
        n_redundant=0, n_clusters_per_class=2,
        class_sep=1.0, random_state=RANDOM_STATE,
    )
    X2, y2 = make_classification(
        n_samples=1_000, n_features=20, n_informative=5,
        n_redundant=5, n_clusters_per_class=1,
        class_sep=0.5, flip_y=0.10, random_state=RANDOM_STATE,
    )
    X3, y3r = make_friedman1(
        n_samples=1_000, n_features=10, noise=1.0, random_state=RANDOM_STATE
    )
    y3 = (y3r > np.median(y3r)).astype(int)

    X4, y4 = make_classification(
        n_samples=1_000, n_features=15, n_informative=8,
        n_redundant=2, n_clusters_per_class=2,
        class_sep=1.5, random_state=RANDOM_STATE + 1,
    )
    return {
        "clean_binary":   (X1, y1, "clean, sep=1.0"),
        "noisy_binary":   (X2, y2, "flip_y=0.10, sep=0.5"),
        "friedman_binar": (X3, y3, "friedman1 binarised at median"),
        "easy_binary":    (X4, y4, "clean, sep=1.5, 8 informative"),
    }


# ---------------------------------------------------------------------------
# Worker -- module-level for Windows pickling
# ---------------------------------------------------------------------------

def _eval_worker(
    model_type: str,   # "reg" or "clf"
    n_rounds: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple:
    import warnings as _w
    _w.filterwarnings("ignore")
    from geoxgb import GeoXGBClassifier, GeoXGBRegressor
    from sklearn.metrics import roc_auc_score
    import numpy as np

    params = dict(**GEO_FIXED, n_rounds=n_rounds)

    if model_type == "reg":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx].astype(float))
        scores = m.predict(X[val_idx])
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        scores = m.predict_proba(X[val_idx])[:, 1]

    auc        = float(roc_auc_score(y[val_idx], scores))
    mean_noise = float(np.mean([e["noise_modulation"]
                                for e in m._resample_history]))
    return model_type, n_rounds, auc, mean_noise


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_count = multiprocessing.cpu_count()
    datasets  = _make_datasets()

    n_jobs_total = (
        len(datasets) * len(ROUND_CHECKPOINTS) * 2 * N_FOLDS
    )

    _section("GeoXGB -- Regressor vs Classifier on Binary AUC")
    print(
        f"\n  Regressor : MSE loss on raw {{0,1}} targets"
        f"\n  Classifier: log-loss (binary)"
        f"\n  Metric    : AUC  (ranking only, calibration irrelevant)"
        f"\n  Delta     : reg_auc - clf_auc  (positive = regressor wins)"
        f"\n  Checkpoints: {ROUND_CHECKPOINTS}"
        f"\n  CV folds  : {N_FOLDS}"
        f"\n  Total jobs: {n_jobs_total}"
        f"\n  CPUs      : {cpu_count}"
    )

    # Build CV splits once per dataset
    splits = {}
    for ds_name, (X, y, _) in datasets.items():
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        splits[ds_name] = list(skf.split(X, y))

    # Build job list
    jobs      = []
    job_keys  = []
    for ds_name, (X, y, _) in datasets.items():
        for n_rounds in ROUND_CHECKPOINTS:
            for model_type in ("reg", "clf"):
                for fi, (tr, te) in enumerate(splits[ds_name]):
                    jobs.append((model_type, n_rounds, X, y, tr, te))
                    job_keys.append((ds_name, model_type, n_rounds, fi))

    print(f"\n  Launching {n_jobs_total} jobs across {cpu_count} workers...")
    import time
    t0  = time.perf_counter()
    raw = Parallel(n_jobs=cpu_count, prefer="processes", verbose=2)(
        delayed(_eval_worker)(*job) for job in jobs
    )
    wall = time.perf_counter() - t0
    print(f"\n  Done. Wall time: {wall:.1f}s")

    # Aggregate
    scores  = defaultdict(list)   # (ds, model, n_rounds) -> [auc]
    noises  = defaultdict(list)   # (ds, model, n_rounds) -> [noise]

    for (ds_name, model_type, n_rounds, _fi), (_, _, auc, noise) in zip(job_keys, raw):
        scores[(ds_name, model_type, n_rounds)].append(auc)
        noises[(ds_name, model_type, n_rounds)].append(noise)

    # -----------------------------------------------------------------------
    # Per-dataset round-curve tables
    # -----------------------------------------------------------------------
    _section("AUC vs Round Count  (5-fold CV mean)")

    for ds_name, (X, y, note) in datasets.items():
        _subsection(f"{ds_name}  ({note})")
        print(
            f"\n  {'rounds':>7s}  {'reg AUC':>9s}  {'clf AUC':>9s}  "
            f"{'delta':>8s}  {'reg noise':>9s}  {'clf noise':>9s}  winner"
        )
        print(
            f"  {'-'*7}  {'-'*9}  {'-'*9}  "
            f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}"
        )
        for n_rounds in ROUND_CHECKPOINTS:
            reg_auc  = float(np.mean(scores[(ds_name, "reg", n_rounds)]))
            clf_auc  = float(np.mean(scores[(ds_name, "clf", n_rounds)]))
            delta    = reg_auc - clf_auc
            reg_n    = float(np.mean(noises[(ds_name, "reg", n_rounds)]))
            clf_n    = float(np.mean(noises[(ds_name, "clf", n_rounds)]))
            winner   = "reg  <--" if delta >  0.0005 else (
                       "clf  <--" if delta < -0.0005 else "tied")
            print(
                f"  {n_rounds:>7d}  {reg_auc:>9.5f}  {clf_auc:>9.5f}  "
                f"  {delta:>+7.5f}  {reg_n:>9.4f}  {clf_n:>9.4f}  {winner}"
            )

    # -----------------------------------------------------------------------
    # Summary at n_rounds=1000
    # -----------------------------------------------------------------------
    _section("Summary at n_rounds=1000")
    print(
        f"\n  {'dataset':<18s}  {'reg':>9s}  {'clf':>9s}  "
        f"{'delta':>8s}  {'noise reg':>9s}  {'noise clf':>9s}  winner"
    )
    print(
        f"  {'-'*18}  {'-'*9}  {'-'*9}  "
        f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}"
    )
    reg_wins = clf_wins = ties = 0
    for ds_name in datasets:
        reg_auc = float(np.mean(scores[(ds_name, "reg", 1000)]))
        clf_auc = float(np.mean(scores[(ds_name, "clf", 1000)]))
        delta   = reg_auc - clf_auc
        reg_n   = float(np.mean(noises[(ds_name, "reg", 1000)]))
        clf_n   = float(np.mean(noises[(ds_name, "clf", 1000)]))
        winner  = "reg" if delta > 0.0005 else ("clf" if delta < -0.0005 else "tied")
        if winner == "reg":   reg_wins += 1
        elif winner == "clf": clf_wins += 1
        else:                 ties += 1
        print(
            f"  {ds_name:<18s}  {reg_auc:>9.5f}  {clf_auc:>9.5f}  "
            f"  {delta:>+7.5f}  {reg_n:>9.4f}  {clf_n:>9.4f}  {winner}"
        )
    print(
        f"\n  Regressor wins: {reg_wins}  |  "
        f"Classifier wins: {clf_wins}  |  Tied: {ties}"
    )

    # -----------------------------------------------------------------------
    # Noise trajectory: does clf noise drop more than reg across rounds?
    # -----------------------------------------------------------------------
    _section("Noise Modulation Trajectory  (mean across datasets)")
    print(
        f"\n  {'rounds':>7s}  {'reg noise':>10s}  {'clf noise':>10s}  "
        f"{'clf - reg':>10s}  note"
    )
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*30}")
    for n_rounds in ROUND_CHECKPOINTS:
        reg_ns = [float(np.mean(noises[(ds, "reg", n_rounds)])) for ds in datasets]
        clf_ns = [float(np.mean(noises[(ds, "clf", n_rounds)])) for ds in datasets]
        rn = float(np.mean(reg_ns))
        cn = float(np.mean(clf_ns))
        diff = cn - rn
        note = ("clf sees weaker signal" if diff < -0.05 else
                ("clf sees stronger signal" if diff > 0.05 else "comparable"))
        print(f"  {n_rounds:>7d}  {rn:>10.4f}  {cn:>10.4f}  {diff:>+10.4f}  {note}")

    print(
        f"\n  Interpretation:"
        f"\n    noise_modulation near 1.0 = HVRT sees clean gradient signal"
        f"\n    noise_modulation near 0.0 = HVRT sees compressed / noisy gradients"
        f"\n    If clf_noise drops faster across rounds, log-loss gradient"
        f"\n    compression is the likely cause of the performance gap."
        f"\n"
    )


if __name__ == "__main__":
    main()
