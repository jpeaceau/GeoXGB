"""
GeoXGB -- Multiclass Parallel Training Benchmark
=================================================

Measures the wall-time speedup of n_jobs > 1 for multiclass GeoXGBClassifier.

The boosting loop within each class ensemble is inherently sequential.
Parallelism is across the K one-vs-rest class ensembles, which are
fully independent of one another.

Sweeps:
  n_classes : 3, 5, 8
  n_jobs    : 1 (sequential), 2, 4, -1 (all CPUs)

Fixed settings: n_rounds=500, learning_rate=0.2, refit_interval=20,
                max_depth=4, n_samples=1000, n_features=10

Each (n_classes, n_jobs) combination is run N_REPS times and the
median wall time is reported to reduce noise from process-spawn jitter.

Usage
-----
    python benchmarks/multiclass_parallel_benchmark.py
"""

from __future__ import annotations

import multiprocessing
import time
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from geoxgb import GeoXGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ROUNDS       = 500
LEARNING_RATE  = 0.2
REFIT_INTERVAL = 20
MAX_DEPTH      = 4
N_SAMPLES      = 1_000
N_FEATURES     = 10
N_REPS         = 3
RANDOM_STATE   = 42

N_CLASSES_LIST = [3, 5, 8]
N_JOBS_LIST    = [1, 2, 4, -1]

GEO_FIXED = dict(
    n_rounds       = N_ROUNDS,
    learning_rate  = LEARNING_RATE,
    refit_interval = REFIT_INTERVAL,
    max_depth      = MAX_DEPTH,
    min_samples_leaf = 5,
    reduce_ratio   = 0.7,
    auto_noise     = True,
    cache_geometry = False,
    random_state   = RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


def _make_dataset(n_classes):
    X, y = make_classification(
        n_samples        = N_SAMPLES,
        n_features       = N_FEATURES,
        n_informative    = max(n_classes, 5),
        n_redundant      = 0,
        n_repeated       = 0,
        n_classes        = n_classes,
        n_clusters_per_class = 1,
        random_state     = RANDOM_STATE,
    )
    return X, y


def _one_vs_rest_auc(y_true, proba):
    lb = LabelBinarizer().fit(y_true)
    y_bin = lb.transform(y_true)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    aucs = []
    for k in range(proba.shape[1]):
        try:
            aucs.append(roc_auc_score(y_bin[:, k], proba[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_count = multiprocessing.cpu_count()

    _section("GeoXGB -- Multiclass Parallel Training Benchmark")
    print(
        f"\n  n_rounds       : {N_ROUNDS}"
        f"\n  learning_rate  : {LEARNING_RATE}"
        f"\n  refit_interval : {REFIT_INTERVAL}"
        f"\n  max_depth      : {MAX_DEPTH}"
        f"\n  n_samples      : {N_SAMPLES}"
        f"\n  n_classes      : {N_CLASSES_LIST}"
        f"\n  n_jobs tested  : {N_JOBS_LIST}"
        f"\n  reps per cell  : {N_REPS}  (median reported)"
        f"\n  CPUs available : {cpu_count}"
        f"\n"
        f"\n  Note: parallelism is across K class ensembles only."
        f"\n        The boosting loop within each class is sequential."
    )

    # Warm up the process pool once to amortise spawn overhead
    print("\n  Warming up process pool...")
    _warmup_X, _warmup_y = _make_dataset(3)
    GeoXGBClassifier(n_rounds=5, n_jobs=-1, **{
        k: v for k, v in GEO_FIXED.items() if k != "n_rounds"
    }).fit(_warmup_X, _warmup_y)
    print("  Done.\n")

    results: dict[tuple, dict] = {}

    for n_classes in N_CLASSES_LIST:
        _subsection(f"n_classes = {n_classes}")
        X, y = _make_dataset(n_classes)

        # Split into train/test once
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.permutation(len(X))
        split = int(0.8 * len(X))
        tr, te = idx[:split], idx[split:]

        print(
            f"\n  {'n_jobs':>7s}  {'median_t':>9s}  {'speedup':>8s}  "
            f"{'mean_auc':>9s}  note"
        )
        print(f"  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*20}")

        base_time = None

        for n_jobs in N_JOBS_LIST:
            times = []
            aucs  = []
            for rep in range(N_REPS):
                m = GeoXGBClassifier(n_jobs=n_jobs, **GEO_FIXED)
                t0 = time.perf_counter()
                m.fit(X[tr], y[tr])
                elapsed = time.perf_counter() - t0
                proba   = m.predict_proba(X[te])
                auc     = _one_vs_rest_auc(y[te], proba)
                times.append(elapsed)
                aucs.append(auc)

            med_t = float(np.median(times))
            mean_auc = float(np.mean(aucs))

            if n_jobs == 1:
                base_time = med_t
                speedup_s = "  1.00×"
                note = "[baseline]"
            else:
                ratio = base_time / med_t if med_t > 0 else float("nan")
                speedup_s = f"{ratio:>7.2f}×"
                effective_jobs = cpu_count if n_jobs == -1 else n_jobs
                ideal = min(effective_jobs, n_classes)
                note = f"ideal={ideal}×"

            results[(n_classes, n_jobs)] = {
                "median_t": med_t,
                "mean_auc": mean_auc,
                "speedup":  base_time / med_t if base_time else 1.0,
            }

            jobs_label = "all" if n_jobs == -1 else str(n_jobs)
            print(
                f"  {jobs_label:>7s}  {med_t:>8.1f}s  {speedup_s}  "
                f"{mean_auc:>9.5f}  {note}"
            )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    _section("Speedup Summary  (n_jobs=-1 vs n_jobs=1)")
    print(
        f"\n  {'n_classes':>9s}  {'sequential':>11s}  {'parallel':>9s}  "
        f"{'speedup':>8s}  {'auc_delta':>10s}"
    )
    print(f"  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*8}  {'-'*10}")

    for n_classes in N_CLASSES_LIST:
        seq = results[(n_classes, 1)]
        par = results[(n_classes, -1)]
        speedup   = seq["median_t"] / par["median_t"]
        auc_delta = par["mean_auc"] - seq["mean_auc"]
        print(
            f"  {n_classes:>9d}  {seq['median_t']:>10.1f}s  "
            f"{par['median_t']:>8.1f}s  {speedup:>7.2f}×  {auc_delta:>+10.5f}"
        )

    _section("Observations")
    print(
        f"\n  CPUs available : {cpu_count}"
        f"\n"
        f"\n  Ideal speedup ceiling = min(n_jobs, n_classes)."
        f"\n  Overhead sources:"
        f"\n    - Process spawn / data serialisation (one-time per fit call)"
        f"\n    - X must be pickled and sent to each worker"
        f"\n    - Memory: K copies of the tree ensemble are held simultaneously"
        f"\n"
        f"\n  AUC delta should be ~0: parallel training is mathematically"
        f"\n  identical to sequential (same random seeds per class)."
        f"\n"
    )


if __name__ == "__main__":
    main()
