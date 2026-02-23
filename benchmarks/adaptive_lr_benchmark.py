"""
GeoXGB -- Adaptive Learning Rate Benchmark (Multiprocessing)
============================================================

Compares 9 learning rate schedules at n_rounds=1000, learning_rate=0.2 (base)
across three datasets using all available CPU cores.

Schedules
---------
  constant        : 0.2 everywhere (baseline)
  linear_decay    : 0.2 -> 0.02 linearly over 1000 rounds
  exp_decay       : 0.2 * exp(-3 * i/n)  ->  ~0.01 at round 1000
  cosine          : 0.2 * 0.5 * (1 + cos(pi * i/n))  ->  0.0 at round 1000
  cosine_restarts : cosine with 4 equal-length warm restarts (250 rounds each)
  warmup_cosine   : linear warmup for 50 rounds then cosine decay to 0
  step_decay      : halve every 333 rounds  (0.2 -> 0.1 -> 0.05 -> 0.025)
  cyclical        : triangular wave between 0.02 and 0.2, 200-round cycle
  sqrt_decay      : 0.2 / sqrt(1 + i)  (theoretically optimal for convex SGD)

All schedules share the same base lr=0.2 and n_rounds=1000.

Datasets
--------
  friedman1      : Friedman #1 regression  (R^2)
  friedman2      : Friedman #2 regression  (R^2)
  classification : Synthetic binary classification  (AUC)

CV: 5-fold

Usage
-----
    python benchmarks/adaptive_lr_benchmark.py

Requirements: geoxgb, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import multiprocessing
import time
import warnings
from math import cos, exp, pi, sqrt

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

BASE_LR      = 0.2
N_ROUNDS     = 1000
N_FOLDS      = 5
RANDOM_STATE = 42

GEO_FIXED = dict(
    n_rounds=N_ROUNDS,
    learning_rate=BASE_LR,
    max_depth=4,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    cache_geometry=False,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Schedule definitions  (all module-level for pickling on Windows)
# Each callable: (round_idx: int, n_rounds: int, base_lr: float) -> float
# ---------------------------------------------------------------------------

def _sched_constant(i, n, lr):
    return lr


def _sched_linear_decay(i, n, lr):
    # lr -> 0.1 * lr (10x decay) by the final round
    return lr * (1.0 - 0.9 * i / max(n - 1, 1))


def _sched_exp_decay(i, n, lr):
    # lr -> ~0.05 * lr (decay constant chosen so half-life ~ n/4)
    return lr * exp(-3.0 * i / max(n - 1, 1))


def _sched_cosine(i, n, lr):
    return lr * 0.5 * (1.0 + cos(pi * i / max(n - 1, 1)))


def _sched_cosine_restarts(i, n, lr, n_cycles=4):
    cycle_len = n / n_cycles
    pos = (i % cycle_len) / cycle_len
    return lr * 0.5 * (1.0 + cos(pi * pos))


def _sched_warmup_cosine(i, n, lr, warmup=50):
    if i < warmup:
        return lr * (i + 1) / warmup
    progress = (i - warmup) / max(n - warmup - 1, 1)
    return lr * 0.5 * (1.0 + cos(pi * progress))


def _sched_step_decay(i, n, lr, gamma=0.5, steps=3):
    step_size = n // steps
    return lr * (gamma ** (i // step_size))


def _sched_cyclical(i, n, lr, cycle=200, min_factor=0.1):
    pos = i % cycle
    half = cycle // 2
    frac = pos / half if pos < half else (cycle - pos) / half
    return lr * min_factor + (lr - lr * min_factor) * frac


def _sched_sqrt_decay(i, n, lr):
    return lr / sqrt(1.0 + i)


SCHEDULES: dict[str, object] = {
    "constant":         _sched_constant,
    "linear_decay":     _sched_linear_decay,
    "exp_decay":        _sched_exp_decay,
    "cosine":           _sched_cosine,
    "cosine_restarts":  _sched_cosine_restarts,
    "warmup_cosine":    _sched_warmup_cosine,
    "step_decay":       _sched_step_decay,
    "cyclical":         _sched_cyclical,
    "sqrt_decay":       _sched_sqrt_decay,
}

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
    sched_name: str,
    sched_fn: object,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[str, float, float]:
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(**GEO_FIXED, lr_schedule=sched_fn)
    t0 = time.perf_counter()

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(r2_score(y[val_idx], m.predict(X[val_idx])))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(roc_auc_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))

    return sched_name, score, time.perf_counter() - t0

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


def _preview_schedules() -> None:
    """Print a table showing lr at key round indices for each schedule."""
    checkpoints = [0, 50, 100, 200, 333, 500, 667, 800, 999]
    header = f"  {'schedule':<20s}" + "".join(f"  r={r:<4d}" for r in checkpoints)
    print(header)
    print(f"  {'-'*20}" + "".join(f"  {'-'*6}" for _ in checkpoints))
    for name, fn in SCHEDULES.items():
        vals = [fn(r, N_ROUNDS, BASE_LR) for r in checkpoints]
        row = f"  {name:<20s}" + "".join(f"  {v:>6.4f}" for v in vals)
        # Mark effective area under curve (sum, proxy for total step magnitude)
        auc = sum(fn(i, N_ROUNDS, BASE_LR) for i in range(N_ROUNDS))
        row += f"   sum={auc:>7.1f}"
        print(row)
    print(f"\n  (constant sum = {BASE_LR * N_ROUNDS:.1f}  -- reference)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    datasets  = _make_datasets()
    n_jobs    = len(SCHEDULES) * len(datasets) * N_FOLDS

    _section("GeoXGB -- Adaptive Learning Rate Benchmark")
    print(
        f"\n  n_rounds     : {N_ROUNDS}"
        f"\n  base lr      : {BASE_LR}"
        f"\n  Schedules    : {len(SCHEDULES)}"
        f"\n  Datasets     : {', '.join(datasets.keys())}"
        f"\n  CV folds     : {N_FOLDS}"
        f"\n  Total jobs   : {n_jobs}"
        f"\n  CPUs         : {cpu_count}"
    )

    _section("Schedule Preview  (effective lr at selected rounds)")
    _preview_schedules()

    # -----------------------------------------------------------------------
    # Build CV splits
    # -----------------------------------------------------------------------
    splits: dict[str, list] = {}
    for ds_name, (X, y, task, _m) in datasets.items():
        if task == "classification":
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
        else:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits[ds_name] = list(kf.split(X, y))

    # -----------------------------------------------------------------------
    # Build job list
    # -----------------------------------------------------------------------
    jobs: list[tuple] = []
    job_keys: list[tuple[str, str, int]] = []

    for ds_name, (X, y, task, _m) in datasets.items():
        for sched_name, sched_fn in SCHEDULES.items():
            for fi, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((sched_name, sched_fn, task, X, y, tr, val))
                job_keys.append((ds_name, sched_name, fi))

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

    for (ds_name, sched_name, _fi), (_, score, elapsed) in zip(job_keys, raw):
        scores[(ds_name, sched_name)].append(score)
        times[ (ds_name, sched_name)].append(elapsed)

    # -----------------------------------------------------------------------
    # Per-dataset tables
    # -----------------------------------------------------------------------
    _section("Per-Dataset Results")

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name})")

        rows = []
        for sched_name in SCHEDULES:
            key = (ds_name, sched_name)
            sc  = scores[key]
            rows.append({
                "name":  sched_name,
                "mean":  float(np.mean(sc)),
                "std":   float(np.std(sc)),
                "folds": sc,
                "time":  float(np.mean(times[key])),
            })
        rows.sort(key=lambda r: -r["mean"])
        best_mean = rows[0]["mean"]
        base_mean = next(r["mean"] for r in rows if r["name"] == "constant")

        print(
            f"\n  {'schedule':<20s}  {'mean':>8s}  {'std':>6s}"
            f"  {'vs constant':>11s}  {'avg time':>8s}  score"
        )
        print(
            f"  {'-'*20}  {'-'*8}  {'-'*6}"
            f"  {'-'*11}  {'-'*8}  {'-'*28}"
        )
        for r in rows:
            delta    = r["mean"] - base_mean
            delta_s  = f"{delta:>+.5f}"
            bar      = _bar(r["mean"], best_mean)
            best_tag = " [best]" if r is rows[0] else "       "
            base_tag = " [base]" if r["name"] == "constant" else "       "
            tag      = best_tag if r is rows[0] else base_tag
            print(
                f"  {r['name']:<20s}  {r['mean']:>8.5f}  {r['std']:>6.4f}"
                f"  {delta_s:>11s}  {r['time']:>7.1f}s  {bar}{tag}"
            )

    # -----------------------------------------------------------------------
    # Cross-dataset z-score ranking
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Z-Score Ranking")
    print(
        "\n  Z-scores remove scale differences between R^2 and AUC.\n"
        "  Each schedule's score is normalised within each dataset then averaged."
    )

    z_by_sched: dict[str, list[float]] = {s: [] for s in SCHEDULES}
    for ds_name in datasets:
        ds_scores = {s: float(np.mean(scores[(ds_name, s)])) for s in SCHEDULES}
        mu  = float(np.mean(list(ds_scores.values())))
        sig = float(np.std(list(ds_scores.values()))) or 1e-9
        for s, v in ds_scores.items():
            z_by_sched[s].append((v - mu) / sig)

    ranked = sorted(SCHEDULES.keys(), key=lambda s: -float(np.mean(z_by_sched[s])))
    base_z = float(np.mean(z_by_sched["constant"]))

    print(f"\n  {'rank':>4s}  {'schedule':<20s}  {'mean z':>8s}  {'vs constant':>12s}  bar")
    print(f"  {'-'*4}  {'-'*20}  {'-'*8}  {'-'*12}  {'-'*28}")
    max_z = float(np.mean(z_by_sched[ranked[0]]))
    for rank, sched_name in enumerate(ranked, 1):
        z    = float(np.mean(z_by_sched[sched_name]))
        dz   = z - base_z
        bar  = _bar(max(z, 0), max(max_z, 1e-9))
        flag = " [best]" if rank == 1 else (" [base]" if sched_name == "constant" else "")
        print(
            f"  {rank:>4d}  {sched_name:<20s}  {z:>8.4f}"
            f"  {dz:>+12.4f}  {bar}{flag}"
        )

    # -----------------------------------------------------------------------
    # Schedule character analysis
    # -----------------------------------------------------------------------
    _section("Schedule Character Analysis")
    print(
        "\n  Effective lr statistics across 1000 rounds:\n"
        "  mean_lr = average lr applied per round\n"
        "  total   = sum of all lr values (proportional to total update magnitude)\n"
        "  early   = mean lr in rounds 0-249  (aggressive early learning)\n"
        "  late    = mean lr in rounds 750-999 (fine-tuning phase)"
    )
    print(
        f"\n  {'schedule':<20s}  {'mean_lr':>7s}  {'total':>7s}"
        f"  {'early':>7s}  {'late':>7s}  {'early/late':>10s}"
    )
    print(
        f"  {'-'*20}  {'-'*7}  {'-'*7}"
        f"  {'-'*7}  {'-'*7}  {'-'*10}"
    )
    for sched_name in ranked:
        fn     = SCHEDULES[sched_name]
        lrs    = [fn(i, N_ROUNDS, BASE_LR) for i in range(N_ROUNDS)]
        mean_  = float(np.mean(lrs))
        total_ = float(np.sum(lrs))
        early_ = float(np.mean(lrs[:250]))
        late_  = float(np.mean(lrs[750:]))
        ratio  = early_ / max(late_, 1e-9)
        print(
            f"  {sched_name:<20s}  {mean_:>7.4f}  {total_:>7.1f}"
            f"  {early_:>7.4f}  {late_:>7.4f}  {ratio:>10.1f}x"
        )

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")
    best_sched = ranked[0]
    best_z_val = float(np.mean(z_by_sched[best_sched]))
    const_z    = float(np.mean(z_by_sched["constant"]))

    print(f"\n  Best schedule overall : {best_sched}  (mean z={best_z_val:.4f})")
    print(f"  Constant baseline     : z={const_z:.4f}")
    print(f"  Improvement over constant : {best_z_val - const_z:+.4f} z-score units")
    print()

    for ds_name, (X, y, task, metric_name) in datasets.items():
        ds_scores = {s: float(np.mean(scores[(ds_name, s)])) for s in SCHEDULES}
        ds_best   = max(ds_scores, key=ds_scores.__getitem__)
        print(
            f"  {ds_name:<18s}  best={ds_best:<20s}"
            f"  {metric_name}={ds_scores[ds_best]:.5f}"
            f"  (constant={ds_scores['constant']:.5f}"
            f"  delta={ds_scores[ds_best]-ds_scores['constant']:+.5f})"
        )

    print(
        f"\n  Notes"
        f"\n  -----"
        f"\n  - All schedules share base lr={BASE_LR} and n_rounds={N_ROUNDS}."
        f"\n  - Schedules with higher early/late ratios front-load the learning"
        f"\n    but risk under-fitting in late rounds if lr decays too far."
        f"\n  - GeoXGB refits HVRT geometry at each refit_interval, providing"
        f"\n    an implicit curriculum that may interact with the lr schedule."
        f"\n  - Results are 5-fold CV averages; std is reported per schedule."
        f"\n"
        f"\n  Wall time : {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used : {cpu_count}"
        f"\n"
    )


if __name__ == "__main__":
    main()
