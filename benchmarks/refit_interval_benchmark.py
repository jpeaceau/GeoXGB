"""
GeoXGB -- Refit Interval Sensitivity Benchmark (Multiprocessing)
================================================================

Sweeps refit_interval across {None, 10, 20, 50, 100, 250, 500} at
n_rounds=1000, learning_rate=0.2 across three datasets using all
available CPU cores.

refit_interval=None disables HVRT refits entirely; only the initial
partition geometry is used for all 1000 rounds.

Datasets
--------
  friedman1      : Friedman #1 regression  (R^2)
  friedman2      : Friedman #2 regression  (R^2)
  classification : Synthetic binary classification  (AUC)

CV: 5-fold

Usage
-----
    python benchmarks/refit_interval_benchmark.py

Requirements: geoxgb, scikit-learn, numpy, joblib
"""

from __future__ import annotations

import multiprocessing
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

N_ROUNDS     = 1000
LEARNING_RATE = 0.2
N_FOLDS      = 5
RANDOM_STATE = 42

REFIT_INTERVALS = [None, 5, 10, 20, 50, 100, 250, 500]

GEO_FIXED = dict(
    n_rounds=N_ROUNDS,
    learning_rate=LEARNING_RATE,
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
    from sklearn.datasets import make_regression

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
    # Sparse high-dimensional regression: 40 features, only 8 informative,
    # high additive noise — tests HVRT geometry in a noisy high-D space
    X4, y4 = make_regression(
        n_samples=1_000, n_features=40, n_informative=8,
        noise=20.0, random_state=RANDOM_STATE,
    )
    # Hard noisy classification: 20 features, 10% label flip, low class
    # separation — tests HVRT robustness to label noise and overlapping classes
    X5, y5 = make_classification(
        n_samples=1_000, n_features=20, n_informative=5,
        n_redundant=5, n_repeated=0, n_clusters_per_class=1,
        class_sep=0.5, flip_y=0.10, random_state=RANDOM_STATE,
    )
    return {
        "friedman1":        (X1, y1, "regression",     "R^2"),
        "friedman2":        (X2, y2, "regression",     "R^2"),
        "classification":   (X3, y3, "classification", "AUC"),
        "sparse_highdim":   (X4, y4, "regression",     "R^2"),
        "noisy_clf":        (X5, y5, "classification", "AUC"),
    }


# ---------------------------------------------------------------------------
# Worker -- module-level for Windows pickling
# ---------------------------------------------------------------------------

def _eval_fold(
    interval,
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple:
    import warnings as _w
    _w.filterwarnings("ignore")

    params = dict(**GEO_FIXED, refit_interval=interval)
    t0 = time.perf_counter()

    if task == "regression":
        m = GeoXGBRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(r2_score(y[val_idx], m.predict(X[val_idx])))
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[train_idx], y[train_idx])
        score = float(roc_auc_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))

    return interval, score, time.perf_counter() - t0


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


def _interval_label(v) -> str:
    return "None" if v is None else str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    datasets  = _make_datasets()
    n_configs = len(REFIT_INTERVALS)
    n_jobs    = n_configs * len(datasets) * N_FOLDS

    _section("GeoXGB -- Refit Interval Sensitivity Benchmark")
    dataset_notes = {
        "friedman1":      "10 features, noise=1.0 (moderate noise, nonlinear)",
        "friedman2":      "10 features, noise=0.0 (clean, nonlinear)",
        "classification": "10 features, 5 informative, class_sep=1.0 (clean)",
        "sparse_highdim": "40 features, 8 informative, noise=20 (sparse, high-D, noisy)",
        "noisy_clf":      "20 features, 5 informative, flip_y=0.10, sep=0.5 (hard, label noise)",
    }
    print(
        f"\n  n_rounds        : {N_ROUNDS}"
        f"\n  learning_rate   : {LEARNING_RATE}"
        f"\n  refit_intervals : {[_interval_label(v) for v in REFIT_INTERVALS]}"
        f"\n  CV folds        : {N_FOLDS}"
        f"\n  Total jobs      : {n_jobs}"
        f"\n  CPUs            : {cpu_count}"
    )
    print(f"\n  Datasets:")
    for ds_name, note in dataset_notes.items():
        print(f"    {ds_name:<18s}  {note}")
    print(
        f"\n  Note: refit_interval=None disables HVRT refits after the initial fit."
        f"\n        refit_interval=10 is the package default."
    )

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
    job_keys: list[tuple] = []

    for ds_name, (X, y, task, _m) in datasets.items():
        for interval in REFIT_INTERVALS:
            for fi, (tr, val) in enumerate(splits[ds_name]):
                jobs.append((interval, task, X, y, tr, val))
                job_keys.append((ds_name, interval, fi))

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

    for (ds_name, interval, _fi), (_, score, elapsed) in zip(job_keys, raw):
        scores[(ds_name, interval)].append(score)
        times[ (ds_name, interval)].append(elapsed)

    # -----------------------------------------------------------------------
    # Per-dataset tables
    # -----------------------------------------------------------------------
    _section("Per-Dataset Results")

    baseline_interval = 10  # package default

    for ds_name, (X, y, task, metric_name) in datasets.items():
        _subsection(f"{ds_name}  (metric: {metric_name})")

        rows = []
        for interval in REFIT_INTERVALS:
            key = (ds_name, interval)
            sc  = scores[key]
            rows.append({
                "interval": interval,
                "label":    _interval_label(interval),
                "mean":     float(np.mean(sc)),
                "std":      float(np.std(sc)),
                "folds":    sc,
                "time":     float(np.mean(times[key])),
            })

        base_mean = next(
            r["mean"] for r in rows if r["interval"] == baseline_interval
        )
        best_mean = max(r["mean"] for r in rows)

        print(
            f"\n  {'interval':>10s}  {'mean':>8s}  {'std':>6s}"
            f"  {'vs ri=10':>8s}  {'avg time':>8s}  score"
        )
        print(
            f"  {'-'*10}  {'-'*8}  {'-'*6}"
            f"  {'-'*8}  {'-'*8}  {'-'*28}"
        )
        for r in rows:
            delta    = r["mean"] - base_mean
            delta_s  = f"{delta:>+.5f}"
            bar      = _bar(r["mean"], best_mean)
            best_tag = " [best]" if abs(r["mean"] - best_mean) < 1e-9 else ""
            base_tag = " [default]" if r["interval"] == baseline_interval else ""
            tag      = best_tag or base_tag
            print(
                f"  {r['label']:>10s}  {r['mean']:>8.5f}  {r['std']:>6.4f}"
                f"  {delta_s:>8s}  {r['time']:>7.1f}s  {bar}{tag}"
            )

    # -----------------------------------------------------------------------
    # Cross-dataset z-score ranking
    # -----------------------------------------------------------------------
    _section("Cross-Dataset Z-Score Ranking")
    print(
        "\n  Z-scores remove scale differences between R^2 and AUC.\n"
        "  Each interval's score is normalised within each dataset then averaged."
    )

    interval_labels = [_interval_label(v) for v in REFIT_INTERVALS]
    z_by_interval: dict = {_interval_label(v): [] for v in REFIT_INTERVALS}

    for ds_name in datasets:
        ds_scores = {
            _interval_label(iv): float(np.mean(scores[(ds_name, iv)]))
            for iv in REFIT_INTERVALS
        }
        mu  = float(np.mean(list(ds_scores.values())))
        sig = float(np.std(list(ds_scores.values()))) or 1e-9
        for lbl, v in ds_scores.items():
            z_by_interval[lbl].append((v - mu) / sig)

    ranked_labels = sorted(
        interval_labels, key=lambda lbl: -float(np.mean(z_by_interval[lbl]))
    )
    base_z = float(np.mean(z_by_interval[_interval_label(baseline_interval)]))

    print(f"\n  {'rank':>4s}  {'interval':>10s}  {'mean z':>8s}  {'vs ri=10':>10s}  bar")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*28}")
    max_z = float(np.mean(z_by_interval[ranked_labels[0]]))
    for rank, lbl in enumerate(ranked_labels, 1):
        z    = float(np.mean(z_by_interval[lbl]))
        dz   = z - base_z
        bar  = _bar(max(z, 0), max(max_z, 1e-9))
        flag = " [best]" if rank == 1 else (" [default]" if lbl == _interval_label(baseline_interval) else "")
        print(
            f"  {rank:>4d}  {lbl:>10s}  {z:>8.4f}"
            f"  {dz:>+10.4f}  {bar}{flag}"
        )

    # -----------------------------------------------------------------------
    # Refit frequency analysis
    # -----------------------------------------------------------------------
    _section("Refit Frequency Analysis")
    print(
        f"\n  At n_rounds={N_ROUNDS}, each interval implies the following number"
        f"\n  of HVRT refits (not counting the initial fit):"
    )
    print(f"\n  {'interval':>10s}  {'n_refits':>10s}  {'approx % of rounds with geometry update':>40s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*40}")
    for iv in REFIT_INTERVALS:
        if iv is None:
            n_refits = 0
            pct = "0%  (static geometry)"
        else:
            n_refits = (N_ROUNDS - 1) // iv
            pct = f"{100 * iv / N_ROUNDS:.1f}% interval spacing"
        print(f"  {_interval_label(iv):>10s}  {str(n_refits):>10s}  {pct}")

    # -----------------------------------------------------------------------
    # Recommendation
    # -----------------------------------------------------------------------
    _section("Recommendation")
    best_lbl = ranked_labels[0]
    best_z_v = float(np.mean(z_by_interval[best_lbl]))

    print(f"\n  Best refit_interval overall : {best_lbl}  (mean z={best_z_v:.4f})")
    print(f"  Default baseline (ri=10)    : z={base_z:.4f}")
    print(f"  Improvement over default    : {best_z_v - base_z:+.4f} z-score units")
    print()

    for ds_name, (X, y, task, metric_name) in datasets.items():
        ds_scores = {
            _interval_label(iv): float(np.mean(scores[(ds_name, iv)]))
            for iv in REFIT_INTERVALS
        }
        ds_best  = max(ds_scores, key=ds_scores.__getitem__)
        base_val = ds_scores[_interval_label(baseline_interval)]
        print(
            f"  {ds_name:<18s}  best=ri={ds_best:<6s}"
            f"  {metric_name}={ds_scores[ds_best]:.5f}"
            f"  (ri=10={base_val:.5f}"
            f"  delta={ds_scores[ds_best]-base_val:+.5f})"
        )

    # -----------------------------------------------------------------------
    # Heuristic analysis: per-dataset best interval vs dataset properties
    # -----------------------------------------------------------------------
    _section("Heuristic Analysis: Best Interval vs Dataset Properties")
    print(
        "\n  For each dataset, record the best refit_interval and compare"
        "\n  against dataset properties to surface any predictive pattern."
    )

    ds_properties = {
        "friedman1":      dict(n_features=10, n_informative=10, noise="moderate", task="regression"),
        "friedman2":      dict(n_features=10, n_informative=10, noise="zero",     task="regression"),
        "classification": dict(n_features=10, n_informative=5,  noise="none",     task="classification"),
        "sparse_highdim": dict(n_features=40, n_informative=8,  noise="high",     task="regression"),
        "noisy_clf":      dict(n_features=20, n_informative=5,  noise="label10%", task="classification"),
    }

    print(
        f"\n  {'dataset':<18s}  {'n_feat':>6s}  {'n_info':>6s}  {'noise':<10s}"
        f"  {'task':<14s}  {'best_ri':>7s}  {'best_score':>10s}"
    )
    print(
        f"  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*10}"
        f"  {'-'*14}  {'-'*7}  {'-'*10}"
    )

    best_ris: list[tuple] = []
    for ds_name, (X, y, task, metric_name) in datasets.items():
        ds_scores_raw = {
            iv: float(np.mean(scores[(ds_name, iv)]))
            for iv in REFIT_INTERVALS
        }
        # exclude None from best (it's always worse; we want the best enabled value)
        enabled = {k: v for k, v in ds_scores_raw.items() if k is not None}
        best_iv  = max(enabled, key=enabled.__getitem__)
        best_sc  = enabled[best_iv]
        props    = ds_properties.get(ds_name, {})
        best_ris.append((ds_name, best_iv, best_sc, props))

        print(
            f"  {ds_name:<18s}  {props.get('n_features', '?'):>6}  "
            f"{props.get('n_informative', '?'):>6}  "
            f"{props.get('noise', '?'):<10s}  "
            f"{props.get('task', '?'):<14s}  "
            f"{_interval_label(best_iv):>7s}  {best_sc:>10.5f}"
        )

    print(
        f"\n  Heuristic observations:"
        f"\n  - Note which interval wins across dataset types."
        f"\n  - A universal best value would indicate a robust default."
        f"\n  - Per-property patterns (e.g., noisy -> larger ri) would"
        f"\n    suggest an adaptive heuristic (e.g., ri = f(noise, n_features))."
    )

    print(
        f"\n  Notes"
        f"\n  -----"
        f"\n  - Smaller refit_interval = more frequent HVRT geometry updates."
        f"\n  - refit_interval=None = HVRT geometry frozen after the initial fit."
        f"\n  - More frequent refits improve adaptability but increase runtime"
        f"\n    (each refit is O(n log n) HVRT + O(n) sampling)."
        f"\n  - For very large n_rounds (5000+) consider increasing refit_interval"
        f"\n    to reduce total resampling cost while retaining adaptability."
        f"\n"
        f"\n  Wall time : {t_wall:.1f}s  ({t_wall / 60:.1f} min)"
        f"\n  CPUs used : {cpu_count}"
        f"\n"
    )


if __name__ == "__main__":
    main()
