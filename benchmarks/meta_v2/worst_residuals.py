"""
GeoXGB Geometric Explanation — 50 Worst Predictions on friedman1
================================================================
Exercises GeoXGBExplainer on the canonical fold (seed=0, fold=0).

For the 50 samples with the largest |model_error| (noise-decomposed,
since the friedman1 ground truth is known):

  - Full per-sample geometric report via print_explanation()
  - Aggregate summary via print_summary()

The true noiseless function:
    y = 10·sin(π·x1·x2) + 20·(x3−0.5)² + 10·x4 + 5·x5

Run from benchmarks/meta_v2/:
    python worst_residuals.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from meta_reg import _make_datasets                                    # noqa: E402
from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor   # noqa: E402
from geoxgb.explain import (                                           # noqa: E402
    GeoXGBExplainer, print_explanation, print_summary,
)

# ── Config ──────────────────────────────────────────────────────────────────

DATASET     = "friedman1"
CANON_SEED  = 0
CANON_FOLD  = 0
N_FOLDS     = 4
TOP_N       = 50
PRINT_TOP_N = 10   # print full per-sample report for this many; summary covers all 50

OPT = dict(
    n_rounds=3000, learning_rate=0.02, max_depth=2, min_samples_leaf=5,
    reduce_ratio=0.7, expand_ratio=0.1, y_weight=0.5, refit_interval=5,
    auto_noise=False, noise_guard=False, refit_noise_floor=0.05,
    auto_expand=True, min_train_samples=5000, bandwidth="auto",
    variance_weighted=True, hvrt_min_samples_leaf=-1, n_partitions=-1, n_bins=64,
)


def _friedman1_raw(X: np.ndarray) -> np.ndarray:
    """Noiseless friedman1 in raw (unnormalised) units."""
    X = np.asarray(X)
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5  * X[:, 4]
    )


def make_friedman1_true(X_full: np.ndarray, y_normalised: np.ndarray):
    """
    Return a true_fn callable that maps X → normalised noiseless y.

    _make_datasets normalises y to (mean=0, std=1) before returning it.
    Recover the affine mapping raw→normalised via OLS on the full dataset
    (noise cancels in expectation; R² of fit ≈ r2_ceil ≈ 0.96).
    """
    raw = _friedman1_raw(X_full)
    # fit: y_normalised ≈ a * raw + b
    a, b = np.polyfit(raw, y_normalised, 1)

    def _true_fn(X):
        return a * _friedman1_raw(np.asarray(X)) + b

    return _true_fn


def main() -> None:
    datasets = _make_datasets()
    X, y, sigma, r2_ceil = datasets[DATASET]
    n, d = X.shape

    print(f"\n{'='*66}")
    print(f"GeoXGB Geometric Explanation — {DATASET}  (n={n}, d={d})")
    print(f"  Canonical fold: seed={CANON_SEED}, fold={CANON_FOLD}")
    print(f"  Reporting top {PRINT_TOP_N} worst by |model_error|; "
          f"summary over top {TOP_N}")
    print(f"{'='*66}\n")

    # ── Canonical fold split ─────────────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CANON_SEED)
    splits = list(kf.split(X))
    tr_idx, val_idx = splits[CANON_FOLD]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  Train: {len(tr_idx)}   Val: {len(val_idx)}\n")

    # ── Fit GeoXGB ───────────────────────────────────────────────────────
    print("Fitting CppGeoXGBRegressor ...", end=" ", flush=True)
    t0 = time.perf_counter()
    cfg   = make_cpp_config(**dict(OPT, random_state=CANON_SEED))
    model = CppGeoXGBRegressor(cfg)
    model.fit(X_tr, y_tr)
    print(f"done ({time.perf_counter()-t0:.0f}s)")

    # ── Build explainer ──────────────────────────────────────────────────
    print("Building GeoXGBExplainer (fits Python HVRT + precomputes z→X corr) ...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    true_fn = make_friedman1_true(X, y)   # fit normalisation on full 1000-sample dataset

    exp = GeoXGBExplainer(
        model, X_tr, y_tr,
        feature_names=[f"x{i+1}" for i in range(d)],
        k=5,
        true_fn=true_fn,
    )
    print(f"done ({time.perf_counter()-t0:.0f}s)")

    # ── Explain val set, rank by |model_error| ───────────────────────────
    print(f"\nExplaining {len(val_idx)} val samples ...", end=" ", flush=True)
    t0 = time.perf_counter()
    explanations = exp.explain(X_val, y_val, sort_by="abs_model_error", top_n=TOP_N)
    print(f"done ({time.perf_counter()-t0:.0f}s)")

    # ── Per-sample reports ───────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"Per-sample reports — top {PRINT_TOP_N} worst by |model_error|")
    print(f"{'='*66}\n")

    for e in explanations[:PRINT_TOP_N]:
        print_explanation(e, show_all_x=True)
        print()

    # ── Aggregate summary ────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"Aggregate summary — top {TOP_N} worst predictions")
    print(f"{'='*66}\n")

    s = exp.summary(explanations)
    print_summary(s)

    # ── Quick structural observations ────────────────────────────────────
    print(f"\n{'='*66}")
    print("Structural observations")
    print(f"{'='*66}")

    # Do worst predictions cluster in low-T partitions?
    t_vals = [e.partition.t_value for e in explanations
              if not (e.partition.t_value != e.partition.t_value)]  # filter NaN
    global_t = float(np.mean(t_vals)) if t_vals else float("nan")

    # Are their z-space neighbourhoods ambiguous (high y_std)?
    nb_y_stds = [e.neighbour_y_std for e in explanations]

    # How often does the nearest neighbour share the partition?
    frac_nn_same = np.mean([e.neighbours[0].same_partition for e in explanations
                            if e.neighbours])

    print(f"  Mean T-value (worst {TOP_N}): {global_t:+.4f}"
          f"  (negative = below-average cooperation)")
    print(f"  Mean neighbour y_std:         {np.mean(nb_y_stds):.4f}"
          f"  (measures local signal ambiguity)")
    print(f"  Nearest neighbour same-partition: {100*frac_nn_same:.0f}%"
          f"  (z-NN outside partition → partition boundary effect)")

    # Partition concentration: how many distinct partitions in worst-50?
    pids = [e.partition.partition_id for e in explanations]
    n_distinct = len(set(pids))
    print(f"  Distinct partitions in worst {TOP_N}: {n_distinct} "
          f"({'concentrated' if n_distinct < 8 else 'spread'} — "
          f"{'structural blind spot' if n_distinct < 8 else 'distributed noise'})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
