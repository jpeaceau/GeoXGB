"""
y_weight_mode experiment — compare fixed, adaptive, linear, adaptive_linear.

Tests all four modes across a range of y_weight values on three synthetic
dataset archetypes, measuring RMSE on a held-out validation set plus
within-partition y variance (a proxy for how well HVRT's geometry captures
the label structure).

Archetypes
----------
A. Feature-cooperative + strong y signal:
   y = sum of pairwise products + small noise.
   The pairwise target and y_component should align — 'adaptive' mode should
   attenuate y_weight automatically since they carry the same information.

B. Partially orthogonal:
   y depends on a subset of features that are NOT driving pairwise interactions.
   The x_component and y_component are orthogonal — both signals are
   independently useful, so full y_weight should help.

C. Noisy:
   y is mostly noise with a weak linear trend.
   High y_weight should hurt; 'adaptive' should auto-dampen it.

Usage
-----
    python benchmarks/y_weight_mode_benchmark.py
"""

import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_cooperative(n=1000, d=10, noise=0.1, seed=0):
    """y = z-scored pairwise sum of feature interactions + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = sum(X[:, i] * X[:, j]
            for i in range(d - 1) for j in range(i + 1, d))
    y = (y - y.mean()) / (y.std() + 1e-10)
    y += rng.standard_normal(n) * noise
    return X, y


def make_orthogonal(n=1000, d=10, noise=0.2, seed=0):
    """y driven by last 2 features; pairwise interactions among first 8."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    # Pairwise interaction among the first d-2 features (what HVRT's target sees)
    x_part = sum(X[:, i] * X[:, j]
                 for i in range(d - 2) for j in range(i + 1, d - 2))
    # y driven by the last 2 features only (orthogonal to pairwise signal)
    y = 3.0 * X[:, -1] + 2.0 * X[:, -2]
    y = (y - y.mean()) / (y.std() + 1e-10)
    y += rng.standard_normal(n) * noise
    return X, y


def make_noisy(n=1000, d=10, signal_ratio=0.15, seed=0):
    """y is mostly noise with a weak linear trend on one feature."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    signal = X[:, 0]
    signal = (signal - signal.mean()) / (signal.std() + 1e-10)
    noise = rng.standard_normal(n)
    noise = (noise - noise.mean()) / (noise.std() + 1e-10)
    y = signal_ratio * signal + (1 - signal_ratio) * noise
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def within_partition_y_var(hvrt_model, y):
    """Mean within-partition variance of y — lower means better label capture."""
    pid = hvrt_model.partition_ids_
    parts = hvrt_model.unique_partitions_
    vars_ = []
    for p in parts:
        yp = y[pid == p]
        if len(yp) > 1:
            vars_.append(float(yp.var()))
    return float(np.mean(vars_)) if vars_ else float("nan")


def run_hvrt_metrics(X_tr, y_tr, X_val, y_val, y_weight, mode):
    """Fit HVRT, return within-partition variance + naive reduce RMSE."""
    from hvrt import HVRT
    model = HVRT(y_weight=y_weight, y_weight_mode=mode, random_state=42)
    model.fit(X_tr, y_tr)

    wpv = within_partition_y_var(model, y_tr)

    # Naive predictor: predict partition mean of y_train for each val sample
    pid_val = model.apply_raw(X_val)
    pid_tr  = model.partition_ids_
    part_means = {p: float(y_tr[pid_tr == p].mean())
                  for p in model.unique_partitions_}
    global_mean = float(y_tr.mean())
    y_pred = np.array([part_means.get(p, global_mean) for p in pid_val],
                      dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))

    return wpv, rmse, model.n_partitions_


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

MODES       = ["fixed", "adaptive", "linear", "adaptive_linear"]
Y_WEIGHTS   = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
DATASETS    = {
    "cooperative": make_cooperative,
    "orthogonal":  make_orthogonal,
    "noisy":       make_noisy,
}

HEADER = f"{'dataset':<15} {'mode':<18} {'y_weight':>8}  {'wpv':>8}  {'rmse':>8}  {'n_parts':>7}"
SEP    = "-" * len(HEADER)


def main():
    print("y_weight_mode experiment")
    print("=" * 70)
    print("Metrics: wpv = within-partition y variance (lower is better)")
    print("         rmse = partition-mean predictor on validation (lower is better)")
    print()

    results = {}

    for ds_name, ds_fn in DATASETS.items():
        print(f"\nDataset: {ds_name.upper()}")
        print(HEADER)
        print(SEP)

        X, y = ds_fn(n=1500, seed=1)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25,
                                                     random_state=0)
        results[ds_name] = {}

        for mode in MODES:
            best_wpv  = float("inf")
            best_rmse = float("inf")
            best_yw   = None

            for yw in Y_WEIGHTS:
                wpv, rmse, n_parts = run_hvrt_metrics(X_tr, y_tr, X_val, y_val, yw, mode)
                tag = ""
                if wpv  < best_wpv:  best_wpv  = wpv;  best_yw = yw; tag = " << best wpv"
                if rmse < best_rmse: best_rmse = rmse;                tag += " << best rmse" if tag else " << best rmse"
                print(f"{ds_name:<15} {mode:<18} {yw:>8.1f}  {wpv:>8.4f}  {rmse:>8.4f}  {n_parts:>7}{tag}")

            results[ds_name][mode] = {"best_wpv": best_wpv, "best_rmse": best_rmse,
                                      "best_yw": best_yw}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — best across all y_weight values per (dataset, mode)")
    print("=" * 70)
    print(f"{'dataset':<15} {'mode':<18} {'best_wpv':>10}  {'best_rmse':>10}  {'best_yw':>8}")
    print("-" * 70)
    for ds_name in DATASETS:
        for mode in MODES:
            r = results[ds_name][mode]
            print(f"{ds_name:<15} {mode:<18} {r['best_wpv']:>10.4f}  {r['best_rmse']:>10.4f}  {r['best_yw']:>8.1f}")

    # Best mode per dataset
    print("\n" + "=" * 70)
    print("WINNER per dataset (lowest RMSE)")
    print("=" * 70)
    for ds_name in DATASETS:
        best = min(results[ds_name].items(), key=lambda x: x[1]["best_rmse"])
        mode, r = best
        print(f"  {ds_name:<15}  ->  {mode}  (rmse={r['best_rmse']:.4f}, best_yw={r['best_yw']})")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
