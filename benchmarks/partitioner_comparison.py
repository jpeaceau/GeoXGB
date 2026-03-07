"""
HVRT vs PyramidHART benchmark.

Two experiments:
  1. Imbalanced datasets  — minority cluster embedded in majority; compare R²
     at class-ratio 1:5, 1:10, 1:20, 1:50.
  2. Noise sensitivity    — additive Gaussian noise on features; compare R²
     degradation as sigma scales from 0 to 3.

Both use 5-fold CV, seeds 0-4, n=1000, Friedman-1 signal.
"""

import numpy as np
from sklearn.metrics import r2_score
from geoxgb import GeoXGBRegressor

RNG   = np.random.default_rng(0)
SEEDS = list(range(5))

SHARED = dict(
    n_rounds=500, learning_rate=0.02, max_depth=3,
    reduce_ratio=0.8, refit_interval=50,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def friedman1(n, noise=0.0, rng=None):
    """Friedman-1 with optional additive feature noise."""
    rng = rng or np.random.default_rng(0)
    X = rng.uniform(0, 1, (n, 5))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5)**2
         + 10 * X[:, 3]
         + 5  * X[:, 4])
    y += rng.standard_normal(n) * 1.0          # base label noise
    if noise > 0:
        X = X + rng.standard_normal(X.shape) * noise   # feature noise
    return X, y


def cv_r2(partitioner, X, y, seed):
    """Manual 5-fold CV — avoids sklearn BaseEstimator requirement."""
    n = len(X)
    fold_size = n // 5
    rng_cv = np.random.default_rng(seed)
    idx = rng_cv.permutation(n)
    scores = []
    for k in range(5):
        val_idx = idx[k * fold_size : (k + 1) * fold_size]
        trn_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
        model = GeoXGBRegressor(
            partitioner=partitioner, random_state=seed, **SHARED
        )
        model.fit(X[trn_idx], y[trn_idx])
        scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))
    return float(np.mean(scores))


def mean_r2(partitioner, X, y):
    return float(np.mean([cv_r2(partitioner, X, y, s) for s in SEEDS]))


# ---------------------------------------------------------------------------
# Experiment 1: Imbalanced datasets
# ---------------------------------------------------------------------------
# Two clusters: majority (low signal) + minority (high signal, dense).
# Minority cluster sits at a different mean so the signal is concentrated there.
# Ratio = minority_n / majority_n.

print("=" * 65)
print("EXPERIMENT 1: Imbalanced datasets (Friedman-1 signal in minority)")
print("=" * 65)
print(f"{'Ratio':>10}  {'n_minority':>10}  {'HVRT R²':>10}  {'PyramHART R²':>13}  {'Delta':>8}")
print("-" * 65)

for ratio_label, n_min, n_maj in [
    ("1:5",  167, 833),
    ("1:10",  91, 909),
    ("1:20",  48, 952),
    ("1:50",  20, 980),
]:
    n = n_min + n_maj
    rng = np.random.default_rng(7)

    # Minority: Friedman-1 signal in [0,1]^5
    X_min, y_min = friedman1(n_min, noise=0.0, rng=rng)

    # Majority: weak linear signal in a different region of feature space
    X_maj = rng.uniform(2, 3, (n_maj, 5))   # shifted away from minority
    y_maj = X_maj[:, 0] + rng.standard_normal(n_maj) * 0.5

    X = np.vstack([X_min, X_maj])
    y = np.concatenate([y_min, y_maj])
    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    r2_hvrt = mean_r2("hvrt",         X, y)
    r2_pyr  = mean_r2("pyramid_hart", X, y)
    delta   = r2_hvrt - r2_pyr

    print(f"{ratio_label:>10}  {n_min:>10}  {r2_hvrt:>10.4f}  {r2_pyr:>13.4f}  {delta:>+8.4f}")


# ---------------------------------------------------------------------------
# Experiment 2: Noise sensitivity
# ---------------------------------------------------------------------------
# Pure Friedman-1, n=1000, increasing Gaussian feature noise sigma.
# Measures how much each partitioner's R² degrades as noise grows.

print()
print("=" * 60)
print("EXPERIMENT 2: R2 degradation with additive feature noise")
print("=" * 60)
print(f"{'Noise s':>10}  {'HVRT R2':>10}  {'PyramHART R2':>13}  {'Delta':>8}")
print("-" * 60)

baseline_hvrt = None
baseline_pyr  = None

for sigma in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    rng = np.random.default_rng(42)
    X, y = friedman1(1000, noise=sigma, rng=rng)

    r2_hvrt = mean_r2("hvrt",         X, y)
    r2_pyr  = mean_r2("pyramid_hart", X, y)

    if sigma == 0.0:
        baseline_hvrt = r2_hvrt
        baseline_pyr  = r2_pyr

    drop_hvrt = baseline_hvrt - r2_hvrt
    drop_pyr  = baseline_pyr  - r2_pyr
    delta     = r2_hvrt - r2_pyr

    print(f"{sigma:>10.2f}  {r2_hvrt:>10.4f}  {r2_pyr:>13.4f}  {delta:>+8.4f}"
          f"   (drop: HVRT {drop_hvrt:+.4f}, Pyr {drop_pyr:+.4f})")
