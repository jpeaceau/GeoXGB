"""
local_model() validation on a deterministic synthetic dataset
==============================================================

Ground truth:
    f(x1, x2, x3) = 2*x1 + 3*x2 - 1*x3 + 0.5*x1*x2 - 1.5*x2*x3

Known coefficients (global, in original feature space):
    additive:    alpha_1=2,   alpha_2=3,   alpha_3=-1
    pairwise:    beta_12=0.5, beta_23=-1.5, beta_13=0

Dataset: 5x5x5 = 125 fully deterministic grid points over [0, 2]^3
No noise. No random seed needed.

Validation checks:
1. local_model() at a specific sample approximates the ensemble prediction
   (local_r2 should be high; prediction should be close to model.predict(x))
2. The local ADDITIVE coefficients are close to the ground truth linear
   approximation at the test point (2 + 0.5*x2_mean, 3 + 0.5*x1_mean - 1.5*x3_mean,
   -1 - 1.5*x2_mean where means are over the local partition)
3. The pairwise coefficients (beta_12, beta_23) are recoverable with correct sign
4. Different train/validation splits give consistent local_model() results
   (validates that the cooperation geometry is stable across data subsets)

Usage:
    python benchmarks/local_model_validation.py
"""

import numpy as np
from sklearn.metrics import r2_score
from geoxgb import GeoXGBRegressor


def _p(*a, **k):
    print(*a, **k, flush=True)


# =============================================================================
# Dataset: fully deterministic 5x5x5 grid over [0, 2]^3
# =============================================================================

VALS = [0.0, 0.5, 1.0, 1.5, 2.0]

X = np.array([(x1, x2, x3)
              for x1 in VALS
              for x2 in VALS
              for x3 in VALS], dtype=np.float64)

# Ground truth: additive + multiplicative
y = (2.0 * X[:, 0]
     + 3.0 * X[:, 1]
     - 1.0 * X[:, 2]
     + 0.5 * X[:, 0] * X[:, 1]
     - 1.5 * X[:, 1] * X[:, 2])

FEATURE_NAMES = ["x1", "x2", "x3"]
FT = ["continuous", "continuous", "continuous"]

# Test point: x = [1, 1, 1]  (grid center)
X_TEST = np.array([[1.0, 1.0, 1.0]])
Y_TEST_TRUE = float(2 * 1 + 3 * 1 - 1 + 0.5 * 1 * 1 - 1.5 * 1 * 1)  # = 3.0

_p(f"\n{'='*60}")
_p("local_model() validation -- deterministic synthetic dataset")
_p(f"{'='*60}")
_p(f"  n_samples = {len(X)}, n_features = {X.shape[1]}")
_p(f"  Ground truth: f = 2*x1 + 3*x2 - x3 + 0.5*x1*x2 - 1.5*x2*x3")
_p(f"  Test point:   x = [1, 1, 1],  f(x) = {Y_TEST_TRUE:.4f}")

# =============================================================================
# Fit GeoXGB on full dataset
# =============================================================================

_p(f"\n--- Fitting GeoXGB (n_rounds=2000) on full dataset ---")
model = GeoXGBRegressor(
    n_rounds=2000,
    learning_rate=0.02,
    max_depth=3,
    y_weight=0.25,
    refit_interval=50,
    random_state=42,
)
model.fit(X, y, feature_types=FT)
full_r2 = r2_score(y, model.predict(X))
pred_test = float(model.predict(X_TEST)[0])
_p(f"  Train R2: {full_r2:.4f}")
_p(f"  model.predict([1,1,1]) = {pred_test:.4f}  (true = {Y_TEST_TRUE:.4f})")

# =============================================================================
# local_model() at test point
# =============================================================================

_p(f"\n--- local_model() at x=[1,1,1] ---")
lm = model.local_model(X_TEST[0], feature_names=FEATURE_NAMES, min_pair_coop=0.05)

_p(f"  Partition size: {lm['partition_size']} training points")
_p(f"  Local R2 (polynomial fit on partition): {lm['local_r2']:.4f}")
_p(f"  Polynomial prediction at x: {lm['prediction']:.4f}  "
   f"(ensemble: {pred_test:.4f}, diff: {abs(lm['prediction'] - pred_test):.4f})")

_p(f"\n  Additive coefficients (in z-space):")
for i, (name, coef) in enumerate(zip(FEATURE_NAMES, lm["additive"])):
    _p(f"    alpha_{name} = {coef:+.4f}")

_p(f"\n  Pairwise coefficients (cooperation-active pairs):")
if lm["pairwise"]:
    for (i, j), coef in sorted(lm["pairwise"].items()):
        name_ij = f"({FEATURE_NAMES[i]}, {FEATURE_NAMES[j]})"
        _p(f"    beta_{name_ij} = {coef:+.4f}")
else:
    _p("    (no pairs above min_pair_coop threshold)")

# Ground-truth local linear approximation at partition center
# At x=[1,1,1] the local linear terms of the true function are:
# df/dx1 = 2 + 0.5*x2 = 2.5  (evaluated at partition mean, approx x2~1)
# df/dx2 = 3 + 0.5*x1 - 1.5*x3 = 2.0
# df/dx3 = -1 - 1.5*x2 = -2.5
# These are in original-x space. In z-space they're scaled by std(xi).
_p(f"\n  NOTE: additive coefficients are in z-space.")
_p(f"  The true local linear approx in x-space at (x1,x2,x3)~(1,1,1) is:")
_p(f"    df/dx1 ~ 2 + 0.5*1 = 2.50")
_p(f"    df/dx2 ~ 3 + 0.5*1 - 1.5*1 = 2.00")
_p(f"    df/dx3 ~ -1 - 1.5*1 = -2.50")
std_x = X.std(axis=0)
_p(f"  Feature stds on training set: {std_x}")
_p(f"  Expected alpha in z-space (coef * std(xi)):")
_p(f"    alpha_x1_zspace ~ 2.50 * {std_x[0]:.3f} = {2.50 * std_x[0]:.4f}")
_p(f"    alpha_x2_zspace ~ 2.00 * {std_x[1]:.3f} = {2.00 * std_x[1]:.4f}")
_p(f"    alpha_x3_zspace ~ -2.50 * {std_x[2]:.3f} = {-2.50 * std_x[2]:.4f}")

# =============================================================================
# Check 1: prediction accuracy
# =============================================================================

_p(f"\n--- Check 1: local polynomial approximation quality ---")
err = abs(lm["prediction"] - pred_test)
_p(f"  |polynomial(x) - ensemble(x)| = {err:.4f}")
if err < 0.5:
    _p(f"  PASS: local polynomial closely approximates ensemble prediction")
else:
    _p(f"  WARN: gap is large (check partition size or increase n_rounds)")

# =============================================================================
# Check 2: sign of pairwise coefficients matches ground truth
# =============================================================================

_p(f"\n--- Check 2: pairwise coefficient signs ---")
# True: beta_12 > 0, beta_23 < 0, beta_13 = 0 (in original x-space)
# In z-space: signs are preserved since z = (x - mu) / sigma with sigma > 0
pairs_found = lm["pairwise"]
beta_12 = pairs_found.get((0, 1), None)
beta_23 = pairs_found.get((1, 2), None)
beta_13 = pairs_found.get((0, 2), None)

for name, val, expected_sign in [
    ("beta_(x1,x2)", beta_12, "+"),
    ("beta_(x2,x3)", beta_23, "-"),
    ("beta_(x1,x3)", beta_13,  "0"),
]:
    if val is None:
        status = "not in model (filtered by min_pair_coop)"
    elif expected_sign == "+" and val > 0:
        status = f"PASS  ({val:+.4f} > 0, expected positive)"
    elif expected_sign == "-" and val < 0:
        status = f"PASS  ({val:+.4f} < 0, expected negative)"
    elif expected_sign == "0":
        status = f"INFO  ({val:+.4f}, expected near zero)"
    else:
        status = f"WARN  ({val:+.4f}, expected {expected_sign})"
    _p(f"  {name}: {status}")

# =============================================================================
# Check 3: stability across 3 train/val splits
# =============================================================================

_p(f"\n--- Check 3: local_model() stability across data splits ---")
_p(f"  (same test point x=[1,1,1], different 80/20 random train/val splits)")

# Use deterministic seeds for fully reproducible splits
results_splits = []
for seed in [10, 20, 30]:
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(X))
    n_tr  = int(0.8 * len(X))
    tr_idx = perm[:n_tr]

    m_s = GeoXGBRegressor(
        n_rounds=2000, learning_rate=0.02, max_depth=3,
        y_weight=0.25, refit_interval=50, random_state=seed,
    )
    m_s.fit(X[tr_idx], y[tr_idx], feature_types=FT)
    lm_s = m_s.local_model(X_TEST[0], feature_names=FEATURE_NAMES, min_pair_coop=0.05)
    results_splits.append(lm_s)

    pred_s   = float(m_s.predict(X_TEST)[0])
    _p(f"\n  seed={seed}: partition_size={lm_s['partition_size']},  "
       f"local_r2={lm_s['local_r2']:.4f},  "
       f"polynomial(x)={lm_s['prediction']:.4f}  ensemble(x)={pred_s:.4f}")
    _p(f"    additive: {[f'{a:+.3f}' for a in lm_s['additive']]}")
    _p(f"    pairwise: {[(f'({FEATURE_NAMES[i]},{FEATURE_NAMES[j]})', f'{v:+.3f}') for (i,j),v in sorted(lm_s['pairwise'].items())]}")

# Check if additive coefficient signs are consistent
_p(f"\n  Consistency check: do all 3 splits agree on alpha sign?")
for i, name in enumerate(FEATURE_NAMES):
    signs = [np.sign(r["additive"][i]) for r in results_splits]
    consistent = len(set(signs)) == 1
    vals  = [f"{r['additive'][i]:+.3f}" for r in results_splits]
    status = "consistent" if consistent else "INCONSISTENT"
    _p(f"    alpha_{name}: {vals}  -> {status}")

_p(f"\n  Consistency check: do all 3 splits agree on pairwise sign?")
for key_name, key in [("(x1,x2)", (0,1)), ("(x2,x3)", (1,2))]:
    vals = [r["pairwise"].get(key, 0.0) for r in results_splits]
    signs = [np.sign(v) for v in vals]
    consistent = len(set(signs)) == 1
    v_str = [f"{v:+.3f}" for v in vals]
    status = "consistent" if consistent else "INCONSISTENT"
    _p(f"    beta_{key_name}: {v_str}  -> {status}")

_p(f"\nDone.")
