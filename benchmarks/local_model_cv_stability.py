"""
local_model() coefficient stability across training splits
===========================================================

Tests whether local_model() explanations at fixed test points are consistent
when the model is fitted on different 80/20 subsets of the training data.

This is a different question from HPO ranking stability:
  - HPO ranking: "does cv=1 pick the same best params as cv=3?" (answer: NO)
  - Explanation stability: "does local_model(x) return the same coefficients
    regardless of which 80% subset we trained on?" (this study)

If explanations are consistent, a single fit is sufficient for interpretation
and users need not run cross-validated ensembles for local_model() output.

Metrics
-------
For each test point x and coefficient (additive_i, pairwise_(i,j)):
  - sign_rate   : fraction of splits where the sign matches the majority sign
  - magnitude_cv: coefficient of variation = std(coef) / |mean(coef)| + eps
    -- low CV means stable magnitude; high CV means the value is noisy

Datasets (regression only — local_model targets ensemble log-odds for binary clf,
          which is valid, but we focus on regression for cleaner ground truth):
  diabetes         -- n=442,  d=10
  california       -- n=20640, d=8  (4000-sample subsample)
  friedman1        -- n=2000,  d=10
  boston-proxy     -- make_regression(n=500, d=13, noise=20)

Test points: 25th / 50th / 75th percentile samples from the full dataset
             (chosen deterministically — no randomness in test point selection)

Usage
-----
    python benchmarks/local_model_cv_stability.py [--n-splits N] [--seed S]
"""

import argparse
import time
import warnings
import numpy as np
from sklearn.datasets import (
    load_diabetes, fetch_california_housing, make_friedman1, make_regression
)
from sklearn.metrics import r2_score

from geoxgb import GeoXGBRegressor


def _p(*a, **k):
    print(*a, **k, flush=True)


# --- GeoXGB default params (not HPO'd — we want to study explanation stability,
#     not find optimal params; defaults are representative) ------------------
_MODEL_PARAMS = dict(
    n_rounds       = 1000,
    learning_rate  = 0.02,
    max_depth      = 3,
    y_weight       = 0.25,
    refit_interval = 50,
    convergence_tol= 0.01,   # keep enabled — this is a real fit, not a study artefact
)


# --- stability analysis for one test point across splits -------------------

def analyse_stability(coeffs_per_split, feature_names, eps=1e-6):
    """
    coeffs_per_split: list of dicts, each from local_model().
    Returns a summary of sign consistency and magnitude CV per coefficient.
    """
    d = len(feature_names)
    # Collect additive and pairwise values across splits
    add_vals = {i: [] for i in range(d)}   # feature index -> list of coefs
    pair_vals = {}                          # (i,j) -> list of coefs

    # Union of all active pairs across splits
    all_pairs = set()
    for lm in coeffs_per_split:
        all_pairs.update(lm["pairwise"].keys())

    for lm in coeffs_per_split:
        for i in range(d):
            add_vals[i].append(lm["additive"][i])
        for pair in all_pairs:
            pair_vals.setdefault(pair, []).append(lm["pairwise"].get(pair, 0.0))

    results = {}

    # Additive
    for i in range(d):
        v = np.array(add_vals[i])
        signs = np.sign(v)
        maj_sign = np.sign(np.sum(signs))
        sign_rate = float(np.mean(signs == maj_sign))
        cv = float(np.std(v) / (abs(np.mean(v)) + eps))
        results[("add", i)] = {
            "name":      feature_names[i],
            "mean":      float(np.mean(v)),
            "std":       float(np.std(v)),
            "sign_rate": sign_rate,
            "mag_cv":    cv,
        }

    # Pairwise
    for (i, j) in sorted(all_pairs):
        v = np.array(pair_vals[(i, j)])
        signs = np.sign(v)
        maj_sign = np.sign(np.sum(signs))
        sign_rate = float(np.mean(signs == maj_sign))
        cv = float(np.std(v) / (abs(np.mean(v)) + eps))
        results[("pair", i, j)] = {
            "name":      f"({feature_names[i]},{feature_names[j]})",
            "mean":      float(np.mean(v)),
            "std":       float(np.std(v)),
            "sign_rate": sign_rate,
            "mag_cv":    cv,
        }

    return results


def coop_matrix_stability(X_z_train, part_ids, x_z, tree, d):
    """Pearson corr matrix for the partition containing x_z."""
    leaf = int(tree.apply(x_z.reshape(1, -1).astype(np.float32))[0])
    mask = part_ids == leaf
    Z_p  = X_z_train[mask]
    m    = len(Z_p)
    if m < 2:
        return np.eye(d)
    Z_c  = Z_p - Z_p.mean(axis=0, keepdims=True)
    std  = Z_c.std(axis=0)
    std[std < 1e-10] = 1.0
    Z_n  = Z_c / std
    return (Z_n.T @ Z_n) / m


def run_dataset(name, X, y, n_splits, base_seed, min_pair_coop=0.30):
    _p(f"\n{'='*65}")
    _p(f"Dataset: {name}  (n={len(X)}, d={X.shape[1]})")
    _p(f"{'='*65}")

    feature_names = [f"x{i}" for i in range(X.shape[1])]
    FT = ["continuous"] * X.shape[1]
    d  = X.shape[1]

    # Choose 3 test points deterministically: 25/50/75 percentile by y-rank
    order = np.argsort(y)
    test_indices = [
        int(order[int(len(order) * 0.25)]),
        int(order[int(len(order) * 0.50)]),
        int(order[int(len(order) * 0.75)]),
    ]
    X_test_pts = X[test_indices]
    y_test_pts = y[test_indices]
    _p(f"  Test points: indices {test_indices}  "
       f"(y = {[f'{v:.2f}' for v in y_test_pts]})")
    _p(f"  min_pair_coop={min_pair_coop} (pairs must have |C_ij| >= this)")

    # Fit on n_splits different 80/20 subsets
    lm_per_point    = {ti: [] for ti in range(len(test_indices))}
    coop_per_point  = {ti: [] for ti in range(len(test_indices))}
    pred_per_point  = {ti: [] for ti in range(len(test_indices))}
    r2_vals = []

    for s in range(n_splits):
        rng = np.random.default_rng(base_seed + s * 7)
        perm = rng.permutation(len(X))
        n_tr = int(0.8 * len(X))
        tr = perm[:n_tr]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GeoXGBRegressor(**_MODEL_PARAMS, random_state=base_seed + s)
            m.fit(X[tr], y[tr], feature_types=FT)

        r2_vals.append(r2_score(y[perm[n_tr:]], m.predict(X[perm[n_tr:]])))
        hm = m._resample_history[-1]["hvrt_model"]

        for ti, x_pt in enumerate(X_test_pts):
            lm = m.local_model(x_pt, feature_names=feature_names,
                               min_pair_coop=min_pair_coop)
            lm_per_point[ti].append(lm)
            pred_per_point[ti].append(float(m.predict(x_pt.reshape(1, -1))[0]))

            # Cooperation matrix at test point
            x_z = hm._to_z(x_pt.reshape(1, -1))[0]
            C   = coop_matrix_stability(hm.X_z_, hm.partition_ids_, x_z, hm.tree_, d)
            coop_per_point[ti].append(C)

    _p(f"  Val R2 across {n_splits} splits: "
       f"mean={np.mean(r2_vals):.4f}  std={np.std(r2_vals):.4f}")

    grand_add_sign   = []
    grand_add_cv     = []
    grand_coop_sign  = []

    for ti, x_pt in enumerate(X_test_pts):
        lm_list    = lm_per_point[ti]
        preds      = pred_per_point[ti]
        coop_mats  = np.array(coop_per_point[ti])   # (n_splits, d, d)
        part_sizes = [lm["partition_size"] for lm in lm_list]

        # Ensemble prediction stability
        pred_mean = np.mean(preds)
        pred_cv   = float(np.std(preds) / (abs(pred_mean) + 1e-6))

        _p(f"\n  Test point {ti+1}: y={y_test_pts[ti]:.2f}  "
           f"pred_mean={pred_mean:.2f}  pred_cv={pred_cv:.3f}  "
           f"partition_size={int(np.mean(part_sizes)):.0f}+/-{int(np.std(part_sizes)):.0f}")

        # Additive coefficient stability
        stability = analyse_stability(lm_list, feature_names)
        add_sign_v = [stability[("add", i)]["sign_rate"] for i in range(d)]
        add_cv_v   = [stability[("add", i)]["mag_cv"]    for i in range(d)]
        mean_add_sign = float(np.mean(add_sign_v))
        mean_add_cv   = float(np.mean(add_cv_v))

        # Pairwise coefficient stability (local_model, filtered pairs only)
        pair_items = [(k, v) for k, v in stability.items() if k[0] == "pair"]
        if pair_items:
            pair_sign_v = [v["sign_rate"] for _, v in pair_items]
            pair_cv_v   = [v["mag_cv"]    for _, v in pair_items]
            mean_pair_sign = float(np.mean(pair_sign_v))
            mean_pair_cv   = float(np.mean(pair_cv_v))
            n_pairs = len(pair_items)
        else:
            mean_pair_sign = float('nan')
            mean_pair_cv   = float('nan')
            n_pairs = 0

        # Cooperation MATRIX stability: upper triangle sign consistency
        coop_sign_rates = []
        for i in range(d):
            for j in range(i + 1, d):
                vals = coop_mats[:, i, j]
                signs = np.sign(vals)
                maj = np.sign(np.sum(signs))
                coop_sign_rates.append(float(np.mean(signs == maj)))
        mean_coop_sign = float(np.mean(coop_sign_rates))

        _p(f"    Pred stability: pred_cv={pred_cv:.3f}")
        _p(f"    local_model additive:  sign_rate={mean_add_sign:.3f}  mag_cv={mean_add_cv:.3f}")
        if n_pairs > 0:
            _p(f"    local_model pairwise:  sign_rate={mean_pair_sign:.3f}  mag_cv={mean_pair_cv:.3f}"
               f"  (n_pairs={n_pairs}, filtered by min_pair_coop={min_pair_coop})")
        else:
            _p(f"    local_model pairwise:  0 pairs above min_pair_coop={min_pair_coop}")
        _p(f"    cooperation_matrix:    sign_rate={mean_coop_sign:.3f}  "
           f"(all {d*(d-1)//2} pairs, direct geometry)")

        grand_add_sign.append(mean_add_sign)
        grand_add_cv.append(mean_add_cv)
        grand_coop_sign.append(mean_coop_sign)

    # Dataset-level summary
    ds_add_sign  = float(np.mean(grand_add_sign))
    ds_add_cv    = float(np.mean(grand_add_cv))
    ds_coop_sign = float(np.mean(grand_coop_sign))

    _p(f"\n  >> Dataset summary (n_splits={n_splits}):")
    _p(f"     local_model additive sign_rate: {ds_add_sign:.3f}  mag_cv: {ds_add_cv:.3f}")
    _p(f"     cooperation_matrix  sign_rate:  {ds_coop_sign:.3f}  (direct geometry, no OLS)")

    stable_add  = (ds_add_sign  >= 0.85 and ds_add_cv <= 0.40)
    stable_coop = (ds_coop_sign >= 0.85)
    _p(f"     local_model additive: {'STABLE' if stable_add  else 'VARIABLE'}")
    _p(f"     cooperation_matrix:   {'STABLE' if stable_coop else 'VARIABLE'}")

    return {
        "dataset":       name,
        "n":             len(X),
        "d":             X.shape[1],
        "mean_r2":       float(np.mean(r2_vals)),
        "add_sign_rate": ds_add_sign,
        "add_mag_cv":    ds_add_cv,
        "coop_sign":     ds_coop_sign,
        "stable_add":    stable_add,
        "stable_coop":   stable_coop,
    }


def load_datasets(seed):
    ds = []

    d = load_diabetes()
    ds.append(("diabetes", d.data, d.target))

    X_ca, y_ca = fetch_california_housing(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_ca), size=4000, replace=False)
    ds.append(("california_4k", X_ca[idx], y_ca[idx]))

    X_f, y_f = make_friedman1(n_samples=2000, noise=1.0, random_state=seed)
    ds.append(("friedman1", X_f, y_f))

    X_r, y_r = make_regression(n_samples=500, n_features=13, noise=20.0,
                                random_state=seed)
    ds.append(("make_reg_500", X_r, y_r))

    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=8,
                    help="Number of different 80/20 train splits per dataset")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    datasets = load_datasets(args.seed)

    _p(f"\n{'='*65}")
    _p(f"local_model() stability study -- {args.n_splits} splits per dataset")
    _p(f"{'='*65}")
    _p(f"  Sign rate: fraction of splits where sign matches majority")
    _p(f"  Mag CV:    std / |mean| -- lower = more stable magnitude")
    _p(f"  Threshold: sign_rate >= 0.85 AND mag_cv <= 0.40 = STABLE")

    summaries = []
    t_total = time.time()
    for name, X, y in datasets:
        t0 = time.time()
        r = run_dataset(name, X, y, args.n_splits, args.seed)
        summaries.append(r)
        _p(f"\n  [Elapsed: {time.time()-t0:.0f}s]")

    _p(f"\n{'='*72}")
    _p(f"FINAL SUMMARY  (total: {time.time()-t_total:.0f}s)")
    _p(f"{'='*72}")
    _p(f"  {'Dataset':<18}  {'n':>5}  {'d':>4}  {'R2':>6}  "
       f"{'add_sign':>9}  {'add_cv':>7}  {'coop_sign':>10}  verdict")
    _p(f"  {'-'*72}")
    for r in summaries:
        v_add  = "ok" if r["stable_add"]  else "noisy"
        v_coop = "ok" if r["stable_coop"] else "noisy"
        _p(f"  {r['dataset']:<18}  {r['n']:>5}  {r['d']:>4}  "
           f"{r['mean_r2']:>6.3f}  "
           f"{r['add_sign_rate']:>9.3f}  {r['add_mag_cv']:>7.3f}  "
           f"{r['coop_sign']:>10.3f}  "
           f"add={v_add}  coop={v_coop}")

    _p(f"\nConclusion:")
    n_add_stable  = sum(r["stable_add"]  for r in summaries)
    n_coop_stable = sum(r["stable_coop"] for r in summaries)
    _p(f"  local_model additive stable: {n_add_stable}/{len(summaries)}")
    _p(f"  cooperation_matrix   stable: {n_coop_stable}/{len(summaries)}")
    _p()
    if n_coop_stable == len(summaries):
        _p(f"  cooperation_matrix: consistently stable across ALL datasets.")
        _p(f"  Use cooperation_matrix() for robust local explanations (no OLS, pure geometry).")
    if n_add_stable >= len(summaries) // 2:
        _p(f"  local_model additive: stable on majority of datasets.")
        _p(f"  Additive direction (sign of alpha_i) is trustworthy from a single fit.")
        _p(f"  NOTE: pairwise coefficients from local_model() are unreliable for small")
        _p(f"  partitions (OLS non-identifiability). Use cooperation_matrix() for pairwise.")
    else:
        _p(f"  local_model additive unstable on majority; multiple fits advised.")

    _p(f"\nDone.")


if __name__ == "__main__":
    main()
