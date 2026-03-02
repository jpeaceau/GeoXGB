"""
Dual Strategy Exploration — Friedman1 Residuals
================================================
Following residual_diag.py's finding:
  - 100% of partitions have z-space CV gain > 0 (mean 0.1255) → real heteroscedasticity
  - Sign consistency ≈ 0.52 → mean corrections inject noise

Two strategies tested on friedman1 (5 seeds × 4 folds = 20 folds total):

  Strategy A — Shrinkage:
      delta_shrunk = delta * n_child / (n_child + λ)
      Sweep λ ∈ [0, 5, 10, 25, 50, 100, 200, 500, 1000] to find break-even.
      λ=0 → raw correction (same as surgery_test).
      Large λ → correction → 0 (baseline, no surgery).

  Strategy B — Heteroscedastic intervals:
      sigma_child = std(resid_child_train) per child.
      Test 1 — Calibration: Spearman(sigma_train, sigma_val) across all children.
      Test 2 — Coverage: actual 90%/95% coverage vs nominal for child/partition/global σ.
      Test 3 — Sharpness: Spearman(sigma_child_assigned, |resid_val|).

No GAIN_THRESHOLD filter — use every partition that yields a valid split (both
children ≥ MIN_LEAF = 5).

Run from benchmarks/meta_v2/:
    python dual_strategy.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ── local imports ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from meta_reg import _make_datasets                                    # noqa: E402
from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor   # noqa: E402
from hvrt import HVRT                                                  # noqa: E402

# ── Configuration ───────────────────────────────────────────────────────────────

DATASET  = "friedman1"
N_SEEDS  = 5
N_FOLDS  = 4
MIN_LEAF = 5

OPT = dict(
    n_rounds=3000, learning_rate=0.02, max_depth=2, min_samples_leaf=5,
    reduce_ratio=0.7, expand_ratio=0.1, y_weight=0.5, refit_interval=5,
    auto_noise=False, noise_guard=False, refit_noise_floor=0.05,
    auto_expand=True, min_train_samples=5000, bandwidth="auto",
    variance_weighted=True, hvrt_min_samples_leaf=-1, n_partitions=-1, n_bins=64,
)

LAMBDAS = [0, 5, 10, 25, 50, 100, 200, 500, 1000]
Z_90    = 1.645
Z_95    = 1.960


# ── Core helper ─────────────────────────────────────────────────────────────────

def _best_split(
    X_part_z: np.ndarray,
    resid_part: np.ndarray,
    min_leaf: int,
) -> tuple[int, float | None, float]:
    """
    Exhaustive variance-minimising split search over z-space features.
    Tries 9 percentile thresholds (10th–90th step 10) for each feature.
    Returns (feat_idx, threshold, gain_ratio) where
        gain_ratio = 1 - var_after / var_before.
    Returns (-1, None, 0.0) if no valid split exists.
    """
    n, d = X_part_z.shape
    var_before = np.var(resid_part) * n
    if var_before < 1e-14:
        return -1, None, 0.0
    best_gain, best_feat, best_thresh = 0.0, -1, None
    for feat in range(d):
        vals = X_part_z[:, feat]
        for thresh in np.unique(np.percentile(vals, np.arange(10, 100, 10))):
            left  = resid_part[vals <= thresh]
            right = resid_part[vals >  thresh]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            var_after = np.var(left) * len(left) + np.var(right) * len(right)
            gain = 1.0 - var_after / var_before
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_feat, best_thresh, best_gain


# ── Per-fold computation ────────────────────────────────────────────────────────

def _run_fold(
    X_tr:    np.ndarray,
    y_tr:    np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    r2_ceil: float,
    seed:    int,
) -> dict:
    """
    Fit GeoXGB + HVRT; compute per-partition split statistics;
    return fold-level results for both strategies.
    """
    # 1. Fit GeoXGB
    cfg   = make_cpp_config(**dict(OPT, random_state=seed))
    model = CppGeoXGBRegressor(cfg)
    model.fit(X_tr, y_tr)
    pred_tr   = model.predict(X_tr)
    pred_val  = model.predict(X_val)
    resid_tr  = y_tr  - pred_tr
    resid_val = y_val - pred_val

    r2_base = r2_score(y_val, pred_val) / r2_ceil

    # 2. Fit Python HVRT for partition geometry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hvrt = HVRT(y_weight=0.5, random_state=seed)
        hvrt.fit(X_tr, y_tr)

    X_tr_z   = hvrt._to_z(X_tr)
    X_val_z  = hvrt._to_z(X_val)
    tr_pids  = hvrt.tree_.apply(X_tr_z)
    val_pids = hvrt.tree_.apply(X_val_z)

    sigma_global_tr = float(np.std(resid_tr))
    n_val = len(y_val)

    # 3. Find best split per partition; collect child statistics
    part_splits: dict[int, dict] = {}

    for pid in np.unique(tr_pids):
        m_tr = tr_pids == pid
        if m_tr.sum() < 2 * MIN_LEAF:
            continue
        X_p = X_tr_z[m_tr]
        r_p = resid_tr[m_tr]

        feat, thresh, gain = _best_split(X_p, r_p, MIN_LEAF)
        if feat == -1:
            continue

        goes_l = X_p[:, feat] <= thresh
        n_l = int(goes_l.sum())
        n_r = int((~goes_l).sum())
        if n_l < MIN_LEAF or n_r < MIN_LEAF:
            continue

        delta_l       = float(r_p[goes_l].mean())
        delta_r       = float(r_p[~goes_l].mean())
        sigma_l_tr    = float(np.std(r_p[goes_l]))
        sigma_r_tr    = float(np.std(r_p[~goes_l]))
        sigma_part_tr = float(np.std(r_p))

        # Val samples in this partition
        m_val = val_pids == pid
        if not m_val.any():
            continue

        X_val_p    = X_val_z[m_val]
        r_val_p    = resid_val[m_val]
        goes_l_val = X_val_p[:, feat] <= thresh

        n_l_val = int(goes_l_val.sum())
        n_r_val = int((~goes_l_val).sum())
        sigma_l_val = float(np.std(r_val_p[goes_l_val]))   if n_l_val >= 2 else float("nan")
        sigma_r_val = float(np.std(r_val_p[~goes_l_val]))  if n_r_val >= 2 else float("nan")

        part_splits[int(pid)] = dict(
            feat          = feat,
            thresh        = thresh,
            gain          = gain,
            delta_l       = delta_l,
            delta_r       = delta_r,
            n_l           = n_l,
            n_r           = n_r,
            sigma_l_tr    = sigma_l_tr,
            sigma_r_tr    = sigma_r_tr,
            sigma_part_tr = sigma_part_tr,
            sigma_l_val   = sigma_l_val,
            sigma_r_val   = sigma_r_val,
            m_val         = m_val,
            goes_l_val    = goes_l_val,
            r_val_p       = r_val_p,
        )

    # ── Strategy A: shrinkage sweep ─────────────────────────────────────────────
    # Track which val samples are covered by a valid split
    covered = np.zeros(n_val, dtype=bool)
    for s in part_splits.values():
        covered[s["m_val"]] = True

    strat_a: dict[int, dict] = {}
    for lam in LAMBDAS:
        corrections = np.zeros(n_val)
        for s in part_splits.values():
            dl = s["delta_l"] * s["n_l"] / (s["n_l"] + lam)
            dr = s["delta_r"] * s["n_r"] / (s["n_r"] + lam)
            corr = np.where(s["goes_l_val"], dl, dr)
            corrections[s["m_val"]] = corr

        r2_lam = r2_score(y_val, pred_val + corrections) / r2_ceil
        n_cov  = int(covered.sum())
        mean_abs = float(np.mean(np.abs(corrections[covered]))) if n_cov > 0 else 0.0

        strat_a[lam] = dict(
            delta_r2  = r2_lam - r2_base,
            mean_abs  = mean_abs,
        )

    # ── Strategy B: heteroscedastic intervals ───────────────────────────────────

    # Calibration pairs: (sigma_child_tr, sigma_child_val) for each child
    cal_pairs: list[tuple[float, float]] = []

    # Sigma assignments for covered val samples
    sigma_child_arr  = np.zeros(n_val)
    sigma_part_arr   = np.zeros(n_val)
    sigma_global_arr = np.full(n_val, sigma_global_tr)

    for s in part_splits.values():
        m_val = s["m_val"]
        sigma_child = np.where(s["goes_l_val"], s["sigma_l_tr"], s["sigma_r_tr"])
        sigma_child_arr[m_val] = sigma_child
        sigma_part_arr[m_val]  = s["sigma_part_tr"]

        if not np.isnan(s["sigma_l_val"]):
            cal_pairs.append((s["sigma_l_tr"], s["sigma_l_val"]))
        if not np.isnan(s["sigma_r_val"]):
            cal_pairs.append((s["sigma_r_tr"], s["sigma_r_val"]))

    # Test 1 — Calibration
    cal_spearman = float("nan")
    cal_p        = float("nan")
    n_pairs      = len(cal_pairs)
    if n_pairs >= 3:
        tr_arr  = np.array([x[0] for x in cal_pairs])
        val_arr = np.array([x[1] for x in cal_pairs])
        res = spearmanr(tr_arr, val_arr)
        cal_spearman = float(res.statistic)
        cal_p        = float(res.pvalue)

    # Test 2 — Coverage (restricted to covered val samples for fair comparison)
    def _cov_width(sigma_arr: np.ndarray, z: float) -> tuple[float, float]:
        mask = covered & (sigma_arr > 0)
        if mask.sum() == 0:
            return float("nan"), float("nan")
        cov   = float(np.mean(np.abs(resid_val[mask]) <= z * sigma_arr[mask]))
        width = float(2 * z * np.mean(sigma_arr[mask]))
        return cov, width

    cov_child_90,  w_child_90  = _cov_width(sigma_child_arr,  Z_90)
    cov_child_95,  _c95        = _cov_width(sigma_child_arr,  Z_95)
    cov_part_90,   w_part_90   = _cov_width(sigma_part_arr,   Z_90)
    cov_part_95,   _p95        = _cov_width(sigma_part_arr,   Z_95)
    cov_global_90, w_global_90 = _cov_width(sigma_global_arr, Z_90)
    cov_global_95, _g95        = _cov_width(sigma_global_arr, Z_95)

    # Test 3 — Sharpness
    sharp_spearman = float("nan")
    sharp_p        = float("nan")
    sharp_mask = covered & (sigma_child_arr > 0)
    if sharp_mask.sum() >= 3:
        res3 = spearmanr(sigma_child_arr[sharp_mask], np.abs(resid_val[sharp_mask]))
        sharp_spearman = float(res3.statistic)
        sharp_p        = float(res3.pvalue)

    return dict(
        r2_base       = r2_base,
        n_splits      = len(part_splits),
        n_covered_val = int(covered.sum()),
        # Strategy A
        strat_a       = strat_a,
        # Strategy B
        cal_spearman  = cal_spearman,
        cal_p         = cal_p,
        n_pairs       = n_pairs,
        cov_child_90  = cov_child_90,
        cov_child_95  = cov_child_95,
        w_child_90    = w_child_90,
        cov_part_90   = cov_part_90,
        cov_part_95   = cov_part_95,
        w_part_90     = w_part_90,
        cov_global_90 = cov_global_90,
        cov_global_95 = cov_global_95,
        w_global_90   = w_global_90,
        sharp_spearman = sharp_spearman,
        sharp_p        = sharp_p,
    )


# ── Aggregation helper ──────────────────────────────────────────────────────────

def _nanmean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if not np.isnan(float(v))])
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _fmt(v: float, fmt: str = ".4f", na: str = "   n/a") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    datasets = _make_datasets()
    X, y, sigma_noise, r2_ceil = datasets[DATASET]
    n, d = X.shape

    print(f"\n{'='*70}")
    print(f"Dual Strategy Exploration — {DATASET}")
    print(f"  n={n}  d={d}  noise_sigma={sigma_noise}  R²_ceil={r2_ceil:.4f}")
    print(f"  {N_SEEDS} seeds × {N_FOLDS} folds = {N_SEEDS*N_FOLDS} folds total")
    print(f"  MIN_LEAF={MIN_LEAF}  (no GAIN_THRESHOLD — use every valid split)")
    print(f"{'='*70}\n")

    fold_results: list[dict] = []
    t_start = time.perf_counter()

    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  seed={seed}  fold={fold_i+1}/{N_FOLDS} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                res = _run_fold(
                    X[tr_idx], y[tr_idx],
                    X[val_idx], y[val_idx],
                    r2_ceil, seed,
                )
                fold_results.append(res)
                elapsed = time.perf_counter() - t0
                print(
                    f"  n_splits={res['n_splits']}  "
                    f"cov={res['n_covered_val']}  "
                    f"ΔR²(λ=0)={res['strat_a'][0]['delta_r2']:+.4f}  "
                    f"cal_ρ={_fmt(res['cal_spearman'], '+.3f')}  "
                    f"({elapsed:.0f}s)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")

    n_folds = len(fold_results)
    elapsed_total = time.perf_counter() - t_start
    print(f"\n  [{n_folds} folds completed in {elapsed_total/60:.1f} min]\n")

    if not fold_results:
        print("No results — exiting.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Part A — Shrinkage Sweep
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Part A — Shrinkage Sweep")
    print(f"  delta_shrunk = delta * n_child / (n_child + λ)")
    print(f"  λ=0 → raw correction; large λ → correction→0 (baseline)\n")

    hdr = f"  {'lambda':>7}  {'mean ΔR²_adj':>14}  {'std':>8}  {'% folds +ve':>12}  {'mean |corr|':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    first_nonneg_lam = None
    for lam in LAMBDAS:
        dr2_vals = [r["strat_a"][lam]["delta_r2"] for r in fold_results]
        abs_vals = [r["strat_a"][lam]["mean_abs"]  for r in fold_results]
        dr2_m, dr2_s = _nanmean_std(dr2_vals)
        abs_m, _     = _nanmean_std(abs_vals)
        pct_pos      = 100.0 * sum(v > 0 for v in dr2_vals) / len(dr2_vals)

        if first_nonneg_lam is None and dr2_m >= 0:
            first_nonneg_lam = lam

        print(
            f"  {lam:>7}  {dr2_m:>+14.4f}  {dr2_s:>8.4f}"
            f"  {pct_pos:>11.1f}%  {abs_m:>12.4f}"
        )

    print()
    if first_nonneg_lam is not None:
        print(f"  First λ where mean ΔR²_adj ≥ 0: λ = {first_nonneg_lam}")
    else:
        print(f"  No λ in sweep achieves mean ΔR²_adj ≥ 0.")

    # ──────────────────────────────────────────────────────────────────────────
    # Part B — Heteroscedastic Intervals
    # ──────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Part B — Heteroscedastic Intervals")
    print(f"  sigma_child = std(resid_child_train); evaluated on val samples")
    print()

    # Test 1 — Calibration
    cal_rho_vals = [r["cal_spearman"] for r in fold_results]
    cal_p_vals   = [r["cal_p"]        for r in fold_results]
    n_pairs_vals = [r["n_pairs"]      for r in fold_results]
    cal_rho_m, cal_rho_s = _nanmean_std(cal_rho_vals)
    cal_p_m, _           = _nanmean_std(cal_p_vals)
    n_pairs_m, _         = _nanmean_std(n_pairs_vals)

    print("  Test 1 — Calibration (Spearman sigma_child_train vs sigma_child_val"
          " across all children):")
    print(f"    mean ρ: {_fmt(cal_rho_m, '+.3f')} ± {_fmt(cal_rho_s, '.3f')}"
          f"  (mean p={_fmt(cal_p_m, '.3f')}  n_children/fold ≈ {n_pairs_m:.0f})")

    # Test 2 — Coverage
    def _agg_cov(key: str) -> str:
        vals = [r[key] for r in fold_results]
        m, _ = _nanmean_std(vals)
        return f"{100*m:.1f}%" if not np.isnan(m) else "  n/a"

    def _agg_width(key: str) -> str:
        vals = [r[key] for r in fold_results]
        m, _ = _nanmean_std(vals)
        return _fmt(m, ".4f") if not np.isnan(m) else "  n/a"

    print()
    print("  Test 2 — Coverage (aggregated over 20 folds)")
    print(f"  {'Method':<18} | {'90% cov':>8} | {'95% cov':>8} | {'mean width (90%)':>16}")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*16}")
    for method, k90, k95, kw in [
        ("child sigma",    "cov_child_90",  "cov_child_95",  "w_child_90"),
        ("partition sigma","cov_part_90",   "cov_part_95",   "w_part_90"),
        ("global sigma",   "cov_global_90", "cov_global_95", "w_global_90"),
    ]:
        print(
            f"  {method:<18} | {_agg_cov(k90):>8} | {_agg_cov(k95):>8}"
            f" | {_agg_width(kw):>16}"
        )

    # Test 3 — Sharpness
    sh_rho_vals = [r["sharp_spearman"] for r in fold_results]
    sh_p_vals   = [r["sharp_p"]        for r in fold_results]
    sh_rho_m, sh_rho_s = _nanmean_std(sh_rho_vals)
    sh_p_m, _          = _nanmean_std(sh_p_vals)

    print()
    print("  Test 3 — Sharpness: Spearman(sigma_child_assigned, |resid_val|)")
    print(f"    mean ρ: {_fmt(sh_rho_m, '+.3f')} ± {_fmt(sh_rho_s, '.3f')}"
          f"  (mean p={_fmt(sh_p_m, '.3f')})")

    # ──────────────────────────────────────────────────────────────────────────
    # Joint interpretation
    # ──────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Joint Interpretation")
    print()

    # Gather key values for decisions
    dr2_raw_m, _ = _nanmean_std([r["strat_a"][0]["delta_r2"]   for r in fold_results])
    dr2_best_m   = max(
        (_nanmean_std([r["strat_a"][lam]["delta_r2"] for r in fold_results])[0]
         for lam in LAMBDAS),
        default=float("nan"),
    )
    a_wins = not np.isnan(dr2_best_m) and dr2_best_m > 0
    b_calibrated = not np.isnan(cal_rho_m) and cal_rho_m > 0.3 and not np.isnan(cal_p_m) and cal_p_m < 0.05
    b_sharp      = not np.isnan(sh_rho_m)  and sh_rho_m > 0.1

    best_lam_str = (
        f"λ={first_nonneg_lam}" if first_nonneg_lam is not None
        else "no λ in [0…1000]"
    )

    print(f"  Strategy A (shrinkage):")
    print(f"    Raw (λ=0) ΔR²_adj: {_fmt(dr2_raw_m, '+.4f')}")
    if a_wins:
        print(f"    Best mean ΔR²_adj across sweep: {_fmt(dr2_best_m, '+.4f')}")
        print(f"    Break-even first achieved at {best_lam_str}.")
        print(f"    VERDICT: Shrinkage CAN rescue point correction.")
        print(f"    → Re-run surgery_test with delta_shrunk at this λ.")
    else:
        print(f"    Best mean ΔR²_adj across sweep: {_fmt(dr2_best_m, '+.4f')} (≤ 0)")
        print(f"    VERDICT: Shrinkage does NOT rescue point correction.")
        print(f"    Even with optimal λ, mean corrections remain harmful.")

    print()
    print(f"  Strategy B (heteroscedastic intervals):")
    print(f"    Calibration ρ: {_fmt(cal_rho_m, '+.3f')}  "
          f"({'significant' if b_calibrated else 'not significant or weak'})")
    print(f"    Sharpness   ρ: {_fmt(sh_rho_m,  '+.3f')}  "
          f"({'informative' if b_sharp else 'not informative'})")
    if b_calibrated and b_sharp:
        print(f"    VERDICT: Partition-split sigmas ARE calibrated and sharp.")
        print(f"    → Pivot to UQ framing: expose sigma_child as prediction interval.")
        print(f"    → This enables conformal / heteroscedastic interval prediction")
        print(f"       without injecting point-correction noise.")
    elif b_calibrated:
        print(f"    VERDICT: Sigmas calibrate but don't rank-order errors well.")
        print(f"    → Interval widths correct in expectation; point ranking unreliable.")
        print(f"    → Cautious UQ framing still preferred over point correction.")
    elif b_sharp:
        print(f"    VERDICT: Sigmas rank-order errors but are miscalibrated.")
        print(f"    → Need rescaling (e.g. conformal calibration) before use.")
    else:
        print(f"    VERDICT: Training-child sigmas do NOT generalise to val.")
        print(f"    → Partition sizes too small to estimate variance reliably.")
        print(f"    → Neither point correction nor UQ is exploitable at this scale.")

    print()
    print("  Recommended next step:")
    if a_wins and b_calibrated:
        print("    Both strategies show promise. Prefer B (UQ) as it is lower-variance")
        print("    and avoids the sign-consistency problem entirely.")
    elif a_wins:
        print("    Tune λ in surgery_test and recheck sign consistency.")
        print("    If sign consistency improves with shrinkage, proceed to C++ integration.")
    elif b_calibrated or b_sharp:
        print("    Pivot to UQ framing. Expose per-child sigma estimates via the")
        print("    C++ layer for use in conformal prediction intervals downstream.")
    else:
        print("    Residual structure exists (CV gain > 0) but is not exploitable")
        print("    at current partition sizes. Options:")
        print("      (a) Increase n_samples or reduce n_partitions to grow partitions.")
        print("      (b) Use a different partition scheme (e.g. finer HVRT).")
        print("      (c) Accept that the residuals are at the noise floor and stop.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
