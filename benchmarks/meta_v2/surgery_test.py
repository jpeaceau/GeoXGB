"""GeoXGB Surgical Residual Split — Empirical Validation
=======================================================
Hypothesis: within high-T partitions, a single variance-minimising split in
z-space captures exploitable signal from GeoXGB training residuals.

Method
------
For each train/val fold:
  1. Fit CppGeoXGBRegressor (optimised config from meta-analysis).
  2. Fit Python HVRT on the same training data to expose partition geometry.
  3. For each training partition, search exhaustively for the z-space split that
     minimises within-child residual variance (gain = 1 - var_after/var_before).
  4. Apply the per-child mean correction to val predictions.
  5. Record R²_adj before/after, per-partition gain vs T_value correlations, and
     what fraction of eligible partitions actually helped on val.

Success criteria (proceed to C++ integration if met):
  - R²_adj improves > 0.002 on reg_sparse or reg_large
  - Spearman(gain, T_value) > 0  (T predicts where surgery is beneficial)
  - Majority of eligible partitions help on val (no overfitting)

Run from benchmarks/meta_v2/:
    python surgery_test.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# ── local imports ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                       # for meta_reg
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))     # for geoxgb

from meta_reg import _make_datasets  # noqa: E402
from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor  # noqa: E402
from hvrt import HVRT  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

# Optimised config from meta-analysis phases 2–5
OPT = dict(
    n_rounds              = 3000,
    learning_rate         = 0.02,
    max_depth             = 2,
    min_samples_leaf      = 5,
    reduce_ratio          = 0.7,
    expand_ratio          = 0.1,
    y_weight              = 0.5,
    refit_interval        = 5,
    auto_noise            = False,
    noise_guard           = False,
    refit_noise_floor     = 0.05,
    auto_expand           = True,
    min_train_samples     = 5000,
    bandwidth             = "auto",
    variance_weighted     = True,
    hvrt_min_samples_leaf = -1,
    n_partitions          = -1,
    n_bins                = 64,
)

GAIN_THRESHOLD = 0.01   # 1% variance reduction to qualify for surgery
MIN_LEAF       = 5      # both children must have >= this many samples
N_SEEDS        = 5
N_FOLDS        = 4

# ── Core helpers ───────────────────────────────────────────────────────────────

def _best_split(
    X_part_z: np.ndarray,
    resid_part: np.ndarray,
    min_leaf: int,
) -> tuple[int, float | None, float]:
    """
    Exhaustive variance-minimising split search over z-space features.

    Tries 9 percentile thresholds (10th–90th step 10) for each feature.
    Returns (feat_idx, threshold, gain_ratio) where
        gain_ratio = 1 - var_after / var_before
    Returns (-1, None, 0.0) if no valid split exists.
    """
    n, d = X_part_z.shape
    var_before = np.var(resid_part) * n   # total (unnormalised) variance
    if var_before < 1e-14:
        return -1, None, 0.0

    best_gain, best_feat, best_thresh = 0.0, -1, None
    percentiles = np.arange(10, 100, 10)   # [10, 20, …, 90]

    for feat in range(d):
        vals = X_part_z[:, feat]
        for thresh in np.unique(np.percentile(vals, percentiles)):
            left  = resid_part[vals <= thresh]
            right = resid_part[vals >  thresh]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            var_after = (
                np.var(left)  * len(left) +
                np.var(right) * len(right)
            )
            gain = 1.0 - var_after / var_before
            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain


def _run_fold(
    X_tr:   np.ndarray,
    y_tr:   np.ndarray,
    X_val:  np.ndarray,
    y_val:  np.ndarray,
    r2_ceil: float,
    seed:   int,
) -> dict:
    """Run one train/val fold; return per-fold surgery statistics."""

    # ── 1. Fit GeoXGB ──────────────────────────────────────────────────────────
    cfg   = make_cpp_config(**dict(OPT, random_state=seed))
    model = CppGeoXGBRegressor(cfg)
    model.fit(X_tr, y_tr)

    pred_tr  = model.predict(X_tr)
    pred_val = model.predict(X_val)
    resid_tr = y_tr - pred_tr

    r2_before = r2_score(y_val, pred_val) / r2_ceil

    # ── 2. Fit Python HVRT for partition geometry ───────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hvrt = HVRT(y_weight=0.5, random_state=seed)
        hvrt.fit(X_tr, y_tr)

    X_tr_z  = hvrt._to_z(X_tr)
    tr_pids = hvrt.tree_.apply(X_tr_z)   # leaf IDs for training samples

    geom_raw = hvrt.geometry_stats()
    geom = {pg["id"]: pg for pg in geom_raw.get("partitions", [])}

    # ── 3. Surgery — find best split per qualifying partition ──────────────────
    surgery: dict[int, dict] = {}
    for pid in np.unique(tr_pids):
        mask = tr_pids == pid
        if mask.sum() < 2 * MIN_LEAF:
            continue
        X_p = X_tr_z[mask]
        r_p = resid_tr[mask]

        feat, thresh, gain = _best_split(X_p, r_p, MIN_LEAF)
        if gain < GAIN_THRESHOLD:
            continue

        goes_left = X_p[:, feat] <= thresh
        surgery[int(pid)] = dict(
            feat    = feat,
            thresh  = thresh,
            gain    = gain,
            delta_l = float(r_p[goes_left].mean()),
            delta_r = float(r_p[~goes_left].mean()),
            T_value = float(geom.get(pid, {}).get("E_T", float("nan"))),
            n       = int(mask.sum()),
        )

    # ── 4. Apply corrections to validation set ─────────────────────────────────
    X_val_z  = hvrt._to_z(X_val)
    val_pids = hvrt.tree_.apply(X_val_z)
    corrections = np.zeros(len(X_val))

    # Track per-partition: (train gain, val ΔR²) for correlation analysis
    pid_gain_dr2: list[tuple[float, float]] = []

    for pid, s in surgery.items():
        mask_v = val_pids == pid
        if not mask_v.any():
            continue
        goes_left_v = X_val_z[mask_v, s["feat"]] <= s["thresh"]
        corr = np.where(goes_left_v, s["delta_l"], s["delta_r"])
        corrections[mask_v] = corr

        # Per-partition R² contribution (need ≥ 2 val samples for r2_score)
        y_true_p   = y_val[mask_v]
        if len(y_true_p) < 2:
            continue
        pred_val_p = pred_val[mask_v]
        dr2_p = (
            r2_score(y_true_p, pred_val_p + corr) -
            r2_score(y_true_p, pred_val_p)
        )
        pid_gain_dr2.append((s["gain"], dr2_p))

    y_pred_corrected = pred_val + corrections
    r2_after = r2_score(y_val, y_pred_corrected) / r2_ceil

    # ── 5. Aggregate statistics ────────────────────────────────────────────────
    n_total_partitions = int(len(np.unique(tr_pids)))
    n_eligible         = len(surgery)
    gains   = [s["gain"]    for s in surgery.values()]
    t_vals  = [s["T_value"] for s in surgery.values()]
    t_all   = [
        float(geom.get(int(pid), {}).get("E_T", float("nan")))
        for pid in np.unique(tr_pids)
    ]

    # Spearman(gain, T_value) over eligible partitions with valid T
    valid_mask = [not np.isnan(t) for t in t_vals]
    if sum(valid_mask) >= 3:
        g_valid = [g for g, m in zip(gains, valid_mask) if m]
        t_valid = [t for t, m in zip(t_vals, valid_mask) if m]
        corr_gain_T, p_gain_T = spearmanr(g_valid, t_valid)
    else:
        corr_gain_T, p_gain_T = float("nan"), float("nan")

    # Spearman(ΔR²_val, gain) over partitions present in val
    if len(pid_gain_dr2) >= 3:
        gains_v, dr2_v = zip(*pid_gain_dr2)
        corr_dr2_gain, p_dr2_gain = spearmanr(gains_v, dr2_v)
    else:
        corr_dr2_gain, p_dr2_gain = float("nan"), float("nan")

    n_helped = sum(1 for _, dr2 in pid_gain_dr2 if dr2 > 0)
    n_hurt   = sum(1 for _, dr2 in pid_gain_dr2 if dr2 < 0)

    return dict(
        r2_before          = r2_before,
        r2_after           = r2_after,
        delta_r2           = r2_after - r2_before,
        n_total_partitions = n_total_partitions,
        n_eligible         = n_eligible,
        mean_gain          = float(np.mean(gains)) if gains else 0.0,
        mean_T_eligible    = float(np.nanmean(t_vals)) if t_vals else float("nan"),
        mean_T_all         = float(np.nanmean(t_all)),
        corr_gain_T        = corr_gain_T,
        p_gain_T           = p_gain_T,
        corr_dr2_gain      = corr_dr2_gain,
        p_dr2_gain         = p_dr2_gain,
        n_helped           = n_helped,
        n_hurt             = n_hurt,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("GeoXGB Surgical Residual Split — Empirical Validation")
    print(f"  GAIN_THRESHOLD={GAIN_THRESHOLD}  MIN_LEAF={MIN_LEAF}  "
          f"N_SEEDS={N_SEEDS}  N_FOLDS={N_FOLDS}")

    datasets = _make_datasets()
    t_start  = time.perf_counter()

    # ── summary table: dataset -> pass/fail per criterion ─────────────────────
    summary: dict[str, dict] = {}

    for ds_name, (X, y, sigma, r2_ceil) in datasets.items():
        n, d = X.shape
        print(f"\n{'='*62}")
        print(f"=== {ds_name}  (n={n}, d={d}, sigma={sigma})  "
              f"R²_ceil={r2_ceil:.4f} ===")

        fold_results: list[dict] = []

        for seed in range(N_SEEDS):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
                print(f"  seed={seed}  fold={fold_idx + 1}/{N_FOLDS} ... ",
                      end="", flush=True)
                t0 = time.perf_counter()
                try:
                    result = _run_fold(
                        X[tr_idx], y[tr_idx],
                        X[val_idx], y[val_idx],
                        r2_ceil, seed,
                    )
                    fold_results.append(result)
                    elapsed = time.perf_counter() - t0
                    print(
                        f"ΔR²_adj={result['delta_r2']:+.4f}  "
                        f"eligible={result['n_eligible']}/"
                        f"{result['n_total_partitions']}  "
                        f"({elapsed:.0f}s)"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"ERROR: {exc}")

        if not fold_results:
            print("  No results — skipping.")
            continue

        # ── aggregate over seeds × folds ───────────────────────────────────────
        def _agg(key: str) -> tuple[float, float]:
            vals = [
                r[key] for r in fold_results
                if not np.isnan(float(r.get(key, float("nan"))))
            ]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        r2b_m,  r2b_s  = _agg("r2_before")
        r2a_m,  r2a_s  = _agg("r2_after")
        dr2_m,  dr2_s  = _agg("delta_r2")
        gain_m, _      = _agg("mean_gain")
        T_elig_m, _    = _agg("mean_T_eligible")
        T_all_m,  _    = _agg("mean_T_all")

        n_elig_vals  = [r["n_eligible"]        for r in fold_results]
        n_total_vals = [r["n_total_partitions"] for r in fold_results]

        n_helped_tot = sum(r["n_helped"] for r in fold_results)
        n_hurt_tot   = sum(r["n_hurt"]   for r in fold_results)

        corrs_gT = [
            r["corr_gain_T"] for r in fold_results
            if not np.isnan(float(r["corr_gain_T"]))
        ]
        ps_gT = [
            r["p_gain_T"] for r in fold_results
            if not np.isnan(float(r["p_gain_T"]))
        ]
        corrs_dg = [
            r["corr_dr2_gain"] for r in fold_results
            if not np.isnan(float(r["corr_dr2_gain"]))
        ]

        print()
        print(f"  R²_adj before surgery  : {r2b_m:.4f} ± {r2b_s:.4f}")
        print(f"  R²_adj after  surgery  : {r2a_m:.4f} ± {r2a_s:.4f}"
              f"   (Δ = {dr2_m:+.4f} ± {dr2_s:.4f})")
        print(f"  Eligible partitions    : "
              f"{np.mean(n_elig_vals):.1f} / {np.mean(n_total_vals):.1f}"
              f"  (gain ≥ {GAIN_THRESHOLD})")
        print(f"  Mean gain (eligible)   : {gain_m:.4f}")
        print(f"  Mean T_value: eligible={T_elig_m:.4f}  all={T_all_m:.4f}")

        if corrs_gT:
            print(f"  Spearman(gain, T_value): {np.mean(corrs_gT):+.3f}"
                  f"  (mean p={np.mean(ps_gT):.3f}  n_folds={len(corrs_gT)})")
        else:
            print("  Spearman(gain, T_value): n/a (too few eligible partitions)")

        if corrs_dg:
            print(f"  Spearman(ΔR²_val, gain): {np.mean(corrs_dg):+.3f}"
                  f"  (per-partition contribution  n_folds={len(corrs_dg)})")
        else:
            print("  Spearman(ΔR²_val, gain): n/a")

        n_decided = n_helped_tot + n_hurt_tot
        if n_decided > 0:
            pct = 100 * n_helped_tot / n_decided
            print(f"  Surgery on val: {n_helped_tot}/{n_decided} eligible partitions "
                  f"helped ({pct:.0f}%);  {n_hurt_tot} hurt")
        else:
            print("  Surgery on val: no eligible partitions reached val set")

        # ── success criteria ───────────────────────────────────────────────────
        c1 = dr2_m > 0.002
        c2 = bool(corrs_gT and np.mean(corrs_gT) > 0)
        c3 = n_helped_tot > n_hurt_tot if n_decided > 0 else False

        corr_gT_str = f"{np.mean(corrs_gT):.3f}" if corrs_gT else "n/a"
        print("\n  [Success criteria]")
        print(f"    ΔR²_adj > 0.002:           {'PASS' if c1 else 'FAIL'}"
              f"  ({dr2_m:+.4f})")
        print(f"    Spearman(gain, T) > 0:     {'PASS' if c2 else 'FAIL'}"
              f"  ({corr_gT_str})")
        print(f"    Majority partitions helped:{'PASS' if c3 else 'FAIL'}"
              f"  ({n_helped_tot} helped, {n_hurt_tot} hurt)")

        summary[ds_name] = dict(
            dr2=dr2_m, c1=c1, c2=c2, c3=c3,
            all_pass=c1 and c2 and c3,
        )

    # ── final summary ──────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    print(f"\n{'='*62}")
    print(f"SUMMARY  (total elapsed: {elapsed_total/60:.1f} min)")
    print(f"{'Dataset':<14} {'ΔR²_adj':>9} {'C1':>5} {'C2':>5} {'C3':>5} {'All':>5}")
    print("-" * 44)
    for ds, s in summary.items():
        print(
            f"  {ds:<12} {s['dr2']:>+9.4f} "
            f"{'Y' if s['c1'] else 'N':>5} "
            f"{'Y' if s['c2'] else 'N':>5} "
            f"{'Y' if s['c3'] else 'N':>5} "
            f"{'Y' if s['all_pass'] else 'N':>5}"
        )

    any_pass = any(s["all_pass"] for s in summary.values())
    print()
    if any_pass:
        print("OUTCOME: At least one dataset passed all criteria.")
        print("         -> Proceed to C++ integration of surgical split.")
    else:
        c1_any = any(s["c1"] for s in summary.values())
        c2_any = any(s["c2"] for s in summary.values())
        if not c1_any:
            print("OUTCOME: ΔR²_adj never exceeded 0.002.")
            print("         -> Residuals near noise floor; no exploitable signal.")
        elif not c2_any:
            print("OUTCOME: Spearman(gain, T) ≤ 0.")
            print("         -> T-value is not a useful surgery predictor.")
        else:
            print("OUTCOME: Surgery helps on train but not on val.")
            print("         -> Corrections overfit to seed-specific residuals.")
    print("Done.")


if __name__ == "__main__":
    main()
