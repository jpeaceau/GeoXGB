"""
Friedman1 Residual Deep Diagnostic
====================================
Investigates why single-split surgery hurt friedman1 (Δ = -0.0048).

One canonical fold (seed=0, fold=0) for detailed per-partition tables;
then N_SEEDS × N_FOLDS aggregate statistics to confirm findings are stable.

Key questions answered
----------------------
Q1. Per-partition residual profile: n, std, skew, effective SNR.
Q2. Z-space feature structure: Spearman(z_feat, resid) and Spearman(z_feat, |resid|).
Q3. Within-partition CV gain: split partition 70/30 — does the best split generalise?
Q4. Sign consistency: when delta_l was applied, was the sign correct on val?
Q5. Is T_value predictive of any of the above?

Run from benchmarks/meta_v2/:
    python residual_diag.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr, skew as scipy_skew
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from meta_reg import _make_datasets                            # noqa: E402
from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor  # noqa: E402
from hvrt import HVRT                                          # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET   = "friedman1"
N_SEEDS   = 5
N_FOLDS   = 4
CANON_SEED, CANON_FOLD = 0, 0    # fold used for detailed per-partition table

OPT = dict(
    n_rounds=3000, learning_rate=0.02, max_depth=2, min_samples_leaf=5,
    reduce_ratio=0.7, expand_ratio=0.1, y_weight=0.5, refit_interval=5,
    auto_noise=False, noise_guard=False, refit_noise_floor=0.05,
    auto_expand=True, min_train_samples=5000, bandwidth="auto",
    variance_weighted=True, hvrt_min_samples_leaf=-1, n_partitions=-1, n_bins=64,
)

CV_SPLIT  = 0.70   # within-partition CV: fraction used to find split
MIN_LEAF  = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_split_on(X_z: np.ndarray, resid: np.ndarray, min_leaf: int):
    """Return (feat, thresh, gain_ratio) for the best variance-min split."""
    n, d = X_z.shape
    var_before = np.var(resid) * n
    if var_before < 1e-14:
        return -1, None, 0.0
    best_gain, best_feat, best_thresh = 0.0, -1, None
    for feat in range(d):
        vals = X_z[:, feat]
        for thresh in np.unique(np.percentile(vals, np.arange(10, 100, 10))):
            l, r = resid[vals <= thresh], resid[vals > thresh]
            if len(l) < min_leaf or len(r) < min_leaf:
                continue
            gain = 1.0 - (np.var(l) * len(l) + np.var(r) * len(r)) / var_before
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_feat, best_thresh, best_gain


def _cv_gain(X_z: np.ndarray, resid: np.ndarray, cv_split: float, min_leaf: int, rng):
    """
    Within-partition cross-validated gain.
    Find best split on a random cv_split fraction; evaluate on the remainder.
    Returns (gain_train, gain_test, feat_used).
    gain_test can be negative (split hurts on test half).
    """
    n = len(resid)
    idx = rng.permutation(n)
    n_tr = max(2 * min_leaf, int(np.ceil(n * cv_split)))
    if n - n_tr < min_leaf:
        return 0.0, float("nan"), -1
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    feat, thresh, gain_tr = _best_split_on(X_z[tr_idx], resid[tr_idx], min_leaf)
    if feat == -1:
        return 0.0, float("nan"), -1
    # Evaluate same split on test half
    var_before_te = np.var(resid[te_idx]) * len(te_idx)
    if var_before_te < 1e-14:
        return gain_tr, 0.0, feat
    vals_te = X_z[te_idx, feat]
    l_te = resid[te_idx][vals_te <= thresh]
    r_te = resid[te_idx][vals_te >  thresh]
    if len(l_te) < 1 or len(r_te) < 1:
        return gain_tr, float("nan"), feat
    gain_te = 1.0 - (np.var(l_te) * len(l_te) + np.var(r_te) * len(r_te)) / var_before_te
    return gain_tr, gain_te, feat


def _sign_consistency(delta: float, val_resid_child: np.ndarray) -> float:
    """
    Fraction of val samples in this child where the correction delta
    has the same sign as the actual val residual.
    """
    if len(val_resid_child) == 0:
        return float("nan")
    return float(np.mean(np.sign(val_resid_child) == np.sign(delta)))


def _fit_and_diagnose(
    X_tr, y_tr, X_val, y_val, r2_ceil, seed, detailed: bool
) -> dict:
    """
    Fit GeoXGB + HVRT, compute per-partition diagnostics.
    If detailed=True, return per-partition rows for printing.
    Always returns aggregate statistics.
    """
    # ── Fit GeoXGB ─────────────────────────────────────────────────────────────
    cfg   = make_cpp_config(**dict(OPT, random_state=seed))
    model = CppGeoXGBRegressor(cfg)
    model.fit(X_tr, y_tr)
    pred_tr  = model.predict(X_tr)
    pred_val = model.predict(X_val)
    resid_tr = y_tr  - pred_tr
    resid_val= y_val - pred_val

    from sklearn.metrics import r2_score
    r2_before = r2_score(y_val, pred_val) / r2_ceil

    # ── Fit HVRT ───────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hvrt = HVRT(y_weight=0.5, random_state=seed)
        hvrt.fit(X_tr, y_tr)

    X_tr_z  = hvrt._to_z(X_tr)
    X_val_z = hvrt._to_z(X_val)
    tr_pids  = hvrt.tree_.apply(X_tr_z)
    val_pids = hvrt.tree_.apply(X_val_z)
    geom_raw = hvrt.geometry_stats()
    geom     = {pg["id"]: pg for pg in geom_raw.get("partitions", [])}

    n_tr_global  = len(resid_tr)
    std_global   = float(np.std(resid_tr))
    rng = np.random.RandomState(seed ^ 0xDEAD)

    rows = []           # per-partition diagnostic rows
    corrections = np.zeros(len(X_val))

    for pid in np.unique(tr_pids):
        m_tr  = tr_pids  == pid
        m_val = val_pids == pid
        X_p   = X_tr_z[m_tr]
        r_p   = resid_tr[m_tr]
        n_p   = len(r_p)
        n_val_p = int(m_val.sum())
        pg    = geom.get(int(pid), {})

        # ── Q1: residual profile ───────────────────────────────────────────────
        std_p   = float(np.std(r_p))
        mean_p  = float(np.mean(r_p))
        skew_p  = float(scipy_skew(r_p)) if n_p >= 4 else float("nan")
        # Effective SNR: ratio of partition residual std to the known noise sigma
        # (sigma ≈ 1.0 in original units; after normalisation, effective_noise_std
        #  = sigma / std(y_raw). We use global std_resid as a proxy for noise floor.)
        snr_p   = std_p / std_global if std_global > 0 else float("nan")

        # ── Q2: z-space feature correlations with resid and |resid| ───────────
        max_corr_resid, max_corr_absresid = 0.0, 0.0
        best_feat_resid, best_feat_abs    = -1, -1
        if n_p >= 6:
            for feat in range(X_p.shape[1]):
                c_r, _  = spearmanr(X_p[:, feat], r_p)
                c_a, _  = spearmanr(X_p[:, feat], np.abs(r_p))
                if abs(c_r) > abs(max_corr_resid):
                    max_corr_resid, best_feat_resid = c_r, feat
                if abs(c_a) > abs(max_corr_absresid):
                    max_corr_absresid, best_feat_abs = c_a, feat
        else:
            max_corr_resid = max_corr_absresid = float("nan")

        # ── Q3: within-partition CV gain ───────────────────────────────────────
        if n_p >= 2 * MIN_LEAF + MIN_LEAF:   # need enough for split + test
            gain_tr, gain_te, cv_feat = _cv_gain(X_p, r_p, CV_SPLIT, MIN_LEAF, rng)
        else:
            gain_tr, gain_te, cv_feat = 0.0, float("nan"), -1

        # ── Full split (for Q4 sign consistency) ──────────────────────────────
        feat, thresh, gain_full = _best_split_on(X_p, r_p, MIN_LEAF)
        sc_l = sc_r = float("nan")
        delta_l = delta_r = float("nan")
        corr_delta = float("nan")
        if feat >= 0 and n_val_p > 0:
            goes_l_tr = X_p[:, feat] <= thresh
            delta_l   = float(r_p[goes_l_tr].mean()) if goes_l_tr.sum() >= 1 else float("nan")
            delta_r   = float(r_p[~goes_l_tr].mean()) if (~goes_l_tr).sum() >= 1 else float("nan")

            X_val_p   = X_val_z[m_val]
            r_val_p   = resid_val[m_val]
            goes_l_val = X_val_p[:, feat] <= thresh

            # ── Q4: sign consistency ──────────────────────────────────────────
            sc_l = _sign_consistency(delta_l, r_val_p[goes_l_val])
            sc_r = _sign_consistency(delta_r, r_val_p[~goes_l_val])

            # Apply correction and track
            corr = np.where(goes_l_val, delta_l, delta_r)
            corrections[m_val] = corr
            # Correlation of correction vector with val residuals
            if len(r_val_p) >= 3:
                corr_delta, _ = spearmanr(corr, r_val_p)
            else:
                corr_delta = float("nan")

        row = dict(
            pid             = int(pid),
            n_tr            = n_p,
            n_val           = n_val_p,
            mean_resid      = mean_p,
            std_resid       = std_p,
            skew_resid      = skew_p,
            snr_vs_global   = snr_p,
            T_value         = float(pg.get("E_T", float("nan"))),
            frac_in_cone    = float(pg.get("frac_in_cone", float("nan"))),
            max_corr_resid  = max_corr_resid,
            best_feat_resid = best_feat_resid,
            max_corr_abs    = max_corr_absresid,
            best_feat_abs   = best_feat_abs,
            gain_train_full = gain_full,
            gain_cv_train   = gain_tr,
            gain_cv_test    = gain_te,
            cv_generalises  = bool(gain_te > 0) if not np.isnan(gain_te) else False,
            feat_used       = feat,
            delta_l         = delta_l,
            delta_r         = delta_r,
            sc_l            = sc_l,
            sc_r            = sc_r,
            corr_correction = corr_delta,
        )
        rows.append(row)

    # Overall after applying all corrections
    from sklearn.metrics import r2_score as _r2
    r2_after = _r2(y_val, pred_val + corrections) / r2_ceil

    return dict(
        r2_before = r2_before,
        r2_after  = r2_after,
        delta_r2  = r2_after - r2_before,
        rows      = rows if detailed else [],
        n_partitions = len(rows),
        # aggregate across partitions
        frac_cv_generalise = float(np.mean([
            r["cv_generalises"] for r in rows
            if not np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan")))
        ])) if rows else float("nan"),
        mean_gain_full = float(np.mean([r["gain_train_full"] for r in rows])),
        mean_gain_cv_te = float(np.nanmean([r["gain_cv_test"] for r in rows])),
        mean_sc_l = float(np.nanmean([r["sc_l"] for r in rows])),
        mean_sc_r = float(np.nanmean([r["sc_r"] for r in rows])),
        corr_T_stdresid = float(spearmanr(
            [r["T_value"]   for r in rows if not np.isnan(r["T_value"])],
            [r["std_resid"] for r in rows if not np.isnan(r["T_value"])],
        ).statistic) if sum(not np.isnan(r["T_value"]) for r in rows) >= 3 else float("nan"),
        corr_T_cvgain = float(spearmanr(
            [r["T_value"]    for r in rows if not np.isnan(r["T_value"]) and not np.isnan(float(r["gain_cv_test"] or float("nan")))],
            [r["gain_cv_test"] for r in rows if not np.isnan(r["T_value"]) and not np.isnan(float(r["gain_cv_test"] or float("nan")))],
        ).statistic) if sum(
            not np.isnan(r["T_value"]) and not np.isnan(float(r["gain_cv_test"] or float("nan")))
            for r in rows
        ) >= 3 else float("nan"),
    )


# ── Printing helpers ──────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f", na="  n/a"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return format(v, fmt)


def _print_partition_table(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["std_resid"], reverse=True)
    hdr = (
        f"{'pid':>5} {'n_tr':>5} {'n_val':>5} "
        f"{'mean_r':>8} {'std_r':>7} {'skew':>6} "
        f"{'T_val':>7} {'corr_r':>7} {'corr|r|':>8} "
        f"{'gain_full':>10} {'gain_cv_tr':>11} {'gain_cv_te':>11} "
        f"{'CV_ok':>6} {'sc_L':>6} {'sc_R':>6} {'corr_δ':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows_sorted:
        cv_ok = "Y" if r["cv_generalises"] else ("?" if np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan"))) else "N")
        print(
            f"{r['pid']:>5} {r['n_tr']:>5} {r['n_val']:>5} "
            f"{_fmt(r['mean_resid'], '+.4f'):>8} {_fmt(r['std_resid']):>7} "
            f"{_fmt(r['skew_resid'], '+.3f'):>6} "
            f"{_fmt(r['T_value'], '+.4f'):>7} "
            f"{_fmt(r['max_corr_resid'], '+.4f'):>7} "
            f"{_fmt(r['max_corr_abs'], '+.4f'):>8} "
            f"{_fmt(r['gain_train_full']):>10} "
            f"{_fmt(r['gain_cv_train']):>11} "
            f"{_fmt(r['gain_cv_test'], '.4f'):>11} "
            f"{cv_ok:>6} "
            f"{_fmt(r['sc_l'], '.3f'):>6} "
            f"{_fmt(r['sc_r'], '.3f'):>6} "
            f"{_fmt(r['corr_correction'], '+.3f'):>7}"
        )


def _print_feature_leaderboard(rows: list[dict], d: int) -> None:
    """Count how often each z-feature is the best predictor of resid / |resid|."""
    feat_resid = np.zeros(d, dtype=int)
    feat_abs   = np.zeros(d, dtype=int)
    for r in rows:
        if r["best_feat_resid"] >= 0:
            feat_resid[r["best_feat_resid"]] += 1
        if r["best_feat_abs"] >= 0:
            feat_abs[r["best_feat_abs"]] += 1
    print(f"  {'feat':>5}  {'#best_resid':>12}  {'#best_|resid|':>14}")
    for f in range(d):
        if feat_resid[f] > 0 or feat_abs[f] > 0:
            print(f"  z[{f:02d}]  {feat_resid[f]:>12}  {feat_abs[f]:>14}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    datasets = _make_datasets()
    X, y, sigma, r2_ceil = datasets[DATASET]
    n, d = X.shape
    print(f"\n{'='*70}")
    print(f"DATASET: {DATASET}  n={n}  d={d}  sigma={sigma}  "
          f"R²_ceil={r2_ceil:.4f}")
    print(f"Noise floor in R²_adj: {1.0 - r2_ceil:.4f}  "
          f"(irreducible from sigma)")
    print(f"{'='*70}\n")

    # ── PART 1: detailed canonical fold ───────────────────────────────────────
    kf_c = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CANON_SEED)
    splits_c = list(kf_c.split(X))
    tr_idx, val_idx = splits_c[CANON_FOLD]

    print(f"── PART 1: Canonical fold  (seed={CANON_SEED}, fold={CANON_FOLD}) ──\n")
    print(f"  Train: {len(tr_idx)}   Val: {len(val_idx)}\n")

    result_c = _fit_and_diagnose(
        X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
        r2_ceil, CANON_SEED, detailed=True,
    )
    print(f"  R²_adj before: {result_c['r2_before']:.4f}   "
          f"after surgery: {result_c['r2_after']:.4f}   "
          f"Δ = {result_c['delta_r2']:+.4f}\n")

    rows = result_c["rows"]

    # ── Q1+Q2+Q3+Q4 table ─────────────────────────────────────────────────────
    print("PER-PARTITION TABLE  (sorted by std_resid ↓)")
    print("Columns: pid | n_train | n_val | mean_resid | std_resid | skew | "
          "T_val | max_corr(resid) | max_corr(|resid|) | "
          "gain_full | gain_cv_train | gain_cv_test | CV_ok | "
          "sign_consistency_L | sign_consistency_R | corr(correction,val_resid)\n")
    _print_partition_table(rows)

    # ── Q2: feature leaderboard ───────────────────────────────────────────────
    print(f"\nFEATURE LEADERBOARD  (how often each z[i] is the best predictor)")
    _print_feature_leaderboard(rows, d)

    # ── Q3: CV generalisation summary ─────────────────────────────────────────
    cv_gains_te = [r["gain_cv_test"] for r in rows if not np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan")))]
    cv_gains_tr = [r["gain_cv_train"] for r in rows if not np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan")))]
    n_pos = sum(1 for g in cv_gains_te if g > 0)
    print(f"\nCV GENERALISATION SUMMARY")
    print(f"  Partitions with CV result:     {len(cv_gains_te)} / {len(rows)}")
    print(f"  CV gain > 0 (split generalises): {n_pos} / {len(cv_gains_te)} "
          f"({100*n_pos/len(cv_gains_te):.0f}%)" if cv_gains_te else "  (no CV results)")
    print(f"  Mean gain_train (70%):  {np.mean(cv_gains_tr):.4f}"
          if cv_gains_tr else "")
    print(f"  Mean gain_test  (30%):  {np.nanmean(cv_gains_te):.4f}"
          if cv_gains_te else "")
    if cv_gains_te and cv_gains_tr:
        print(f"  Gain shrinkage (te/tr): "
              f"{np.nanmean(cv_gains_te)/np.mean(cv_gains_tr):.3f}")

    # ── Q4: sign consistency summary ─────────────────────────────────────────
    sc_ls = [r["sc_l"] for r in rows if not np.isnan(float(r["sc_l"]))]
    sc_rs = [r["sc_r"] for r in rows if not np.isnan(float(r["sc_r"]))]
    corr_deltas = [r["corr_correction"] for r in rows if not np.isnan(float(r["corr_correction"]))]
    print(f"\nSIGN CONSISTENCY (val residuals agree with training correction direction)")
    print(f"  Mean sign_consistency left  child: {np.mean(sc_ls):.3f}"
          f"  (0.5=random, 1.0=perfect)"  if sc_ls else "  n/a")
    print(f"  Mean sign_consistency right child: {np.mean(sc_rs):.3f}"
          if sc_rs else "  n/a")
    print(f"  Mean Spearman(correction, val_resid): {np.nanmean(corr_deltas):+.3f}"
          if corr_deltas else "  n/a")

    # ── Q5: T_value predictors ────────────────────────────────────────────────
    print(f"\nT_VALUE CORRELATIONS (Spearman)")
    T_vals   = [r["T_value"]          for r in rows if not np.isnan(r["T_value"])]
    stds     = [r["std_resid"]        for r in rows if not np.isnan(r["T_value"])]
    gains_f  = [r["gain_train_full"]  for r in rows if not np.isnan(r["T_value"])]
    sc_all   = [(r["sc_l"]+r["sc_r"])/2 for r in rows
                if not np.isnan(r["T_value"]) and not np.isnan(r["sc_l"])]
    T_for_sc = [r["T_value"] for r in rows
                if not np.isnan(r["T_value"]) and not np.isnan(r["sc_l"])]
    T_for_cv = [r["T_value"] for r in rows
                if not np.isnan(r["T_value"]) and
                not np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan")))]
    cvte_for_T = [r["gain_cv_test"] for r in rows
                  if not np.isnan(r["T_value"]) and
                  not np.isnan(float(r["gain_cv_test"] if r["gain_cv_test"] is not None else float("nan")))]

    def _sp(a, b, label):
        if len(a) >= 3 and len(b) >= 3:
            c, p = spearmanr(a, b)
            print(f"  Spearman(T, {label}): {c:+.3f}  (p={p:.3f}  n={len(a)})")
        else:
            print(f"  Spearman(T, {label}): n/a")

    _sp(T_vals, stds,   "std_resid    ")
    _sp(T_vals, gains_f,"gain_full    ")
    if T_for_cv:  _sp(T_for_cv, cvte_for_T, "gain_cv_test ")
    if T_for_sc:  _sp(T_for_sc, sc_all,     "sign_consist ")

    # ── PART 2: aggregate across all folds ────────────────────────────────────
    print(f"\n\n── PART 2: Aggregate across {N_SEEDS} seeds × {N_FOLDS} folds ──\n")

    agg_results = []
    for seed in range(N_SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold_i, (tri, vali) in enumerate(kf.split(X)):
            canon = (seed == CANON_SEED and fold_i == CANON_FOLD)
            print(f"  seed={seed} fold={fold_i+1}/{N_FOLDS} {'(canonical)' if canon else '          '}",
                  end="", flush=True)
            res = _fit_and_diagnose(
                X[tri], y[tri], X[vali], y[vali],
                r2_ceil, seed, detailed=False,
            )
            agg_results.append(res)
            print(f"  Δ={res['delta_r2']:+.4f}  "
                  f"CV_ok={res['frac_cv_generalise']:.2f}  "
                  f"sc=({res['mean_sc_l']:.3f},{res['mean_sc_r']:.3f})  "
                  f"gain_cv_te={_fmt(res['mean_gain_cv_te'], '+.4f')}")

    def _agg_key(key):
        vals = [r[key] for r in agg_results if not np.isnan(float(r[key]))]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    dr2_m, dr2_s   = _agg_key("delta_r2")
    frac_m, frac_s = _agg_key("frac_cv_generalise")
    sc_l_m, _      = _agg_key("mean_sc_l")
    sc_r_m, _      = _agg_key("mean_sc_r")
    gcvte_m, _     = _agg_key("mean_gain_cv_te")
    gcvtr_m, _     = _agg_key("mean_gain_full")
    cT_std_m, _    = _agg_key("corr_T_stdresid")
    cT_cv_m, _     = _agg_key("corr_T_cvgain")

    print(f"\nAGGREGATE SUMMARY ({len(agg_results)} folds)")
    print(f"  ΔR²_adj (surgery):             {dr2_m:+.4f} ± {dr2_s:.4f}")
    print(f"  Frac. partitions CV_ok:        {frac_m:.3f} ± {frac_s:.3f}")
    print(f"  Mean gain_train (full split):  {gcvtr_m:.4f}")
    print(f"  Mean gain_cv_test (30% held):  {_fmt(gcvte_m, '+.4f')}")
    print(f"  Mean sign_consistency left:    {sc_l_m:.3f}")
    print(f"  Mean sign_consistency right:   {sc_r_m:.3f}")
    print(f"  Spearman(T, std_resid):        {_fmt(cT_std_m, '+.3f')}")
    print(f"  Spearman(T, gain_cv_test):     {_fmt(cT_cv_m, '+.3f')}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n── INTERPRETATION ──\n")

    is_random_sign = (sc_l_m < 0.55 and sc_r_m < 0.55)
    cv_fails       = (frac_m < 0.35)
    gain_shrinks   = (not np.isnan(gcvte_m) and gcvte_m < 0.01)

    if is_random_sign:
        print("  [SIGN] Sign consistency ≈ 0.50 → corrections are in the wrong")
        print("         direction roughly half the time. The within-partition mean")
        print("         residual on train does not predict sign of val residuals.")
    elif sc_l_m < 0.60:
        print("  [SIGN] Sign consistency is only slightly above random (0.5).")
        print("         Corrections are unreliable but not fully inverted.")
    else:
        print("  [SIGN] Sign consistency > 0.60: corrections direction is reliable.")

    if cv_fails:
        print("  [CV]   < 35% of partitions have a split that generalises to")
        print("         the held-out 30%. The within-partition structure on train")
        print("         is noise, not signal.")
    elif frac_m < 0.55:
        print("  [CV]   Marginal CV generalisation (~35-55%). Some real structure")
        print("         but swamped by noise — need larger partition sizes or")
        print("         stronger regularisation of the split selection.")
    else:
        print("  [CV]   > 55% of partitions generalise. Structure is real.")

    if gain_shrinks:
        print("  [GAIN] CV-test gain ≈ 0 or negative. High training gain is")
        print("         artefact of overfitting to the 70% subsample's noise.")

    print()
    print("  Key failure mechanism for friedman1:")
    if is_random_sign and cv_fails:
        print("  Both sign consistency and CV gain point to the SAME root cause:")
        print("  the within-partition residuals after GeoXGB convergence are")
        print("  dominated by the sigma=1.0 noise, not by structural signal.")
        print("  A single split cannot reliably capture signal at this partition size.")
    elif not cv_fails and is_random_sign:
        print("  There IS within-partition structure (CV_ok), but the mean")
        print("  correction (delta_l/delta_r) is the wrong estimator — the sign")
        print("  of the mean isn't consistent. Need a more robust estimator.")
    elif cv_fails and not is_random_sign:
        print("  Sign of corrections is right, but the splits don't generalise.")
        print("  Regularise the split search: raise MIN_LEAF or use higher threshold.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
