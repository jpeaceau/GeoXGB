"""
GeoXGB vs XGBoost — Normalized MAE Comparison
==============================================
Motivation: R² uses squared error, which disproportionately punishes the
occasional large GeoXGB errors (concrete XGB-wins region: std=7.95).
A model that loses rarely but badly looks worse under R² than a model that
loses often but narrowly.  MAE treats every error unit equally.

Metrics reported per dataset:
  R²               (current default — kept for reference)
  MAE              (absolute, in target units)
  NMAE             = MAE / std(y_train)   [dimensionless, cross-dataset comparable]
  NMAE_skill       = 1 - NMAE_model / NMAE_baseline   [0=baseline, 1=perfect]

Per-sample relative advantage:
  rel_adv_i = (|e_X_i| - |e_G_i|) / (|e_X_i| + |e_G_i| + eps)   in [-1, +1]
  +1 = XGBoost perfect, GeoXGB maximally wrong
  -1 = GeoXGB perfect, XGBoost maximally wrong
  0  = equal absolute error (or both zero)

This normalises the win magnitude so that a 0.001-unit win when both
models achieve 0.001 error is not equivalent to a 0.001-unit win when
one model achieves 10.0 error.

Datasets: california_housing, concrete_compressive, breast_cancer,
          ionosphere, friedman1.
3 seeds x 5-fold CV = 15 folds per dataset.
"""

from __future__ import annotations

import io, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                               fetch_openml, make_friedman1)
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from geoxgb._cpp_backend import (
    CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config)

# ── Noise floor estimation ─────────────────────────────────────────────────────

def noise_floor_mae_nn(X: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate irreducible MAE from nearest-neighbour label differences.

    For IID noise ε_i ~ N(0, σ²), Var(y_i - y_j) = 2σ² when x_i ≈ x_j.
    So σ_noise = std(pair_diffs) / sqrt(2),
    and MAE_irreducible = σ_noise × sqrt(2/π)  [mean of half-normal].

    The NN must be in the ORIGINAL feature space (not z-space) since we want
    pairs that are similar in the data-generating process.  Uses 1-NN so the
    pairs are as close as possible; for very small n (<30) returns 0 (no floor).
    """
    n = len(y)
    if n < 30:
        return 0.0
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)          # indices[:, 0] is self; [:, 1] is 1-NN
    pair_diffs  = y - y[indices[:, 1]]
    sigma_noise = float(np.std(pair_diffs)) / np.sqrt(2.0)
    return max(sigma_noise * np.sqrt(2.0 / np.pi), 0.0)


# ── Config ─────────────────────────────────────────────────────────────────────

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]
EPS      = 1e-9   # floor for relative-advantage denominator

# GeoXGB regression defaults updated to MAE-optimal values from meta-regression
# OAT + pairwise sweep (phases 2-5). Key changes vs previous config:
#   learning_rate  0.10 -> 0.02  (+0.060 skill_adj, rank-1 parameter)
#   max_depth      3    -> 2     (+0.051 importance)
#   reduce_ratio   0.70 -> 0.80  (+0.002)
#   y_weight       0.20 -> 0.50  (aligns with GeoXGBRegressor default)
#   auto_noise     True -> False  (+0.025)
#   noise_guard    True -> False  (+0.036, rank-1 secondary parameter)
#   n_bins         64   -> 128   (+0.004)
#   n_rounds scaled from 500 (lr=0.10) -> 1000 (lr=0.02) for comparable depth
GEO_REG = dict(n_rounds=1000, learning_rate=0.02, max_depth=2,
               min_samples_leaf=5, reduce_ratio=0.8, y_weight=0.5,
               refit_interval=5, auto_expand=True, expand_ratio=0.1,
               auto_noise=False, noise_guard=False, variance_weighted=False,
               min_train_samples=100, n_bins=128)
GEO_CLF = dict(n_rounds=500, learning_rate=0.1, max_depth=5,
               min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
               refit_interval=5, auto_expand=True, expand_ratio=0.1,
               min_train_samples=100, n_bins=64)
XGB_REG = dict(n_estimators=500, learning_rate=0.1, max_depth=3,
               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
               verbosity=0, n_jobs=-1)
XGB_CLF = dict(n_estimators=500, learning_rate=0.1, max_depth=5,
               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
               verbosity=0, n_jobs=-1)

# ── Data loading ───────────────────────────────────────────────────────────────

def clean_X(df):
    if not hasattr(df, "select_dtypes"):
        arr = np.asarray(df, dtype=np.float64)
        return SimpleImputer(strategy="median").fit_transform(arr) if np.isnan(arr).any() else arr
    df = df.copy()
    num_c = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_c = df.select_dtypes(include=["object","category"]).columns.tolist()
    if num_c and df[num_c].isnull().any().any():
        df[num_c] = SimpleImputer(strategy="median").fit_transform(df[num_c])
    if cat_c:
        df[cat_c] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_c])
        df[cat_c] = OrdinalEncoder().fit_transform(df[cat_c])
    return df.values.astype(np.float64)


def load_datasets():
    ds = {}
    rng = np.random.RandomState(RNG)

    d = fetch_california_housing()
    idx = rng.choice(len(d.data), 8000, replace=False)
    ds["california"] = ("reg", d.data[idx].astype(np.float64), d.target[idx])

    try:
        raw = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        ds["concrete"] = ("reg", clean_X(raw.data),
                          np.asarray(raw.target, dtype=np.float64))
    except Exception as e:
        print(f"  concrete SKIP: {e}")

    d2 = load_breast_cancer()
    ds["breast_cancer"] = ("clf", d2.data.astype(np.float64),
                           d2.target.astype(np.float64))

    try:
        raw2 = fetch_openml(data_id=59, as_frame=True, parser="auto")
        ds["ionosphere"] = ("clf", clean_X(raw2.data),
                            (raw2.target == "g").astype(np.float64).values)
    except Exception as e:
        print(f"  ionosphere SKIP: {e}")

    Xf, yf = make_friedman1(n_samples=1000, random_state=RNG)
    ds["friedman1"] = ("reg", Xf.astype(np.float64), yf.astype(np.float64))

    return ds


# ── Per-fold metrics ───────────────────────────────────────────────────────────

def fold_metrics(y_val, pred_g, pred_x, y_tr, task, mae_floor=0.0):
    """
    Return dict of metrics for one fold.

    skill_adj: noise-corrected MAE skill score
        = (mae_naive - mae_model) / (mae_naive - mae_floor)
    where mae_floor is the estimated irreducible MAE (from noise_floor_mae_nn).
    When mae_floor=0 this collapses to the plain skill score.
    """
    e_g = np.abs(y_val - pred_g)
    e_x = np.abs(y_val - pred_x)

    # Per-sample relative advantage: +1 = XGB perfect, -1 = Geo perfect
    rel_adv = (e_x - e_g) / (e_x + e_g + EPS)   # negative = GeoXGB wins

    if task == "reg":
        y_std   = float(np.std(y_tr)) if len(y_tr) > 1 else 1.0
        mae_naive = float(np.mean(np.abs(y_val - y_tr.mean())))
        mae_g = float(np.mean(e_g))
        mae_x = float(np.mean(e_x))
        nmae_g = mae_g   / (y_std + EPS)
        nmae_x = mae_x   / (y_std + EPS)
        nmae_base  = mae_naive  / (y_std + EPS)
        nmae_floor = mae_floor  / (y_std + EPS)
        # Plain skill: fraction of naive MAE captured
        skill_g = 1.0 - nmae_g / (nmae_base + EPS)
        skill_x = 1.0 - nmae_x / (nmae_base + EPS)
        # Noise-corrected: fraction of EXPLOITABLE MAE captured
        denom = nmae_base - nmae_floor
        skill_adj_g = float((nmae_base - nmae_g) / (denom + EPS)) if denom > 1e-6 else float("nan")
        skill_adj_x = float((nmae_base - nmae_x) / (denom + EPS)) if denom > 1e-6 else float("nan")
        return dict(
            r2_g       = r2_score(y_val, pred_g),
            r2_x       = r2_score(y_val, pred_x),
            mae_g      = mae_g,        mae_x      = mae_x,
            nmae_g     = nmae_g,       nmae_x     = nmae_x,
            skill_g    = skill_g,      skill_x    = skill_x,
            skill_adj_g= skill_adj_g,  skill_adj_x= skill_adj_x,
            mae_floor  = mae_floor,
            rel_adv    = rel_adv,
        )
    else:
        auc_g = roc_auc_score(y_val, pred_g)
        auc_x = roc_auc_score(y_val, pred_x)
        mae_g = float(np.mean(e_g))
        mae_x = float(np.mean(e_x))
        # Classification: NMAE normalised by class-rate baseline
        p_base   = float(y_tr.mean())
        mae_base = float(np.mean(np.abs(y_val - p_base)))
        nmae_g   = mae_g / (mae_base + EPS)
        nmae_x   = mae_x / (mae_base + EPS)
        skill_g  = 1.0 - nmae_g
        skill_x  = 1.0 - nmae_x
        # For classification labels are 0/1 — noise floor ~ 0 (clean labels)
        skill_adj_g = skill_g
        skill_adj_x = skill_x
        return dict(
            auc_g      = auc_g,        auc_x      = auc_x,
            mae_g      = mae_g,        mae_x      = mae_x,
            nmae_g     = nmae_g,       nmae_x     = nmae_x,
            skill_g    = skill_g,      skill_x    = skill_x,
            skill_adj_g= skill_adj_g,  skill_adj_x= skill_adj_x,
            mae_floor  = mae_floor,
            rel_adv    = rel_adv,
        )


def run_dataset(ds_name, task, X, y):
    kf_cls = StratifiedKFold if task == "clf" else KFold
    fold_results = []

    for seed in SEEDS:
        kf = kf_cls(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        split_y = y.astype(int) if task == "clf" else y
        for tr, va in kf.split(X, split_y):
            base = GEO_REG if task == "reg" else GEO_CLF
            cfg  = make_cpp_config(**{**base, "random_state": seed})

            # Noise floor from training fold (NN-pair estimator, regression only)
            mae_floor = noise_floor_mae_nn(X[tr], y[tr]) if task == "reg" else 0.0

            if task == "reg":
                geo = CppGeoXGBRegressor(cfg)
                geo.fit(X[tr], y[tr])
                pg = geo.predict(X[va])
                xgm = xgb.XGBRegressor(**{**XGB_REG, "random_state": seed})
                xgm.fit(X[tr], y[tr])
                px = xgm.predict(X[va])
            else:
                geo = CppGeoXGBClassifier(cfg)
                geo.fit(X[tr], y[tr])
                pg = geo.predict_proba(X[va])[:, 1]
                xgm = xgb.XGBClassifier(**{**XGB_CLF, "random_state": seed})
                xgm.fit(X[tr], y[tr])
                px = xgm.predict_proba(X[va])[:, 1]

            fold_results.append(fold_metrics(y[va], pg, px, y[tr], task, mae_floor))

        print(f"    seed={seed}", end=" ", flush=True)
    print()
    return fold_results


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_dataset(ds_name, task, fold_results):
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {ds_name}   ({task}   {len(fold_results)} folds)")
    print(sep)

    def mean_std(key):
        vals = [f[key] for f in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    # ── Primary metrics ────────────────────────────────────────────────────────
    if task == "reg":
        r2g, r2g_s  = mean_std("r2_g");   r2x, r2x_s  = mean_std("r2_x")
        print(f"\n  R²      GeoXGB={r2g:+.4f} (±{r2g_s:.4f})  "
              f"XGBoost={r2x:+.4f} (±{r2x_s:.4f})  "
              f"delta={r2g-r2x:+.4f}")
    else:
        aug, aug_s  = mean_std("auc_g");  aux, aux_s  = mean_std("auc_x")
        print(f"\n  AUC     GeoXGB={aug:.4f} (±{aug_s:.4f})  "
              f"XGBoost={aux:.4f} (±{aux_s:.4f})  "
              f"delta={aug-aux:+.4f}")

    mag, mag_s   = mean_std("mae_g");    max_, max_s  = mean_std("mae_x")
    nmg, nmg_s   = mean_std("nmae_g");   nmx, nmx_s   = mean_std("nmae_x")
    skg, skg_s   = mean_std("skill_g");  skx, skx_s   = mean_std("skill_x")
    sag, sag_s   = mean_std("skill_adj_g"); sax, sax_s = mean_std("skill_adj_x")
    flr, _       = mean_std("mae_floor")

    print(f"\n  MAE     GeoXGB={mag:.4f} (±{mag_s:.4f})  "
          f"XGBoost={max_:.4f} (±{max_s:.4f})  "
          f"delta={mag-max_:+.4f}  ({'GeoXGB better' if mag < max_ else 'XGBoost better'})")
    print(f"  NMAE    GeoXGB={nmg:.4f} (±{nmg_s:.4f})  "
          f"XGBoost={nmx:.4f} (±{nmx_s:.4f})  "
          f"delta={nmg-nmx:+.4f}  ({'GeoXGB better' if nmg < nmx else 'XGBoost better'})")
    print(f"  Skill   GeoXGB={skg:.4f} (±{skg_s:.4f})  "
          f"XGBoost={skx:.4f} (±{skx_s:.4f})  "
          f"delta={skg-skx:+.4f}")
    if flr > 1e-4:
        print(f"  Skill*  GeoXGB={sag:.4f} (±{sag_s:.4f})  "
              f"XGBoost={sax:.4f} (±{sax_s:.4f})  "
              f"delta={sag-sax:+.4f}  [noise-corrected, floor≈{flr:.4f}]")
    else:
        print(f"  Skill*  (noise floor ≈ 0 — labels are clean; Skill* = Skill)")

    # ── Per-sample relative advantage distribution ─────────────────────────────
    all_rel = np.concatenate([f["rel_adv"] for f in fold_results])
    # rel_adv: negative = GeoXGB wins, positive = XGB wins
    geo_wins_mask = all_rel < 0
    xgb_wins_mask = all_rel > 0
    ties_mask     = all_rel == 0.0

    geo_pct = 100.0 * geo_wins_mask.mean()
    xgb_pct = 100.0 * xgb_wins_mask.mean()
    tie_pct = 100.0 * ties_mask.mean()

    geo_margin = float(-all_rel[geo_wins_mask].mean()) if geo_wins_mask.any() else 0.0
    xgb_margin = float( all_rel[xgb_wins_mask].mean()) if xgb_wins_mask.any() else 0.0

    print(f"\n  Per-sample relative advantage  rel_adv = (|e_X| - |e_G|) / (|e_X| + |e_G|)")
    print(f"  GeoXGB wins: {geo_pct:.1f}%  mean rel margin={geo_margin:.3f}")
    print(f"  XGBoost wins:{xgb_pct:.1f}%  mean rel margin={xgb_margin:.3f}")
    if tie_pct > 0.5:
        print(f"  Ties:         {tie_pct:.1f}%  (both models accurate to EPS)")

    # Quantile distribution of rel_adv
    qs = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
    counts = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (all_rel >= lo) & (all_rel < hi)
        counts.append(int(mask.sum()))
    total = len(all_rel)
    print(f"\n  rel_adv distribution  (- = GeoXGB wins, + = XGBoost wins):")
    labels = ["(-1,-0.5]", "(-0.5,-0.2]", "(-0.2,-0.1]", "(-0.1,0]",
              "(0,0.1]", "(0.1,0.2]", "(0.2,0.5]", "(0.5,1]"]
    bar_width = 30
    for label, cnt in zip(labels, counts):
        pct = 100.0 * cnt / total
        bar = "#" * int(pct * bar_width / 30)
        print(f"  {label:>12}  {pct:5.1f}%  {bar}")

    # Net relative advantage (pooled mean: negative = GeoXGB wins on average)
    net = float(all_rel.mean())
    winner = "GeoXGB" if net < 0 else "XGBoost"
    print(f"\n  Net mean rel_adv = {net:+.4f}  -> {winner} wins under normalized criterion")

    return dict(
        ds=ds_name, task=task,
        nmae_delta=nmg - nmx, skill_delta=skg - skx,
        skill_adj_delta=sag - sax,
        mae_delta=mag - max_,
        net_rel_adv=net,
        geo_margin=geo_margin, xgb_margin=xgb_margin,
        geo_pct=geo_pct, xgb_pct=xgb_pct,
        r2_delta=(r2g - r2x) if task == "reg" else None,
        auc_delta=(aug - aux) if task == "clf" else None,
    )


def global_summary(summaries):
    print("\n" + "=" * 68)
    print("  GLOBAL SUMMARY — Normalized MAE perspective")
    print("=" * 68)

    print(f"\n  {'Dataset':<16}  {'M':>3}  {'dR2/AUC':>8}  {'dMAE':>7}  "
          f"{'dNMAE':>7}  {'dSkill*':>8}  {'net_rel':>8}  winner")
    print("  " + "-" * 76)
    for s in summaries:
        primary_d = s['r2_delta'] if s['task'] == 'reg' else s['auc_delta']
        winner = "GeoXGB" if s['net_rel_adv'] < 0 else "XGBoost"
        print(f"  {s['ds']:<16}  {s['task']:>3}  {primary_d or 0:>+8.4f}  "
              f"{s['mae_delta']:>+7.4f}  {s['nmae_delta']:>+7.4f}  "
              f"{s['skill_adj_delta']:>+8.4f}  {s['net_rel_adv']:>+8.4f}  {winner}")

    print(f"\n  Interpretation of columns:")
    print(f"    delta_R2/AUC  = GeoXGB - XGBoost  (+ = GeoXGB better)")
    print(f"    delta_MAE     = GeoXGB - XGBoost  (- = GeoXGB better)")
    print(f"    delta_NMAE    = GeoXGB - XGBoost  (- = GeoXGB better)")
    print(f"    net_rel_adv   = mean (|e_X|-|e_G|)/(|e_X|+|e_G|)  (- = GeoXGB better)")

    # How does the ranking change across metrics?
    print(f"\n  Rank order under each metric (best GeoXGB performance first):")
    for metric_key, metric_name, reverse in [
        ("r2_delta",       "R2/AUC delta",  True),
        ("nmae_delta",     "NMAE delta",    False),
        ("skill_adj_delta","Skill* delta",  True),
        ("net_rel_adv",    "rel_adv",       False),
    ]:
        ordered = sorted(summaries, key=lambda s: s[metric_key] or 0, reverse=reverse)
        print(f"    {metric_name:<14}: " +
              "  ".join(f"{s['ds']}({(s[metric_key] or 0):+.3f})" for s in ordered))

    # Summary verdict
    geo_wins_r2   = sum(1 for s in summaries if (s['r2_delta'] or 0) > 0 or (s['auc_delta'] or 0) > 0)
    geo_wins_nmae = sum(1 for s in summaries if s['nmae_delta'] < 0)
    geo_wins_skill= sum(1 for s in summaries if s['skill_adj_delta'] > 0)
    geo_wins_rel  = sum(1 for s in summaries if s['net_rel_adv'] < 0)
    n = len(summaries)
    print(f"\n  GeoXGB wins count:  R2/AUC={geo_wins_r2}/{n}  "
          f"NMAE={geo_wins_nmae}/{n}  Skill*={geo_wins_skill}/{n}  rel_adv={geo_wins_rel}/{n}")
    print(f"  (* Skill* = noise-corrected NMAE skill, primary deployment metric)")

    # Margin asymmetry table
    print(f"\n  Win-margin asymmetry (per-sample relative advantage):")
    print(f"  {'Dataset':<16}  {'Geo wins':>8}  {'Geo margin':>11}  "
          f"{'XGB wins':>8}  {'XGB margin':>11}  {'ratio Geo/XGB':>14}")
    print("  " + "-" * 76)
    for s in summaries:
        ratio = (s['geo_margin'] / s['xgb_margin']
                 if s['xgb_margin'] > 1e-6 else float('nan'))
        print(f"  {s['ds']:<16}  {s['geo_pct']:>7.1f}%  {s['geo_margin']:>11.4f}  "
              f"{s['xgb_pct']:>7.1f}%  {s['xgb_margin']:>11.4f}  {ratio:>14.2f}x")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  GeoXGB vs XGBoost — Normalized MAE Comparison")
    print("=" * 68)
    print(f"  Seeds: {SEEDS}  |  Folds: {N_SPLITS}  |  Folds/dataset: {len(SEEDS)*N_SPLITS}")
    print()

    datasets = load_datasets()
    print(f"  Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}\n")

    summaries = []
    for ds_name, (task, X, y) in datasets.items():
        print(f"  Running {ds_name} ...", flush=True)
        t0 = time.perf_counter()
        fold_results = run_dataset(ds_name, task, X, y)
        print(f"  ({time.perf_counter()-t0:.0f}s)", flush=True)
        s = report_dataset(ds_name, task, fold_results)
        summaries.append(s)

    global_summary(summaries)
    print()


if __name__ == "__main__":
    main()
