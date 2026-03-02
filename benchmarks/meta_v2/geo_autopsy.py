"""
GeoXGB vs XGBoost — Geometric Autopsy
======================================
For every prediction on every dataset, computes the complete geometric
fingerprint (T, Q, S, cone, novelty, partition) and asks:

  "Does HVRT cooperation geometry predict when GeoXGB beats XGBoost?"

Mathematical framework
----------------------
Let z* = whitener.transform(x*) be the whitened feature vector of a test sample.

  Q*       = ||z*||²             = Σ_k z*_k²          Mahalanobis norm
  S*       = Σ_k z*_k                                  cooperation sum
  T_self*  = (S*² − Q*) / 2     = Σ_{k<l} z*_k z*_l  self-cooperation (exact)
             Note: T_self > 0 ⟺ S² > Q ⟺ features move coherently (same sign)

  T_approx* = Σ_{k<l} (z*_k z*_l − μ_kl) / σ_kl      training-normalised T
              where μ_kl = mean(z_nk * z_nl over training), σ_kl = std(...)
              For well-whitened data μ_kl ≈ 0, σ_kl ≈ 1 → T_approx ≈ T_self.

  Cone membership: T_approx* > 0  (sample in the HVRT cooperation cone)

  Novelty*  = min_j ||z* − z_j||   min Euclidean distance to training z

Prediction-level quantities
----------------------------
  ε_G = y* − ŷ_GeoXGB    ε_X = y* − ŷ_XGB
  Δ_abs = |ε_G| − |ε_X|  (+ve → GeoXGB worse,  −ve → GeoXGB better)

Key hypotheses tested
---------------------
  H1: Spearman(T_approx*, Δ_abs) < 0  → samples with high T (in cone) favour GeoXGB
  H2: Spearman(Q*, Δ_abs) > 0         → high-norm samples (outliers) disfavour GeoXGB
  H3: E[Δ_abs | in_cone] < E[Δ_abs | out_cone]  (cone is a valid partition)
  H4: OLS R² > 0  (geometric features collectively explain the advantage)

Datasets
--------
  california_housing  — reg n=8000 d=8  (XGBoost wins by ~0.02 R²)
  concrete_compressive— reg n=1030 d=8  (XGBoost wins by ~0.03 R²)
  breast_cancer       — clf n=569  d=30 (GeoXGB wins by ~0.002 AUC)
  ionosphere          — clf n=351  d=34 (GeoXGB wins by ~0.011 AUC)
  friedman1           — reg n=1000 d=10 (synthetic, known structure)

5-fold CV, 3 seeds each.
"""

from __future__ import annotations

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_friedman1
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

from geoxgb._cpp_backend import (            # noqa: E402
    CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config)
from hvrt import HVRT                        # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]

GEO_REG = dict(
    n_rounds=500, learning_rate=0.1, max_depth=3, min_samples_leaf=5,
    reduce_ratio=0.7, y_weight=0.2, refit_interval=5, auto_expand=True,
    expand_ratio=0.1, min_train_samples=100, n_bins=64,
)
GEO_CLF = dict(
    n_rounds=500, learning_rate=0.1, max_depth=5, min_samples_leaf=5,
    reduce_ratio=0.7, y_weight=0.2, refit_interval=5, auto_expand=True,
    expand_ratio=0.1, min_train_samples=100, n_bins=64,
)
XGB_REG = dict(n_estimators=500, learning_rate=0.1, max_depth=3,
               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
               random_state=RNG, verbosity=0, n_jobs=-1)
XGB_CLF = dict(n_estimators=500, learning_rate=0.1, max_depth=5,
               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
               random_state=RNG, verbosity=0, n_jobs=-1)

HVRT_PARAMS = dict(y_weight=0.5)

# ── Data loading ──────────────────────────────────────────────────────────────

def clean_X(df):
    if not isinstance(df, pd.DataFrame):
        arr = np.asarray(df, dtype=np.float64)
        if np.isnan(arr).any():
            arr = SimpleImputer(strategy="median").fit_transform(arr)
        return arr
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

    X, y = fetch_california_housing(return_X_y=True)
    idx = rng.choice(len(X), 8000, replace=False)
    ds["california_housing"] = ("reg", X[idx].astype(np.float64), y[idx])

    try:
        d = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        ds["concrete_compressive"] = ("reg", clean_X(d.data),
                                       np.asarray(d.target, dtype=np.float64))
    except Exception: pass

    Xb, yb = load_breast_cancer(return_X_y=True)
    ds["breast_cancer"] = ("clf", Xb.astype(np.float64), yb.astype(np.float64))

    try:
        d = fetch_openml(data_id=59, as_frame=True, parser="auto")
        ds["ionosphere"] = ("clf", clean_X(d.data),
                             (d.target == "g").astype(np.float64).values)
    except Exception: pass

    X_f, y_f = make_friedman1(n_samples=1000, random_state=RNG)
    ds["friedman1"] = ("reg", X_f.astype(np.float64), y_f.astype(np.float64))

    return ds


# ── Geometric fingerprint computation ─────────────────────────────────────────

def compute_pair_stats(X_z: np.ndarray):
    """
    Precompute training pair statistics (μ_kl, σ_kl) for T_approx.
    Returns list of (k, l, mu, sig) for all pairs k < l.
    """
    d = X_z.shape[1]
    stats_list = []
    for k in range(d):
        for l in range(k + 1, d):
            prod = X_z[:, k] * X_z[:, l]
            mu   = float(prod.mean())
            sig  = float(prod.std())
            if sig < 1e-10:
                sig = 1.0
            stats_list.append((k, l, mu, sig))
    return stats_list


def compute_geometric_fingerprint(
    X_val:       np.ndarray,
    X_tr:        np.ndarray,
    hvrt_model:  HVRT,
    pair_stats:  list,
) -> dict[str, np.ndarray]:
    """
    Compute complete geometric fingerprint for validation samples.

    Returns dict with arrays of shape (n_val,):
      z_val     : whitened coordinates           (n_val, d)
      Q         : Mahalanobis norm squared
      S         : cooperation sum
      T_self    : (S² - Q) / 2  [exact self-cooperation]
      T_approx  : training-normalised cooperation target [HVRT approximation]
      cone_in   : T_approx > 0  [bool]
      novelty   : min distance to training set in z-space
      partition : HVRT partition id
    """
    X_tr_z  = hvrt_model._to_z(X_tr)
    X_val_z = hvrt_model._to_z(X_val)
    n_val, d = X_val_z.shape

    # ── Q, S, T_self ──────────────────────────────────────────────────────────
    Q      = (X_val_z ** 2).sum(axis=1)
    S      = X_val_z.sum(axis=1)
    T_self = (S ** 2 - Q) / 2.0

    # ── T_approx ─────────────────────────────────────────────────────────────
    T_approx = np.zeros(n_val)
    for k, l, mu, sig in pair_stats:
        pair_val  = X_val_z[:, k] * X_val_z[:, l]
        T_approx += (pair_val - mu) / sig

    # ── Novelty (min L2 to training z) ───────────────────────────────────────
    # BLAS: (n_val, d) @ (d, n_tr) → (n_val, n_tr)
    n_tr = X_tr_z.shape[0]
    norm_val = (X_val_z ** 2).sum(axis=1)
    norm_tr  = (X_tr_z  ** 2).sum(axis=1)
    D = norm_val[:, None] - 2.0 * (X_val_z @ X_tr_z.T) + norm_tr[None, :]
    D = np.clip(D, 0.0, None)
    novelty = np.sqrt(D.min(axis=1))

    # ── Partition assignment ──────────────────────────────────────────────────
    partition = hvrt_model.tree_.apply(X_val_z).astype(int)

    return dict(
        z_val=X_val_z, Q=Q, S=S, T_self=T_self,
        T_approx=T_approx, cone_in=(T_approx > 0),
        novelty=novelty, partition=partition,
    )


def pearson_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (r, p-value) ignoring NaNs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(a[mask], b[mask])
    return float(r), float(p)


def spearman_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    res = stats.spearmanr(a[mask], b[mask])
    return float(res.statistic), float(res.pvalue)


# ── Per-fold analysis ─────────────────────────────────────────────────────────

def analyse_fold(
    X_tr:   np.ndarray, y_tr:  np.ndarray,
    X_val:  np.ndarray, y_val: np.ndarray,
    task:   str,
    seed:   int,
) -> dict:
    """
    Fit GeoXGB + XGBoost + HVRT on training, compute geometric fingerprints
    for val, return full per-sample table and aggregate statistics.
    """
    # ── Fit models ────────────────────────────────────────────────────────────
    base = GEO_REG if task == "reg" else GEO_CLF
    cfg  = make_cpp_config(**{**base, "random_state": seed})

    if task == "reg":
        geo  = CppGeoXGBRegressor(cfg)
        geo.fit(X_tr, y_tr)
        pred_g = geo.predict(X_val)
        xgm  = xgb.XGBRegressor(**{**XGB_REG, "random_state": seed})
        xgm.fit(X_tr, y_tr)
        pred_x = xgm.predict(X_val)
        eps_g  = y_val - pred_g
        eps_x  = y_val - pred_x
        metric_g = r2_score(y_val, pred_g)
        metric_x = r2_score(y_val, pred_x)
    else:
        geo  = CppGeoXGBClassifier(cfg)
        geo.fit(X_tr, y_tr)
        pred_g = geo.predict_proba(X_val)[:, 1]
        xgm  = xgb.XGBClassifier(**{**XGB_CLF, "random_state": seed})
        xgm.fit(X_tr, y_tr)
        pred_x = xgm.predict_proba(X_val)[:, 1]
        # For classification, use log-odds residuals for error analysis
        eps_g  = y_val - pred_g
        eps_x  = y_val - pred_x
        metric_g = roc_auc_score(y_val, pred_g)
        metric_x = roc_auc_score(y_val, pred_x)

    abs_eps_g = np.abs(eps_g)
    abs_eps_x = np.abs(eps_x)
    delta_abs = abs_eps_g - abs_eps_x   # +ve → GeoXGB worse

    # ── Fit Python HVRT for geometry ──────────────────────────────────────────
    hvrt_m = HVRT(**HVRT_PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hvrt_m.fit(X_tr, y_tr)

    pair_stats = compute_pair_stats(hvrt_m._to_z(X_tr))
    geo_fp     = compute_geometric_fingerprint(X_val, X_tr, hvrt_m, pair_stats)

    T   = geo_fp["T_approx"]
    T_s = geo_fp["T_self"]
    Q   = geo_fp["Q"]
    S   = geo_fp["S"]
    nov = geo_fp["novelty"]
    cin = geo_fp["cone_in"]

    # ── Correlations ─────────────────────────────────────────────────────────
    corrs = {}
    for name, arr in [("T_approx", T), ("T_self", T_s), ("Q", Q),
                       ("S", S), ("novelty", nov)]:
        rho, pv = spearman_r(arr, delta_abs)
        corrs[name] = (rho, pv)
        # Also correlate separately with each model's error
        rho_g, _ = spearman_r(arr, abs_eps_g)
        rho_x, _ = spearman_r(arr, abs_eps_x)
        corrs[f"{name}_vs_G"] = rho_g
        corrs[f"{name}_vs_X"] = rho_x

    # ── Cone test (two-sample: in-cone vs out-of-cone Δ_abs) ─────────────────
    n_in  = int(cin.sum())
    n_out = int((~cin).sum())
    cone_result = {}
    if n_in >= 3 and n_out >= 3:
        mean_in  = float(delta_abs[cin].mean())
        mean_out = float(delta_abs[~cin].mean())
        t_stat, t_pval = stats.ttest_ind(delta_abs[cin], delta_abs[~cin])
        cone_result = dict(n_in=n_in, n_out=n_out,
                           mean_in=mean_in, mean_out=mean_out,
                           t_stat=float(t_stat), t_pval=float(t_pval))
    else:
        cone_result = dict(n_in=n_in, n_out=n_out,
                           mean_in=float("nan"), mean_out=float("nan"),
                           t_stat=float("nan"), t_pval=float("nan"))

    # ── OLS: regress Δ_abs on (T, Q, S, novelty) ─────────────────────────────
    feat_names = ["T_approx", "Q", "S", "T_self", "novelty"]
    feats = np.column_stack([T, Q, S, T_s, nov])
    mask_finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(delta_abs)
    ols_r2   = float("nan")
    ols_coef = {}
    if mask_finite.sum() >= 10:
        lr = LinearRegression().fit(feats[mask_finite], delta_abs[mask_finite])
        ols_r2 = float(lr.score(feats[mask_finite], delta_abs[mask_finite]))
        for nm, c in zip(feat_names, lr.coef_):
            ols_coef[nm] = float(c)
        ols_coef["intercept"] = float(lr.intercept_)

    # ── Per-partition breakdown ───────────────────────────────────────────────
    pids   = geo_fp["partition"]
    part_stats_out = {}
    for pid in np.unique(pids):
        m = pids == pid
        n_m = int(m.sum())
        if n_m < 3: continue
        part_stats_out[int(pid)] = dict(
            n        = n_m,
            mean_T   = float(T[m].mean()),
            mean_Q   = float(Q[m].mean()),
            mean_delta = float(delta_abs[m].mean()),
            frac_cone  = float(cin[m].mean()),
            mean_eps_g = float(abs_eps_g[m].mean()),
            mean_eps_x = float(abs_eps_x[m].mean()),
        )

    # ── T distribution summary ────────────────────────────────────────────────
    T_stats = dict(
        mean       = float(np.nanmean(T)),
        std        = float(np.nanstd(T)),
        frac_pos   = float((T > 0).mean()),
        Q_mean     = float(np.nanmean(Q)),
        Q_std      = float(np.nanstd(Q)),
        S_mean     = float(np.nanmean(S)),
        S_std      = float(np.nanstd(S)),
        T_self_corr_T_approx = pearson_r(T_s, T)[0],  # how close are the two T definitions
    )

    return dict(
        metric_g     = metric_g,
        metric_x     = metric_x,
        delta_metric = metric_g - metric_x,
        corrs        = corrs,
        cone         = cone_result,
        ols_r2       = ols_r2,
        ols_coef     = ols_coef,
        T_stats      = T_stats,
        part_stats   = part_stats_out,
        n_val        = int(len(y_val)),
    )


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def _nm(vals):
    """Nanmean ± nanstd string."""
    a = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not a: return "   n/a  "
    return f"{np.mean(a):+.4f} ± {np.std(a):.4f}"


def _fv(v, fmt="+.4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "   n/a"
    return format(v, fmt)


# ── Dataset-level analysis ────────────────────────────────────────────────────

def run_dataset(name: str, task: str, X: np.ndarray, y: np.ndarray) -> list[dict]:
    d = X.shape[1]
    print(f"\n  {'='*66}")
    print(f"  Dataset: {name}  (n={len(X)}, d={d}, task={task})")
    print(f"  {'='*66}")

    kf_cls = KFold if task == "reg" else StratifiedKFold
    fold_results = []

    for seed in SEEDS:
        kf = kf_cls(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fi, (tr_idx, va_idx) in enumerate(kf.split(X, y.astype(int) if task=="clf" else y)):
            t0 = time.perf_counter()
            res = analyse_fold(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], task, seed)
            dt = time.perf_counter() - t0
            fold_results.append(res)
            T_m = res["T_stats"]["mean"]
            fpc = res["T_stats"]["frac_pos"]
            cr  = res["corrs"].get("T_approx", (float("nan"), float("nan")))[0]
            print(f"    seed={seed} fold={fi+1}  metric_G={res['metric_g']:+.4f}  "
                  f"metric_X={res['metric_x']:+.4f}  Δ={res['delta_metric']:+.4f}  "
                  f"T̄={T_m:+.3f}  frac_cone={fpc:.2f}  ρ(T,Δ)={_fv(cr, '+.3f')}  "
                  f"({dt:.0f}s)")

    return fold_results


def print_dataset_summary(name: str, fold_results: list[dict]) -> None:
    nf = len(fold_results)
    print(f"\n  ── {name} ({'=' * (60 - len(name))})")

    # Metrics
    mg = np.mean([r["metric_g"] for r in fold_results])
    mx = np.mean([r["metric_x"] for r in fold_results])
    dm = np.mean([r["delta_metric"] for r in fold_results])
    print(f"  Metric  GeoXGB={mg:.4f}  XGBoost={mx:.4f}  Δ={dm:+.4f}")

    # T distribution
    Tm  = np.mean([r["T_stats"]["mean"]     for r in fold_results])
    Tstd = np.mean([r["T_stats"]["std"]      for r in fold_results])
    fpc = np.mean([r["T_stats"]["frac_pos"] for r in fold_results])
    Qm  = np.mean([r["T_stats"]["Q_mean"]   for r in fold_results])
    Sm  = np.mean([r["T_stats"]["S_mean"]   for r in fold_results])
    corr_T = np.mean([r["T_stats"]["T_self_corr_T_approx"] for r in fold_results
                      if not np.isnan(r["T_stats"]["T_self_corr_T_approx"])])
    print(f"  Geometry   T̄={Tm:+.3f} (σ={Tstd:.3f})  Q̄={Qm:.3f}  S̄={Sm:+.3f}  "
          f"frac_cone={fpc:.3f}  r(T_self,T_approx)={corr_T:+.3f}")

    # Correlations with Δ_abs
    print(f"  Spearman ρ (positive → feature predicts GeoXGB worse):")
    for feat in ["T_approx", "T_self", "Q", "S", "novelty"]:
        rhos  = [r["corrs"][feat][0]    for r in fold_results]
        pvals = [r["corrs"][feat][1]    for r in fold_results]
        rhoG  = [r["corrs"][f"{feat}_vs_G"] for r in fold_results]
        rhoX  = [r["corrs"][f"{feat}_vs_X"] for r in fold_results]
        rho_m = np.nanmean(rhos); pval_m = np.nanmean(pvals)
        rhoG_m = np.nanmean(rhoG); rhoX_m = np.nanmean(rhoX)
        sig = "**" if pval_m < 0.05 else "  "
        print(f"    {feat:<12}  ρ(vs_Δ)={rho_m:+.3f}{sig}  "
              f"ρ(vs_|εG|)={rhoG_m:+.3f}  ρ(vs_|εX|)={rhoX_m:+.3f}")

    # Cone test
    mean_in  = np.nanmean([r["cone"]["mean_in"]  for r in fold_results])
    mean_out = np.nanmean([r["cone"]["mean_out"] for r in fold_results])
    t_pval   = np.nanmean([r["cone"]["t_pval"]   for r in fold_results])
    n_in     = int(np.mean([r["cone"]["n_in"]    for r in fold_results]))
    n_out    = int(np.mean([r["cone"]["n_out"]   for r in fold_results]))
    sig_cone = "** (significant)" if t_pval < 0.05 else "(not significant)"
    print(f"  Cone test  in_cone: Δ̄={mean_in:+.4f} (n≈{n_in})  "
          f"out_cone: Δ̄={mean_out:+.4f} (n≈{n_out})  "
          f"t-pval={t_pval:.3f} {sig_cone}")
    if not np.isnan(mean_in) and not np.isnan(mean_out):
        if mean_in < mean_out:
            print(f"  → IN-cone samples favour GeoXGB (Δ_in < Δ_out)")
        else:
            print(f"  → OUT-cone samples favour GeoXGB (Δ_out < Δ_in)")

    # OLS
    ols_r2 = np.nanmean([r["ols_r2"] for r in fold_results])
    print(f"  OLS R² (T,Q,S,T_self,novelty → Δ_abs): {ols_r2:.4f}")
    # Aggregate OLS coefficients (sign indicates direction)
    coef_names = ["T_approx", "Q", "S", "T_self", "novelty"]
    for cn in coef_names:
        vals = [r["ols_coef"].get(cn, float("nan")) for r in fold_results
                if r["ols_coef"]]
        if vals:
            cm = np.nanmean(vals)
            print(f"    β_{cn:<12} = {cm:+.4f}")

    # Partition breakdown: top partitions by mean Δ_abs
    part_agg: dict[int, list] = {}
    for r in fold_results:
        for pid, ps in r["part_stats"].items():
            if pid not in part_agg:
                part_agg[pid] = []
            part_agg[pid].append(ps)
    part_summary = {pid: {
        "mean_delta":  np.mean([p["mean_delta"]  for p in pss]),
        "mean_T":      np.mean([p["mean_T"]      for p in pss]),
        "mean_Q":      np.mean([p["mean_Q"]      for p in pss]),
        "frac_cone":   np.mean([p["frac_cone"]   for p in pss]),
        "n":           np.mean([p["n"]           for p in pss]),
    } for pid, pss in part_agg.items()}

    if part_summary:
        sorted_parts = sorted(part_summary.items(), key=lambda x: x[1]["mean_delta"])
        print(f"\n  Partitions sorted by mean Δ_abs (top-3 GeoXGB wins, bottom-3 losses):")
        to_show = sorted_parts[:3] + [("...", None)] + sorted_parts[-3:]
        for pid, ps in to_show:
            if ps is None:
                print(f"    ...")
                continue
            print(f"    pid={pid:<4}  Δ̄={ps['mean_delta']:+.4f}  "
                  f"T̄={ps['mean_T']:+.4f}  Q̄={ps['mean_Q']:.4f}  "
                  f"frac_cone={ps['frac_cone']:.2f}  n≈{ps['n']:.0f}")

        # Correlation: partition mean_T vs partition mean_delta
        pts = [ps for ps in part_summary.values() if ps["n"] >= 3]
        if len(pts) >= 4:
            part_T     = np.array([p["mean_T"]     for p in pts])
            part_delta = np.array([p["mean_delta"] for p in pts])
            rp, pp = spearman_r(part_T, part_delta)
            print(f"  Partition-level Spearman(T̄_p, Δ̄_p) = {rp:+.3f} (p={pp:.3f})")
            if rp < -0.2 and pp < 0.1:
                print(f"  → HIGH-T partitions systematically favour GeoXGB")
            elif rp > 0.2 and pp < 0.1:
                print(f"  → LOW-T partitions systematically favour GeoXGB")
            else:
                print(f"  → No strong partition-level T vs Δ relationship")


# ── Global summary across all datasets ───────────────────────────────────────

def print_global_summary(all_fold_results: dict[str, list[dict]]) -> None:
    print(f"\n{'='*68}")
    print("  GLOBAL MATHEMATICAL SUMMARY")
    print(f"{'='*68}\n")

    print("  HYPOTHESIS TEST RESULTS")
    print("  " + "-"*64)
    print(f"  {'Dataset':<26} {'ρ(T,Δ)':>8} {'cone_test':>12} {'OLS_R²':>8} "
          f"{'GeoXGB_wins':>12}")
    print("  " + "-"*64)

    all_T_corrs = []
    all_cone_diffs = []
    for ds_name, fold_results in all_fold_results.items():
        rho_T = np.nanmean([r["corrs"]["T_approx"][0] for r in fold_results])
        pval  = np.nanmean([r["corrs"]["T_approx"][1] for r in fold_results])
        sig   = "*" if pval < 0.05 else " "
        cone_in  = np.nanmean([r["cone"]["mean_in"]  for r in fold_results])
        cone_out = np.nanmean([r["cone"]["mean_out"] for r in fold_results])
        cone_diff = cone_in - cone_out  # negative → in-cone better
        ols_r2 = np.nanmean([r["ols_r2"] for r in fold_results])
        dm = np.mean([r["delta_metric"] for r in fold_results])
        winner = "GeoXGB" if dm > 0 else "XGBoost"
        all_T_corrs.append(rho_T)
        all_cone_diffs.append(cone_diff)
        print(f"  {ds_name:<26} {rho_T:>+7.3f}{sig}  "
              f"{cone_diff:>+11.4f}  {ols_r2:>7.4f}  {winner:>12}")

    print()
    print(f"  Mean ρ(T_approx, Δ_abs): {np.nanmean(all_T_corrs):+.3f}")
    print(f"  Mean cone diff (in - out Δ_abs): {np.nanmean(all_cone_diffs):+.4f}")
    print()

    print("  INTERPRETATION")
    print("  " + "-"*64)

    mean_rho = np.nanmean(all_T_corrs)
    mean_cone = np.nanmean(all_cone_diffs)

    if mean_rho < -0.1:
        print(f"  H1 SUPPORTED: ρ(T, Δ_abs) = {mean_rho:+.3f}")
        print(f"  High-cooperation (in-cone) samples favour GeoXGB.")
        print(f"  T_approx > 0 is a valid geometric selector for GeoXGB.")
    elif mean_rho > 0.1:
        print(f"  H1 REJECTED (reversed): ρ(T, Δ_abs) = {mean_rho:+.3f}")
        print(f"  High-cooperation samples HURT GeoXGB — HVRT over-partitions them.")
    else:
        print(f"  H1 INCONCLUSIVE: ρ(T, Δ_abs) = {mean_rho:+.3f} (near zero)")
        print(f"  T-cooperation is not the primary predictor of model advantage.")

    if mean_cone < -0.005:
        print(f"\n  H3 SUPPORTED: in-cone Δ_abs < out-cone Δ_abs")
        print(f"  Cone condition T > 0 partitions predictions: GeoXGB wins inside.")
    elif mean_cone > 0.005:
        print(f"\n  H3 REVERSED: in-cone samples are where GeoXGB loses most.")
        print(f"  Possibly HVRT over-smooths within-cone regions.")
    else:
        print(f"\n  H3 INCONCLUSIVE: cone condition has no predictive power for advantage.")

    print()
    print("  MATHEMATICAL RELATIONSHIP: T_self vs T_approx")
    print("  T_self = (S² - Q)/2 = Σ_{k<l} z_k z_l  (exact, no training stats needed)")
    print("  T_approx = Σ_{k<l} (z_k z_l - μ_kl)/σ_kl  (training-normalized)")
    corr_vals = [r["T_stats"]["T_self_corr_T_approx"]
                 for frl in all_fold_results.values()
                 for r in frl
                 if not np.isnan(r["T_stats"]["T_self_corr_T_approx"])]
    if corr_vals:
        mc = np.mean(corr_vals)
        print(f"  Mean Pearson(T_self, T_approx) = {mc:.3f}")
        if mc > 0.9:
            print(f"  → T_self ≈ T_approx: whitening fully decorrelates feature pairs.")
            print(f"  → The closed-form T_self = (S²-Q)/2 is a sufficient approximation.")
        else:
            print(f"  → T_self ≠ T_approx: feature pair covariances survive whitening.")
            print(f"  → T_approx provides additional information beyond T_self.")

    print()
    print("  PRACTICAL RECOMMENDATION")
    print("  " + "-"*64)
    # Determine best predictor of advantage
    all_Q_corrs = np.nanmean([np.nanmean([r["corrs"]["Q"][0] for r in frl])
                               for frl in all_fold_results.values()])
    all_nov_corrs = np.nanmean([np.nanmean([r["corrs"]["novelty"][0] for r in frl])
                                 for frl in all_fold_results.values()])
    best = max([("T_approx", abs(mean_rho)),
                ("Q",        abs(all_Q_corrs)),
                ("novelty",  abs(all_nov_corrs))],
               key=lambda x: x[1])
    print(f"  Strongest per-prediction predictor of advantage: {best[0]}  (|ρ|={best[1]:.3f})")
    if best[1] > 0.05:
        print(f"  Use {best[0]} as a routing signal: compute it cheaply at inference time")
        print(f"  and select GeoXGB vs XGBoost per-sample accordingly.")
    else:
        print(f"  No single geometric feature strongly predicts per-prediction advantage.")
        print(f"  Hybrid selection requires a learned meta-classifier, not a closed-form rule.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*68)
    print("  GeoXGB Geometric Autopsy  —  T / Q / S / Cone / Novelty")
    print("="*68)
    print(f"  Seeds: {SEEDS}  |  Folds: {N_SPLITS}  |  "
          f"Folds/dataset: {len(SEEDS)*N_SPLITS}")
    print()

    datasets = load_datasets()
    print(f"  Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")

    all_fold_results: dict[str, list[dict]] = {}

    for ds_name, (task, X, y) in datasets.items():
        fold_results = run_dataset(ds_name, task, X, y)
        all_fold_results[ds_name] = fold_results
        print_dataset_summary(ds_name, fold_results)

    print_global_summary(all_fold_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
