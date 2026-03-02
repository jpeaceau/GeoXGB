"""
GeoXGB vs XGBoost — Residual Pattern Analysis
==============================================
Asks: what structural properties of a validation sample predict whether
XGBoost beats GeoXGB (or vice versa)?

This is distinct from the geometric autopsy (which looked at T, Q, S in
z-space). Here we operate entirely in the original feature space and ask
whether there is a learnable rule in X that explains model advantage.

Per dataset, pools predictions across all seeds × folds, then runs:
  A. Feature Spearman correlations with Δ_abs
  B. Depth-3 decision tree trained to predict "XGB wins"
  C. Error direction — does GeoXGB over- or under-predict in loss regions?
  D. Target quantile analysis — does one model dominate at high/low y?
  E. Density proxy — 5-NN distance in X vs advantage
  F. Win-margin distribution — are losses large or marginal?

Datasets: california_housing, concrete_compressive, breast_cancer,
          ionosphere, friedman1
3 seeds × 5 folds = 15 folds per dataset.
"""

from __future__ import annotations

import io, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats
from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                               fetch_openml, make_friedman1)
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
from geoxgb._cpp_backend import (
    CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config)

# ── Config ─────────────────────────────────────────────────────────────────────

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]

GEO_REG = dict(n_rounds=500, learning_rate=0.1, max_depth=3,
               min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
               refit_interval=5, auto_expand=True, expand_ratio=0.1,
               min_train_samples=100, n_bins=64)
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
    ds, feat_names = {}, {}
    rng = np.random.RandomState(RNG)

    d = fetch_california_housing()
    idx = rng.choice(len(d.data), 8000, replace=False)
    ds["california_housing"] = ("reg", d.data[idx].astype(np.float64), d.target[idx])
    feat_names["california_housing"] = list(d.feature_names)

    try:
        raw = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = clean_X(raw.data); yc = np.asarray(raw.target, dtype=np.float64)
        ds["concrete_compressive"] = ("reg", Xc, yc)
        feat_names["concrete_compressive"] = list(raw.data.columns)
    except Exception as e:
        print(f"  concrete SKIP: {e}")

    d2 = load_breast_cancer()
    ds["breast_cancer"] = ("clf", d2.data.astype(np.float64), d2.target.astype(np.float64))
    feat_names["breast_cancer"] = list(d2.feature_names)

    try:
        raw2 = fetch_openml(data_id=59, as_frame=True, parser="auto")
        Xi = clean_X(raw2.data); yi = (raw2.target == "g").astype(np.float64).values
        ds["ionosphere"] = ("clf", Xi, yi)
        feat_names["ionosphere"] = [f"f{i}" for i in range(Xi.shape[1])]
    except Exception as e:
        print(f"  ionosphere SKIP: {e}")

    Xf, yf = make_friedman1(n_samples=1000, random_state=RNG)
    ds["friedman1"] = ("reg", Xf.astype(np.float64), yf.astype(np.float64))
    feat_names["friedman1"] = [f"x{i}" for i in range(10)]

    return ds, feat_names


# ── Per-fold prediction collection ────────────────────────────────────────────

def collect_fold(X_tr, y_tr, X_val, y_val, task, seed):
    base = GEO_REG if task == "reg" else GEO_CLF
    cfg  = make_cpp_config(**{**base, "random_state": seed})

    if task == "reg":
        geo = CppGeoXGBRegressor(cfg)
        geo.fit(X_tr, y_tr)
        pred_g = geo.predict(X_val)
        xgm = xgb.XGBRegressor(**{**XGB_REG, "random_state": seed})
        xgm.fit(X_tr, y_tr)
        pred_x = xgm.predict(X_val)
    else:
        geo = CppGeoXGBClassifier(cfg)
        geo.fit(X_tr, y_tr)
        pred_g = geo.predict_proba(X_val)[:, 1]
        xgm = xgb.XGBClassifier(**{**XGB_CLF, "random_state": seed})
        xgm.fit(X_tr, y_tr)
        pred_x = xgm.predict_proba(X_val)[:, 1]

    return pred_g, pred_x


def run_dataset(ds_name, task, X, y):
    """Pool predictions across all seeds and folds."""
    all_X, all_y, all_pg, all_px = [], [], [], []
    kf_cls = StratifiedKFold if task == "clf" else KFold

    for seed in SEEDS:
        kf = kf_cls(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        split_y = y.astype(int) if task == "clf" else y
        for tr, va in kf.split(X, split_y):
            pg, px = collect_fold(X[tr], y[tr], X[va], y[va], task, seed)
            all_X.append(X[va])
            all_y.append(y[va])
            all_pg.append(pg)
            all_px.append(px)
        print(f"    seed={seed} done", flush=True)

    X_pool  = np.vstack(all_X)
    y_pool  = np.concatenate(all_y)
    pg_pool = np.concatenate(all_pg)
    px_pool = np.concatenate(all_px)
    return X_pool, y_pool, pg_pool, px_pool


# ── Analysis functions ─────────────────────────────────────────────────────────

def feature_correlations(X, delta_abs, feat_names):
    """Spearman(feature_j, delta_abs) for each j. Positive = XGB wins when high."""
    results = []
    for j in range(X.shape[1]):
        r, p = stats.spearmanr(X[:, j], delta_abs)
        results.append((feat_names[j], float(r), float(p)))
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


def decision_tree_analysis(X, xgb_wins, feat_names, n_val):
    """Depth-3 decision tree predicting 'XGB wins' from X.

    Returns (accuracy, baseline_acc, tree_text, feature_importances).
    n_val: number of unique validation samples per seed (for train/test split).
    """
    # Use first n_val samples (seed=0 folds) as test, rest as train
    # This avoids the same samples appearing in both train and test
    X_test,  y_test  = X[:n_val],  xgb_wins[:n_val]
    X_train, y_train = X[n_val:],  xgb_wins[n_val:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
    dt.fit(X_train_s, y_train)

    acc       = float(dt.score(X_test_s, y_test))
    baseline  = float(max(y_test.mean(), 1 - y_test.mean()))
    tree_text = export_text(dt, feature_names=feat_names, decimals=3)

    # Restore original-scale thresholds in the text (approximately)
    fi = [(feat_names[i], float(v)) for i, v in enumerate(dt.feature_importances_)]
    fi.sort(key=lambda x: x[1], reverse=True)
    return acc, baseline, tree_text, fi


def error_direction(eps_g, eps_x, delta_abs, task):
    """Signed error bias in XGB-win vs GeoXGB-win regions."""
    pct75 = np.percentile(delta_abs, 75)
    pct25 = np.percentile(delta_abs, 25)
    xgb_wins_mask  = delta_abs >= pct75
    geo_wins_mask  = delta_abs <= pct25

    if task == "reg":
        return {
            "xgb_wins_mean_eps_g":  float(eps_g[xgb_wins_mask].mean()),
            "xgb_wins_mean_eps_x":  float(eps_x[xgb_wins_mask].mean()),
            "geo_wins_mean_eps_g":  float(eps_g[geo_wins_mask].mean()),
            "geo_wins_mean_eps_x":  float(eps_x[geo_wins_mask].mean()),
            "xgb_wins_std_eps_g":   float(eps_g[xgb_wins_mask].std()),
            "geo_wins_std_eps_g":   float(eps_g[geo_wins_mask].std()),
        }
    else:
        # For classification: confidence proximity to 0.5
        pg_abs  = np.abs(eps_g)   # |true - prob_G|
        px_abs  = np.abs(eps_x)
        return {
            "xgb_wins_mean_absepsg": float(pg_abs[xgb_wins_mask].mean()),
            "xgb_wins_mean_absepsx": float(px_abs[xgb_wins_mask].mean()),
            "geo_wins_mean_absepsg": float(pg_abs[geo_wins_mask].mean()),
            "geo_wins_mean_absepsx": float(px_abs[geo_wins_mask].mean()),
        }


def target_quantile(y, delta_abs, n_quantiles=4):
    """Mean delta_abs per target quantile."""
    qs = np.quantile(y, np.linspace(0, 1, n_quantiles + 1))
    rows = []
    for i in range(n_quantiles):
        mask = (y >= qs[i]) & (y < qs[i+1]) if i < n_quantiles - 1 else (y >= qs[i])
        rows.append((float(qs[i]), float(qs[i+1]), float(delta_abs[mask].mean()),
                     int(mask.sum())))
    return rows


def density_analysis(X, delta_abs, k=5):
    """5-NN distance in normalised X as density proxy; Spearman with delta_abs."""
    Xs = StandardScaler().fit_transform(X)
    # Subsample for speed if large
    cap = 5000
    if len(X) > cap:
        idx = np.random.RandomState(42).choice(len(X), cap, replace=False)
        Xs_s, d_s = Xs[idx], delta_abs[idx]
    else:
        Xs_s, d_s = Xs, delta_abs

    norms = (Xs_s ** 2).sum(axis=1)
    D = norms[:, None] - 2.0 * (Xs_s @ Xs_s.T) + norms[None, :]
    D = np.clip(D, 0.0, None)
    np.fill_diagonal(D, np.inf)
    knn_dists = np.sort(D, axis=1)[:, :k].mean(axis=1)

    r, p = stats.spearmanr(knn_dists, d_s)
    return float(r), float(p), knn_dists


def win_margin(delta_abs):
    """Distribution of win margins: how big are XGB's wins vs GeoXGB's wins."""
    xgb_wins  = delta_abs[delta_abs > 0]
    geo_wins  = delta_abs[delta_abs < 0]
    return {
        "xgb_pct":       100.0 * float((delta_abs > 0).mean()),
        "geo_pct":       100.0 * float((delta_abs < 0).mean()),
        "xgb_mean_win":  float(xgb_wins.mean()) if len(xgb_wins) else 0.0,
        "geo_mean_win":  float(-geo_wins.mean()) if len(geo_wins) else 0.0,
        "xgb_p90_win":   float(np.percentile(xgb_wins, 90)) if len(xgb_wins) else 0.0,
        "geo_p90_win":   float(np.percentile(-geo_wins, 90)) if len(geo_wins) else 0.0,
    }


# ── Dataset-level reporting ────────────────────────────────────────────────────

def report_dataset(ds_name, task, X_pool, y_pool, pg_pool, px_pool, fnames):
    n = len(y_pool)
    eps_g = y_pool - pg_pool
    eps_x = y_pool - px_pool
    delta_abs = np.abs(eps_g) - np.abs(eps_x)   # +ve = XGB wins

    n_val = len(y_pool) // (len(SEEDS) * N_SPLITS) * N_SPLITS  # samples from seed 0

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Dataset: {ds_name}   (n_pooled={n}, d={X_pool.shape[1]}, {task})")
    print(sep)

    # ── Win margin ───────────────────────────────────────────────────────────
    wm = win_margin(delta_abs)
    print(f"\n  Win rates:  XGBoost={wm['xgb_pct']:.1f}%  GeoXGB={wm['geo_pct']:.1f}%")
    print(f"  Mean win margin:  XGBoost={wm['xgb_mean_win']:.4f}  GeoXGB={wm['geo_mean_win']:.4f}")
    print(f"  90th-pct win:     XGBoost={wm['xgb_p90_win']:.4f}  GeoXGB={wm['geo_p90_win']:.4f}")

    # ── A: Feature correlations ───────────────────────────────────────────────
    print(f"\n  A. Feature Spearman correlations with delta_abs  (+ = XGB wins when high)")
    print(f"     {'Feature':<30}  {'rho':>7}  {'p':>8}  sig")
    corrs = feature_correlations(X_pool, delta_abs, fnames)
    for fname, r, p in corrs[:8]:
        sig = "***" if p < 0.001 else ("** " if p < 0.01 else (" * " if p < 0.05 else "   "))
        print(f"     {fname:<30}  {r:>+7.3f}  {p:>8.4f}  {sig}")
    if len(corrs) > 8:
        print(f"     ... ({len(corrs) - 8} more features not shown)")

    # ── B: Decision tree ─────────────────────────────────────────────────────
    print(f"\n  B. Decision tree (depth=3): can X predict 'XGB wins'?")
    xgb_wins_label = (delta_abs > 0).astype(int)
    acc, baseline, tree_txt, fi = decision_tree_analysis(
        X_pool, xgb_wins_label, fnames, n_val)
    lift = acc - baseline
    print(f"     Accuracy={acc:.3f}  baseline={baseline:.3f}  lift={lift:+.3f}")
    if abs(lift) > 0.01:
        print(f"     Top features: " +
              "  ".join(f"{f}({v:.2f})" for f, v in fi[:5] if v > 0.01))
        # Print tree with indent, trimmed to reasonable length
        lines = tree_txt.strip().split("\n")
        for ln in lines[:30]:
            print(f"       {ln}")
        if len(lines) > 30:
            print(f"       ... ({len(lines)-30} more lines)")
    else:
        print(f"     No meaningful pattern — lift < 0.01 (decision tree cannot predict winner)")

    # ── C: Error direction ────────────────────────────────────────────────────
    print(f"\n  C. Error direction in extreme-advantage regions (top/bottom 25%)")
    ed = error_direction(eps_g, eps_x, delta_abs, task)
    if task == "reg":
        print(f"     {'Region':<22}  {'mean(eps_G)':>12}  {'mean(eps_X)':>12}  {'std(eps_G)':>12}")
        print(f"     {'XGB wins (Q4 delta)':22}  {ed['xgb_wins_mean_eps_g']:>+12.4f}"
              f"  {ed['xgb_wins_mean_eps_x']:>+12.4f}  {ed['xgb_wins_std_eps_g']:>12.4f}")
        print(f"     {'GeoXGB wins (Q1 delta)':22}  {ed['geo_wins_mean_eps_g']:>+12.4f}"
              f"  {ed['geo_wins_mean_eps_x']:>+12.4f}  {ed['geo_wins_std_eps_g']:>12.4f}")
        g_xgb = ed['xgb_wins_mean_eps_g']
        direction = "OVER-predicts" if g_xgb > 0.05 * np.std(eps_g) else \
                    ("UNDER-predicts" if g_xgb < -0.05 * np.std(eps_g) else "centred")
        print(f"     GeoXGB {direction} in regions where XGBoost wins")
    else:
        print(f"     {'Region':<22}  {'|eps_G|':>8}  {'|eps_X|':>8}")
        print(f"     {'XGB wins (Q4 delta)':22}  {ed['xgb_wins_mean_absepsg']:>8.4f}"
              f"  {ed['xgb_wins_mean_absepsx']:>8.4f}")
        print(f"     {'GeoXGB wins (Q1 delta)':22}  {ed['geo_wins_mean_absepsg']:>8.4f}"
              f"  {ed['geo_wins_mean_absepsx']:>8.4f}")

    # ── D: Target quantile ────────────────────────────────────────────────────
    print(f"\n  D. Target (y) quantile analysis")
    print(f"     {'Quantile':>10}  {'y_range':>20}  {'mean_delta':>12}  {'n':>6}")
    rows = target_quantile(y_pool, delta_abs, n_quantiles=4)
    for i, (lo, hi, md, cnt) in enumerate(rows):
        label = f"Q{i+1}"
        print(f"     {label:>10}  [{lo:>8.3f}, {hi:>8.3f}]  {md:>+12.4f}  {cnt:>6}")
    diffs = [r[2] for r in rows]
    trend = "increases" if diffs[-1] > diffs[0] + 0.001 else \
            ("decreases" if diffs[-1] < diffs[0] - 0.001 else "flat")
    print(f"     XGB advantage {trend} with target value")

    # ── E: Density ────────────────────────────────────────────────────────────
    print(f"\n  E. Local density vs advantage")
    dr, dp, knn_d = density_analysis(X_pool, delta_abs)
    sig_d = "***" if dp < 0.001 else ("** " if dp < 0.01 else (" * " if dp < 0.05 else "   "))
    interpretation = ("XGB wins more in SPARSE regions" if dr > 0.02 else
                      ("XGB wins more in DENSE regions" if dr < -0.02 else
                       "Density does not predict advantage"))
    print(f"     rho(5NN_dist, delta_abs) = {dr:+.3f}  p={dp:.4f}  {sig_d}")
    print(f"     -> {interpretation}")

    return {
        "ds": ds_name, "task": task,
        "xgb_pct": wm["xgb_pct"], "geo_pct": wm["geo_pct"],
        "xgb_mean_win": wm["xgb_mean_win"], "geo_mean_win": wm["geo_mean_win"],
        "top_feat": corrs[0][0] if corrs else "?",
        "top_rho":  corrs[0][1] if corrs else 0.0,
        "tree_lift": lift,
        "density_rho": dr, "density_sig": dp,
        "target_trend": trend,
        "error_dir": ed,
        "quantile_rows": rows,
        "feat_corrs": corrs,
    }


# ── Global summary ─────────────────────────────────────────────────────────────

def global_summary(summaries):
    print("\n" + "=" * 68)
    print("  GLOBAL SUMMARY")
    print("=" * 68)

    print(f"\n  {'Dataset':<28}  {'XGB%':>6}  {'Geo%':>6}  "
          f"{'XGB_win':>8}  {'Geo_win':>8}  {'top_feat':<22}  {'lift':>6}")
    print("  " + "-" * 92)
    for s in summaries:
        print(f"  {s['ds']:<28}  {s['xgb_pct']:>5.1f}%  {s['geo_pct']:>5.1f}%  "
              f"  {s['xgb_mean_win']:>7.4f}  {s['geo_mean_win']:>7.4f}  "
              f"{s['top_feat']:<22}  {s['tree_lift']:>+6.3f}")

    # Density consensus
    print(f"\n  Density pattern (rho > 0 = XGB wins in sparse):")
    for s in summaries:
        sig = "*" if s['density_sig'] < 0.05 else " "
        print(f"    {s['ds']:<30}  rho={s['density_rho']:+.3f}{sig}")

    # Target quantile trend consensus
    print(f"\n  Target quantile trend:")
    for s in summaries:
        print(f"    {s['ds']:<30}  {s['target_trend']}")

    # Feature consistency: which features appear most in top positions?
    print(f"\n  Top predictive features (most consistently correlated with XGB advantage):")
    feat_votes: dict[str, list[float]] = {}
    for s in summaries:
        for fname, r, p in s["feat_corrs"][:3]:
            if fname not in feat_votes:
                feat_votes[fname] = []
            feat_votes[fname].append(r)
    for fname, rhos in sorted(feat_votes.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        if len(rhos) >= 2:
            print(f"    {fname:<30}  n_datasets={len(rhos)}  mean_rho={np.mean(rhos):+.3f}")

    # Error direction consensus for regression
    reg_sums = [s for s in summaries if s["task"] == "reg"]
    if reg_sums:
        print(f"\n  Error direction in XGB-wins regions (regression only):")
        for s in reg_sums:
            ed = s["error_dir"]
            bias = ed["xgb_wins_mean_eps_g"]
            std_g = ed.get("xgb_wins_std_eps_g", 1.0)
            direction = "over-predicts" if bias > 0.05 * std_g else \
                        ("under-predicts" if bias < -0.05 * std_g else "centred")
            print(f"    {s['ds']:<30}  GeoXGB {direction:>15}  "
                  f"(mean eps_G={bias:+.4f})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  GeoXGB vs XGBoost — Residual Pattern Analysis")
    print("=" * 68)
    print(f"  Seeds: {SEEDS}  |  Folds: {N_SPLITS}  |  Folds/dataset: {len(SEEDS)*N_SPLITS}")
    print()

    datasets, feat_names = load_datasets()
    print(f"  Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}\n")

    summaries = []
    for ds_name, (task, X, y) in datasets.items():
        print(f"  Running {ds_name} ...", flush=True)
        t0 = time.perf_counter()
        X_pool, y_pool, pg_pool, px_pool = run_dataset(ds_name, task, X, y)
        print(f"  done ({time.perf_counter()-t0:.0f}s)", flush=True)
        fnames = feat_names.get(ds_name, [f"f{i}" for i in range(X.shape[1])])
        s = report_dataset(ds_name, task, X_pool, y_pool, pg_pool, px_pool, fnames)
        summaries.append(s)

    global_summary(summaries)
    print()


if __name__ == "__main__":
    main()
