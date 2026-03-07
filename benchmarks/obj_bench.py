"""
Objective Function Comparison — L2 vs L1 vs Huber
==================================================
Tests whether switching the GBT loss objective from MSE (L2) to MAE (L1)
or Huber changes the competitive landscape between GeoXGB and XGBoost.

Background
----------
mae_bench.py showed that GeoXGB wins 4/5 datasets under normalized per-sample
relative advantage, but loses R² on california and concrete.  The R² loss is
driven by a minority of high-y samples where GeoXGB makes catastrophically large
errors.  L1 / Huber objectives de-emphasize those samples — they contribute
±1 gradient instead of ±Δ gradient, so the trees stop chasing them.

Three objectives tested for XGBoost:
  L2      reg:squarederror       (current baseline)
  L1      reg:absoluteerror
  Huber   reg:pseudohubererror   (delta=1.0 by default in XGBoost)

GeoXGB is shown for reference (C++ L2, current production config).

Note: GeoXGB C++ currently only implements L2.  The experiment answers whether
implementing L1 in GeoXGB would be worthwhile:
  - If XGB_L1 >> XGB_L2 on california/concrete → yes, L1 helps in those regimes
  - If XGB_L1 ≈ XGB_L2                         → loss function isn't the bottleneck

Metrics: R², MAE, NMAE=MAE/std(y_train), net relative advantage (per-sample).
5-fold CV, 3 seeds (15 folds per dataset).
Regression datasets only (objective matters more for regression; classification
probabilities are always Brier/log-loss minimized for both models).
"""

from __future__ import annotations

import io, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml, make_friedman1
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config

# ── Config ─────────────────────────────────────────────────────────────────────

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]
EPS      = 1e-9

GEO_REG = dict(n_rounds=500, learning_rate=0.1, max_depth=3,
               min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
               refit_interval=5, auto_expand=True, expand_ratio=0.1,
               min_train_samples=100, n_bins=64)

_XGB_COMMON = dict(n_estimators=500, learning_rate=0.1, max_depth=3,
                   min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                   verbosity=0, n_jobs=-1)

# XGBoost objectives for regression
OBJECTIVES = {
    "XGB_L2":    "reg:squarederror",       # standard MSE
    "XGB_L1":    "reg:absoluteerror",      # MAE
    "XGB_Huber": "reg:pseudohubererror",   # pseudo-Huber
}

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

    print("  california_housing...", end=" ", flush=True)
    d = fetch_california_housing()
    idx = rng.choice(len(d.data), 8000, replace=False)
    ds["california"] = (d.data[idx].astype(np.float64), d.target[idx])
    print(f"n={len(idx)}, d={d.data.shape[1]}")

    print("  concrete_compressive...", end=" ", flush=True)
    try:
        raw = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = clean_X(raw.data)
        yc = np.asarray(raw.target, dtype=np.float64)
        ds["concrete"] = (Xc, yc)
        print(f"n={len(yc)}, d={Xc.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  friedman1...", end=" ", flush=True)
    Xf, yf = make_friedman1(n_samples=1000, random_state=RNG)
    ds["friedman1"] = (Xf.astype(np.float64), yf.astype(np.float64))
    print(f"n={len(yf)}, d={Xf.shape[1]}")

    return ds


# ── Per-fold evaluation ────────────────────────────────────────────────────────

def fold_metrics_pair(y_val, y_tr, pred_a, pred_b):
    """Metrics comparing two predictions (a=GeoXGB, b=XGB variant)."""
    e_a = np.abs(y_val - pred_a)
    e_b = np.abs(y_val - pred_b)
    rel_adv = (e_b - e_a) / (e_b + e_a + EPS)   # negative = a wins
    y_std = float(np.std(y_tr)) if len(y_tr) > 1 else 1.0
    return dict(
        r2_a  = r2_score(y_val, pred_a),
        r2_b  = r2_score(y_val, pred_b),
        mae_a = float(np.mean(e_a)),
        mae_b = float(np.mean(e_b)),
        nmae_a= float(np.mean(e_a)) / (y_std + EPS),
        nmae_b= float(np.mean(e_b)) / (y_std + EPS),
        rel   = rel_adv,
    )


def run_dataset(ds_name, X, y):
    """Run all XGB objectives + GeoXGB across all seeds/folds."""
    results = {name: [] for name in list(OBJECTIVES.keys()) + ["GeoXGB"]}

    for seed in SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        # Pre-fit GeoXGB once per fold (same seed for fair comparison)
        geo_preds = []
        for tr, va in kf.split(X, y):
            cfg = make_cpp_config(**{**GEO_REG, "random_state": seed})
            geo = CppGeoXGBRegressor(cfg)
            geo.fit(X[tr], y[tr])
            geo_preds.append((tr, va, geo.predict(X[va])))

        # XGB objectives
        for obj_name, objective in OBJECTIVES.items():
            kf2 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for (tr, va, pred_geo), (tr2, va2) in zip(geo_preds, kf2.split(X, y)):
                assert np.array_equal(tr, tr2), "fold mismatch"
                m = xgb.XGBRegressor(**{**_XGB_COMMON, "random_state": seed,
                                        "objective": objective})
                m.fit(X[tr], y[tr])
                pred_xgb = m.predict(X[va])
                fm = fold_metrics_pair(y[va], y[tr], pred_geo, pred_xgb)
                results[obj_name].append(fm)

        # Store GeoXGB scores vs L2 baseline for reference
        kf3 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for (tr, va, pred_geo), (tr3, va3) in zip(geo_preds, kf3.split(X, y)):
            m_l2 = xgb.XGBRegressor(**{**_XGB_COMMON, "random_state": seed,
                                        "objective": "reg:squarederror"})
            m_l2.fit(X[tr], y[tr])
            pred_l2 = m_l2.predict(X[va])
            fm = fold_metrics_pair(y[va], y[tr], pred_geo, pred_l2)
            results["GeoXGB"].append(fm)

        print(f"    seed={seed}", end=" ", flush=True)

    print()
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def agg(results, key):
    vals = [f[key] for f in results]
    return float(np.mean(vals)), float(np.std(vals))


def report_dataset(ds_name, results):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {ds_name}   ({len(results['GeoXGB'])} folds)")
    print(sep)

    # GeoXGB absolute performance
    r2g_m, r2g_s = agg(results["GeoXGB"], "r2_a")
    mag_m, mag_s = agg(results["GeoXGB"], "mae_a")
    nmg_m, nmg_s = agg(results["GeoXGB"], "nmae_a")
    print(f"\n  GeoXGB (L2, reference): R²={r2g_m:+.4f}(±{r2g_s:.4f})  "
          f"MAE={mag_m:.4f}  NMAE={nmg_m:.4f}")

    # Compare each XGB objective vs GeoXGB
    print(f"\n  {'Objective':<14}  {'XGB R²':>9}  {'GeoXGB wins R²?':>16}  "
          f"{'XGB NMAE':>9}  {'GeoXGB wins NMAE?':>18}  "
          f"{'net_rel_adv':>12}")
    print("  " + "-" * 82)

    summary_rows = []
    for obj_name in list(OBJECTIVES.keys()):
        res = results[obj_name]
        r2b_m, _ = agg(res, "r2_b")
        mab_m, _ = agg(res, "mae_b")
        nmb_m, _ = agg(res, "nmae_b")
        rel_all   = np.concatenate([f["rel"] for f in res])
        net_rel   = float(rel_all.mean())

        geo_r2_better  = "YES" if r2g_m > r2b_m else "no"
        geo_nmae_better = "YES" if nmg_m < nmb_m else "no"
        print(f"  {obj_name:<14}  {r2b_m:>+9.4f}  {geo_r2_better:>16}  "
              f"{nmb_m:>9.4f}  {geo_nmae_better:>18}  {net_rel:>+12.4f}")
        summary_rows.append(dict(obj=obj_name, r2_xgb=r2b_m, nmae_xgb=nmb_m,
                                  net_rel=net_rel))

    # Key insight: does switching XGB from L2→L1/Huber change whether GeoXGB wins?
    l2 = next(r for r in summary_rows if r["obj"] == "XGB_L2")
    l1 = next(r for r in summary_rows if r["obj"] == "XGB_L1")
    hb = next(r for r in summary_rows if r["obj"] == "XGB_Huber")

    print(f"\n  XGB objective effect (relative to L2 baseline):")
    print(f"    L1 vs L2 XGB R² delta:    {l1['r2_xgb'] - l2['r2_xgb']:+.4f}  "
          f"({'L1 better' if l1['r2_xgb'] > l2['r2_xgb'] else 'L2 better'})")
    print(f"    Huber vs L2 XGB R² delta: {hb['r2_xgb'] - l2['r2_xgb']:+.4f}  "
          f"({'Huber better' if hb['r2_xgb'] > l2['r2_xgb'] else 'L2 better'})")
    print(f"    GeoXGB net_rel_adv vs L1: {l1['net_rel']:+.4f}  "
          f"({'GeoXGB wins' if l1['net_rel'] < 0 else 'XGB_L1 wins'})")

    return summary_rows


def global_summary(all_summaries):
    print("\n" + "=" * 72)
    print("  GLOBAL SUMMARY")
    print("=" * 72)

    print(f"\n  {'Dataset':<14}  {'XGB_L2 R²':>10}  {'XGB_L1 R²':>10}  "
          f"{'XGB_Hub R²':>10}  {'GeoXGB R²':>10}")
    print("  " + "-" * 58)
    for ds_name, rows, geo_r2 in all_summaries:
        l2 = next(r for r in rows if r["obj"] == "XGB_L2")
        l1 = next(r for r in rows if r["obj"] == "XGB_L1")
        hb = next(r for r in rows if r["obj"] == "XGB_Huber")
        print(f"  {ds_name:<14}  {l2['r2_xgb']:>+10.4f}  {l1['r2_xgb']:>+10.4f}  "
              f"{hb['r2_xgb']:>+10.4f}  {geo_r2:>+10.4f}")

    print(f"\n  Net relative advantage (GeoXGB vs each XGB objective), lower = GeoXGB better:")
    print(f"  {'Dataset':<14}  {'vs L2':>8}  {'vs L1':>8}  {'vs Huber':>9}")
    print("  " + "-" * 42)
    for ds_name, rows, _ in all_summaries:
        l2 = next(r for r in rows if r["obj"] == "XGB_L2")
        l1 = next(r for r in rows if r["obj"] == "XGB_L1")
        hb = next(r for r in rows if r["obj"] == "XGB_Huber")
        print(f"  {ds_name:<14}  {l2['net_rel']:>+8.4f}  {l1['net_rel']:>+8.4f}  "
              f"{hb['net_rel']:>+9.4f}")

    print(f"\n  Interpretation:")
    print(f"    If XGB_L1 R² << XGB_L2 R² → MAE objective hurts XGBoost → L2 is the right baseline")
    print(f"    If GeoXGB net_rel_adv improves vs XGB_L1 → GeoXGB is more L1-compatible")
    print(f"    If GeoXGB net_rel_adv degrades vs XGB_L1 → GeoXGB needs L1 training objective too")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  GeoXGB — Objective Function Comparison (L2 vs L1 vs Huber)")
    print("=" * 72)
    print()

    print("Loading datasets...")
    datasets = load_datasets()
    print(f"  Loaded {len(datasets)} regression datasets.\n")

    all_summaries = []
    for ds_name, (X, y) in datasets.items():
        print(f"  Running {ds_name} ...", flush=True)
        t0 = time.perf_counter()
        results = run_dataset(ds_name, X, y)
        print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
        rows = report_dataset(ds_name, results)
        geo_r2, _ = agg(results["GeoXGB"], "r2_a")
        all_summaries.append((ds_name, rows, geo_r2))

    global_summary(all_summaries)


if __name__ == "__main__":
    main()
