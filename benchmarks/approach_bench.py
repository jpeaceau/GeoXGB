"""
GeoXGB Architecture Approaches Benchmark
=========================================
Tests two new architectural strategies against baseline and XGBoost on 4
real-world datasets (2 regression where GeoXGB currently loses, 2 classification
where it currently wins).

Approach 1 — Selective target (selective_target=True):
  At each HVRT refit, replaces the static full-pairwise cooperation target with
  the top-k feature-pair products ranked by |Pearson(pair, residuals)|.  For low-d
  data (d=8, 28 pairs), most pairs are orthogonal to the gradient — this focuses
  the HVRT partition tree on the few pairs that actually predict the residuals.

Approach 2 — Adaptive d-threshold (d_geom_threshold=12):
  When d <= 12, skip HVRT entirely and run pure GBT on the full dataset.
  The pairwise cooperation target requires d*(d-1)/2 pairs; for d<=8 that is
  only 28 pairs, too sparse to reliably guide partitions.  Eliminating HVRT
  avoids discarding informative training points and removes the resampling overhead.

Combined — A1+A2:
  A2 handles low-d (skip HVRT); A1 handles high-d (guide HVRT with selective target).
  A1 is inert when A2 triggers (HVRT disabled), so no interaction.

Datasets:
  california_housing  — regression, n=8000, d=8   (XGBoost wins by ~0.02)
  concrete_compressive— regression, n=1030, d=8   (XGBoost wins by ~0.03)
  breast_cancer       — classification, n=569, d=30 (GeoXGB wins by ~0.002)
  ionosphere          — classification, n=351, d=34 (GeoXGB wins by ~0.011)

5-fold CV, 3 seeds.  Reports metric ± std, delta vs baseline, and vs XGBoost.
"""
import sys, io, warnings, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

from geoxgb._cpp_backend import CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]

BASE_REG = dict(
    n_rounds=500, learning_rate=0.1, max_depth=3,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64,
)
BASE_CLF = dict(
    n_rounds=500, learning_rate=0.1, max_depth=5,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64,
)
XGB_REG = dict(
    n_estimators=500, learning_rate=0.1, max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RNG, verbosity=0, n_jobs=-1,
)
XGB_CLF = dict(
    n_estimators=500, learning_rate=0.1, max_depth=5,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RNG, verbosity=0, n_jobs=-1,
)

VARIANTS = [
    ("baseline",    {}),
    ("A1_selective",   dict(selective_target=True)),
    ("A2_dthresh12",   dict(d_geom_threshold=12)),
    ("A1+A2",          dict(selective_target=True, d_geom_threshold=12)),
    ("A3_rc25",        dict(residual_correct_lambda=25.0)),
    ("A3_rc50",        dict(residual_correct_lambda=50.0)),
    ("A1+A3_rc25",     dict(selective_target=True, residual_correct_lambda=25.0)),
    ("XGBoost",        None),   # sentinel; handled separately
]


def clean_X(df):
    if not isinstance(df, pd.DataFrame):
        arr = np.asarray(df, dtype=np.float64)
        if np.isnan(arr).any():
            arr = SimpleImputer(strategy="median").fit_transform(arr)
        return arr
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if num_cols and df[num_cols].isnull().any().any():
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols])
    return df.values.astype(np.float64)


def load_datasets():
    datasets = {}
    print("Loading datasets...")

    print("  california_housing...", end=" ", flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.RandomState(RNG)
    idx = rng.choice(len(X), 8000, replace=False)
    datasets["california_housing"] = ("reg", X[idx].astype(np.float64), y[idx])
    print(f"n={len(idx)}, d={X.shape[1]}")

    print("  concrete_compressive...", end=" ", flush=True)
    try:
        data = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = clean_X(data.data); yc = np.asarray(data.target, dtype=np.float64)
        datasets["concrete_compressive"] = ("reg", Xc, yc)
        print(f"n={len(yc)}, d={Xc.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  breast_cancer...", end=" ", flush=True)
    Xb, yb = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer"] = ("clf", Xb.astype(np.float64), yb.astype(np.float64))
    print(f"n={len(yb)}, d={Xb.shape[1]}")

    print("  ionosphere...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=59, as_frame=True, parser="auto")
        Xi = clean_X(data.data); yi = (data.target == "g").astype(np.float64).values
        datasets["ionosphere"] = ("clf", Xi, yi)
        print(f"n={len(yi)}, d={Xi.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    return datasets


def cv_geo(task, X, y, extra_kwargs, seed):
    base = BASE_REG if task == "reg" else BASE_CLF
    params = {**base, "random_state": seed, **extra_kwargs}
    cfg = make_cpp_config(**params)
    kf = (KFold if task == "reg" else StratifiedKFold)(
        n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X, y.astype(int) if task == "clf" else y):
        if task == "reg":
            m = CppGeoXGBRegressor(cfg)
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[va], m.predict(X[va])))
        else:
            m = CppGeoXGBClassifier(cfg)
            m.fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return scores


def cv_xgb(task, X, y, seed):
    params = {**(XGB_REG if task == "reg" else XGB_CLF), "random_state": seed}
    kf = (KFold if task == "reg" else StratifiedKFold)(
        n_splits=N_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X, y.astype(int) if task == "clf" else y):
        if task == "reg":
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[va], m.predict(X[va])))
        else:
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return scores


def run_all(datasets):
    all_results = {var: {} for var, _ in VARIANTS}

    for var_name, extra_kwargs in VARIANTS:
        t0 = time.perf_counter()
        print(f"  Running {var_name}...", end=" ", flush=True)
        for ds_name, (task, X, y) in datasets.items():
            folds = []
            for seed in SEEDS:
                if extra_kwargs is None:
                    folds.extend(cv_xgb(task, X, y, seed))
                else:
                    folds.extend(cv_geo(task, X, y, extra_kwargs, seed))
            all_results[var_name][ds_name] = np.array(folds)
        print(f"done  ({time.perf_counter()-t0:.0f}s)")

    return all_results


def print_table(datasets, all_results):
    metric_map = {"california_housing": "R2", "concrete_compressive": "R2",
                  "breast_cancer": "AUC", "ionosphere": "AUC"}
    baseline = all_results["baseline"]
    xgb_res  = all_results["XGBoost"]

    GEO_VARIANTS = ["A1_selective", "A2_dthresh12", "A1+A2",
                    "A3_rc25", "A3_rc50", "A1+A3_rc25"]

    print()
    print(f"{'Dataset':<26} {'M':>3}  {'baseline':>10}  "
          + "  ".join(f"{v:>14}" for v in GEO_VARIANTS)
          + f"  {'XGBoost':>14}")
    print("-" * (26 + 4 + 12 + len(GEO_VARIANTS) * 16 + 16))
    for ds_name, (task, X, y) in datasets.items():
        metric = metric_map.get(ds_name, "?")
        row = f"  {ds_name:<24} {metric:>3}"
        b_mean = baseline[ds_name].mean()
        row += f"  {b_mean:>+.4f}    "
        for vname in GEO_VARIANTS:
            s = all_results[vname][ds_name]
            row += f"  {s.mean():+.4f}({s.mean()-b_mean:+.4f})"
        xm = xgb_res[ds_name].mean()
        row += f"  {xm:>+.4f}({xm-b_mean:+.4f})"
        print(row)

    print()
    print("  Format: mean(delta_vs_baseline)")
    print()

    print(f"  {'Variant':<16}  {'mean_d_vs_base':>16}  {'reg_d':>8}  {'clf_d':>8}  {'vs_XGB':>8}")
    print("  " + "-" * 64)
    reg_ds = [k for k, (t, _, _) in datasets.items() if t == "reg"]
    clf_ds = [k for k, (t, _, _) in datasets.items() if t == "clf"]

    for vname in GEO_VARIANTS + ["XGBoost"]:
        res = all_results[vname]
        deltas_b = [res[d].mean() - baseline[d].mean() for d in datasets]
        delta_xgb = np.mean([res[d].mean() - xgb_res[d].mean() for d in datasets])
        rdelta = np.mean([res[d].mean() - baseline[d].mean() for d in reg_ds if d in res])
        cdelta = np.mean([res[d].mean() - baseline[d].mean() for d in clf_ds if d in res])
        print(f"  {vname:<16}  {np.mean(deltas_b):>+16.4f}  {rdelta:>+8.4f}  "
              f"{cdelta:>+8.4f}  {delta_xgb:>+8.4f}")
    print()


def main():
    print("=" * 72)
    print("  GeoXGB Approach 1 + 2 Benchmark")
    print("=" * 72)
    datasets = load_datasets()
    if not datasets:
        print("No datasets. Exiting."); return

    print()
    all_results = run_all(datasets)
    print_table(datasets, all_results)

    # Recommendation
    geo_names = [v for v, _ in VARIANTS if v != "XGBoost"]
    gains = {v: np.mean([all_results[v][d].mean() - all_results["baseline"][d].mean()
                          for d in datasets])
             for v in geo_names if v != "baseline"}
    best = max(gains, key=gains.get)
    print(f"  Best GeoXGB variant: {best}  (mean Δ_vs_baseline = {gains[best]:+.4f})")
    xgb_gap = np.mean([all_results["baseline"][d].mean() - all_results["XGBoost"][d].mean()
                        for d in datasets])
    best_gap = np.mean([all_results[best][d].mean() - all_results["XGBoost"][d].mean()
                         for d in datasets])
    print(f"  Baseline vs XGBoost: {xgb_gap:+.4f}  |  {best} vs XGBoost: {best_gap:+.4f}")
    if best_gap > xgb_gap:
        print(f"  → {best} closes the XGBoost gap by {abs(best_gap - xgb_gap):.4f}.")
    else:
        print(f"  → No approach closes the XGBoost gap on mean score.")
    print()


if __name__ == "__main__":
    main()
