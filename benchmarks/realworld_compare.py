"""
GeoXGB vs XGBoost — Real-World Dataset Benchmark
=================================================
Purely real-world datasets only (no synthetic, no hybrid).
5-fold CV, matched parameter budgets, no pre-normalization of X.

Regression  (metric: R²):
  california_housing        — 1990 US Census geographic/housing data
  concrete_compressive      — Yeh (1998) compressive strength measurements
  abalone                   — UCI physical measurements of abalones
  wine_quality              — Portuguese red wine physicochemistry

Classification  (metric: ROC-AUC):
  breast_cancer             — fine-needle aspirate (medical imaging, UCI)
  credit-g                  — German bank credit risk (Statlog)
  adult                     — 1994 US Census income (10k subsample)
  vehicle                   — Statlog vehicle silhouette images (4-class)
"""
import sys, io, warnings, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

from geoxgb._cpp_backend import CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config

RNG = 42
N_SPLITS = 5

GEO_REG_PARAMS = dict(
    n_rounds=500, learning_rate=0.1, max_depth=3,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64, random_state=RNG,
)
GEO_CLF_PARAMS = dict(
    n_rounds=500, learning_rate=0.1, max_depth=5,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64, random_state=RNG,
)
XGB_REG_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RNG, verbosity=0, n_jobs=-1,
)
XGB_CLF_PARAMS = dict(
    n_estimators=500, learning_rate=0.1, max_depth=5,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=RNG, verbosity=0, n_jobs=-1,
)


def clean_X(df):
    """
    Impute NaN then ordinal-encode categoricals. Returns float64 numpy array.
    No z-score normalization — GeoXGB handles its own internal normalization.
    """
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


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_datasets():
    datasets = {}
    print("Loading datasets (this may download from OpenML on first run)...")

    # ── Regression ────────────────────────────────────────────────────────────
    print("  california_housing...", end=" ", flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.RandomState(RNG)
    idx = rng.choice(len(X), 8000, replace=False)
    datasets["california_housing"] = ("reg", X[idx].astype(np.float64), y[idx])
    print(f"n={len(idx)}, d={X.shape[1]}")

    print("  concrete_compressive...", end=" ", flush=True)
    try:
        data = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = clean_X(data.data)
        yc = np.asarray(data.target, dtype=np.float64)
        datasets["concrete_compressive"] = ("reg", Xc, yc)
        print(f"n={len(yc)}, d={Xc.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  abalone...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=183, as_frame=True, parser="auto")
        Xa = clean_X(data.data)
        ya = np.asarray(data.target, dtype=np.float64)
        datasets["abalone"] = ("reg", Xa, ya)
        print(f"n={len(ya)}, d={Xa.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  wine_quality...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=40691, as_frame=True, parser="auto")
        Xw = clean_X(data.data)
        yw = np.asarray(data.target, dtype=np.float64)
        datasets["wine_quality"] = ("reg", Xw, yw)
        print(f"n={len(yw)}, d={Xw.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    # ── Classification ────────────────────────────────────────────────────────
    print("  breast_cancer...", end=" ", flush=True)
    Xb, yb = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer"] = ("clf", Xb.astype(np.float64), yb.astype(np.float64))
    print(f"n={len(yb)}, d={Xb.shape[1]}")

    print("  credit-g...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        Xcr = clean_X(data.data)
        ycr = (data.target == "good").astype(np.float64).values
        datasets["credit-g"] = ("clf", Xcr, ycr)
        print(f"n={len(ycr)}, d={Xcr.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  adult (income)...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        Xad = clean_X(data.data)   # imputes the 2k+ NaN rows in workclass/occupation
        yad = (data.target.str.strip() == ">50K").astype(np.float64).values
        rng2 = np.random.RandomState(RNG)
        idx2 = rng2.choice(len(yad), 10000, replace=False)
        datasets["adult"] = ("clf", Xad[idx2], yad[idx2])
        print(f"n=10000, d={Xad.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    # vehicle is 4-class; CppGeoXGBClassifier is binary-only in 0.2.0 — skip.
    # Replace with ionosphere (binary, real radar measurements).
    print("  ionosphere...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=59, as_frame=True, parser="auto")
        Xi = clean_X(data.data)
        yi = LabelEncoder().fit_transform(data.target).astype(np.float64)
        datasets["ionosphere"] = ("clf", Xi, yi)
        print(f"n={len(yi)}, d={Xi.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print()
    return datasets


# ── CV ────────────────────────────────────────────────────────────────────────
def run_cv(task, X, y):
    is_reg = (task == "reg")
    is_multi = (task == "multiclf")
    n_classes = int(y.max()) + 1 if is_multi else 2

    kf = (KFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG) if is_reg
          else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG))
    splits = list(kf.split(X, None if is_reg else y.astype(int)))

    geo_scores, xgb_scores = [], []

    for tr, va in splits:
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # ── GeoXGB ──────────────────────────────────────────────────────────
        if is_reg:
            cfg = make_cpp_config(**GEO_REG_PARAMS)
            gm = CppGeoXGBRegressor(cfg)
            gm.fit(Xtr, ytr)
            geo_scores.append(r2_score(yva, gm.predict(Xva)))
        else:
            cfg = make_cpp_config(**GEO_CLF_PARAMS)
            gm = CppGeoXGBClassifier(cfg)
            gm.fit(Xtr, ytr)
            gprob = gm.predict_proba(Xva)
            if is_multi:
                labels = np.arange(n_classes)
                geo_scores.append(roc_auc_score(
                    yva, gprob, multi_class="ovr", average="macro", labels=labels))
            else:
                geo_scores.append(roc_auc_score(yva, gprob[:, 1]))

        # ── XGBoost ─────────────────────────────────────────────────────────
        if is_reg:
            xm = xgb.XGBRegressor(**XGB_REG_PARAMS)
            xm.fit(Xtr, ytr)
            xgb_scores.append(r2_score(yva, xm.predict(Xva)))
        elif is_multi:
            xm = xgb.XGBClassifier(**XGB_CLF_PARAMS,
                                    objective="multi:softprob",
                                    num_class=n_classes)
            xm.fit(Xtr, ytr.astype(int))
            xprob = xm.predict_proba(Xva)
            xgb_scores.append(roc_auc_score(
                yva, xprob, multi_class="ovr", average="macro",
                labels=np.arange(n_classes)))
        else:
            xm = xgb.XGBClassifier(**XGB_CLF_PARAMS)
            xm.fit(Xtr, ytr.astype(int))
            xgb_scores.append(roc_auc_score(yva, xm.predict_proba(Xva)[:, 1]))

    return {
        "metric": "R2" if is_reg else "AUC",
        "geo":  (float(np.mean(geo_scores)), float(np.std(geo_scores))),
        "xgb":  (float(np.mean(xgb_scores)), float(np.std(xgb_scores))),
        "delta": float(np.mean(geo_scores) - np.mean(xgb_scores)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  GeoXGB 0.2.0 (C++) vs XGBoost 3.x  —  Real-World Benchmark")
    print("=" * 72)
    print(f"  GeoXGB reg : depth=3, n_rounds=500, lr=0.1, y_weight=0.2, er=0.1")
    print(f"  GeoXGB clf : depth=5, n_rounds=500, lr=0.1, y_weight=0.2, er=0.1")
    print(f"  XGBoost reg: depth=3, n_estimators=500, lr=0.1, sub=0.8")
    print(f"  XGBoost clf: depth=5, n_estimators=500, lr=0.1, sub=0.8")
    print(f"  {N_SPLITS}-fold CV  |  no X normalization  |  NaN imputed (median/mode)")
    print()

    datasets = load_datasets()
    results = []

    print(f"{'Dataset':<24} {'M':^3} {'GeoXGB':>13} {'XGBoost':>13} {'Delta':>8}  Winner")
    print("-" * 75)

    for ds_name, (task, X, y) in datasets.items():
        t0 = time.perf_counter()
        res = run_cv(task, X, y)
        elapsed = time.perf_counter() - t0

        g_mean, g_std = res["geo"]
        x_mean, x_std = res["xgb"]
        delta = res["delta"]
        m = res["metric"]

        winner = "GeoXGB " if delta > 0 else "XGBoost"
        sign = "+" if delta >= 0 else ""
        print(f"  {ds_name:<22} {m:^3}  "
              f"{g_mean:.4f}+/-{g_std:.4f}  "
              f"{x_mean:.4f}+/-{x_std:.4f}  "
              f"{sign}{delta:.4f}  {winner}  ({elapsed:.0f}s)")

        results.append({"dataset": ds_name, "task": task, "metric": m,
                         "geo": g_mean, "geo_std": g_std,
                         "xgb": x_mean, "xgb_std": x_std,
                         "delta": delta})

    print("-" * 75)
    n_geo = sum(1 for r in results if r["delta"] > 0)
    n_xgb = len(results) - n_geo
    mean_delta = np.mean([r["delta"] for r in results])

    reg = [r for r in results if r["task"] == "reg"]
    clf = [r for r in results if r["task"] in ("clf", "multiclf")]

    print(f"\n  GeoXGB wins: {n_geo}/{len(results)}   XGBoost wins: {n_xgb}/{len(results)}")
    print(f"  Overall mean delta (GeoXGB - XGBoost): {mean_delta:+.4f}")
    if reg:
        print(f"  Regression  (R2)  mean delta: {np.mean([r['delta'] for r in reg]):+.4f}")
    if clf:
        print(f"  Classification (AUC) mean delta: {np.mean([r['delta'] for r in clf]):+.4f}")
    print()


if __name__ == "__main__":
    main()
