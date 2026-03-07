"""
GeoXGB defaults vs XGBoost defaults — comprehensive stability audit.

Both models use their own defaults (no HPO). Tests regression + classification
across small-n, medium-n, and large-n regimes with varied dimensionality.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_friedman1, make_friedman3, make_regression,
    fetch_california_housing, load_diabetes,
    make_classification, load_wine, load_breast_cancer,
)
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor, GeoXGBClassifier


# ── Regression datasets ─────────────────────────────────────────────────────

def ds_diabetes(n=None):
    X, y = load_diabetes(return_X_y=True)
    return X, y, "regression", 10

def ds_friedman1(n):
    X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)
    return X, y, "regression", 10

def ds_friedman3(n):
    X, y = make_friedman3(n_samples=n, noise=0.5, random_state=42)
    return X, y, "regression", 4

def ds_california(n=None):
    X_full, y_full = fetch_california_housing(return_X_y=True)
    if n and n < len(X_full):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_full), n, replace=False)
        return X_full[idx], y_full[idx], "regression", 8
    return X_full, y_full, "regression", 8

def ds_linreg_20d(n):
    X, y = make_regression(n_samples=n, n_features=20, n_informative=15,
                           noise=1.0, random_state=42)
    return X, y, "regression", 20

def ds_reg_4d(n):
    X, y = make_regression(n_samples=n, n_features=4, n_informative=4,
                           noise=1.0, random_state=42)
    return X, y, "regression", 4

# ── Classification datasets ─────────────────────────────────────────────────

def ds_breast_cancer(n=None):
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, "classification", 30

def ds_wine(n=None):
    X, y = load_wine(return_X_y=True)
    return X, y, "classification", 13

def ds_binary_clf(n):
    X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                                n_redundant=2, random_state=42)
    return X, y, "classification", 10

def ds_multiclass(n):
    X, y = make_classification(n_samples=n, n_features=15, n_informative=10,
                                n_classes=5, n_clusters_per_class=1,
                                random_state=42)
    return X, y, "classification", 15


# ── Test configurations ─────────────────────────────────────────────────────

CONFIGS = [
    # (name, factory, n)  — n=None means use built-in size
    # Small n (< 1k)
    ("diabetes", ds_diabetes, None),
    ("breast_cancer", ds_breast_cancer, None),
    ("wine", ds_wine, None),
    # Medium n (1k-10k)
    ("friedman1_2k", ds_friedman1, 2000),
    ("friedman3_2k", ds_friedman3, 2000),
    ("california_5k", ds_california, 5000),
    ("linreg_20d_2k", ds_linreg_20d, 2000),
    ("binary_clf_5k", ds_binary_clf, 5000),
    ("multiclass_5k", ds_multiclass, 5000),
    # Large n (10k-100k)
    ("friedman1_10k", ds_friedman1, 10000),
    ("friedman1_50k", ds_friedman1, 50000),
    ("california_10k", ds_california, 10000),
    ("california_full", ds_california, None),
    ("linreg_20d_10k", ds_linreg_20d, 10000),
    ("linreg_20d_50k", ds_linreg_20d, 50000),
    ("reg_4d_10k", ds_reg_4d, 10000),
    ("reg_4d_50k", ds_reg_4d, 50000),
    ("binary_clf_20k", ds_binary_clf, 20000),
]


# ── Evaluation ──────────────────────────────────────────────────────────────

def eval_regression(geo_cls, xgb_cls, X, y, n_folds=3):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    geo_scores, xgb_scores = [], []
    geo_times, xgb_times = [], []

    for tr, val in kf.split(X):
        # GeoXGB defaults
        m = geo_cls(random_state=42)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        geo_times.append(time.perf_counter() - t0)
        geo_scores.append(r2_score(y[val], m.predict(X[val])))

        # XGBoost defaults (matching n_estimators for fairness)
        mx = xgb_cls(
            n_estimators=1000, learning_rate=0.02, max_depth=3,
            random_state=42, verbosity=0, tree_method="hist",
        )
        t0 = time.perf_counter()
        mx.fit(X[tr], y[tr])
        xgb_times.append(time.perf_counter() - t0)
        xgb_scores.append(r2_score(y[val], mx.predict(X[val])))

    return (np.mean(geo_scores), np.mean(geo_times),
            np.mean(xgb_scores), np.mean(xgb_times))


def eval_classification(geo_cls, xgb_cls, X, y, n_folds=3):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    geo_scores, xgb_scores = [], []
    geo_times, xgb_times = [], []
    n_classes = len(np.unique(y))

    for tr, val in kf.split(X):
        # GeoXGB
        m = geo_cls(random_state=42)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        geo_times.append(time.perf_counter() - t0)
        proba = m.predict_proba(X[val])
        if n_classes == 2:
            geo_scores.append(roc_auc_score(y[val], proba[:, 1]))
        else:
            geo_scores.append(roc_auc_score(y[val], proba, multi_class='ovr'))

        # XGBoost
        mx = xgb_cls(
            n_estimators=1000, learning_rate=0.02, max_depth=3,
            random_state=42, verbosity=0, tree_method="hist",
            use_label_encoder=False, eval_metric='logloss',
        )
        t0 = time.perf_counter()
        mx.fit(X[tr], y[tr])
        xgb_times.append(time.perf_counter() - t0)
        proba_x = mx.predict_proba(X[val])
        if n_classes == 2:
            xgb_scores.append(roc_auc_score(y[val], proba_x[:, 1]))
        else:
            xgb_scores.append(roc_auc_score(y[val], proba_x, multi_class='ovr'))

    return (np.mean(geo_scores), np.mean(geo_times),
            np.mean(xgb_scores), np.mean(xgb_times))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    results = []

    for name, factory, n in CONFIGS:
        if n is not None:
            X, y, task, d = factory(n)
        else:
            X, y, task, d = factory()
        actual_n = len(X)

        print(f"\n{'='*70}")
        print(f"  {name}  n={actual_n:,}  d={d}  task={task}")
        print(f"{'='*70}")

        try:
            if task == "regression":
                geo_score, geo_time, xgb_score, xgb_time = eval_regression(
                    GeoXGBRegressor, xgb.XGBRegressor, X, y)
                metric = "R2"
            else:
                geo_score, geo_time, xgb_score, xgb_time = eval_classification(
                    GeoXGBClassifier, xgb.XGBClassifier, X, y)
                metric = "AUC"
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        delta = geo_score - xgb_score
        speed = geo_time / max(xgb_time, 0.001)
        winner = "GeoXGB" if delta > 0.001 else ("XGBoost" if delta < -0.001 else "tie")

        print(f"  GeoXGB:  {metric}={geo_score:.4f}  time={geo_time:.2f}s")
        print(f"  XGBoost: {metric}={xgb_score:.4f}  time={xgb_time:.2f}s")
        print(f"  Delta: {delta:+.4f}  Speed: {speed:.1f}x  Winner: {winner}")

        results.append({
            "dataset": name, "n": actual_n, "d": d, "task": task,
            "metric": metric,
            "geo_score": round(geo_score, 5),
            "xgb_score": round(xgb_score, 5),
            "delta": round(delta, 5),
            "geo_time": round(geo_time, 3),
            "xgb_time": round(xgb_time, 3),
            "speed_ratio": round(speed, 2),
            "winner": winner,
        })

        # Save incrementally
        pd.DataFrame(results).to_csv(
            "opus_study/results_defaults_stability.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("opus_study/results_defaults_stability.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SUMMARY — GeoXGB defaults vs XGBoost (matched hyperparams)")
    print(f"{'='*70}")

    print(f"\n  {'Dataset':<22} {'n':>7} {'d':>3} {'Metric':>6} "
          f"{'GeoXGB':>8} {'XGBoost':>8} {'Delta':>8} {'Speed':>6} {'Winner':>8}")
    print(f"  {'-'*22} {'-'*7} {'-'*3} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:<22} {r['n']:>7,} {r['d']:>3} {r['metric']:>6} "
              f"{r['geo_score']:>8.4f} {r['xgb_score']:>8.4f} {r['delta']:>+8.4f} "
              f"{r['speed_ratio']:>5.1f}x {r['winner']:>8}")

    geo_wins = (df['winner'] == 'GeoXGB').sum()
    xgb_wins = (df['winner'] == 'XGBoost').sum()
    ties = (df['winner'] == 'tie').sum()

    # By regime
    small = df[df['n'] <= 1000]
    medium = df[(df['n'] > 1000) & (df['n'] <= 10000)]
    large = df[df['n'] > 10000]

    print(f"\n  Overall: GeoXGB wins {geo_wins}, XGBoost wins {xgb_wins}, ties {ties}")
    print(f"  Mean delta: {df['delta'].mean():+.4f}")

    for label, sub in [("Small (n<=1k)", small), ("Medium (1k<n<=10k)", medium), ("Large (n>10k)", large)]:
        if sub.empty:
            continue
        sw = (sub['winner'] == 'GeoXGB').sum()
        xw = (sub['winner'] == 'XGBoost').sum()
        tw = (sub['winner'] == 'tie').sum()
        print(f"  {label}: GeoXGB {sw}, XGBoost {xw}, ties {tw}  "
              f"mean_delta={sub['delta'].mean():+.4f}")


if __name__ == "__main__":
    main()
