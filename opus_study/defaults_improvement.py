"""
Test default improvements: convergence_tol, and other tweaks.

Goal: find defaults that improve GeoXGB's win rate without HPO.
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


# ── Datasets ────────────────────────────────────────────────────────────────

DATASETS = {
    "diabetes":       (lambda: load_diabetes(return_X_y=True), "reg", 10),
    "breast_cancer":  (lambda: load_breast_cancer(return_X_y=True), "clf", 30),
    "wine":           (lambda: load_wine(return_X_y=True), "clf", 13),
    "friedman1_2k":   (lambda: make_friedman1(2000, n_features=10, noise=1.0, random_state=42), "reg", 10),
    "friedman3_2k":   (lambda: make_friedman3(2000, noise=0.5, random_state=42), "reg", 4),
    "california_5k":  (None, "reg", 8),  # special handling
    "linreg_20d_2k":  (lambda: make_regression(2000, 20, n_informative=15, noise=1.0, random_state=42), "reg", 20),
    "binary_clf_5k":  (lambda: make_classification(5000, 10, n_informative=6, n_redundant=2, random_state=42), "clf", 10),
    "multiclass_5k":  (lambda: make_classification(5000, 15, n_informative=10, n_classes=5, n_clusters_per_class=1, random_state=42), "clf", 15),
    "friedman1_10k":  (lambda: make_friedman1(10000, n_features=10, noise=1.0, random_state=42), "reg", 10),
    "friedman1_50k":  (lambda: make_friedman1(50000, n_features=10, noise=1.0, random_state=42), "reg", 10),
    "california_10k": (None, "reg", 8),
    "linreg_20d_10k": (lambda: make_regression(10000, 20, n_informative=15, noise=1.0, random_state=42), "reg", 20),
    "reg_4d_10k":     (lambda: make_regression(10000, 4, n_informative=4, noise=1.0, random_state=42), "reg", 4),
}


def load_california(n=None):
    X, y = fetch_california_housing(return_X_y=True)
    if n and n < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n, replace=False)
        return X[idx], y[idx]
    return X, y


def get_data(name):
    if name == "california_5k":
        return load_california(5000)
    elif name == "california_10k":
        return load_california(10000)
    fn, _, _ = DATASETS[name]
    return fn()


# ── Configs to test ─────────────────────────────────────────────────────────

# Each config: (label, reg_kwargs, clf_kwargs)
CONFIGS = [
    ("current_defaults", {}, {}),
    ("conv_tol_01", {"convergence_tol": 0.01}, {"convergence_tol": 0.01}),
    ("conv_tol_005", {"convergence_tol": 0.005}, {"convergence_tol": 0.005}),
    ("lr003_depth3", {"learning_rate": 0.03}, {"learning_rate": 0.03}),
    ("lr003_conv01", {"learning_rate": 0.03, "convergence_tol": 0.01}, {"learning_rate": 0.03, "convergence_tol": 0.01}),
    ("ri25_conv01", {"refit_interval": 25, "convergence_tol": 0.01}, {"refit_interval": 25, "convergence_tol": 0.01}),
]


# ── Evaluation ──────────────────────────────────────────────────────────────

def eval_config(ds_name, task, X, y, reg_kw, clf_kw):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores, times = [], []

    for tr, val in kf.split(X):
        if task == "reg":
            m = GeoXGBRegressor(random_state=42, **reg_kw)
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[val], m.predict(X[val])))
        else:
            m = GeoXGBClassifier(random_state=42, **clf_kw)
            t0 = time.perf_counter()
            m.fit(X[tr], y[tr])
            times.append(time.perf_counter() - t0)
            proba = m.predict_proba(X[val])
            n_classes = len(np.unique(y))
            if n_classes == 2:
                scores.append(roc_auc_score(y[val], proba[:, 1]))
            else:
                scores.append(roc_auc_score(y[val], proba, multi_class='ovr'))
    return np.mean(scores)


def eval_xgb(ds_name, task, X, y):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, val in kf.split(X):
        if task == "reg":
            m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=3,
                                  random_state=42, verbosity=0, tree_method="hist")
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[val], m.predict(X[val])))
        else:
            m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3,
                                   random_state=42, verbosity=0, tree_method="hist",
                                   use_label_encoder=False, eval_metric='logloss')
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[val])
            n_classes = len(np.unique(y))
            if n_classes == 2:
                scores.append(roc_auc_score(y[val], proba[:, 1]))
            else:
                scores.append(roc_auc_score(y[val], proba, multi_class='ovr'))
    return np.mean(scores)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    results = []

    for ds_name, (_, task, d) in DATASETS.items():
        X, y = get_data(ds_name)
        n = len(X)
        metric = "R2" if task == "reg" else "AUC"

        print(f"\n{'='*70}")
        print(f"  {ds_name}  n={n:,}  d={d}  {metric}")
        print(f"{'='*70}")

        xgb_score = eval_xgb(ds_name, task, X, y)
        print(f"  XGBoost:           {metric}={xgb_score:.4f}")

        for label, reg_kw, clf_kw in CONFIGS:
            try:
                geo_score = eval_config(ds_name, task, X, y, reg_kw, clf_kw)
            except Exception as e:
                print(f"  {label:<20} FAIL: {e}")
                continue

            delta = geo_score - xgb_score
            print(f"  {label:<20} {metric}={geo_score:.4f} ({delta:+.4f})")

            results.append({
                "dataset": ds_name, "n": n, "d": d, "task": task,
                "config": label,
                "geo_score": round(geo_score, 5),
                "xgb_score": round(xgb_score, 5),
                "delta": round(delta, 5),
            })

        pd.DataFrame(results).to_csv("opus_study/results_defaults_improvement.csv", index=False)

    df = pd.DataFrame(results)

    # ── Summary per config ───────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  CONFIG COMPARISON")
    print(f"{'='*70}")

    for label, _, _ in CONFIGS:
        sub = df[df["config"] == label]
        wins = (sub["delta"] > 0.001).sum()
        losses = (sub["delta"] < -0.001).sum()
        ties = len(sub) - wins - losses
        mean_d = sub["delta"].mean()

        # Small n
        small = sub[sub["n"] <= 1000]
        small_d = small["delta"].mean() if len(small) > 0 else 0

        # Medium n
        med = sub[(sub["n"] > 1000) & (sub["n"] <= 10000)]
        med_d = med["delta"].mean() if len(med) > 0 else 0

        # Large n
        large = sub[sub["n"] > 10000]
        large_d = large["delta"].mean() if len(large) > 0 else 0

        print(f"\n  {label}:")
        print(f"    Overall: wins={wins} losses={losses} ties={ties}  mean_delta={mean_d:+.4f}")
        print(f"    Small (n<=1k): {small_d:+.4f}   Medium (1k-10k): {med_d:+.4f}   Large (>10k): {large_d:+.4f}")

    # ── Per-dataset comparison ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  PER-DATASET: best config vs current defaults")
    print(f"{'='*70}")
    for ds_name in DATASETS:
        sub = df[df["dataset"] == ds_name]
        curr = sub[sub["config"] == "current_defaults"]
        if curr.empty:
            continue
        curr_delta = curr["delta"].values[0]
        best = sub.loc[sub["delta"].idxmax()]
        if best["config"] != "current_defaults":
            improvement = best["delta"] - curr_delta
            print(f"  {ds_name:<20} current={curr_delta:+.4f}  "
                  f"best={best['delta']:+.4f} ({best['config']})  "
                  f"improvement={improvement:+.4f}")
        else:
            print(f"  {ds_name:<20} current={curr_delta:+.4f}  (already best)")


if __name__ == "__main__":
    main()
