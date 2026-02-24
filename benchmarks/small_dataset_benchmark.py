"""
Small Dataset Benchmark — GeoXGB auto_expand vs XGBoost
=========================================================

Tests whether GeoXGB's geometry-aware synthetic expansion (auto_expand=True)
provides a meaningful advantage over XGBoost at very small sample sizes.

When n_reduced falls below min_train_samples (default 5000), auto_expand
generates synthetic samples from the within-partition KDE to bring the
effective training set up to size.  At small n, this is the primary intended
use of the expansion mechanism.

Models
------
  GeoXGB (auto_expand=True)   — default; expands to min_train_samples=5000
  GeoXGB (auto_expand=False)  — geometry-aware reduction only, no expansion
  XGBoost                     — n_estimators=500, lr=0.1, max_depth=4

Dataset sizes: n = 50, 100, 200, 300, 500
Tasks: Classification (make_classification) and Regression (make_friedman1)
Features: 10 total, 6 informative
Seeds: 10 per size (higher than usual for stability at small n)
Train/test: 80/20 split

Metric: AUC (classification), R2 (regression)
"""
import warnings
import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

N_SIZES      = [50, 100, 200, 300, 500]
RANDOM_SEEDS = list(range(10))
GEO_ROUNDS   = 500
XGB_ROUNDS   = 500
COL_W        = 26


def evaluate(n, seed, task):
    rng = np.random.default_rng(seed)

    if task == "clf":
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=6,
            n_redundant=2, random_state=seed,
        )
        X = X.astype(np.float64)
        stratify = y
    else:
        X, y = make_friedman1(
            n_samples=n, n_features=10, noise=1.0, random_state=seed,
        )
        X = X.astype(np.float64)
        stratify = None

    n_test = max(1, int(n * 0.20))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=n_test, stratify=stratify, random_state=seed,
    )

    def score_geo(auto_exp):
        cls = GeoXGBClassifier if task == "clf" else GeoXGBRegressor
        m = cls(
            n_rounds=GEO_ROUNDS,
            auto_expand=auto_exp,
            random_state=seed,
        )
        m.fit(X_tr, y_tr)
        if task == "clf":
            return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        return r2_score(y_te, m.predict(X_te))

    def score_xgb():
        kw = dict(n_estimators=XGB_ROUNDS, learning_rate=0.1, max_depth=4,
                  tree_method="hist", verbosity=0, random_state=seed)
        if task == "clf":
            m = XGBClassifier(**kw, eval_metric="auc")
            m.fit(X_tr, y_tr)
            return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        m = XGBRegressor(**kw)
        m.fit(X_tr, y_tr)
        return r2_score(y_te, m.predict(X_te))

    return {
        "GeoXGB(expand)":   score_geo(True),
        "GeoXGB(no-expand)": score_geo(False),
        "XGBoost":           score_xgb(),
    }


MODEL_ORDER = ["GeoXGB(expand)", "GeoXGB(no-expand)", "XGBoost"]

print(f"Small Dataset Benchmark  |  {len(N_SIZES)} sizes × {len(RANDOM_SEEDS)} seeds,"
      f" rounds={GEO_ROUNDS}")
print()

for task, label, metric in [
    ("clf", "Classification", "AUC"),
    ("reg", "Regression / Friedman #1", "R2 "),
]:
    print(f"{'='*70}")
    print(f"  TASK: {label}  ({metric})")
    print(f"{'='*70}")
    print(f"  {'Model':<{COL_W}}", end="")
    for n in N_SIZES:
        print(f"  n={n:>3}", end="")
    print("  wins")
    print("  " + "-" * (COL_W + len(N_SIZES) * 8 + 6))

    all_means = {m: {} for m in MODEL_ORDER}

    for n in N_SIZES:
        scores = {m: [] for m in MODEL_ORDER}
        for seed in RANDOM_SEEDS:
            r = evaluate(n, seed, task)
            for m in MODEL_ORDER:
                scores[m].append(r[m])
        for m in MODEL_ORDER:
            all_means[m][n] = np.mean(scores[m])

    for m in MODEL_ORDER:
        print(f"  {m:<{COL_W}}", end="")
        wins = 0
        for n in N_SIZES:
            val  = all_means[m][n]
            best = max(all_means[mm][n] for mm in MODEL_ORDER)
            star = "*" if val == best else " "
            print(f"  {val:.3f}{star}", end="")
            if val == best:
                wins += 1
        print(f"  {wins}/{len(N_SIZES)}")

    # Delta rows: GeoXGB(expand) vs XGBoost
    print()
    print(f"  {'GeoXGB-expand margin':<{COL_W}}", end="")
    for n in N_SIZES:
        delta = all_means["GeoXGB(expand)"][n] - all_means["XGBoost"][n]
        print(f"  {delta:+.3f} ", end="")
    print()
    print()

print("=== DONE ===")
