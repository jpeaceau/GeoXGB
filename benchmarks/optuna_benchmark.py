"""
GeoXGB vs XGBoost -- Optuna TPE Benchmark
==========================================

Fair head-to-head: GeoXGBOptimizer (25 trials, 3-fold CV, fast=True) vs an equivalent
Optuna study tuning XGBoost over the same budget on the same 5 standard datasets.

GeoXGB fast=True: trials use cache_geometry=True, auto_expand=False, convergence_tol=0.01
for practical HPO speed; the final best_model_ is refit at full quality.

Both models search analogous parameter sets:

  GeoXGB search space
  --------------------
    n_rounds       : [100, 200, 500, 1000, 2000]
    learning_rate  : [0.05, 0.1, 0.15, 0.2, 0.3]
    max_depth      : [3, 4, 5, 6]
    refit_interval : [10, 20, 50]

  XGBoost search space (analogous roles, same number of params)
  ---------------------------------------------------------------
    n_estimators   : [100, 200, 500, 1000, 2000]   -- equiv. n_rounds
    learning_rate  : [0.05, 0.1, 0.15, 0.2, 0.3]
    max_depth      : [3, 4, 5, 6]
    subsample      : [0.7, 0.8, 0.9, 1.0]          -- equiv. reduce_ratio

Both models receive a warm-start trial 0 at their respective defaults.
XGBoost is also given a larger subsample search axis (4 options) compared
to GeoXGB's refit_interval (3 options), compensating for the fact that
GeoXGB already has geometry-tuned defaults.

Datasets (same 5 as refit_interval_benchmark / default_parameter_final_benchmark)
-----------------------------------------------------------------------------------
  friedman1      1000 samples, 10 features, noise=1.0         regression  R^2
  friedman2      1000 samples,  4 features, noise=0.0         regression  R^2
  classification 1000 samples, 10 features, sep=1.0           binary AUC
  sparse_highdim 1000 samples, 40 features, noise=20          regression  R^2
  noisy_clf      1000 samples, 20 features, flip_y=0.10       binary AUC

Train/test split: 80/20 stratified (for classification) or random.

Usage
-----
    python benchmarks/optuna_benchmark.py

Requirements: geoxgb, xgboost, optuna, scikit-learn, numpy
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import optuna
from sklearn.datasets import make_classification, make_friedman1, make_friedman2, make_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from geoxgb import GeoXGBOptimizer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRIALS     = 25
N_CV_FOLDS   = 3
RANDOM_STATE = 42
TEST_SIZE    = 0.20

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _make_datasets():
    X1, y1 = make_friedman1(n_samples=1_000, n_features=10, noise=1.0,
                             random_state=RANDOM_STATE)
    X2, y2 = make_friedman2(n_samples=1_000, noise=0.0,
                             random_state=RANDOM_STATE)
    X3, y3 = make_classification(n_samples=1_000, n_features=10, n_informative=5,
                                  n_redundant=0, n_clusters_per_class=2,
                                  class_sep=1.0, random_state=RANDOM_STATE)
    X4, y4 = make_regression(n_samples=1_000, n_features=40, n_informative=8,
                              noise=20.0, random_state=RANDOM_STATE)
    X5, y5 = make_classification(n_samples=1_000, n_features=20, n_informative=5,
                                  n_redundant=5, n_clusters_per_class=1,
                                  class_sep=0.5, flip_y=0.10,
                                  random_state=RANDOM_STATE)
    return {
        "friedman1":      (X1, y1, "regression",     "R^2"),
        "friedman2":      (X2, y2, "regression",     "R^2"),
        "classification": (X3, y3, "classification", "AUC"),
        "sparse_highdim": (X4, y4, "regression",     "R^2"),
        "noisy_clf":      (X5, y5, "classification", "AUC"),
    }


# ---------------------------------------------------------------------------
# XGBoost Optuna study (module-level for Windows pickling)
# ---------------------------------------------------------------------------

# XGBoost defaults for warm start
_XGB_DEFAULTS = dict(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=1.0)

_XGB_SEARCH = {
    "n_estimators":  [100, 200, 500, 1000, 2000],
    "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.3],
    "max_depth":     [3, 4, 5, 6],
    "subsample":     [0.7, 0.8, 0.9, 1.0],
}


def _run_xgb_optuna(X_train, y_train, task, n_trials=N_TRIALS, cv=N_CV_FOLDS,
                    random_state=RANDOM_STATE):
    """Run Optuna TPE study on XGBoost. Returns (best_params, best_cv_score)."""
    from xgboost import XGBClassifier, XGBRegressor

    if task == "classification":
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train, y_train))
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train))

    def objective(trial):
        params = {name: trial.suggest_categorical(name, choices)
                  for name, choices in _XGB_SEARCH.items()}
        if task == "classification":
            model_cls = XGBClassifier
            scorer = lambda m, X, y: roc_auc_score(y, m.predict_proba(X)[:, 1])
        else:
            model_cls = XGBRegressor
            scorer = lambda m, X, y: r2_score(y, m.predict(X))

        fold_scores = []
        for tr, val in splits:
            m = model_cls(**params, random_state=random_state,
                          eval_metric="logloss" if task == "classification" else "rmse",
                          verbosity=0)
            m.fit(X_train[tr], y_train[tr])
            fold_scores.append(scorer(m, X_train[val], y_train[val]))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(_XGB_DEFAULTS)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value), study


def _xgb_test_score(params, X_train, y_train, X_test, y_test, task,
                    random_state=RANDOM_STATE):
    from xgboost import XGBClassifier, XGBRegressor
    if task == "classification":
        m = XGBClassifier(**params, random_state=random_state,
                          eval_metric="logloss", verbosity=0)
        m.fit(X_train, y_train)
        return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    else:
        m = XGBRegressor(**params, random_state=random_state,
                         eval_metric="rmse", verbosity=0)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _section("GEOXGB vs XGBOOST -- OPTUNA TPE BENCHMARK")
    print(
        f"\n  5 datasets x {N_TRIALS} trials x {N_CV_FOLDS}-fold CV each"
        f"\n  GeoXGB: GeoXGBOptimizer fast=True (cache_geometry, no expansion during HPO)"
        f"\n          Final model refit at full quality (cache_geometry=False, auto_expand=True)"
        f"\n  XGBoost: Optuna TPE (n_estimators, learning_rate, max_depth, subsample)"
        f"\n  Warm start: trial 0 = model defaults for both"
        f"\n  Train/test split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}"
    )

    datasets = _make_datasets()

    results = {}   # results[ds_name] = {geo_cv, geo_test, xgb_cv, xgb_test, geo_bp, xgb_bp}

    for ds_name, (X, y, task, metric) in datasets.items():
        _subsection(f"{ds_name}  ({task}, metric={metric})")

        # Train/test split
        if task == "classification":
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        print(f"\n  Train: {len(X_tr)}  |  Test: {len(X_te)}")

        # --- GeoXGB Optuna ---
        print(f"\n  [GeoXGB] Running {N_TRIALS} Optuna trials ({N_CV_FOLDS}-fold CV, fast=True)...")
        t0 = time.perf_counter()
        geo_opt = GeoXGBOptimizer(
            task=task, n_trials=N_TRIALS, cv=N_CV_FOLDS,
            random_state=RANDOM_STATE, verbose=False, fast=True,
        )
        geo_opt.fit(X_tr, y_tr)
        geo_elapsed = time.perf_counter() - t0

        # Test score with best model
        if task == "classification":
            geo_test = roc_auc_score(y_te, geo_opt.predict_proba(X_te)[:, 1])
        else:
            geo_test = r2_score(y_te, geo_opt.predict(X_te))

        print(f"  [GeoXGB] Done in {geo_elapsed:.1f}s")
        print(f"  [GeoXGB] Best config: {geo_opt.best_params_}")
        print(f"  [GeoXGB] Best CV {metric}: {geo_opt.best_score_:.4f}  |  Test {metric}: {geo_test:.4f}")

        # --- XGBoost Optuna ---
        print(f"\n  [XGBoost] Running {N_TRIALS} Optuna trials ({N_CV_FOLDS}-fold CV)...")
        t0 = time.perf_counter()
        xgb_bp, xgb_cv, xgb_study = _run_xgb_optuna(
            X_tr, y_tr, task, n_trials=N_TRIALS, cv=N_CV_FOLDS,
            random_state=RANDOM_STATE,
        )
        xgb_elapsed = time.perf_counter() - t0

        xgb_test = _xgb_test_score(xgb_bp, X_tr, y_tr, X_te, y_te, task)

        print(f"  [XGBoost] Done in {xgb_elapsed:.1f}s")
        print(f"  [XGBoost] Best config: {xgb_bp}")
        print(f"  [XGBoost] Best CV {metric}: {xgb_cv:.4f}  |  Test {metric}: {xgb_test:.4f}")

        margin_cv   = geo_opt.best_score_ - xgb_cv
        margin_test = geo_test - xgb_test
        winner = "GeoXGB" if margin_test > 0 else "XGBoost"
        print(f"\n  Margin (test): {margin_test:+.4f} {metric}  ({winner} wins)")

        results[ds_name] = {
            "task":        task,
            "metric":      metric,
            "geo_cv":      geo_opt.best_score_,
            "geo_test":    geo_test,
            "geo_params":  geo_opt.best_params_,
            "xgb_cv":      xgb_cv,
            "xgb_test":    xgb_test,
            "xgb_params":  xgb_bp,
        }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _section(f"SUMMARY -- GEOXGB vs XGBOOST (Optuna HPO, {N_TRIALS} trials)")

    print(f"\n  {'Dataset':<18s}  {'Metric':>5s}  "
          f"{'GeoXGB CV':>10s}  {'GeoXGB Test':>12s}  "
          f"{'XGB CV':>8s}  {'XGB Test':>10s}  {'Margin (test)':>14s}  {'Winner':>8s}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*8}")

    geo_wins = 0
    margins  = []
    for ds, r in results.items():
        m    = r["metric"]
        margin = r["geo_test"] - r["xgb_test"]
        margins.append(margin)
        winner = "GeoXGB" if margin > 0 else "XGBoost"
        if margin > 0:
            geo_wins += 1
        print(f"  {ds:<18s}  {m:>5s}  "
              f"{r['geo_cv']:>10.4f}  {r['geo_test']:>12.4f}  "
              f"{r['xgb_cv']:>8.4f}  {r['xgb_test']:>10.4f}  "
              f"{margin:>+14.4f}  {winner:>8s}")

    print(f"\n  Win record: GeoXGB {geo_wins}/{len(results)}  |  "
          f"XGBoost {len(results)-geo_wins}/{len(results)}")
    print(f"  Mean margin: {np.mean(margins):+.4f}  |  "
          f"Min: {min(margins):+.4f}  |  Max: {max(margins):+.4f}")

    if geo_wins == len(results):
        print(f"\n  GeoXGB wins ALL {len(results)} datasets under Optuna HPO.")
    elif geo_wins > len(results) // 2:
        print(f"\n  GeoXGB wins {geo_wins}/{len(results)} datasets under Optuna HPO.")
    else:
        print(f"\n  XGBoost wins {len(results)-geo_wins}/{len(results)} datasets under Optuna HPO.")

    print()


if __name__ == "__main__":
    main()
