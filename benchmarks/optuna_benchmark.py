"""
GeoXGB vs XGBoost -- Optuna TPE Benchmark  (v2)
================================================

Fair head-to-head: custom Optuna TPE study for GeoXGB (50 trials, 3-fold CV,
full quality -- no fast=True proxy) vs an equivalent XGBoost study over the
same trial budget on the same 5 standard datasets.

Changes from v1
---------------
  - fast=True dropped: HVRT 2.5.0 vectorisation makes full-quality trials as
    fast as cached-geometry trials at n=1000 (measured ratio 1.00x, Sec 20).
  - GeoXGB search space extended beyond GeoXGBOptimizer defaults to include
    hvrt_min_samples_leaf, reduce_ratio, expand_ratio, and min_samples_leaf.
  - Trials increased from 25 to 50.
  - Parallel trial execution via study.optimize(n_jobs=GEO_N_JOBS) using
    joblib threads sharing in-memory storage (no external storage needed;
    numpy/cython ops release the GIL so threading gives genuine concurrency).
  - n_rounds ceiling 2000 (3000 was impractically slow: 1750s for one dataset).
  - XGBoost search space extended with colsample_bytree and min_child_weight
    to match GeoXGB's additional axes.

GeoXGB search space (9 parameters)
------------------------------------
  n_rounds              : [200, 500, 1000, 2000]
  learning_rate         : [0.05, 0.1, 0.15, 0.2]
  max_depth             : [3, 4, 5, 6]
  refit_interval        : [10, 20, 50]
  y_weight              : [0.1, 0.3, 0.5, 0.7, 0.9]
  hvrt_min_samples_leaf : [5, 10, 20, 30]      <- new
  reduce_ratio          : [0.7, 0.8, 0.9]      <- new
  expand_ratio          : [0.0, 0.1, 0.2]      <- new
  min_samples_leaf      : [1, 5, 10]            <- new

XGBoost search space (6 parameters, analogous roles)
------------------------------------------------------
  n_estimators    : [200, 500, 1000, 2000]
  learning_rate   : [0.05, 0.1, 0.15, 0.2]
  max_depth       : [3, 4, 5, 6]
  subsample       : [0.7, 0.8, 0.9, 1.0]
  colsample_bytree: [0.7, 0.8, 0.9, 1.0]      <- new
  min_child_weight: [1, 3, 5]                  <- new

Both receive warm-start trial 0 at their respective defaults.
Final models: GeoXGB refit at full quality (cache_geometry=False,
auto_expand=True) with best params; XGBoost refit with best params.

Datasets (same 5 as previous benchmarks)
-----------------------------------------
  friedman1      1000 samples, 10 features, noise=1.0         regression  R^2
  friedman2      1000 samples,  4 features, noise=0.0         regression  R^2
  classification 1000 samples, 10 features, sep=1.0           binary AUC
  sparse_highdim 1000 samples, 40 features, noise=20          regression  R^2
  noisy_clf      1000 samples, 20 features, flip_y=0.10       binary AUC

Train/test split: 80/20  (stratified for classification).

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

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_TRIALS     = 50
N_CV_FOLDS   = 3
RANDOM_STATE = 42
TEST_SIZE    = 0.20
GEO_N_JOBS   = 4    # parallel GeoXGB trials via joblib threads (shared in-memory storage)

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}", flush=True)


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}", flush=True)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

GEO_SEARCH = {
    "n_rounds":              [200, 500, 1000, 2000],
    "learning_rate":         [0.05, 0.1, 0.15, 0.2],
    "max_depth":             [3, 4, 5, 6],
    "refit_interval":        [10, 20, 50],
    "y_weight":              [0.1, 0.3, 0.5, 0.7, 0.9],
    "hvrt_min_samples_leaf": [5, 10, 20, 30],
    "reduce_ratio":          [0.7, 0.8, 0.9],
    "expand_ratio":          [0.0, 0.1, 0.2],
    "min_samples_leaf":      [1, 5, 10],
}

GEO_WARM_START = {
    "n_rounds":              1000,
    "learning_rate":         0.1,
    "max_depth":             4,
    "refit_interval":        20,
    "y_weight":              0.5,
    "hvrt_min_samples_leaf": 20,
    "reduce_ratio":          0.9,
    "expand_ratio":          0.0,
    "min_samples_leaf":      1,
}

# Full-quality overrides applied to every trial and the final model
GEO_FIXED = {
    "cache_geometry":    False,
    "auto_expand":       True,
    "generation_strategy": "epanechnikov",
    "random_state":      RANDOM_STATE,
}

XGB_SEARCH = {
    "n_estimators":     [200, 500, 1000, 2000],
    "learning_rate":    [0.05, 0.1, 0.15, 0.2],
    "max_depth":        [3, 4, 5, 6],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

XGB_WARM_START = {
    "n_estimators":     200,
    "learning_rate":    0.1,
    "max_depth":        6,
    "subsample":        1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
}


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
# Scoring helper
# ---------------------------------------------------------------------------

def _score(m, X, y, task):
    if task == "classification":
        p = m.predict_proba(X)
        return float(roc_auc_score(y, p[:, 1] if p.shape[1] == 2 else p,
                                    multi_class="ovr"))
    return float(r2_score(y, m.predict(X)))


# ---------------------------------------------------------------------------
# GeoXGB Optuna study
# ---------------------------------------------------------------------------

def _run_geo_optuna(X_train, y_train, task):
    """
    Custom Optuna TPE study for GeoXGB with extended search space.
    Uses full-quality settings (no fast=True proxy) and parallel trials.
    Returns (best_params, best_cv_score, elapsed, study).
    """
    from geoxgb import GeoXGBClassifier, GeoXGBRegressor

    model_cls = GeoXGBClassifier if task == "classification" else GeoXGBRegressor

    if task == "classification":
        kf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        splits = list(kf.split(X_train, y_train))
    else:
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_train))

    def objective(trial):
        params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in GEO_SEARCH.items()
        }
        run_params = {**params, **GEO_FIXED}
        fold_scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr, val in splits:
                m = model_cls(**run_params)
                m.fit(X_train[tr], y_train[tr])
                fold_scores.append(_score(m, X_train[val], y_train[val], task))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(GEO_WARM_START)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=GEO_N_JOBS)
    elapsed = time.perf_counter() - t0

    best = {k: study.best_params[k] for k in GEO_SEARCH}
    return best, float(study.best_value), elapsed, study


def _geo_final_score(best_params, X_train, y_train, X_test, y_test, task):
    """Refit at full quality with best params; return test score."""
    from geoxgb import GeoXGBClassifier, GeoXGBRegressor

    model_cls  = GeoXGBClassifier if task == "classification" else GeoXGBRegressor
    run_params = {**best_params, **GEO_FIXED}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = model_cls(**run_params)
        m.fit(X_train, y_train)
    return _score(m, X_test, y_test, task)


# ---------------------------------------------------------------------------
# XGBoost Optuna study
# ---------------------------------------------------------------------------

def _run_xgb_optuna(X_train, y_train, task):
    """Run Optuna TPE study on XGBoost. Returns (best_params, best_cv_score, elapsed, study)."""
    from xgboost import XGBClassifier, XGBRegressor

    model_cls = XGBClassifier if task == "classification" else XGBRegressor
    xgb_kw   = dict(random_state=RANDOM_STATE, verbosity=0,
                     eval_metric="logloss" if task == "classification" else "rmse")

    if task == "classification":
        kf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        splits = list(kf.split(X_train, y_train))
    else:
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_train))

    def objective(trial):
        params = {name: trial.suggest_categorical(name, choices)
                  for name, choices in XGB_SEARCH.items()}
        fold_scores = []
        for tr, val in splits:
            m = model_cls(**params, **xgb_kw)
            m.fit(X_train[tr], y_train[tr])
            fold_scores.append(_score(m, X_train[val], y_train[val], task))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(XGB_WARM_START)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.perf_counter() - t0

    best = {k: study.best_params[k] for k in XGB_SEARCH}
    return best, float(study.best_value), elapsed, study


def _xgb_final_score(best_params, X_train, y_train, X_test, y_test, task):
    from xgboost import XGBClassifier, XGBRegressor

    model_cls = XGBClassifier if task == "classification" else XGBRegressor
    xgb_kw   = dict(random_state=RANDOM_STATE, verbosity=0,
                     eval_metric="logloss" if task == "classification" else "rmse")
    m = model_cls(**best_params, **xgb_kw)
    m.fit(X_train, y_train)
    return _score(m, X_test, y_test, task)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _section("GEOXGB vs XGBOOST -- OPTUNA TPE BENCHMARK  (v2)")
    print(
        f"\n  5 datasets x {N_TRIALS} trials x {N_CV_FOLDS}-fold CV each"
        f"\n  GeoXGB: full quality (no fast=True proxy); {GEO_N_JOBS} parallel trials"
        f"\n          9-parameter search space (incl. hvrt_msl, reduce_ratio, expand_ratio, msl)"
        f"\n          Final model refit at full quality (cache=False, expand=True)"
        f"\n  XGBoost: Optuna TPE, 1 trial at a time; 6-parameter search space"
        f"\n           (incl. colsample_bytree, min_child_weight)"
        f"\n  Warm start: trial 0 = model defaults for both"
        f"\n  Train/test split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}"
        f"\n  Seed: {RANDOM_STATE}",
        flush=True,
    )

    datasets = _make_datasets()
    results  = {}

    for ds_name, (X, y, task, metric) in datasets.items():
        _subsection(f"{ds_name}  ({task}, metric={metric})")

        if task == "classification":
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        print(f"\n  Train: {len(X_tr)}  |  Test: {len(X_te)}", flush=True)

        # --- GeoXGB ---
        print(f"\n  [GeoXGB] Running {N_TRIALS} Optuna trials "
              f"({N_CV_FOLDS}-fold CV, full quality, n_jobs={GEO_N_JOBS})...",
              flush=True)
        geo_bp, geo_cv, geo_elapsed, _ = _run_geo_optuna(X_tr, y_tr, task)
        geo_test = _geo_final_score(geo_bp, X_tr, y_tr, X_te, y_te, task)

        print(f"  [GeoXGB] Done in {geo_elapsed:.1f}s")
        print(f"  [GeoXGB] Best config: {geo_bp}")
        print(f"  [GeoXGB] Best CV {metric}: {geo_cv:.4f}  |  Test {metric}: {geo_test:.4f}",
              flush=True)

        # --- XGBoost ---
        print(f"\n  [XGBoost] Running {N_TRIALS} Optuna trials "
              f"({N_CV_FOLDS}-fold CV)...", flush=True)
        xgb_bp, xgb_cv, xgb_elapsed, _ = _run_xgb_optuna(X_tr, y_tr, task)
        xgb_test = _xgb_final_score(xgb_bp, X_tr, y_tr, X_te, y_te, task)

        print(f"  [XGBoost] Done in {xgb_elapsed:.1f}s")
        print(f"  [XGBoost] Best config: {xgb_bp}")
        print(f"  [XGBoost] Best CV {metric}: {xgb_cv:.4f}  |  Test {metric}: {xgb_test:.4f}")

        margin = geo_test - xgb_test
        winner = "GeoXGB" if margin > 0 else "XGBoost"
        print(f"\n  Margin (test): {margin:+.4f} {metric}  ({winner} wins)", flush=True)

        results[ds_name] = {
            "task":       task,
            "metric":     metric,
            "geo_cv":     geo_cv,
            "geo_test":   geo_test,
            "geo_params": geo_bp,
            "geo_time":   geo_elapsed,
            "xgb_cv":     xgb_cv,
            "xgb_test":   xgb_test,
            "xgb_params": xgb_bp,
            "xgb_time":   xgb_elapsed,
        }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _section(f"SUMMARY -- GEOXGB vs XGBOOST (Optuna HPO, {N_TRIALS} trials)")

    print(f"\n  {'Dataset':<18}  {'Metric':>5}  "
          f"{'GeoXGB CV':>10}  {'GeoXGB Test':>12}  "
          f"{'XGB CV':>8}  {'XGB Test':>10}  {'Margin (test)':>14}  {'Winner':>8}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*8}")

    geo_wins = 0
    margins  = []
    for ds, r in results.items():
        m      = r["metric"]
        margin = r["geo_test"] - r["xgb_test"]
        margins.append(margin)
        winner = "GeoXGB" if margin > 0 else "XGBoost"
        if margin > 0:
            geo_wins += 1
        print(f"  {ds:<18}  {m:>5}  "
              f"{r['geo_cv']:>10.4f}  {r['geo_test']:>12.4f}  "
              f"{r['xgb_cv']:>8.4f}  {r['xgb_test']:>10.4f}  "
              f"{margin:>+14.4f}  {winner:>8}")

    print(f"\n  Win record: GeoXGB {geo_wins}/{len(results)}  |  "
          f"XGBoost {len(results)-geo_wins}/{len(results)}")
    print(f"  Mean margin: {np.mean(margins):+.4f}  |  "
          f"Min: {min(margins):+.4f}  |  Max: {max(margins):+.4f}")

    print(f"\n  Timing")
    print(f"  {'Dataset':<18}  {'GeoXGB (s)':>12}  {'XGBoost (s)':>12}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}")
    for ds, r in results.items():
        print(f"  {ds:<18}  {r['geo_time']:>12.1f}  {r['xgb_time']:>12.1f}")

    print()


if __name__ == "__main__":
    main()
