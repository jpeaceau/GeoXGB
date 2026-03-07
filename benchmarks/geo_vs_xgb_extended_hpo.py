"""
Extended HPO: GeoXGB vs XGBoost on California Housing
======================================================

Can GeoXGB beat XGBoost HPO (R²=0.8564) with a wider search space?

Changes vs interpretability_demo.py
------------------------------------
  • 50 trials (vs 20)
  • Expanded search space:
      - learning_rate   adds 0.003
      - max_depth       adds 6, 7
      - n_rounds        adds 4000
      - refit_interval  adds 300, 500
      - expand_ratio    adds 0.4, 0.5
      - hvrt_min_samples_leaf  NEW: [None, 5, 10, 20, 30]
  • Warm-start trial 0 = previous best config (CV R²=0.8295)
  • Trial-by-trial progress printed live via Optuna callback
  • Refit winner on full 16 k training set

Usage
-----
    python benchmarks/geo_vs_xgb_extended_hpo.py

Requirements
------------
    pip install optuna xgboost matplotlib
"""
from __future__ import annotations

import sys
import time
import warnings
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import optuna
import xgboost as xgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

optuna.logging.set_verbosity(optuna.logging.WARNING)

from geoxgb import GeoXGBRegressor, ContributionFrame
from geoxgb.optimizer import GeoXGBOptimizer

def _p(*a, **k): print(*a, **k, flush=True)
def _sec(t0): return f"{time.perf_counter()-t0:.1f}s"
def _banner(s): _p(f"\n{'='*65}\n{s}\n{'='*65}")

# =============================================================================
# DATA
# =============================================================================
_banner("Data: California Housing")
housing    = fetch_california_housing()
X, y       = housing.data, housing.target
feat_names = list(housing.feature_names)
d          = len(feat_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
_p(f"  train={len(X_train)}, test={len(X_test)}, d={d}")
FT = ["continuous"] * d

# 8 k HPO subsample (> 5 k block-cycling threshold)
rng     = np.random.default_rng(42)
sub_idx = rng.choice(len(X_train), size=8000, replace=False)
X_hpo, y_hpo = X_train[sub_idx], y_train[sub_idx]
_p(f"  HPO subsample: {len(X_hpo)} (block cycling active during HPO)")

kf     = KFold(n_splits=3, shuffle=True, random_state=42)
splits = list(kf.split(X_hpo))

# =============================================================================
# GEOXGB — EXTENDED HPO
# =============================================================================
_banner("GeoXGB — Extended HPO (50 trials, expanded search space)")

# --- Search space -------------------------------------------------------
GEO_SPACE = {
    "n_rounds":              [1000, 1500, 2000, 3000, 4000],
    "learning_rate":         [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03],
    "max_depth":             [3, 4, 5, 6, 7],
    "reduce_ratio":          [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    "refit_interval":        [50, 100, 150, 200, 300, 500],
    "expand_ratio":          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "y_weight":              [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "hvrt_min_samples_leaf": [None, 5, 10, 20, 30],
    "sample_block_n":        [None, 400, 800, 1600],
}

# Warm start = previous best from 20-trial run
GEO_WARMSTART = {
    "n_rounds":              1500,
    "learning_rate":         0.03,
    "max_depth":             5,
    "reduce_ratio":          0.7,
    "refit_interval":        200,
    "expand_ratio":          0.3,
    "y_weight":              0.2,
    "hvrt_min_samples_leaf": None,
    "sample_block_n":        None,
}

_p("\n  Search space:")
for k, v in GEO_SPACE.items():
    _p(f"    {k:<24} {v}")
_p(f"\n  Warm-start trial 0 = previous best (CV R²≈0.83)")

def _geo_cv(params):
    run = {**params, "random_state": 42, "convergence_tol": 0.01}
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, val in splits:
            m = GeoXGBRegressor(**run)
            m.fit(X_hpo[tr], y_hpo[tr])
            scores.append(r2_score(y_hpo[val], m.predict(X_hpo[val])))
    return float(np.mean(scores))

def _geo_objective(trial):
    params = {k: trial.suggest_categorical(k, v) for k, v in GEO_SPACE.items()}
    return _geo_cv(params)

_best_geo_cv = [-np.inf]

def _geo_callback(study, trial):
    if trial.value is None:
        return
    new_best = trial.value > _best_geo_cv[0]
    if new_best:
        _best_geo_cv[0] = trial.value
    mark = "★" if new_best else " "
    _p(f"  #{trial.number:3d}  CV R²={trial.value:.4f}  "
       f"best={_best_geo_cv[0]:.4f} {mark}"
       + (f"  <- {trial.params}" if new_best else ""))

geo_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
geo_study.enqueue_trial(GEO_WARMSTART)

t0 = time.perf_counter()
_p()
geo_study.optimize(_geo_objective, n_trials=50, callbacks=[_geo_callback])
geo_hpo_time = _sec(t0)

geo_best_params = geo_study.best_params
geo_best_cv     = geo_study.best_value
_p(f"\n  HPO done in {geo_hpo_time}")
_p(f"  Best CV R² = {geo_best_cv:.4f}")
_p(f"  Best params: {geo_best_params}")

# Refit on full training set
_p("\n  Refitting on full 16 k training set with feature_types...")
t0 = time.perf_counter()
geo_final = GeoXGBRegressor(**geo_best_params, random_state=42)
geo_final.fit(X_train, y_train, feature_types=FT)
geo_test_r2 = r2_score(y_test, geo_final.predict(X_test))
_p(f"  Test R² = {geo_test_r2:.4f}  ({_sec(t0)})")

# =============================================================================
# XGBOOST — HPO BASELINE (30 trials, same 8 k subsample)
# =============================================================================
_banner("XGBoost — HPO (30 trials, 8 k subsample)")

XGB_SPACE = {
    "n_estimators":     (200, 1500),
    "max_depth":        (3, 8),
    "learning_rate":    (0.005, 0.3),
    "subsample":        (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "reg_alpha":        (1e-4, 10.0),
    "reg_lambda":       (1e-4, 10.0),
    "min_child_weight": (1, 20),
}
xgb_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X_hpo))

_best_xgb_cv = [-np.inf]

def _xgb_callback(study, trial):
    if trial.value is None:
        return
    new_best = trial.value > _best_xgb_cv[0]
    if new_best:
        _best_xgb_cv[0] = trial.value
    mark = "★" if new_best else " "
    _p(f"  #{trial.number:3d}  CV R²={trial.value:.4f}  "
       f"best={_best_xgb_cv[0]:.4f} {mark}"
       + (f"  <- {trial.params}" if new_best else ""))

def _xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", *XGB_SPACE["n_estimators"]),
        "max_depth":        trial.suggest_int("max_depth", *XGB_SPACE["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate", *XGB_SPACE["learning_rate"], log=True),
        "subsample":        trial.suggest_float("subsample", *XGB_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_SPACE["colsample_bytree"]),
        "reg_alpha":        trial.suggest_float("reg_alpha", *XGB_SPACE["reg_alpha"], log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", *XGB_SPACE["reg_lambda"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *XGB_SPACE["min_child_weight"]),
        "tree_method": "hist", "verbosity": 0, "random_state": 42,
    }
    scores = []
    for tr, val in xgb_splits:
        m = xgb.XGBRegressor(**params)
        m.fit(X_hpo[tr], y_hpo[tr])
        scores.append(r2_score(y_hpo[val], m.predict(X_hpo[val])))
    return float(np.mean(scores))

xgb_study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)

t0 = time.perf_counter()
_p()
xgb_study.optimize(_xgb_objective, n_trials=30, callbacks=[_xgb_callback])
xgb_hpo_time = _sec(t0)

xgb_best_params = {**xgb_study.best_params,
                   "tree_method": "hist", "verbosity": 0, "random_state": 42}
xgb_best_cv     = xgb_study.best_value
_p(f"\n  HPO done in {xgb_hpo_time}")
_p(f"  Best CV R² = {xgb_best_cv:.4f}")
_p(f"  Best params: {xgb_best_params}")

t0 = time.perf_counter()
xgb_final = xgb.XGBRegressor(**xgb_best_params)
xgb_final.fit(X_train, y_train)
xgb_test_r2 = r2_score(y_test, xgb_final.predict(X_test))
_p(f"  Test R² = {xgb_test_r2:.4f}  ({_sec(t0)})")

# =============================================================================
# RESULTS
# =============================================================================
_banner("RESULTS")
_p(f"  {'Model':<35}  {'CV R² (8k sub)':>14}  {'Test R² (16k)':>13}")
_p("  " + "-" * 65)
_p(f"  {'GeoXGB HPO (50 trials, extended)':<35}  {geo_best_cv:>14.4f}  {geo_test_r2:>13.4f}")
_p(f"  {'XGBoost HPO (30 trials)':<35}  {xgb_best_cv:>14.4f}  {xgb_test_r2:>13.4f}")

delta = geo_test_r2 - xgb_test_r2
_p(f"\n  GeoXGB - XGBoost: {delta:+.4f}  "
   f"({'GeoXGB wins' if delta > 0 else f'XGBoost wins by {-delta:.4f}'})")

_p("\n  GeoXGB best params:")
for k, v in geo_best_params.items():
    _p(f"    {k:<24} = {v}")

# =============================================================================
# TRIAL HISTORY SUMMARY (top 10 GeoXGB trials)
# =============================================================================
_banner("Top-10 GeoXGB Trials")
geo_trials = sorted(
    [t for t in geo_study.trials if t.value is not None],
    key=lambda t: -t.value
)
_p(f"  {'#':>4}  {'CV R²':>7}  params")
_p("  " + "-" * 80)
for t in geo_trials[:10]:
    _p(f"  #{t.number:3d}  {t.value:.4f}  {t.params}")

# =============================================================================
# PARAMETER IMPORTANCE (Optuna's built-in FAnova)
# =============================================================================
_banner("GeoXGB Parameter Importance (FAnova)")
try:
    importances = optuna.importance.get_param_importances(geo_study)
    total = sum(importances.values())
    for param, imp in importances.items():
        bar = "█" * int(imp / max(importances.values()) * 30)
        _p(f"  {param:<24} {imp:.4f}  {bar}")
except Exception as e:
    _p(f"  [skipped: {e}]")

# =============================================================================
# SAVE TRIAL HISTORY
# =============================================================================
import os, csv
out_dir = "benchmarks/meta_v2/results"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/extended_hpo_geo_trials.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["trial", "cv_r2"] + list(GEO_SPACE.keys())
    w.writerow(header)
    for t in sorted(geo_study.trials, key=lambda t: t.number):
        if t.value is not None:
            row = [t.number, t.value] + [t.params.get(k) for k in GEO_SPACE.keys()]
            w.writerow(row)
_p(f"\n  Trial history saved to {csv_path}")

_p("\nDone.")
