"""
GeoXGB -- KDE Bandwidth and Generation Strategy Benchmark
==========================================================

Tests the effect of KDE bandwidth values and generation strategies on GeoXGB
predictive performance.  HVRT's 'auto' default (h=0.10 Gaussian or Epanechnikov
based on mean partition size) is the baseline; all other conditions are compared
against it.

Background
----------
HVRT 2.2.0 changed the bandwidth default from 0.5 to 'auto'.  Prior benchmarks
(hvrt_findings.md) measured bandwidth effect on synthetic data quality (marginal
fidelity + TSTR).  This benchmark measures the downstream effect on the GeoXGB
boosting model:

  - Does the tighter h=0.10 Gaussian help, hurt, or make no difference?
  - Does Epanechnikov (covariance-free product kernel) suit the GeoXGB setting
    better than Gaussian?  It was not tested in the original HVRT benchmark.
  - How do Scott/Silverman (data-driven rules) compare to fixed scalars?
  - Does adaptive_bandwidth (per-partition bandwidth scaling by expansion ratio)
    add value on top of a fixed scalar?
  - How do alternative strategies (univariate_kde_copula, bootstrap_noise)
    perform relative to the multivariate_kde default?

Conditions tested
-----------------
  Group A: Gaussian KDE (generation_strategy=None), varying bandwidth
    auto (baseline)   bandwidth='auto',    strategy=None,                 adaptive=False
    bw=0.05           bandwidth=0.05,      strategy=None,                 adaptive=False
    bw=0.10           bandwidth=0.10,      strategy=None,                 adaptive=False
    bw=0.25           bandwidth=0.25,      strategy=None,                 adaptive=False
    bw=0.50           bandwidth=0.50,      strategy=None,                 adaptive=False
    bw=0.75           bandwidth=0.75,      strategy=None,                 adaptive=False
    scott             bandwidth='scott',   strategy=None,                 adaptive=False
    silverman         bandwidth='silverman', strategy=None,               adaptive=False

  Group B: Adaptive bandwidth (only valid with multivariate_kde)
    bw=0.10+adapt     bandwidth=0.10,      strategy=None,                 adaptive=True
    scott+adapt       bandwidth='scott',   strategy=None,                 adaptive=True

  Group C: Alternative generation strategies (bandwidth='auto' for HVRT init)
    epanechnikov      bandwidth='auto',    strategy='epanechnikov',       adaptive=False
    kde_copula        bandwidth='auto',    strategy='univariate_kde_copula', adaptive=False
    bootstrap_noise   bandwidth='auto',    strategy='bootstrap_noise',    adaptive=False

Datasets
--------
  Friedman #1 (GeoXGBRegressor, R^2 / RMSE)
  Synthetic binary classification (GeoXGBClassifier, AUC)

Each condition: 5-fold CV on 800 training samples.
Fixed GeoXGB params: n_rounds=150, lr=0.2, max_depth=4, reduce_ratio=0.7,
                     refit_interval=20, auto_noise=True, auto_expand=True.

Usage
-----
    python benchmarks/kde_bandwidth_benchmark.py

Requirements: geoxgb, xgboost, scikit-learn, numpy
"""

from __future__ import annotations

import time
import warnings
from itertools import product as iproduct

import numpy as np
from sklearn.datasets import make_friedman1, make_classification
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed

from geoxgb import GeoXGBRegressor, GeoXGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 1_000
N_FOLDS      = 5
N_JOBS       = -1   # use all cores

# Fixed GeoXGB params (same across all conditions)
GEO_FIXED = dict(
    n_rounds=150,
    learning_rate=0.2,
    max_depth=4,
    reduce_ratio=0.7,
    refit_interval=20,
    auto_noise=True,
    auto_expand=True,
    cache_geometry=False,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Condition definitions: (label, bandwidth, generation_strategy, adaptive_bw)
# ---------------------------------------------------------------------------
# Note on Epanechnikov: generation_strategy='epanechnikov' uses its own
# Scott-rule bandwidth internally; the 'bandwidth' param still governs HVRT
# geometry fitting (fit()), so 'auto' is appropriate for the geometry side.

CONDITIONS = [
    # -- Group A: Gaussian KDE, varying bandwidth --
    ("auto  (baseline)",  "auto",       None,                      False),
    ("bw=0.05",           0.05,         None,                      False),
    ("bw=0.10",           0.10,         None,                      False),
    ("bw=0.25",           0.25,         None,                      False),
    ("bw=0.50 (old)",     0.50,         None,                      False),
    ("bw=0.75",           0.75,         None,                      False),
    ("scott",             "scott",      None,                      False),
    ("silverman",         "silverman",  None,                      False),
    # -- Group B: Adaptive bandwidth --
    ("bw=0.10+adaptive",  0.10,         None,                      True),
    ("scott+adaptive",    "scott",      None,                      True),
    # -- Group C: Alternative generation strategies --
    ("epanechnikov",      "auto",       "epanechnikov",            False),
    ("kde_copula",        "auto",       "univariate_kde_copula",   False),
    ("bootstrap_noise",   "auto",       "bootstrap_noise",         False),
]

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# CV worker (module-level for Windows pickling)
# ---------------------------------------------------------------------------

def _cv_worker(
    model_cls, fixed_params, bw, strategy, adaptive, X_tr, y_tr, X_val, y_val,
    is_regressor
):
    import warnings as _w
    _w.filterwarnings("ignore")
    try:
        clf = model_cls(
            **fixed_params,
            bandwidth=bw,
            generation_strategy=strategy,
            adaptive_bandwidth=adaptive,
        )
        clf.fit(X_tr, y_tr)
        if is_regressor:
            pred = clf.predict(X_val)
            return float(r2_score(y_val, pred))
        else:
            proba = clf.predict_proba(X_val)[:, 1]
            return float(roc_auc_score(y_val, proba))
    except Exception as exc:
        return float("nan")


# ---------------------------------------------------------------------------
# Run all conditions via 5-fold CV
# ---------------------------------------------------------------------------

def run_conditions(model_cls, X, y, is_regressor, label):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(kf.split(X))

    # Build flat job list: (cond_idx, fold_idx, ...)
    jobs = [
        delayed(_cv_worker)(
            model_cls, GEO_FIXED,
            bw, strategy, adaptive,
            X[tr], y[tr], X[val], y[val],
            is_regressor,
        )
        for cond_idx, (lbl, bw, strategy, adaptive) in enumerate(CONDITIONS)
        for tr, val in splits
    ]

    t0 = time.perf_counter()
    flat_scores = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=0)(jobs)
    elapsed = time.perf_counter() - t0

    # Reshape: (n_conditions, n_folds)
    scores_2d = np.array(flat_scores, dtype=float).reshape(len(CONDITIONS), N_FOLDS)
    return scores_2d, elapsed


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results(scores_2d, metric_name, baseline_idx=0):
    baseline_mean = float(np.nanmean(scores_2d[baseline_idx]))

    print(f"\n  {'Condition':<26s}  {'Mean':>8s}  {'Std':>6s}  {'vs baseline':>12s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*8}")

    # Sort by mean (descending) for display, keeping baseline marked
    order = sorted(range(len(CONDITIONS)), key=lambda i: -float(np.nanmean(scores_2d[i])))
    for i in order:
        lbl, bw, strategy, adaptive = CONDITIONS[i]
        mean = float(np.nanmean(scores_2d[i]))
        std  = float(np.nanstd(scores_2d[i]))
        mn   = float(np.nanmin(scores_2d[i]))
        mx   = float(np.nanmax(scores_2d[i]))
        delta = mean - baseline_mean
        marker = " *" if i == baseline_idx else "  "
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   nan"
        print(f"{marker} {lbl:<26s}  {mean:>8.4f}  {std:>6.4f}  {delta_str:>12s}  {mn:>8.4f}  {mx:>8.4f}")

    print(f"\n  * = baseline condition (bandwidth='auto')")
    print(f"  Metric: {metric_name}  |  {N_FOLDS}-fold CV on {N_SAMPLES} samples")


# ---------------------------------------------------------------------------
# Per-group summary
# ---------------------------------------------------------------------------

def print_group_summary(scores_2d, metric_name, baseline_idx=0):
    baseline_mean = float(np.nanmean(scores_2d[baseline_idx]))
    groups = {
        "A: Gaussian KDE, varying bandwidth":  list(range(0, 8)),
        "B: Adaptive bandwidth":                list(range(8, 10)),
        "C: Alternative strategies":            list(range(10, 13)),
    }
    for group_name, idxs in groups.items():
        best_i = max(idxs, key=lambda i: float(np.nanmean(scores_2d[i])))
        best_m = float(np.nanmean(scores_2d[best_i]))
        print(f"  {group_name}")
        print(f"    Best: {CONDITIONS[best_i][0]:<26s}  {metric_name}={best_m:.4f}  "
              f"vs baseline {best_m - baseline_mean:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _section("KDE BANDWIDTH AND GENERATION STRATEGY BENCHMARK")
    print(
        f"\n  {len(CONDITIONS)} conditions x {N_FOLDS}-fold CV"
        f"\n  Datasets: Friedman #1 (regression) + synthetic binary (classification)"
        f"\n  Fixed params: n_rounds=150, lr=0.2, max_depth=4, reduce_ratio=0.7,"
        f"\n                refit_interval=20, auto_expand=True"
        f"\n  Baseline: bandwidth='auto' (HVRT 2.2.0 default)"
    )

    # -------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------
    X_reg, y_reg = make_friedman1(
        n_samples=N_SAMPLES, n_features=10, noise=1.0, random_state=RANDOM_STATE
    )
    X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )

    X_clf, y_clf = make_classification(
        n_samples=N_SAMPLES, n_features=10, n_informative=5, n_redundant=0,
        n_repeated=0, n_clusters_per_class=2, class_sep=1.0, random_state=RANDOM_STATE
    )
    X_clf_tr, X_clf_te, y_clf_tr, y_clf_te = train_test_split(
        X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=RANDOM_STATE
    )

    print(f"\n  Regression train/test: {len(X_reg_tr)}/{len(X_reg_te)}")
    print(f"  Classification train/test: {len(X_clf_tr)}/{len(X_clf_te)}")

    # -------------------------------------------------------------------
    # [1] Regression
    # -------------------------------------------------------------------
    _section("[1] FRIEDMAN #1 REGRESSION  (GeoXGBRegressor, metric=R^2)")
    print("  Running CV for all conditions...")

    reg_scores, reg_time = run_conditions(
        GeoXGBRegressor, X_reg_tr, y_reg_tr, is_regressor=True, label="regression"
    )
    print(f"  Done in {reg_time:.1f}s")

    print_results(reg_scores, metric_name="R^2")
    _subsection("Per-group best (regression)")
    print_group_summary(reg_scores, "R^2")

    # Final model comparison on held-out test set (best vs baseline)
    _subsection("Test-set scores: top 3 vs baseline (regression)")
    order_reg = sorted(range(len(CONDITIONS)), key=lambda i: -float(np.nanmean(reg_scores[i])))
    top3_plus_base = list(dict.fromkeys(order_reg[:3] + [0]))  # top3 + baseline, dedup
    for i in top3_plus_base:
        lbl, bw, strategy, adaptive = CONDITIONS[i]
        geo = GeoXGBRegressor(
            **GEO_FIXED,
            bandwidth=bw,
            generation_strategy=strategy,
            adaptive_bandwidth=adaptive,
        )
        geo.fit(X_reg_tr, y_reg_tr)
        pred = geo.predict(X_reg_te)
        r2   = float(r2_score(y_reg_te, pred))
        rmse = float(np.sqrt(np.mean((y_reg_te - pred) ** 2)))
        marker = " *" if i == 0 else "  "
        print(f"{marker} {lbl:<26s}  R^2={r2:.4f}  RMSE={rmse:.4f}")

    # -------------------------------------------------------------------
    # [2] Classification
    # -------------------------------------------------------------------
    _section("[2] SYNTHETIC BINARY CLASSIFICATION  (GeoXGBClassifier, metric=AUC)")
    print("  Running CV for all conditions...")

    clf_scores, clf_time = run_conditions(
        GeoXGBClassifier, X_clf_tr, y_clf_tr, is_regressor=False, label="classification"
    )
    print(f"  Done in {clf_time:.1f}s")

    print_results(clf_scores, metric_name="AUC")
    _subsection("Per-group best (classification)")
    print_group_summary(clf_scores, "AUC")

    # Final model comparison on held-out test set
    _subsection("Test-set scores: top 3 vs baseline (classification)")
    order_clf = sorted(range(len(CONDITIONS)), key=lambda i: -float(np.nanmean(clf_scores[i])))
    top3_plus_base_clf = list(dict.fromkeys(order_clf[:3] + [0]))
    for i in top3_plus_base_clf:
        lbl, bw, strategy, adaptive = CONDITIONS[i]
        clf = GeoXGBClassifier(
            **GEO_FIXED,
            bandwidth=bw,
            generation_strategy=strategy,
            adaptive_bandwidth=adaptive,
        )
        clf.fit(X_clf_tr, y_clf_tr)
        proba = clf.predict_proba(X_clf_te)[:, 1]
        auc = float(roc_auc_score(y_clf_te, proba))
        marker = " *" if i == 0 else "  "
        print(f"{marker} {lbl:<26s}  AUC={auc:.4f}")

    # -------------------------------------------------------------------
    # [3] Summary
    # -------------------------------------------------------------------
    _section("[3] SUMMARY")

    reg_baseline  = float(np.nanmean(reg_scores[0]))
    clf_baseline  = float(np.nanmean(clf_scores[0]))

    print(f"\n  Baseline (bandwidth='auto'): R^2={reg_baseline:.4f}  AUC={clf_baseline:.4f}")
    print(f"\n  {'Condition':<26s}  {'R^2 delta':>10s}  {'AUC delta':>10s}  {'Combined z':>12s}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*12}")

    # z-score ranking across both tasks
    reg_means = np.nanmean(reg_scores, axis=1)
    clf_means = np.nanmean(clf_scores, axis=1)
    reg_z = (reg_means - reg_means.mean()) / (reg_means.std() + 1e-12)
    clf_z = (clf_means - clf_means.mean()) / (clf_means.std() + 1e-12)
    combined_z = (reg_z + clf_z) / 2.0

    order_combined = np.argsort(-combined_z)
    for i in order_combined:
        lbl = CONDITIONS[i][0]
        rd  = reg_means[i] - reg_baseline
        cd  = clf_means[i] - clf_baseline
        marker = " *" if i == 0 else "  "
        print(f"{marker} {lbl:<26s}  {rd:>+10.4f}  {cd:>+10.4f}  {combined_z[i]:>12.4f}")

    print(f"\n  * = baseline condition")

    # Highlight winner
    best_i = int(order_combined[0])
    best_lbl = CONDITIONS[best_i][0]
    best_bw  = CONDITIONS[best_i][1]
    best_st  = CONDITIONS[best_i][2]
    best_ad  = CONDITIONS[best_i][3]
    print(f"\n  Combined winner : {best_lbl}")
    print(f"    bandwidth={best_bw}  generation_strategy={best_st}  adaptive_bandwidth={best_ad}")
    print(f"    R^2 delta vs baseline : {reg_means[best_i] - reg_baseline:+.4f}")
    print(f"    AUC delta vs baseline : {clf_means[best_i] - clf_baseline:+.4f}")

    if best_i == 0:
        print(f"\n  Conclusion: 'auto' is the optimal choice -- no condition improves on it.")
    else:
        print(f"\n  Conclusion: {best_lbl} outperforms 'auto' on a combined basis.")
        if best_i in range(0, 8):
            print(f"  Insight: A different Gaussian bandwidth scalar outperforms data-driven auto.")
        elif best_i in range(8, 10):
            print(f"  Insight: Adaptive per-partition bandwidth scaling improves on a fixed scalar.")
        elif best_i in range(10, 13):
            print(f"  Insight: The alternative generation strategy outperforms Gaussian KDE.")

    print()


if __name__ == "__main__":
    main()
