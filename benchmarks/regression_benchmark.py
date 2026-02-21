"""
GeoXGB vs XGBoost -- Regression Benchmark
==========================================

Dataset: Friedman #1 (sklearn), 1,000 samples, 10 features
  - 5 signal features (friedman_0 .. friedman_4) -- used in the true formula:
        y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
  - 5 noise  features (noise_0 .. noise_4) -- uniform [0, 1], independent of y

The interaction term sin(pi*x0*x1) is geometrically complex: HVRT partitions
should isolate it into a distinct region.  friedman_3 (linear in y) is easy to
fit but geometrically simple -- expect boosting rank >> partition rank for x3
and partition rank >> boosting rank for the interaction pair x0/x1.

Structure
---------
  [1] Dataset
  [2] Hyperparameter optimisation (random search, 3-fold CV)
        Validates that the GeoXGB defaults are sane.
  [3] Final model performance (best found vs defaults)
  [4] GeoXGB interpretability -- what XGBoost cannot provide
        4.1 Data quality (noise estimate)
        4.2 Sample provenance
        4.3 Dual importance: boosting vs partition
        4.4 Partition tree rules
        4.5 Partition evolution
        4.6 Ground-truth validation
        4.7 Head-to-head comparison
  [5] XGBoost -- available insights only
  [6] Summary

Usage
-----
    python benchmarks/regression_benchmark.py

Requirements: geoxgb, xgboost, scikit-learn, numpy
"""

from __future__ import annotations

import random
import time
import warnings
from itertools import product

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from geoxgb import GeoXGBRegressor
from geoxgb.report import (
    compare_report,
    evolution_report,
    importance_report,
    noise_report,
    partition_report,
    print_report,
    provenance_report,
    validation_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES    = 1_000
N_SIGNAL     = 5    # Friedman features used in the formula
N_NOISE      = 5    # independent uniform features appended by make_friedman1
N_FEATURES   = N_SIGNAL + N_NOISE

FEATURE_NAMES = (
    [f"friedman_{i}" for i in range(N_SIGNAL)]
    + [f"noise_{i}"    for i in range(N_NOISE)]
)

# Known ground truth -- only friedman_0..friedman_4 appear in the formula
GROUND_TRUTH = {
    "signal_features": list(range(N_SIGNAL)),
    "noise_features":  list(range(N_SIGNAL, N_FEATURES)),
    "mechanism": (
        "Friedman #1 regression: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 "
        "+ 10*x3 + 5*x4 + noise. "
        "friedman_0 and friedman_1 interact non-linearly (sin product); "
        "friedman_3 is purely linear. "
        "noise_0..noise_4 are independent of y."
    ),
}

# ---------------------------------------------------------------------------
# Fixed (non-tuned) parameters
# ---------------------------------------------------------------------------

GEO_FIXED = dict(
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    cache_geometry=False,
    random_state=RANDOM_STATE,
)

XGB_FIXED = dict(
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=0,
)

# Defaults under test
GEO_DEFAULTS = dict(n_rounds=100, learning_rate=0.1, max_depth=4)
XGB_DEFAULTS = dict(n_estimators=100, learning_rate=0.1, max_depth=4)

# ---------------------------------------------------------------------------
# HPO search space
# ---------------------------------------------------------------------------

GEO_SEARCH = dict(
    n_rounds=[50, 100, 150],
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[3, 4, 6],
)

XGB_SEARCH = dict(
    n_estimators=[50, 100, 150],
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[3, 4, 6],
)

N_HPO_CONFIGS = 9
N_HPO_FOLDS   = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title: str) -> None:
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


def _bar(val: float, max_val: float, width: int = 30) -> str:
    return "#" * int(round(width * val / max(max_val, 1e-12)))


def _print_xgb_importance(model: XGBRegressor, feature_names: list[str]) -> None:
    scores = model.feature_importances_
    max_s  = scores.max() if scores.max() > 0 else 1.0
    for name, s in sorted(zip(feature_names, scores), key=lambda x: -x[1]):
        print(f"  {name:<20s}  {_bar(s, max_s):<30s}  {s:.4f}")


def _random_configs(space: dict, n: int, rng: int = RANDOM_STATE) -> list[dict]:
    keys   = list(space.keys())
    combos = list(product(*[space[k] for k in keys]))
    rnd    = random.Random(rng)
    chosen = rnd.sample(combos, min(n, len(combos)))
    return [dict(zip(keys, c)) for c in chosen]


def _r2_scorer(model, X_val, y_val) -> float:
    return float(r2_score(y_val, model.predict(X_val)))


def _run_hpo(model_cls, fixed: dict, search: dict, X, y,
             scorer, n_configs=N_HPO_CONFIGS, n_folds=N_HPO_FOLDS,
             must_include: dict | None = None) -> list[dict]:
    """Random-search CV. Returns list of result dicts sorted by mean CV score.

    ``must_include`` is always evaluated (e.g. the default config) even if
    it was not drawn by the random sampler.
    """
    configs = _random_configs(search, n_configs)
    if must_include is not None and must_include not in configs:
        configs.append(must_include)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    splits  = list(kf.split(X))
    results = []
    for cfg in configs:
        params = {**fixed, **cfg}
        scores = []
        for train_idx, val_idx in splits:
            m = model_cls(**params)
            m.fit(X[train_idx], y[train_idx])
            scores.append(scorer(m, X[val_idx], y[val_idx]))
        results.append({
            "params": cfg,
            "mean":   float(np.mean(scores)),
            "std":    float(np.std(scores)),
        })
    return sorted(results, key=lambda r: -r["mean"])


def _print_hpo_table(results: list[dict], defaults: dict, metric: str = "R^2") -> None:
    print(f"\n  {'Config':<45s}  {'CV ' + metric:>8s}  {'Std':>6s}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*6}")
    for r in results:
        tag  = "  [best]   " if r is results[0] else "           "
        desc = "  ".join(f"{k}={v}" for k, v in r["params"].items())
        flag = " *" if r["params"] == defaults else "  "
        print(f"  {flag}{tag}{desc:<43s}  {r['mean']:>8.4f}  {r['std']:>6.4f}")
    print(f"\n  * = default configuration")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------------------
    # [1] Dataset
    # -----------------------------------------------------------------------
    _section("[1] DATASET")
    print(
        "\n  Friedman #1 regression"
        "\n  Samples : 1,000  (800 train / 200 test)"
        "\n  Features: 10  --  5 signal (friedman_0..4) + 5 noise (noise_0..4)"
        "\n  Formula : y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise"
        "\n"
        "\n  friedman_0 and friedman_1 interact non-linearly (sin product):"
        "\n    geometrically complex -- HVRT should partition them prominently."
        "\n  friedman_3 enters y linearly:"
        "\n    geometrically simple  -- boosting rank >> partition rank."
    )

    X, y = make_friedman1(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=1.0,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]  "
          f"mean={y.mean():.2f}  std={y.std():.2f}")

    # -----------------------------------------------------------------------
    # [2] Hyperparameter optimisation
    # -----------------------------------------------------------------------
    _section("[2] HYPERPARAMETER OPTIMISATION -- VALIDATING DEFAULTS")
    print(
        f"\n  Random search over (n_rounds/n_estimators, learning_rate, max_depth)"
        f"\n  {N_HPO_CONFIGS} configurations x {N_HPO_FOLDS}-fold CV on training set"
        f"\n  Goal: confirm defaults are competitive with the best found config."
    )

    # GeoXGB HPO
    _subsection("[2.1] GeoXGB random search")
    print("  Searching...")
    t0          = time.perf_counter()
    geo_hpo     = _run_hpo(GeoXGBRegressor, GEO_FIXED, GEO_SEARCH,
                           X_train, y_train, _r2_scorer,
                           must_include=GEO_DEFAULTS)
    geo_hpo_time = time.perf_counter() - t0
    print(f"  Search complete in {geo_hpo_time:.1f}s")
    _print_hpo_table(geo_hpo, GEO_DEFAULTS, metric="R^2")

    geo_best_cfg = {**GEO_FIXED, **geo_hpo[0]["params"]}
    geo_best_cv  = geo_hpo[0]["mean"]
    geo_def_cv   = next(r["mean"] for r in geo_hpo if r["params"] == GEO_DEFAULTS)
    geo_gap      = geo_best_cv - geo_def_cv
    print(f"\n  Best found : R^2={geo_best_cv:.4f}  {geo_hpo[0]['params']}")
    print(f"  Defaults   : R^2={geo_def_cv:.4f}  {GEO_DEFAULTS}")
    print(f"  Gap        : {geo_gap:+.4f}  "
          f"({'defaults are competitive' if abs(geo_gap) < 0.02 else 'tuning helps'})")

    # XGBoost HPO
    _subsection("[2.2] XGBoost random search")
    print("  Searching...")
    t0           = time.perf_counter()
    xgb_hpo      = _run_hpo(XGBRegressor, XGB_FIXED, XGB_SEARCH,
                            X_train, y_train, _r2_scorer,
                            must_include=XGB_DEFAULTS)
    xgb_hpo_time = time.perf_counter() - t0
    print(f"  Search complete in {xgb_hpo_time:.1f}s")
    _print_hpo_table(xgb_hpo, XGB_DEFAULTS, metric="R^2")

    xgb_best_cfg = {**XGB_FIXED, **xgb_hpo[0]["params"]}
    xgb_best_cv  = xgb_hpo[0]["mean"]
    xgb_def_cv   = next(r["mean"] for r in xgb_hpo if r["params"] == XGB_DEFAULTS)
    xgb_gap      = xgb_best_cv - xgb_def_cv
    print(f"\n  Best found : R^2={xgb_best_cv:.4f}  {xgb_hpo[0]['params']}")
    print(f"  Defaults   : R^2={xgb_def_cv:.4f}  {XGB_DEFAULTS}")
    print(f"  Gap        : {xgb_gap:+.4f}  "
          f"({'defaults are competitive' if abs(xgb_gap) < 0.02 else 'tuning helps'})")

    # -----------------------------------------------------------------------
    # [3] Final model performance
    # -----------------------------------------------------------------------
    _section("[3] FINAL MODEL PERFORMANCE")
    print(f"\n  Using best hyperparameters found in HPO.")

    t0 = time.perf_counter()
    geo = GeoXGBRegressor(**geo_best_cfg)
    geo.fit(X_train, y_train, feature_types=["continuous"] * N_FEATURES)
    geo_time = time.perf_counter() - t0
    geo_pred = geo.predict(X_test)
    geo_r2   = float(r2_score(y_test, geo_pred))
    geo_rmse = float(np.sqrt(np.mean((y_test - geo_pred) ** 2)))

    t0 = time.perf_counter()
    xgb = XGBRegressor(**xgb_best_cfg)
    xgb.fit(X_train, y_train)
    xgb_time = time.perf_counter() - t0
    xgb_pred = xgb.predict(X_test)
    xgb_r2   = float(r2_score(y_test, xgb_pred))
    xgb_rmse = float(np.sqrt(np.mean((y_test - xgb_pred) ** 2)))

    print()
    print(f"  {'Model':<22s}  {'Config':<30s}  {'R^2':>8s}  {'RMSE':>8s}  {'Time':>7s}")
    print(f"  {'-'*22}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*7}")
    print(f"  {'GeoXGB (best)':<22s}  {str(geo_hpo[0]['params']):<30s}  "
          f"{geo_r2:>8.4f}  {geo_rmse:>8.4f}  {geo_time:>6.1f}s")
    print(f"  {'XGBoost (best)':<22s}  {str(xgb_hpo[0]['params']):<30s}  "
          f"{xgb_r2:>8.4f}  {xgb_rmse:>8.4f}  {xgb_time:>6.1f}s")

    baseline_scores = {
        "r2":             xgb_r2,
        "geoxgb_r2":      geo_r2,
        "time":           xgb_time,
        "geoxgb_time":    geo_time,
        "n_samples_used": len(X_train),
    }

    # -----------------------------------------------------------------------
    # [4] GeoXGB interpretability
    # -----------------------------------------------------------------------
    _section("[4] GEOXGB INTERPRETABILITY -- WHAT XGBOOST CANNOT PROVIDE")

    _subsection("[4.1] Data Quality -- Noise Estimate")
    print(
        "  GeoXGB estimates signal-to-noise from partition geometry.\n"
        "  XGBoost trains on all data uniformly with no noise assessment."
    )
    print_report(noise_report(geo))

    _subsection("[4.2] Sample Provenance")
    print(
        "  GeoXGB tracks exactly how training data was curated:\n"
        "  how many samples were kept, discarded, or synthetically generated.\n"
        "  XGBoost has no such accounting -- it uses all samples as given."
    )
    print_report(provenance_report(geo, detail="standard"))

    _subsection("[4.3] Dual Feature Importance -- Boosting vs Partition")
    print(
        "  GeoXGB exposes TWO importance rankings:\n"
        "  - Boosting importance  -- which features drive predictions\n"
        "  - Partition importance -- which features define the data geometry\n"
        "\n"
        "  Friedman #1 reveals a natural duality:\n"
        "  - friedman_0 and friedman_1 interact non-linearly (sin product):\n"
        "    geometrically complex, so HVRT partitions them prominently.\n"
        "  - friedman_3 enters y linearly: easy to fit, weak geometry signal.\n"
        "  Expected: partition rank(friedman_0/1) >> boosting rank;\n"
        "            boosting rank(friedman_3)     >> partition rank.\n"
        "  XGBoost has only one importance score and cannot distinguish these roles."
    )
    print_report(
        importance_report(geo, feature_names=FEATURE_NAMES,
                          ground_truth=GROUND_TRUTH, detail="standard")
    )

    _subsection("[4.4] Partition Tree Rules -- Initial Geometry (round 0)")
    print(
        "  GeoXGB's HVRT partition tree encodes human-readable geometric rules\n"
        "  describing how feature space is divided into coherent regions.\n"
        "  The sin(pi*x0*x1) interaction creates a non-linear decision surface\n"
        "  that HVRT captures in its partition boundaries.\n"
        "  XGBoost has no partition structure -- it uses all data for every tree."
    )
    print_report(
        partition_report(geo, round_idx=0, feature_names=FEATURE_NAMES,
                         detail="standard")
    )

    _subsection("[4.5] Partition Evolution -- How Geometry Adapts Across Refits")
    print(
        "  At each refit interval GeoXGB re-evaluates the data geometry.\n"
        "  As boosting captures the linear signal (friedman_3), remaining\n"
        "  residuals become dominated by the non-linear interaction term,\n"
        "  causing partition importance to shift toward friedman_0/friedman_1.\n"
        "  XGBoost has no refit mechanism -- it never revisits data selection."
    )
    print_report(
        evolution_report(geo, feature_names=FEATURE_NAMES, detail="full")
    )

    _subsection("[4.6] Validation Against Known Ground Truth")
    print(
        "  Because GeoXGB's decisions are fully traceable, they can be verified\n"
        "  against known data properties and fail fast if something is wrong.\n"
        "  XGBoost's internal mechanics cannot be validated this way."
    )
    print_report(
        validation_report(geo, X_train, y_train,
                          feature_names=FEATURE_NAMES,
                          ground_truth=GROUND_TRUTH)
    )

    _subsection("[4.7] Performance Comparison Summary")
    print_report(compare_report(geo, baseline_scores, feature_names=FEATURE_NAMES))

    # -----------------------------------------------------------------------
    # [5] XGBoost
    # -----------------------------------------------------------------------
    _section("[5] XGBOOST -- AVAILABLE INSIGHTS")
    print(
        "  XGBoost provides one interpretability primitive: feature importance.\n"
        "  All other properties (provenance, noise, geometry, evolution)\n"
        "  are opaque."
    )
    print("\n  Feature importance (gain):")
    _print_xgb_importance(xgb, FEATURE_NAMES)
    print("\n  No noise estimate.")
    print("  No sample provenance.")
    print("  No partition structure or tree rules.")
    print("  No dual (geometric vs predictive) importance.")
    print("  No partition evolution.")
    print("  No ground-truth validation API.")

    # -----------------------------------------------------------------------
    # [6] Summary
    # -----------------------------------------------------------------------
    _section("[6] SUMMARY")

    noise = noise_report(geo)
    prov  = provenance_report(geo)
    imp   = importance_report(geo, feature_names=FEATURE_NAMES,
                               ground_truth=GROUND_TRUTH, detail="standard")
    evo   = evolution_report(geo, feature_names=FEATURE_NAMES, detail="standard")

    print(f"\n  HPO validation")
    print(f"    GeoXGB  best CV R^2={geo_best_cv:.4f}  defaults={geo_def_cv:.4f}  "
          f"gap={geo_gap:+.4f}")
    print(f"    XGBoost best CV R^2={xgb_best_cv:.4f}  defaults={xgb_def_cv:.4f}  "
          f"gap={xgb_gap:+.4f}")

    print(f"\n  Final test-set performance")
    print(f"    GeoXGB  R^2={geo_r2:.4f}  RMSE={geo_rmse:.4f}")
    print(f"    XGBoost R^2={xgb_r2:.4f}  RMSE={xgb_rmse:.4f}")

    print(f"\n  GeoXGB-only insights")
    print(f"    Data quality   : {noise['assessment']} "
          f"(noise modulation={noise['initial_modulation']:.3f})")
    print(f"    Samples used   : {prov['reduced_n']:,} of {prov['original_n']:,} real "
          f"+ {prov['expanded_n']:,} synthetic")
    print(f"    Importance agreement (Spearman): {imp['agreement']:.3f}")

    divergent = imp.get("divergent_features", [])
    if divergent:
        print(f"    Divergent features (geometry != prediction role):")
        for d in divergent[:5]:
            b_rank = d["boosting_rank"]
            p_rank = d["partition_rank"]
            if p_rank < b_rank:
                direction = " <- more geometric than predictive"
            elif b_rank < p_rank:
                direction = " <- more predictive than geometric"
            else:
                direction = ""
            print(f"      {d['feature']:<20s}  boost #{b_rank:>2d} | "
                  f"partition #{p_rank:>2d}  (diff={d['rank_diff']}){direction}")
    else:
        print("    No strongly divergent features detected.")

    val = imp.get("validation", {})
    if val:
        print(f"    Signal capture -- boosting: {val.get('signal_pct_boosting',0):.1f}%  "
              f"partition: {val.get('signal_pct_partition',0):.1f}%")
        print(f"    Noise ignored  -- boosting: "
              f"{'YES' if val.get('boosting_ignores_noise') else 'NO'}  "
              f"partition: {'YES' if val.get('partition_ignores_noise') else 'NO'}")

    nt = evo.get("noise_trend", {})
    if nt:
        print(f"    Noise modulation trend: {nt['direction']} "
              f"({nt['start']:.3f} -> {nt['end']:.3f})")
        if nt["direction"] == "decreasing":
            print(
                "    Interpretation: as boosting captured the linear signal\n"
                "    (friedman_3), residuals became less structured -- the model\n"
                "    progressively learned the easy signal first."
            )

    print()


if __name__ == "__main__":
    main()
