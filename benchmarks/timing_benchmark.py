"""
Fit and Inference Timing Benchmark
====================================

Measures wall-clock time for GeoXGB and XGBoost fit/predict across
dataset sizes, isolates HVRT component costs, and reports throughput.

Sections
--------
  1. GeoXGB vs XGBoost  fit + predict time  (diabetes, friedman1, large)
  2. HVRT component breakdown               (fit, reduce, expand in isolation)
  3. GeoXGB scaling profile                 (n=250 -> 8000, fixed p=10)

Usage
-----
  python benchmarks/timing_benchmark.py

Requirements: geoxgb, xgboost, hvrt, scikit-learn, numpy
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.datasets import load_diabetes, make_friedman1, make_regression
from sklearn.model_selection import train_test_split

from geoxgb import GeoXGBRegressor
from hvrt import HVRT

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_REPS       = 3    # repeat each timing N_REPS times, take median

_SEP  = "=" * 72
_SEP2 = "-" * 60


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}", flush=True)


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}", flush=True)


def _median_time(fn, reps=N_REPS):
    """Return median elapsed seconds over ``reps`` calls."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Section 1: GeoXGB vs XGBoost fit + predict
# ---------------------------------------------------------------------------

def _section1():
    _section("1. GeoXGB vs XGBoost -- Fit and Predict Timing")

    from xgboost import XGBRegressor

    datasets = {}

    X_diab, y_diab = load_diabetes(return_X_y=True)
    datasets["diabetes (n=442, p=10)"] = (X_diab, y_diab)

    X1, y1 = make_friedman1(n_samples=1000, n_features=10, noise=1.0,
                             random_state=RANDOM_STATE)
    datasets["friedman1 (n=1000, p=10)"] = (X1, y1)

    X2, y2 = make_friedman1(n_samples=5000, n_features=10, noise=1.0,
                             random_state=RANDOM_STATE)
    datasets["friedman1 (n=5000, p=10)"] = (X2, y2)

    print(f"\n  GeoXGB params: n_rounds=1000, defaults (generation_strategy=epanechnikov)")
    print(f"  XGBoost params: n_estimators=1000, defaults")
    print(f"  Reps: {N_REPS} (median reported)\n")

    header = f"  {'Dataset':<28}  {'Model':<10}  {'Fit (s)':>8}  {'Predict (ms)':>13}  {'Fit ratio':>10}"
    print(header)
    print(f"  {'-'*28}  {'-'*10}  {'-'*8}  {'-'*13}  {'-'*10}")

    for ds_name, (X, y) in datasets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        geo = GeoXGBRegressor(n_rounds=1000, random_state=RANDOM_STATE)
        geo_fit  = _median_time(lambda: geo.fit(X_tr, y_tr))
        geo_pred = _median_time(lambda: geo.predict(X_te)) * 1000.0

        xgb = XGBRegressor(n_estimators=1000, random_state=RANDOM_STATE,
                            verbosity=0, eval_metric="rmse")
        xgb_fit  = _median_time(lambda: xgb.fit(X_tr, y_tr))
        xgb_pred = _median_time(lambda: xgb.predict(X_te)) * 1000.0

        ratio = geo_fit / xgb_fit

        print(f"  {ds_name:<28}  {'GeoXGB':<10}  {geo_fit:>8.2f}  {geo_pred:>11.1f}ms  {ratio:>9.1f}x")
        print(f"  {'':<28}  {'XGBoost':<10}  {xgb_fit:>8.2f}  {xgb_pred:>11.1f}ms  {'(baseline)':>10}")
        print()


# ---------------------------------------------------------------------------
# Section 2: HVRT component breakdown
# ---------------------------------------------------------------------------

def _section2():
    _section("2. HVRT Component Breakdown (fit / reduce / expand)")

    sizes = [
        ("n=442  p=10", 442,  10),
        ("n=1000 p=10", 1000, 10),
        ("n=5000 p=10", 5000, 10),
        ("n=1000 p=20", 1000, 20),
    ]

    header = (f"  {'Dataset':<16}  {'HVRT.fit':>10}  {'reduce(0.7)':>12}"
              f"  {'expand(1x)':>11}  {'total':>8}")
    print(f"\n  generation_strategy=epanechnikov, bandwidth=auto\n")
    print(header)
    print(f"  {'-'*16}  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*8}")

    for label, n, p in sizes:
        rng = np.random.RandomState(RANDOM_STATE)
        X = rng.randn(n, p)
        y = X[:, 0] * 2 + rng.randn(n) * 0.5

        h = HVRT(y_weight=0.5, bandwidth="auto", random_state=RANDOM_STATE)

        t_fit    = _median_time(lambda: h.fit(X, y))
        h.fit(X, y)  # ensure fitted for reduce/expand

        t_reduce = _median_time(lambda: h.reduce(
            n=max(10, int(n * 0.7)), method="fps",
            variance_weighted=True, return_indices=True,
        ))
        t_expand = _median_time(lambda: h.expand(
            n=n, variance_weighted=True,
            generation_strategy="epanechnikov",
        ))

        t_total = t_fit + t_reduce + t_expand
        print(
            f"  {label:<16}  {t_fit*1000:>8.1f}ms  "
            f"{t_reduce*1000:>10.1f}ms  "
            f"{t_expand*1000:>9.1f}ms  "
            f"{t_total*1000:>6.1f}ms"
        )


# ---------------------------------------------------------------------------
# Section 3: GeoXGB scaling profile
# ---------------------------------------------------------------------------

def _section3():
    _section("3. GeoXGB Scaling Profile  (n=250 to 8000, p=10, n_rounds=500)")

    sizes = [250, 500, 1000, 2000, 4000, 8000]

    print(f"\n  n_rounds=500, all other params default\n")
    header = (f"  {'n_train':>8}  {'fit (s)':>9}  {'s/round':>9}"
              f"  {'predict (ms)':>13}  {'fit ratio vs n=1000':>20}")
    print(header)
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*13}  {'-'*20}")

    baseline_fit = None
    for n in sizes:
        X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0,
                               random_state=RANDOM_STATE)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        geo = GeoXGBRegressor(n_rounds=500, random_state=RANDOM_STATE)
        t_fit = _median_time(lambda: geo.fit(X_tr, y_tr))
        geo.fit(X_tr, y_tr)
        t_pred = _median_time(lambda: geo.predict(X_te)) * 1000.0

        if n == 1000:
            baseline_fit = t_fit

        ratio_str = f"{t_fit / baseline_fit:.2f}x" if baseline_fit else "  --"
        print(
            f"  {n:>8}  {t_fit:>9.2f}  {t_fit/500:>9.4f}"
            f"  {t_pred:>11.1f}ms  {ratio_str:>20}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _section("FIT AND INFERENCE TIMING BENCHMARK  (HVRT 2.6.0 + Numba)")

    import hvrt, geoxgb
    print(f"\n  hvrt version : {hvrt.__version__}")
    print(f"  geoxgb version: {geoxgb.__version__}")
    print(f"  Reps per measurement: {N_REPS} (median)")

    _section1()
    _section2()
    _section3()

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
