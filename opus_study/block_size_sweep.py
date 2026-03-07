"""
Block-size calibration study for GeoXGB sample batching.

Goal: find the optimal auto block-size formula as a function of n, comparing
PyramidHART vs HVRT partitioners with deterministic-only reduction methods.

Methodology:
- Sweep n from 2k to 100k across several synthetic + real-ish datasets
- For each n, test multiple block-size formulas and compare R2 vs XGBoost baseline
- Only deterministic reduction methods: variance_ordered (default for HVRT/PyramidHART)
- Track both accuracy (R2) and wall-clock time per fit

Datasets:
  1. friedman1 (5 informative + 5 noise, non-linear interactions)
  2. make_regression (15/20 informative, linear-ish)
  3. california_housing (real, 8 features, ~20k, upsampled for large n)

Block-size formulas tested:
  A. current:    500 + (n-5000)//50          (500 at 5k, 2400 at 100k)
  B. sqrt:       max(500, int(sqrt(n)*10))   (707 at 5k, 3162 at 100k)
  C. log-scaled: max(500, int(n/log2(n)))    (409 at 5k, 5987 at 100k)
  D. tenth:      max(1000, n//10)            (1000 at 5k, 10000 at 100k)
  E. fifth:      max(1000, n//5)             (1000 at 5k, 20000 at 100k)
  F. none:       None (full dataset)

Output: CSV with columns [dataset, n, partitioner, method, block_formula, r2, time_s, xgb_r2, xgb_time_s]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, make_regression, fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor


# ── Block-size formulas ──────────────────────────────────────────────────────

def block_current(n):
    """Current auto formula: 500 + (n-5000)//50"""
    if n <= 5000:
        return None
    return 500 + (n - 5000) // 50

def block_sqrt(n):
    """sqrt-scaled: ~sqrt(n)*10"""
    if n <= 5000:
        return None
    return max(500, int(math.sqrt(n) * 10))

def block_log(n):
    """log-scaled: n / log2(n)"""
    if n <= 5000:
        return None
    return max(500, int(n / math.log2(n)))

def block_tenth(n):
    """n/10"""
    if n <= 5000:
        return None
    return max(1000, n // 10)

def block_fifth(n):
    """n/5"""
    if n <= 5000:
        return None
    return max(1000, n // 5)

def block_none(n):
    """No block cycling"""
    return None


BLOCK_FORMULAS = {
    "current":  block_current,
    "sqrt":     block_sqrt,
    "log":      block_log,
    "tenth":    block_tenth,
    "fifth":    block_fifth,
    "none":     block_none,
}


# ── Dataset generators ───────────────────────────────────────────────────────

def make_friedman(n):
    X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)
    return X, y

def make_linreg(n):
    X, y = make_regression(n_samples=n, n_features=20, n_informative=15,
                           noise=1.0, random_state=42)
    return X, y

def make_california(n):
    """California housing, resampled to target n."""
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    rng = np.random.RandomState(42)
    if n <= len(X_full):
        idx = rng.choice(len(X_full), size=n, replace=False)
    else:
        idx = rng.choice(len(X_full), size=n, replace=True)
    return X_full[idx], y_full[idx]


DATASETS = {
    "friedman1":  make_friedman,
    "linreg_20d": make_linreg,
    "california": make_california,
}


# ── Partitioner x method combos ──────────────────────────────────────────────
# Deterministic-only methods for each partitioner

PARTITIONER_CONFIGS = [
    {"partitioner": "pyramid_hart", "method": "variance_ordered"},
    {"partitioner": "hvrt",         "method": "variance_ordered"},
]


# ── Run parameters ────────────────────────────────────────────────────────────

N_SIZES = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_FOLDS = 3
N_ROUNDS = 1000
REFIT_INTERVAL = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_geoxgb(X, y, partitioner, method, block_size, folds):
    """Fit GeoXGB with given config, return mean R2 and mean time."""
    scores, times = [], []
    for tr, val in folds:
        m = GeoXGBRegressor(
            n_rounds=N_ROUNDS,
            learning_rate=0.02,
            max_depth=3,
            refit_interval=REFIT_INTERVAL,
            partitioner=partitioner,
            method=method,
            sample_block_n=block_size,
            random_state=42,
            expand_ratio=0.1,
            reduce_ratio=0.8,
            auto_noise=True,
            noise_guard=True,
        )
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        dt = time.perf_counter() - t0
        preds = m.predict(X[val])
        scores.append(r2_score(y[val], preds))
        times.append(dt)
    return np.mean(scores), np.mean(times)


def eval_xgboost(X, y, folds):
    """XGBoost baseline with reasonable defaults, return mean R2 and mean time."""
    scores, times = [], []
    for tr, val in folds:
        m = xgb.XGBRegressor(
            n_estimators=N_ROUNDS,
            learning_rate=0.02,
            max_depth=3,
            random_state=42,
            verbosity=0,
            tree_method="hist",
        )
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        dt = time.perf_counter() - t0
        preds = m.predict(X[val])
        scores.append(r2_score(y[val], preds))
        times.append(dt)
    return np.mean(scores), np.mean(times)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []

    for ds_name, ds_fn in DATASETS.items():
        for n in N_SIZES:
            print(f"\n{'='*70}")
            print(f"  Dataset: {ds_name}  |  n = {n:,}")
            print(f"{'='*70}")

            X, y = ds_fn(n)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            folds = list(kf.split(X))

            # XGBoost baseline (once per dataset x n)
            xgb_r2, xgb_time = eval_xgboost(X, y, folds)
            print(f"  XGBoost:  R2={xgb_r2:.4f}  time={xgb_time:.2f}s")

            for pcfg in PARTITIONER_CONFIGS:
                part = pcfg["partitioner"]
                meth = pcfg["method"]

                for fname, ffn in BLOCK_FORMULAS.items():
                    blk = ffn(n)
                    blk_label = f"{blk}" if blk is not None else "full"

                    try:
                        geo_r2, geo_time = eval_geoxgb(
                            X, y, part, meth, blk, folds
                        )
                    except Exception as e:
                        print(f"  FAIL {part}/{meth} blk={blk_label}: {e}")
                        geo_r2, geo_time = float("nan"), float("nan")

                    delta = geo_r2 - xgb_r2
                    print(
                        f"  {part:15s} {meth:20s} blk={blk_label:>6s}  "
                        f"R2={geo_r2:.4f} ({delta:+.4f})  "
                        f"time={geo_time:.2f}s ({geo_time/max(xgb_time,0.001):.1f}x)"
                    )

                    results.append({
                        "dataset": ds_name,
                        "n": n,
                        "partitioner": part,
                        "method": meth,
                        "block_formula": fname,
                        "block_size": blk,
                        "r2": round(geo_r2, 5),
                        "time_s": round(geo_time, 3),
                        "xgb_r2": round(xgb_r2, 5),
                        "xgb_time_s": round(xgb_time, 3),
                        "r2_delta": round(delta, 5),
                        "speed_ratio": round(geo_time / max(xgb_time, 0.001), 2),
                    })

    df = pd.DataFrame(results)
    out_path = "opus_study/results_block_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\n\nResults saved to {out_path}")
    print(f"\n{df.to_string(index=False)}")

    # ── Summary: best formula per dataset x n ─────────────────────────────────
    print("\n\n" + "="*70)
    print("  SUMMARY: Best block formula per (dataset, n)")
    print("="*70)
    for (ds, n_val), grp in df.groupby(["dataset", "n"]):
        best = grp.loc[grp["r2"].idxmax()]
        xr2 = best["xgb_r2"]
        print(
            f"  {ds:15s} n={n_val:>7,}  "
            f"best={best['block_formula']:>8s} ({best['partitioner']:>13s})  "
            f"R2={best['r2']:.4f} vs XGB={xr2:.4f} ({best['r2_delta']:+.4f})  "
            f"time={best['time_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
