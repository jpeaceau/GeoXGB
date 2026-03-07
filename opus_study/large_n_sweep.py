"""
Large-n block-size sensitivity study.

Tests block-size coefficients at n=10k, 20k, 50k, 100k, 300k, 500k across
multiple datasets. No "none"/full baseline (too slow + worse accuracy).

Block formulas tested: sqrt(n)*C for C in [10, 15, 20, 25, 30, 40]
with floor of 2000.

Datasets:
  1. friedman1       (10d, 5 informative, non-linear interactions)
  2. linreg_20d      (20d, 15 informative, linear)
  3. california      (8d, real-world spatial, max ~20k native, upsampled beyond)
  4. make_regression  (10d, 8 informative, moderate noise)
  5. friedman3        (4d, non-linear arctan)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd
from sklearn.datasets import (make_friedman1, make_friedman3, make_regression,
                               fetch_california_housing)
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor


# ── Block-size formulas ──────────────────────────────────────────────────────

COEFFICIENTS = [10, 15, 20, 25, 30, 40]

def block_for(n, coeff):
    if n <= 5000:
        return None
    return max(2000, int(math.sqrt(n) * coeff))


# ── Datasets ─────────────────────────────────────────────────────────────────

def make_friedman1_ds(n):
    return make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)

def make_friedman3_ds(n):
    return make_friedman3(n_samples=n, noise=0.5, random_state=42)

def make_linreg(n):
    return make_regression(n_samples=n, n_features=20, n_informative=15,
                           noise=1.0, random_state=42)

def make_reg_10d(n):
    return make_regression(n_samples=n, n_features=10, n_informative=8,
                           noise=5.0, random_state=42)

def make_california(n):
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    rng = np.random.RandomState(42)
    if n <= len(X_full):
        idx = rng.choice(len(X_full), size=n, replace=False)
    else:
        idx = rng.choice(len(X_full), size=n, replace=True)
    return X_full[idx], y_full[idx]


DATASETS = {
    "friedman1":   make_friedman1_ds,
    "friedman3":   make_friedman3_ds,
    "linreg_20d":  make_linreg,
    "reg_10d":     make_reg_10d,
    "california":  make_california,
}


# ── Config ────────────────────────────────────────────────────────────────────

N_SIZES = [10_000, 20_000, 50_000, 100_000, 300_000, 500_000]
N_FOLDS = 3
N_ROUNDS = 500
REFIT_INTERVAL = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_geoxgb(X, y, block_size, folds):
    scores, times = [], []
    for tr, val in folds:
        m = GeoXGBRegressor(
            n_rounds=N_ROUNDS,
            learning_rate=0.02,
            max_depth=3,
            refit_interval=REFIT_INTERVAL,
            sample_block_n=block_size,
            random_state=42,
        )
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        dt = time.perf_counter() - t0
        preds = m.predict(X[val])
        scores.append(r2_score(y[val], preds))
        times.append(dt)
    return np.mean(scores), np.mean(times)


def eval_xgboost(X, y, folds):
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
            print(f"  {ds_name}  n={n:,}")
            print(f"{'='*70}")

            X, y = ds_fn(n)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            folds = list(kf.split(X))

            xgb_r2, xgb_time = eval_xgboost(X, y, folds)
            print(f"  XGBoost:  R2={xgb_r2:.4f}  time={xgb_time:.2f}s")

            for coeff in COEFFICIENTS:
                blk = block_for(n, coeff)
                blk_label = f"{blk}" if blk is not None else "off"

                try:
                    geo_r2, geo_time = eval_geoxgb(X, y, blk, folds)
                except Exception as e:
                    print(f"  FAIL coeff={coeff} blk={blk_label}: {e}")
                    geo_r2, geo_time = float("nan"), float("nan")

                delta = geo_r2 - xgb_r2
                ratio = geo_time / max(xgb_time, 0.001)
                print(
                    f"  C={coeff:>2}  blk={blk_label:>6s}  "
                    f"R2={geo_r2:.4f} ({delta:+.4f})  "
                    f"time={geo_time:.2f}s ({ratio:.1f}x)"
                )

                results.append({
                    "dataset": ds_name,
                    "n": n,
                    "coeff": coeff,
                    "block_size": blk,
                    "r2": round(geo_r2, 5),
                    "time_s": round(geo_time, 3),
                    "xgb_r2": round(xgb_r2, 5),
                    "xgb_time_s": round(xgb_time, 3),
                    "r2_delta": round(delta, 5),
                    "speed_ratio": round(ratio, 2),
                })

            # Save incrementally after each (dataset, n) combo
            df = pd.DataFrame(results)
            df.to_csv("opus_study/results_large_n_sweep.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("opus_study/results_large_n_sweep.csv", index=False)
    print(f"\n\nResults saved to opus_study/results_large_n_sweep.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  BEST COEFFICIENT PER (dataset, n)")
    print("="*70)
    for (ds, n_val), grp in df.groupby(["dataset", "n"]):
        best = grp.loc[grp["r2"].idxmax()]
        print(
            f"  {ds:15s} n={n_val:>7,}  "
            f"best_C={int(best['coeff']):>2}  blk={str(best['block_size']):>6s}  "
            f"R2={best['r2']:.4f} vs XGB={best['xgb_r2']:.4f} ({best['r2_delta']:+.4f})  "
            f"time={best['time_s']:.2f}s ({best['speed_ratio']:.1f}x)"
        )

    print("\n" + "="*70)
    print("  MEAN DELTA BY COEFFICIENT (across all datasets and n)")
    print("="*70)
    for coeff in COEFFICIENTS:
        sub = df[df["coeff"] == coeff]
        print(
            f"  C={coeff:>2}:  mean_delta={sub['r2_delta'].mean():+.5f}  "
            f"mean_speed={sub['speed_ratio'].mean():.2f}x  "
            f"wins_vs_xgb={int((sub['r2_delta'] > 0).sum())}/{len(sub)}"
        )

    print("\n" + "="*70)
    print("  MEAN DELTA BY COEFFICIENT (n >= 100k only)")
    print("="*70)
    large = df[df["n"] >= 100_000]
    if not large.empty:
        for coeff in COEFFICIENTS:
            sub = large[large["coeff"] == coeff]
            if sub.empty:
                continue
            print(
                f"  C={coeff:>2}:  mean_delta={sub['r2_delta'].mean():+.5f}  "
                f"mean_speed={sub['speed_ratio'].mean():.2f}x  "
                f"wins_vs_xgb={int((sub['r2_delta'] > 0).sum())}/{len(sub)}"
            )


if __name__ == "__main__":
    main()
