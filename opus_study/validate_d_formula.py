"""
Validate the d-dependent geometric block formula vs old sqrt(n)*15.

The new formula: max(2000, 20 * int(26.6 * d))
The old formula: max(2000, int(sqrt(n) * 15))

Key hypothesis: the d-dependent formula should be better at high-d (where
sqrt(n)*15 undersizes blocks) and equal or better at low-d large-n (where
sqrt(n)*15 oversizes blocks wastefully).

Test grid:
  - datasets with varying d: 4, 8, 10, 20
  - n sizes: 10k, 50k, 100k
  - 3-fold CV, 500 rounds
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor


# -- Block formulas -----------------------------------------------------------

def block_d_formula(n, d):
    """New d-dependent geometric formula."""
    if n <= 5000:
        return None
    return max(2000, 20 * int(26.6 * d))

def block_sqrt_formula(n, d):
    """Old sqrt(n)*15 formula."""
    if n <= 5000:
        return None
    return max(2000, int(math.sqrt(n) * 15))


# -- Datasets -----------------------------------------------------------------

def make_reg_4d(n):
    return make_regression(n_samples=n, n_features=4, n_informative=4,
                           noise=1.0, random_state=42)

def make_california(n):
    from sklearn.datasets import fetch_california_housing
    X_full, y_full = fetch_california_housing(return_X_y=True)
    if n <= len(X_full):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_full), n, replace=False)
        return X_full[idx], y_full[idx]
    # Oversample
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_full), n, replace=True)
    return X_full[idx], y_full[idx]

def make_friedman1_ds(n):
    return make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)

def make_reg_20d(n):
    return make_regression(n_samples=n, n_features=20, n_informative=15,
                           noise=1.0, random_state=42)


DATASETS = {
    "reg_4d":      (make_reg_4d, 4),
    "california":  (make_california, 8),
    "friedman1":   (make_friedman1_ds, 10),
    "linreg_20d":  (make_reg_20d, 20),
}

N_SIZES = [10_000, 50_000, 100_000]
N_FOLDS = 3
N_ROUNDS = 500


# -- Helpers ------------------------------------------------------------------

def eval_geo(X, y, block_size, folds):
    scores, times = [], []
    for tr, val in folds:
        m = GeoXGBRegressor(
            n_rounds=N_ROUNDS, learning_rate=0.02, max_depth=3,
            refit_interval=50, sample_block_n=block_size, random_state=42,
        )
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        dt = time.perf_counter() - t0
        scores.append(r2_score(y[val], m.predict(X[val])))
        times.append(dt)
    return np.mean(scores), np.mean(times)


def eval_xgboost(X, y, folds):
    scores, times = [], []
    for tr, val in folds:
        m = xgb.XGBRegressor(
            n_estimators=N_ROUNDS, learning_rate=0.02, max_depth=3,
            random_state=42, verbosity=0, tree_method="hist",
        )
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        dt = time.perf_counter() - t0
        scores.append(r2_score(y[val], m.predict(X[val])))
        times.append(dt)
    return np.mean(scores), np.mean(times)


# -- Main ---------------------------------------------------------------------

def main():
    results = []

    for ds_name, (ds_fn, d) in DATASETS.items():
        for n in N_SIZES:
            blk_d = block_d_formula(n, d)
            blk_sqrt = block_sqrt_formula(n, d)

            print(f"\n{'='*70}")
            print(f"  {ds_name} (d={d})  n={n:,}")
            print(f"  d-formula block={blk_d}  sqrt-formula block={blk_sqrt}")
            print(f"{'='*70}")

            X, y = ds_fn(n)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            folds = list(kf.split(X))

            xgb_r2, xgb_time = eval_xgboost(X, y, folds)
            print(f"  XGBoost:       R2={xgb_r2:.4f}  time={xgb_time:.2f}s")

            # d-dependent formula
            geo_d_r2, geo_d_time = eval_geo(X, y, blk_d, folds)
            delta_d = geo_d_r2 - xgb_r2
            print(f"  GeoXGB d-form: R2={geo_d_r2:.4f} ({delta_d:+.4f})  "
                  f"time={geo_d_time:.2f}s ({geo_d_time/max(xgb_time,0.001):.1f}x)")

            # sqrt(n)*15 formula
            geo_s_r2, geo_s_time = eval_geo(X, y, blk_sqrt, folds)
            delta_s = geo_s_r2 - xgb_r2
            print(f"  GeoXGB sqrt:   R2={geo_s_r2:.4f} ({delta_s:+.4f})  "
                  f"time={geo_s_time:.2f}s ({geo_s_time/max(xgb_time,0.001):.1f}x)")

            diff = geo_d_r2 - geo_s_r2
            winner = "d-form" if diff > 0.0005 else ("sqrt" if diff < -0.0005 else "tie")
            print(f"  >>> d-form vs sqrt: {diff:+.4f}  winner={winner}")

            results.append({
                "dataset": ds_name, "d": d, "n": n,
                "blk_d": blk_d, "blk_sqrt": blk_sqrt,
                "xgb_r2": round(xgb_r2, 5), "xgb_time": round(xgb_time, 3),
                "geo_d_r2": round(geo_d_r2, 5), "geo_d_time": round(geo_d_time, 3),
                "geo_s_r2": round(geo_s_r2, 5), "geo_s_time": round(geo_s_time, 3),
                "delta_d": round(delta_d, 5), "delta_s": round(delta_s, 5),
                "d_vs_sqrt": round(diff, 5), "winner": winner,
            })

            # Save incrementally
            pd.DataFrame(results).to_csv(
                "opus_study/results_d_formula_validation.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("opus_study/results_d_formula_validation.csv", index=False)

    # -- Summary --
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Per-dataset d-form vs sqrt(n)*15:")
    for ds_name in DATASETS:
        sub = df[df["dataset"] == ds_name]
        mean_diff = sub["d_vs_sqrt"].mean()
        mean_d_delta = sub["delta_d"].mean()
        mean_s_delta = sub["delta_s"].mean()
        d_speed = sub["geo_d_time"].mean()
        s_speed = sub["geo_s_time"].mean()
        print(f"    {ds_name:>12}: d-form avg delta={mean_d_delta:+.4f}  "
              f"sqrt avg delta={mean_s_delta:+.4f}  "
              f"d-form advantage={mean_diff:+.4f}  "
              f"speed: d={d_speed:.1f}s sqrt={s_speed:.1f}s")

    print(f"\n  Overall:")
    print(f"    d-form mean delta vs XGB: {df['delta_d'].mean():+.5f}")
    print(f"    sqrt   mean delta vs XGB: {df['delta_s'].mean():+.5f}")
    print(f"    d-form wins: {(df['winner'] == 'd-form').sum()}/{len(df)}")
    print(f"    sqrt   wins: {(df['winner'] == 'sqrt').sum()}/{len(df)}")
    print(f"    ties:        {(df['winner'] == 'tie').sum()}/{len(df)}")


if __name__ == "__main__":
    main()
