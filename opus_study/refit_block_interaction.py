"""
Block-size × Refit-interval interaction study.

The large-n sweep showed block coefficient barely affects accuracy at fixed
refit_interval=50. But effective data coverage depends on BOTH block_size and
refit_interval:

  unique_rows_seen ≈ min(n, (n_rounds / refit_interval) * block_size)

At refit_interval=50, 500 rounds → 10 block switches → high coverage.
At refit_interval=200, 500 rounds → 2-3 block switches → low coverage.

This study tests the interaction to determine if the auto formula should
account for refit_interval.

Test grid:
  - block coefficients: [10, 15, 25, 40]
  - refit_intervals: [25, 50, 100, 200]
  - n_sizes: [20_000, 50_000, 100_000]
  - datasets: friedman1, linreg_20d, reg_10d
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


# ── Block-size formula ────────────────────────────────────────────────────────

COEFFICIENTS = [10, 15, 25, 40]

def block_for(n, coeff):
    return max(2000, int(math.sqrt(n) * coeff))


# ── Datasets ─────────────────────────────────────────────────────────────────

def make_friedman1_ds(n):
    return make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)

def make_linreg(n):
    return make_regression(n_samples=n, n_features=20, n_informative=15,
                           noise=1.0, random_state=42)

def make_reg_10d(n):
    return make_regression(n_samples=n, n_features=10, n_informative=8,
                           noise=5.0, random_state=42)


DATASETS = {
    "friedman1":  make_friedman1_ds,
    "linreg_20d": make_linreg,
    "reg_10d":    make_reg_10d,
}

# ── Config ────────────────────────────────────────────────────────────────────

N_SIZES = [20_000, 50_000, 100_000]
REFIT_INTERVALS = [25, 50, 100, 200]
N_FOLDS = 3
N_ROUNDS = 500


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_geoxgb(X, y, block_size, refit_interval, folds):
    scores, times = [], []
    for tr, val in folds:
        m = GeoXGBRegressor(
            n_rounds=N_ROUNDS,
            learning_rate=0.02,
            max_depth=3,
            refit_interval=refit_interval,
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

            for ri in REFIT_INTERVALS:
                for coeff in COEFFICIENTS:
                    blk = block_for(n, coeff)
                    n_switches = N_ROUNDS // ri
                    coverage = min(1.0, (n_switches * blk) / n)

                    try:
                        geo_r2, geo_time = eval_geoxgb(X, y, blk, ri, folds)
                    except Exception as e:
                        print(f"  FAIL ri={ri} C={coeff}: {e}")
                        geo_r2, geo_time = float("nan"), float("nan")

                    delta = geo_r2 - xgb_r2
                    ratio = geo_time / max(xgb_time, 0.001)
                    print(
                        f"  ri={ri:>3}  C={coeff:>2}  blk={blk:>6}  "
                        f"cov={coverage:.0%}  "
                        f"R2={geo_r2:.4f} ({delta:+.4f})  "
                        f"time={geo_time:.2f}s ({ratio:.1f}x)"
                    )

                    results.append({
                        "dataset": ds_name,
                        "n": n,
                        "refit_interval": ri,
                        "coeff": coeff,
                        "block_size": blk,
                        "coverage": round(coverage, 3),
                        "r2": round(geo_r2, 5),
                        "time_s": round(geo_time, 3),
                        "xgb_r2": round(xgb_r2, 5),
                        "xgb_time_s": round(xgb_time, 3),
                        "r2_delta": round(delta, 5),
                        "speed_ratio": round(ratio, 2),
                    })

            # Save incrementally
            df = pd.DataFrame(results)
            df.to_csv("opus_study/results_refit_block_interaction.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("opus_study/results_refit_block_interaction.csv", index=False)
    print(f"\n\nResults saved to opus_study/results_refit_block_interaction.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  MEAN DELTA BY (refit_interval, coeff)")
    print("="*70)
    for ri in REFIT_INTERVALS:
        for coeff in COEFFICIENTS:
            sub = df[(df["refit_interval"] == ri) & (df["coeff"] == coeff)]
            if sub.empty:
                continue
            print(
                f"  ri={ri:>3}  C={coeff:>2}:  "
                f"mean_delta={sub['r2_delta'].mean():+.5f}  "
                f"mean_cov={sub['coverage'].mean():.2f}  "
                f"mean_speed={sub['speed_ratio'].mean():.2f}x  "
                f"wins={int((sub['r2_delta'] > 0).sum())}/{len(sub)}"
            )

    print("\n" + "="*70)
    print("  MEAN DELTA BY refit_interval (best coeff per dataset/n)")
    print("="*70)
    for ri in REFIT_INTERVALS:
        sub = df[df["refit_interval"] == ri]
        if sub.empty:
            continue
        best_per_combo = sub.groupby(["dataset", "n"])["r2_delta"].max()
        print(
            f"  ri={ri:>3}:  mean_best_delta={best_per_combo.mean():+.5f}  "
            f"wins={int((best_per_combo > 0).sum())}/{len(best_per_combo)}"
        )


if __name__ == "__main__":
    main()
