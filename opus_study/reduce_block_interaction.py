"""
Study 2: Interaction between block_size and reduce_ratio.

The block sweep showed:
  - linreg_20d: GeoXGB beats XGBoost at ALL sizes (even 50k), smallest blocks best
  - friedman1:  GeoXGB falls behind as n grows, larger blocks help
  - california: GeoXGB always behind, gap grows with n

Hypothesis: the accuracy gap comes from each tree seeing too few samples.
Effective samples per tree = block_size * reduce_ratio * (1 - noise_reduction).
If we increase reduce_ratio at large n, each tree sees more real data.

This study sweeps (block_size, reduce_ratio) jointly at n=10k, 20k, 50k on
friedman1 and california (the two where GeoXGB trails).

Also tests expand_ratio=0 to see if synthetic samples hurt at large n.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1, fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb

from geoxgb import GeoXGBRegressor


# ── Datasets ─────────────────────────────────────────────────────────────────

def make_friedman(n):
    return make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=42)

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
    "friedman1":  make_friedman,
    "california": make_california,
}

# ── Config grid ──────────────────────────────────────────────────────────────

N_SIZES = [10_000, 20_000, 50_000]
N_FOLDS = 3
N_ROUNDS = 500

# Block size as fraction of n
BLOCK_FRACS = {
    "current": lambda n: 500 + (n - 5000) // 50 if n > 5000 else None,
    "n/10":    lambda n: max(1000, n // 10) if n > 5000 else None,
    "n/5":     lambda n: max(1000, n // 5) if n > 5000 else None,
    "n/3":     lambda n: max(1000, n // 3) if n > 5000 else None,
    "full":    lambda n: None,
}

REDUCE_RATIOS = [0.7, 0.8, 0.9, 0.95, 1.0]
EXPAND_RATIOS = [0.0, 0.1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_geoxgb(X, y, block_size, reduce_ratio, expand_ratio, folds):
    scores, times = [], []
    for tr, val in folds:
        m = GeoXGBRegressor(
            n_rounds=N_ROUNDS,
            learning_rate=0.02,
            max_depth=3,
            refit_interval=50,
            partitioner="pyramid_hart",
            method="variance_ordered",
            sample_block_n=block_size,
            reduce_ratio=reduce_ratio,
            expand_ratio=expand_ratio,
            random_state=42,
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

            for bname, bfn in BLOCK_FRACS.items():
                blk = bfn(n)
                for rr in REDUCE_RATIOS:
                    for er in EXPAND_RATIOS:
                        blk_label = f"{blk}" if blk is not None else "full"

                        try:
                            geo_r2, geo_time = eval_geoxgb(X, y, blk, rr, er, folds)
                        except Exception as e:
                            print(f"  FAIL blk={blk_label} rr={rr} er={er}: {e}")
                            geo_r2, geo_time = float("nan"), float("nan")

                        delta = geo_r2 - xgb_r2
                        eff_samples = (blk if blk else n) * rr
                        print(
                            f"  blk={blk_label:>6s} rr={rr:.2f} er={er:.1f}  "
                            f"eff~{eff_samples:>6.0f}  "
                            f"R2={geo_r2:.4f} ({delta:+.4f})  "
                            f"t={geo_time:.2f}s"
                        )

                        results.append({
                            "dataset": ds_name,
                            "n": n,
                            "block_formula": bname,
                            "block_size": blk,
                            "reduce_ratio": rr,
                            "expand_ratio": er,
                            "eff_samples_per_tree": eff_samples,
                            "r2": round(geo_r2, 5),
                            "time_s": round(geo_time, 3),
                            "xgb_r2": round(xgb_r2, 5),
                            "xgb_time_s": round(xgb_time, 3),
                            "r2_delta": round(delta, 5),
                        })

    df = pd.DataFrame(results)
    out_path = "opus_study/results_reduce_block.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # ── Key analysis ──────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ANALYSIS: Best config per (dataset, n)")
    print("="*70)
    for (ds, n_val), grp in df.groupby(["dataset", "n"]):
        best = grp.loc[grp["r2"].idxmax()]
        print(
            f"  {ds:15s} n={n_val:>6,}  "
            f"blk={best['block_formula']:>8s} rr={best['reduce_ratio']:.2f} er={best['expand_ratio']:.1f}  "
            f"eff~{best['eff_samples_per_tree']:.0f}  "
            f"R2={best['r2']:.4f} vs XGB={best['xgb_r2']:.4f} ({best['r2_delta']:+.4f})  "
            f"t={best['time_s']:.2f}s"
        )

    # ── Effect of reduce_ratio ────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  EFFECT OF reduce_ratio (averaged across block sizes, er=0.1)")
    print("="*70)
    sub = df[df["expand_ratio"] == 0.1]
    for (ds, n_val), grp in sub.groupby(["dataset", "n"]):
        print(f"  {ds} n={n_val:,}:")
        for rr in REDUCE_RATIOS:
            rr_grp = grp[grp["reduce_ratio"] == rr]
            if rr_grp.empty:
                continue
            print(
                f"    rr={rr:.2f}  mean_R2={rr_grp['r2'].mean():.4f}  "
                f"mean_delta={rr_grp['r2_delta'].mean():+.4f}  "
                f"best_R2={rr_grp['r2'].max():.4f}"
            )

    # ── Effect of expand_ratio ────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  EFFECT OF expand_ratio (averaged across block sizes and reduce_ratios)")
    print("="*70)
    for (ds, n_val), grp in df.groupby(["dataset", "n"]):
        for er in EXPAND_RATIOS:
            er_grp = grp[grp["expand_ratio"] == er]
            if er_grp.empty:
                continue
            print(
                f"  {ds:15s} n={n_val:>6,}  er={er:.1f}  "
                f"mean_R2={er_grp['r2'].mean():.4f}  "
                f"mean_delta={er_grp['r2_delta'].mean():+.4f}"
            )


if __name__ == "__main__":
    main()
