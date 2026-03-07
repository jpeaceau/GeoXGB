"""
n×d Grid Performance Benchmark
================================
Sweeps n in {500, 2000, 10000, 50000} and d in {5, 10, 20} (12 cells).

DGP: Friedman #1 with (d-5) i.i.d. Uniform noise features.
y = 10·sin(π·x1·x2) + 20·(x3-0.5)² + 10·x4 + 5·x5 + ε

Models:
  GeoXGB       — GeoXGBRegressor(n_rounds=1000, lr=0.02, depth=3, sample_block_n='auto')
  XGB-default  — XGBRegressor(n_estimators=1000)                 [XGB defaults: lr=0.1, depth=6]
  XGB-matched  — XGBRegressor(n_estimators=1000, lr=0.02, depth=3)  [capacity-matched]

Protocol: 3-fold CV, R², fit time per fold.

Output: two printed tables + CSV at benchmarks/results/nd_grid_benchmark.csv
"""
import sys
import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geoxgb import GeoXGBRegressor

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID_N = [500, 2_000, 10_000, 50_000]
GRID_D = [5, 10, 20]
N_ROUNDS = 1_000
N_SPLITS = 3
SEED = 42

# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def make_friedman_extended(n, d, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
        + rng.standard_normal(n)
    )
    return X, y

# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def run_kfold(model_fn, X, y, n_splits=N_SPLITS, seed=SEED):
    """Returns (mean_r2, std_r2, mean_fit_sec)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores, times = [], []
    for tr, val in kf.split(X):
        m = model_fn()
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        times.append(time.perf_counter() - t0)
        scores.append(r2_score(y[val], m.predict(X[val])))
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(times))

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def geo_fn():
    return GeoXGBRegressor(
        n_rounds=N_ROUNDS,
        learning_rate=0.02,
        max_depth=3,
        sample_block_n="auto",
        random_state=SEED,
    )

def xgbd_fn():
    return xgb.XGBRegressor(
        n_estimators=N_ROUNDS,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

def xgbm_fn():
    return xgb.XGBRegressor(
        n_estimators=N_ROUNDS,
        learning_rate=0.02,
        max_depth=3,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    rows = []
    total = len(GRID_N) * len(GRID_D)
    done = 0

    for n in GRID_N:
        for d in GRID_D:
            done += 1
            print(f"[{done}/{total}] n={n:>6}, d={d:>2} ... ", end="", flush=True)
            X, y = make_friedman_extended(n, d)

            r_geo,  s_geo,  t_geo  = run_kfold(geo_fn,  X, y)
            r_xgbd, s_xgbd, t_xgbd = run_kfold(xgbd_fn, X, y)
            r_xgbm, s_xgbm, t_xgbm = run_kfold(xgbm_fn, X, y)

            geo_vs_def = r_geo - r_xgbd
            geo_vs_mat = r_geo - r_xgbm
            spd_vs_def = t_geo / t_xgbd if t_xgbd > 0 else float("nan")
            spd_vs_mat = t_geo / t_xgbm if t_xgbm > 0 else float("nan")

            rows.append(dict(
                n=n, d=d,
                geo_r2=r_geo, geo_std=s_geo, geo_t=t_geo,
                xgbd_r2=r_xgbd, xgbd_std=s_xgbd, xgbd_t=t_xgbd,
                xgbm_r2=r_xgbm, xgbm_std=s_xgbm, xgbm_t=t_xgbm,
                geo_vs_def=geo_vs_def, geo_vs_mat=geo_vs_mat,
                spd_vs_def=spd_vs_def, spd_vs_mat=spd_vs_mat,
            ))
            print(
                f"GeoXGB={r_geo:.4f}±{s_geo:.4f}  "
                f"XGBdef={r_xgbd:.4f}  XGBmat={r_xgbm:.4f}  "
                f"t={t_geo:.2f}s/{t_xgbd:.2f}s/{t_xgbm:.2f}s"
            )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "nd_grid_benchmark.csv")
    fieldnames = [
        "n", "d",
        "geo_r2", "geo_std", "geo_t",
        "xgbd_r2", "xgbd_std", "xgbd_t",
        "xgbm_r2", "xgbm_std", "xgbm_t",
        "geo_vs_def", "geo_vs_mat",
        "spd_vs_def", "spd_vs_mat",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved: {csv_path}\n")

    # -----------------------------------------------------------------------
    # Table 1: R² scores
    # -----------------------------------------------------------------------
    hdr1 = (
        f"{'n':>6}  {'d':>2}  "
        f"{'GeoXGB':^16}  {'XGB-default':^16}  {'XGB-matched':^16}  "
        f"{'Geo-XGBdef':>10}  {'Geo-XGBmat':>10}"
    )
    sep1 = "-" * len(hdr1)
    print("Table 1 — R² scores (mean ± std over 3 folds)")
    print(sep1)
    print(hdr1)
    print(sep1)
    for r in rows:
        geo_str  = f"{r['geo_r2']:.4f}±{r['geo_std']:.4f}"
        xgbd_str = f"{r['xgbd_r2']:.4f}±{r['xgbd_std']:.4f}"
        xgbm_str = f"{r['xgbm_r2']:.4f}±{r['xgbm_std']:.4f}"
        delta_d  = f"{r['geo_vs_def']:+.4f}"
        delta_m  = f"{r['geo_vs_mat']:+.4f}"
        print(
            f"{r['n']:>6}  {r['d']:>2}  "
            f"{geo_str:^16}  {xgbd_str:^16}  {xgbm_str:^16}  "
            f"{delta_d:>10}  {delta_m:>10}"
        )
    print(sep1)

    # -----------------------------------------------------------------------
    # Table 2: fit time per fold
    # -----------------------------------------------------------------------
    hdr2 = (
        f"{'n':>6}  {'d':>2}  "
        f"{'GeoXGB':>10}  {'XGB-default':>11}  {'XGB-matched':>11}  "
        f"{'Geo/XGBdef':>10}  {'Geo/XGBmat':>10}"
    )
    sep2 = "-" * len(hdr2)
    print("\nTable 2 — fit time per fold (seconds, mean over 3 folds)")
    print(sep2)
    print(hdr2)
    print(sep2)
    for r in rows:
        print(
            f"{r['n']:>6}  {r['d']:>2}  "
            f"{r['geo_t']:>9.2f}s  {r['xgbd_t']:>10.2f}s  {r['xgbm_t']:>10.2f}s  "
            f"{r['spd_vs_def']:>9.2f}x  {r['spd_vs_mat']:>9.2f}x"
        )
    print(sep2)


if __name__ == "__main__":
    main()
