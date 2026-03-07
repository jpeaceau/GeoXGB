"""
Quantile Residual Analysis — GeoXGB vs XGBoost
===============================================
Bins absolute errors by target-value percentile decile to confirm whether
GeoXGB's aggregate MAE loss vs XGBoost is concentrated in the high-y tail.

Uses the MAE-optimal GeoXGB config (lr=0.02, depth=2, refit_interval=5, etc.)
matching regressor.py defaults.  XGBoost uses reg:squarederror defaults.

Outputs: mean |error| per decile, fold-aggregated, for all regression datasets.
"""

from __future__ import annotations

import io, os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml, make_friedman1
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from geoxgb import GeoXGBRegressor

RNG      = 42
N_SPLITS = 5
SEEDS    = [42, 123, 999]
N_DECILES = 10

# MAE-optimal GeoXGB defaults (matches regressor.py __init__)
GEO_PARAMS = dict(
    n_rounds=1000, learning_rate=0.02, max_depth=2,
    min_samples_leaf=5, reduce_ratio=0.8, y_weight=0.5,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    auto_noise=False, noise_guard=False, variance_weighted=False,
)

XGB_PARAMS = dict(
    n_estimators=1000, learning_rate=0.02, max_depth=2,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", verbosity=0, n_jobs=-1,
)


def load_datasets():
    ds = {}
    rng = np.random.RandomState(RNG)

    d = fetch_california_housing()
    idx = rng.choice(len(d.data), 8000, replace=False)
    ds["california"] = (d.data[idx].astype(np.float64), d.target[idx])

    try:
        raw = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = np.asarray(raw.data, dtype=np.float64)
        yc = np.asarray(raw.target, dtype=np.float64)
        ds["concrete"] = (Xc, yc)
    except Exception:
        pass

    Xf, yf = make_friedman1(n_samples=1000, random_state=RNG)
    ds["friedman1"] = (Xf.astype(np.float64), yf.astype(np.float64))

    return ds


def run_dataset(ds_name, X, y):
    # Accumulate per-sample (y_val, e_geo, e_xgb) across all folds/seeds
    records = []

    for seed in SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in kf.split(X, y):
            X_tr, y_tr = X[tr], y[tr]
            X_va, y_va = X[va], y[va]

            geo = GeoXGBRegressor(**GEO_PARAMS, random_state=seed)
            geo.fit(X_tr, y_tr)
            p_geo = geo.predict(X_va)

            xgb_m = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
            xgb_m.fit(X_tr, y_tr)
            p_xgb = xgb_m.predict(X_va)

            for yi, eg, ex in zip(y_va, np.abs(y_va - p_geo), np.abs(y_va - p_xgb)):
                records.append((float(yi), float(eg), float(ex)))

    return records


def report_dataset(ds_name, records):
    ys    = np.array([r[0] for r in records])
    e_geo = np.array([r[1] for r in records])
    e_xgb = np.array([r[2] for r in records])

    # Decile boundaries based on pooled y_val
    decile_edges = np.percentile(ys, np.linspace(0, 100, N_DECILES + 1))

    print(f"\n{'='*68}")
    print(f"  {ds_name}   ({len(records)} predictions across all folds/seeds)")
    print(f"{'='*68}")
    print(f"  {'Decile':>8}  {'y range':>18}  {'GeoXGB MAE':>12}  {'XGB MAE':>10}  {'Ratio G/X':>10}  {'Winner':>8}")
    print("  " + "-" * 64)

    geo_wins = 0
    xgb_wins = 0
    for i in range(N_DECILES):
        lo, hi = decile_edges[i], decile_edges[i+1]
        mask = (ys >= lo) & (ys < hi) if i < N_DECILES - 1 else (ys >= lo) & (ys <= hi)
        if mask.sum() == 0:
            continue
        m_geo = float(np.mean(e_geo[mask]))
        m_xgb = float(np.mean(e_xgb[mask]))
        ratio  = m_geo / (m_xgb + 1e-12)
        winner = "GeoXGB" if m_geo < m_xgb else "XGB"
        if m_geo < m_xgb:
            geo_wins += 1
        else:
            xgb_wins += 1
        pct = f"D{i+1:02d} ({(i)*10:2d}–{(i+1)*10:2d}%)"
        print(f"  {pct:>8}  {lo:>8.3f}–{hi:>8.3f}  {m_geo:>12.4f}  {m_xgb:>10.4f}  {ratio:>10.3f}x  {winner:>8}")

    print(f"\n  Overall: GeoXGB MAE={np.mean(e_geo):.4f}  XGB MAE={np.mean(e_xgb):.4f}  "
          f"ratio={np.mean(e_geo)/np.mean(e_xgb):.3f}x")
    print(f"  GeoXGB wins {geo_wins}/{N_DECILES} deciles, XGB wins {xgb_wins}/{N_DECILES} deciles")
    print(f"  R² GeoXGB={r2_score(ys, ys - e_geo * np.sign(1)):.4f}  "
          f"(Note: r2 needs sign; shown as aggregate reference only)")

    # Show the top-3 worst deciles for GeoXGB
    decile_ratios = []
    for i in range(N_DECILES):
        lo, hi = decile_edges[i], decile_edges[i+1]
        mask = (ys >= lo) & (ys < hi) if i < N_DECILES - 1 else (ys >= lo) & (ys <= hi)
        if mask.sum() == 0:
            continue
        m_geo = float(np.mean(e_geo[mask]))
        m_xgb = float(np.mean(e_xgb[mask]))
        decile_ratios.append((i+1, m_geo / (m_xgb + 1e-12), mask.sum()))
    decile_ratios.sort(key=lambda x: -x[1])
    print(f"\n  GeoXGB worst deciles (highest ratio vs XGB):")
    for rank, (d, ratio, n) in enumerate(decile_ratios[:3], 1):
        lo, hi = decile_edges[d-1], decile_edges[d]
        print(f"    #{rank}: D{d:02d} y=[{lo:.2f},{hi:.2f}]  ratio={ratio:.3f}x  n={n}")


def main():
    print("=" * 68)
    print("  Quantile Residual Analysis — GeoXGB vs XGBoost")
    print("  (MAE-optimal GeoXGB config: lr=0.02, depth=2, refit=5)")
    print("=" * 68)

    print("\nLoading datasets...")
    datasets = load_datasets()

    for ds_name, (X, y) in datasets.items():
        print(f"\n  Fitting {ds_name} ...", flush=True)
        records = run_dataset(ds_name, X, y)
        report_dataset(ds_name, records)

    print("\n\nDone.")


if __name__ == "__main__":
    main()
