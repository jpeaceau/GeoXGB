"""
XGBoost optimised comparison vs stored GeoXGB Phase 2 results.
Run: python xgb_comparison.py
XGBoost uses n_jobs=-1 internally; runs sequentially across folds.
"""
import warnings; warnings.filterwarnings("ignore")
import csv, numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def main():
    import xgboost as xgb
    from meta_reg import _make_datasets
    datasets = _make_datasets()
    N_SEEDS, N_FOLDS = 10, 4

    raw = []
    total = N_SEEDS * N_FOLDS * len(datasets)
    done = 0
    for ds_name, (X, y, sigma, r2_ceil) in datasets.items():
        for seed in range(N_SEEDS):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr, val) in enumerate(kf.split(X)):
                X_tr, y_tr = X[tr], y[tr]
                X_val, y_val = X[val], y[val]
                out = {}
                for name, lr, depth in [
                    ("XGB_tuned",   0.02, 2),
                    ("XGB_default", 0.20, 3),
                ]:
                    m = xgb.XGBRegressor(
                        n_estimators=3000, learning_rate=lr, max_depth=depth,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=-1, verbosity=0,
                    )
                    m.fit(X_tr, y_tr)
                    r2 = r2_score(y_val, m.predict(X_val))
                    out[name] = r2 / max(r2_ceil, 1e-6)
                raw.append((ds_name, out))
                done += 1
                if done % 16 == 0:
                    print(f"  {done}/{total}", flush=True)

    xgb_res = defaultdict(list)
    for ds, out in raw:
        for model, v in out.items():
            if not np.isnan(v):
                xgb_res[(model, ds)].append(v)

    # Read GeoXGB lr=0.02 from stored Phase 2 CSV
    geo_res = defaultdict(list)
    with open("results/phase2_oat_primary.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and r["param"] == "learning_rate" and r["value"] == "0.02":
                try:
                    v = float(r["val_r2_adj"])
                    if not np.isnan(v):
                        geo_res[("GeoXGB_opt", r["dataset"])].append(v)
                except Exception:
                    pass

    DATASETS = ["friedman1", "friedman2", "reg_sparse", "reg_large"]
    MODELS   = [("GeoXGB_opt", geo_res), ("XGB_tuned", xgb_res), ("XGB_default", xgb_res)]

    print()
    print(f"  {'Model':<16}  {'friedman1':>14}  {'friedman2':>14}  {'reg_sparse':>14}  {'reg_large':>14}  {'MEAN':>8}")
    print("  " + "-" * 86)
    for model, src in MODELS:
        row = f"  {model:<16}"
        means = []
        for ds in DATASETS:
            s = src[(model, ds)]
            m, std = np.mean(s), np.std(s)
            means.append(m)
            row += f"  {m:.4f}±{std:.4f}"
        row += f"  {np.nanmean(means):.4f}"
        print(row)

    print()
    print("  GeoXGB_opt  : lr=0.02, depth=3 baseline (Phase 2 OAT; depth OAT separately found depth=2 best)")
    print("  XGB_tuned   : lr=0.02, depth=2, 3000 rounds, subsample=0.8, colsample=0.8")
    print("  XGB_default : lr=0.20, depth=3, 3000 rounds, subsample=0.8, colsample=0.8")
    print()
    print("  Deltas (GeoXGB_opt vs XGB_tuned):")
    for ds in DATASETS:
        g = np.mean(geo_res[("GeoXGB_opt", ds)])
        x = np.mean(xgb_res[("XGB_tuned", ds)])
        winner = "GeoXGB" if g > x else "XGBoost"
        print(f"    {ds:<12}  GeoXGB={g:.4f}  XGB={x:.4f}  diff={g-x:+.4f}  winner={winner}")


if __name__ == "__main__":
    main()
