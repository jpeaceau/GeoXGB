"""
GeoXGB Y-Coupling Strategies Benchmark
=======================================
Evaluates three orthogonal strategies for coupling the synthetic-Y vector
with geometric and residual structure in GeoXGB's HVRT resampling layer.

Strategy 1 (blend_cross_term): adds x_z*y_comp interaction term to blend_target,
  concentrating HVRT partitions on regions where cooperation AND y-extremality
  co-occur simultaneously.

Strategy 2 (syn_partition_correct): after knn_assign_y, shifts each synthetic
  y_syn[i] by delta_p = (mean_real_in_p − mean_syn_in_p), removing the
  within-partition mean bias introduced by IDW interpolation across boundaries.

Strategy 3 (y_geom_coupling=α): at every HVRT refit, passes
  y_for_refit = (1-α)*z(residuals) + α*geom_target (rescaled to residual range),
  so the partition tree always sees some fixed geometric structure even when
  residuals are noisy.

Datasets (4 representative):
  california_housing  — regression, n=8000, d=8   (XGBoost wins comfortably)
  concrete_compressive— regression, n=1030, d=8   (XGBoost wins)
  breast_cancer       — classification, n=569, d=30 (GeoXGB wins baseline)
  ionosphere          — classification, n=351, d=34 (GeoXGB wins baseline)

5-fold CV, 3 seeds.  Reports metric ± std and delta vs baseline for each variant.
"""
import sys, io, warnings, time
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from geoxgb._cpp_backend import CppGeoXGBRegressor, CppGeoXGBClassifier, make_cpp_config

RNG = 42
N_SPLITS = 5
SEEDS = [42, 123, 999]

# ── Baseline params (matching realworld_compare.py) ──────────────────────────
BASE_REG = dict(
    n_rounds=500, learning_rate=0.1, max_depth=3,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64,
)
BASE_CLF = dict(
    n_rounds=500, learning_rate=0.1, max_depth=5,
    min_samples_leaf=5, reduce_ratio=0.7, y_weight=0.2,
    refit_interval=5, auto_expand=True, expand_ratio=0.1,
    min_train_samples=100, n_bins=64,
)

# ── Variants to test ─────────────────────────────────────────────────────────
# Each entry: (label, extra_kwargs)
VARIANTS = [
    ("baseline",       {}),
    ("S1_cross",       dict(blend_cross_term=True)),
    ("S2_part_corr",   dict(syn_partition_correct=True)),
    ("S3_geom_0.1",    dict(y_geom_coupling=0.1)),
    ("S3_geom_0.3",    dict(y_geom_coupling=0.3)),
    ("S3_geom_0.5",    dict(y_geom_coupling=0.5)),
    ("all3_geom_0.3",  dict(blend_cross_term=True,
                            syn_partition_correct=True,
                            y_geom_coupling=0.3)),
]


def clean_X(df):
    if not isinstance(df, pd.DataFrame):
        arr = np.asarray(df, dtype=np.float64)
        if np.isnan(arr).any():
            arr = SimpleImputer(strategy="median").fit_transform(arr)
        return arr
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if num_cols and df[num_cols].isnull().any().any():
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols])
    return df.values.astype(np.float64)


def load_datasets():
    datasets = {}
    print("Loading datasets...")

    print("  california_housing...", end=" ", flush=True)
    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.RandomState(RNG)
    idx = rng.choice(len(X), 8000, replace=False)
    datasets["california_housing"] = ("reg", X[idx].astype(np.float64), y[idx])
    print(f"n={len(idx)}, d={X.shape[1]}")

    print("  concrete_compressive...", end=" ", flush=True)
    try:
        data = fetch_openml("concrete_compressive_strength", as_frame=True, parser="auto")
        Xc = clean_X(data.data)
        yc = np.asarray(data.target, dtype=np.float64)
        datasets["concrete_compressive"] = ("reg", Xc, yc)
        print(f"n={len(yc)}, d={Xc.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    print("  breast_cancer...", end=" ", flush=True)
    Xb, yb = load_breast_cancer(return_X_y=True)
    datasets["breast_cancer"] = ("clf", Xb.astype(np.float64), yb.astype(np.float64))
    print(f"n={len(yb)}, d={Xb.shape[1]}")

    print("  ionosphere...", end=" ", flush=True)
    try:
        data = fetch_openml(data_id=59, as_frame=True, parser="auto")
        Xi = clean_X(data.data)
        yi = (data.target == "g").astype(np.float64).values
        datasets["ionosphere"] = ("clf", Xi, yi)
        print(f"n={len(yi)}, d={Xi.shape[1]}")
    except Exception as e:
        print(f"SKIP ({e})")

    return datasets


def cv_score(task, X, y, extra_kwargs, seed):
    """5-fold CV; returns list of per-fold scores."""
    base = BASE_REG if task == "reg" else BASE_CLF
    params = {**base, "random_state": seed, **extra_kwargs}
    cfg = make_cpp_config(**params)

    kf_cls = KFold if task == "reg" else StratifiedKFold
    kf = kf_cls(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y_int = y.astype(int) if task == "clf" else y

    scores = []
    for tr, va in kf.split(X, y_int):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        if task == "reg":
            model = CppGeoXGBRegressor(cfg)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            scores.append(r2_score(yva, pred))
        else:
            model = CppGeoXGBClassifier(cfg)
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xva)[:, 1]
            scores.append(roc_auc_score(yva, prob))
    return scores


def run_variant(name, extra_kwargs, datasets):
    """Run all seeds × folds for one variant across all datasets."""
    results = {}
    for ds_name, (task, X, y) in datasets.items():
        all_scores = []
        for seed in SEEDS:
            all_scores.extend(cv_score(task, X, y, extra_kwargs, seed))
        results[ds_name] = np.array(all_scores)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  GeoXGB Y-Coupling Strategies Benchmark")
    print("=" * 72)
    print(f"  Variants: {len(VARIANTS)}  |  Datasets: 4  |  "
          f"Seeds: {len(SEEDS)}  |  Folds: {N_SPLITS}")
    print()

    datasets = load_datasets()
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    metric_map = {
        "california_housing":   "R2",
        "concrete_compressive": "R2",
        "breast_cancer":        "AUC",
        "ionosphere":           "AUC",
    }

    # ── Run all variants ─────────────────────────────────────────────────────
    all_results = {}
    for var_name, extra_kwargs in VARIANTS:
        t0 = time.perf_counter()
        print(f"  Running {var_name}...", end=" ", flush=True)
        res = run_variant(var_name, extra_kwargs, datasets)
        dt = time.perf_counter() - t0
        all_results[var_name] = res
        print(f"done  ({dt:.0f}s)")

    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    baseline_res = all_results["baseline"]
    col_w = 14
    header = f"{'Dataset':<26} {'M':>3}"
    for var_name, _ in VARIANTS:
        label = var_name[:col_w]
        header += f"  {label:>{col_w}}"
    print(header)
    print("-" * (30 + len(VARIANTS) * (col_w + 2)))

    # Per-dataset rows
    for ds_name, (task, X, y) in datasets.items():
        metric = metric_map.get(ds_name, "?")
        base_mean = baseline_res[ds_name].mean()
        row = f"  {ds_name:<24} {metric:>3}"
        for var_name, _ in VARIANTS:
            s = all_results[var_name][ds_name]
            delta = s.mean() - base_mean
            row += f"  {s.mean():+.4f}({delta:+.4f})"
        print(row)

    print()
    print("  Format: mean(Δ_vs_baseline)")
    print()

    # ── Delta-only summary ───────────────────────────────────────────────────
    print("  Delta vs baseline (positive = better)")
    print(f"  {'Variant':<22}  {'mean_Δ':>8}  {'reg_Δ':>8}  {'clf_Δ':>8}  "
          f"{'n_better':>8}")
    print("  " + "-" * 60)

    reg_ds  = [k for k, (t, _, _) in datasets.items() if t == "reg"]
    clf_ds  = [k for k, (t, _, _) in datasets.items() if t == "clf"]

    for var_name, _ in VARIANTS:
        if var_name == "baseline":
            continue
        res = all_results[var_name]
        deltas = []
        for ds_name in datasets:
            delta = res[ds_name].mean() - baseline_res[ds_name].mean()
            deltas.append(delta)

        reg_deltas = [res[d].mean() - baseline_res[d].mean() for d in reg_ds if d in res]
        clf_deltas = [res[d].mean() - baseline_res[d].mean() for d in clf_ds if d in res]
        n_better = sum(1 for d in deltas if d > 0)

        mean_reg = np.mean(reg_deltas) if reg_deltas else float("nan")
        mean_clf = np.mean(clf_deltas) if clf_deltas else float("nan")
        print(f"  {var_name:<22}  {np.mean(deltas):>+.4f}    {mean_reg:>+.4f}    "
              f"{mean_clf:>+.4f}    {n_better}/{len(deltas)}")

    print()

    # ── Recommendation ───────────────────────────────────────────────────────
    best_mean = -1e9
    best_var = None
    for var_name, _ in VARIANTS:
        if var_name == "baseline":
            continue
        res = all_results[var_name]
        m = np.mean([res[d].mean() - baseline_res[d].mean() for d in datasets])
        if m > best_mean:
            best_mean = m
            best_var = var_name

    if best_mean > 0:
        print(f"  Best variant: {best_var} (mean Δ = {best_mean:+.4f})")
        print("  → Promote this strategy as the new default.")
    else:
        print(f"  Best variant: {best_var} (mean Δ = {best_mean:+.4f})")
        print("  → No variant improves mean score. Coupling strategies need revision.")
    print()


if __name__ == "__main__":
    main()
