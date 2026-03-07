"""
Scalability Tests A, B, C on Kaggle Churn (594k, 19 features).

Test A: Lazy refit — skip do_resample when gradient magnitude hasn't changed.
Test B: Fixed geometry — fit HVRT once, freeze partition tree, only re-reduce/expand.
Test C: Hierarchical reduce + progressive expand (expand grows from 0 to expand_ratio).

All tests use PCA(19), y_weight=0, best-known params from HPO4/PCA HPO.
Evaluated on full 475k train → 119k test.
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = (tr["Churn"] == "Yes").astype(int).values
    X = tr.drop(columns=["id", "Churn"])
    cat_cols = X.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X.values.astype(np.float64), y


def run_test(X_train, y_train, X_test, y_test, label, extra_params=None):
    """Train a single config and return test AUC + timing."""
    # Base params: HPO4 best
    params = dict(
        n_rounds=5000,
        learning_rate=0.015,
        max_depth=6,
        min_samples_leaf=6,
        reduce_ratio=0.59,
        refit_interval=500,
        expand_ratio=0.64,
        y_weight=0.0,
        hvrt_min_samples_leaf=50,
        noise_guard=False,
        auto_noise=False,
        auto_expand=False,
        generation_strategy="laplace",
        reduce_method="orthant_stratified",
        variance_weighted=True,
        partitioner="hvrt",
        n_bins=128,
        convergence_tol=0.01,
        random_state=42,
        sample_block_n=-1,  # disable block cycling for clean comparison
    )
    if extra_params:
        params.update(extra_params)

    cfg = make_cpp_config(**params)
    m = CppGeoXGBClassifier(cfg)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_train, y_train.astype(np.float64))
    fit_time = time.perf_counter() - t0

    proba = m.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    cr = m.convergence_round()
    cr_str = str(cr) if cr >= 0 else "full"

    print(f"  {label:40s}  AUC={test_auc:.5f}  conv={cr_str:>5s}  {fit_time:6.1f}s")
    return test_auc, fit_time, cr


if __name__ == "__main__":
    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # PCA rotation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=19, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"\nAll tests: PCA(19), y_weight=0, HPO4 best params")
    print(f"{'='*80}")

    # ── Baseline ──────────────────────────────────────────────────────────
    print(f"\n--- Baseline (standard refit) ---")
    run_test(X_train_pca, y_train, X_test_pca, y_test,
             "Baseline (normal refit)")

    # ── Test A: Lazy refit ────────────────────────────────────────────────
    print(f"\n--- Test A: Lazy refit (skip when gradients stable) ---")
    for tol in [0.01, 0.05, 0.10, 0.20, 0.50]:
        run_test(X_train_pca, y_train, X_test_pca, y_test,
                 f"lazy_refit_tol={tol}",
                 {"lazy_refit_tol": tol})

    # ── Test B: Fixed geometry ────────────────────────────────────────────
    print(f"\n--- Test B: Fixed geometry (HVRT fit once, frozen partitions) ---")
    run_test(X_train_pca, y_train, X_test_pca, y_test,
             "fixed_geometry=True",
             {"fixed_geometry": True})

    # Fixed geometry + different refit intervals
    for ri in [50, 100, 200, 500]:
        run_test(X_train_pca, y_train, X_test_pca, y_test,
                 f"fixed_geom + ri={ri}",
                 {"fixed_geometry": True, "refit_interval": ri})

    # ── Test C: Progressive expand ────────────────────────────────────────
    print(f"\n--- Test C: Progressive expand (expand_ratio grows over training) ---")
    run_test(X_train_pca, y_train, X_test_pca, y_test,
             "progressive_expand=True",
             {"progressive_expand": True})

    # Progressive expand + higher expand_ratio (since early rounds get less)
    for er in [0.64, 0.80, 1.0, 1.5]:
        run_test(X_train_pca, y_train, X_test_pca, y_test,
                 f"progressive + er={er}",
                 {"progressive_expand": True, "expand_ratio": er})

    # ── Combined: B + C ──────────────────────────────────────────────────
    print(f"\n--- Combined: Fixed geometry + Progressive expand ---")
    for er in [0.64, 1.0]:
        for ri in [100, 300, 500]:
            run_test(X_train_pca, y_train, X_test_pca, y_test,
                     f"fixed+prog er={er} ri={ri}",
                     {"fixed_geometry": True, "progressive_expand": True,
                      "expand_ratio": er, "refit_interval": ri})

    # ── Combined: A + B ──────────────────────────────────────────────────
    print(f"\n--- Combined: Lazy refit + Fixed geometry ---")
    for tol in [0.05, 0.10, 0.20]:
        run_test(X_train_pca, y_train, X_test_pca, y_test,
                 f"lazy={tol} + fixed_geom",
                 {"lazy_refit_tol": tol, "fixed_geometry": True})

    # ── No refit at all (for reference) ───────────────────────────────────
    print(f"\n--- Reference: No refit (refit_interval=0) ---")
    run_test(X_train_pca, y_train, X_test_pca, y_test,
             "refit_interval=0 (no refit)",
             {"refit_interval": 0})

    # XGBoost reference
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train_pca, y_train)
    xgb_time = time.perf_counter() - t0
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_pca)[:, 1])
    print(f"\n  {'XGBoost+PCA':40s}  AUC={xgb_auc:.5f}  {'':>11s}  {xgb_time:6.1f}s")
