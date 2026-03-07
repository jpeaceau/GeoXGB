"""
Scalability Test D: fast_refit on Kaggle Churn.

fast_refit = True:
  - Reduce: random (stratified) selection within partitions — O(n)
  - Expand: skipped entirely — no KDE, no knn_assign_y
  - Expander prep skipped in HVRT fit/refit
  - HVRT tree refit still runs (geometry "memory reset")

Tests at multiple dataset sizes to show scaling behavior.
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
    params = dict(
        n_rounds=2000,
        learning_rate=0.015,
        max_depth=6,
        min_samples_leaf=6,
        reduce_ratio=0.59,
        refit_interval=200,
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
        sample_block_n=-1,
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

    print(f"  {label:55s}  AUC={test_auc:.5f}  conv={cr_str:>5s}  {fit_time:6.1f}s")
    return test_auc, fit_time, cr


if __name__ == "__main__":
    X, y = load_data()
    print(f"Full data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # PCA rotation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=19, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # Test at increasing data sizes
    for n_sub in [10000, 25000, 50000, 100000]:
        if n_sub > len(X_train_pca):
            continue
        idx = np.random.RandomState(42).choice(len(X_train_pca), n_sub, replace=False)
        Xt = X_train_pca[idx]
        yt = y_train_full[idx]

        print(f"\n{'='*90}")
        print(f"n_train = {n_sub}")
        print(f"{'='*90}")

        # Normal (full reduce + expand pipeline)
        run_test(Xt, yt, X_test_pca, y_test,
                 f"n={n_sub} normal (full reduce+expand)")

        # fast_refit (random reduce, no expand)
        run_test(Xt, yt, X_test_pca, y_test,
                 f"n={n_sub} fast_refit",
                 {"fast_refit": True})

        # fast_refit + fixed_geometry
        run_test(Xt, yt, X_test_pca, y_test,
                 f"n={n_sub} fast_refit + fixed_geometry",
                 {"fast_refit": True, "fixed_geometry": True})

        # block cycling baseline (for n >= 20k)
        if n_sub >= 20000:
            blk = min(10000, n_sub // 3)
            run_test(Xt, yt, X_test_pca, y_test,
                     f"n={n_sub} block_cycling blk={blk}",
                     {"sample_block_n": blk})

    # XGBoost reference on full train
    print(f"\n{'='*90}")
    print(f"XGBoost reference (full train n={len(X_train_pca)})")
    print(f"{'='*90}")
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train_pca, y_train_full)
    xgb_time = time.perf_counter() - t0
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_pca)[:, 1])
    print(f"  {'XGBoost+PCA (full 475k)':55s}  AUC={xgb_auc:.5f}  {'':>11s}  {xgb_time:6.1f}s")
