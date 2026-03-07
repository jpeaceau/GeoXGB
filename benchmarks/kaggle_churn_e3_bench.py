"""
Benchmark: blended T + λ·e₃ HVRT target on Kaggle Churn dataset.

Tests whether higher-order geometry (degree-3 interactions via e₃) improves
classification on the deep-learning-encoded churn data.

Uses best HPO params from kaggle_churn_hpo3.py as baseline, sweeps λ values.
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

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


if __name__ == "__main__":
    from geoxgb import GeoXGBClassifier

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    # Hold out 20% for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 10k subsample for CV speed
    MAX_SAMPLES = 10_000
    X_hpo, _, y_hpo, _ = train_test_split(
        X_train, y_train, train_size=MAX_SAMPLES,
        stratify=y_train, random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_hpo, y_hpo))

    # Best params from HPO round 3 (500 trials)
    best_params = dict(
        n_rounds=10000,
        learning_rate=0.046,
        max_depth=7,
        min_samples_leaf=20,
        reduce_ratio=0.54,
        refit_interval=300,
        expand_ratio=0.475,
        y_weight=0.31,
        hvrt_min_samples_leaf=30,
        noise_guard=False,
        auto_noise=False,
        auto_expand=True,
        generation_strategy="laplace",
        method="orthant_stratified",
        variance_weighted=True,
        convergence_tol=0.01,
        random_state=42,
        sample_block_n="auto",
    )

    # Lambda sweep
    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"\nSweeping e3_target_lambda on {MAX_SAMPLES} subsample, 3-fold CV")
    print(f"Base params: lr={best_params['learning_rate']}, depth={best_params['max_depth']}, "
          f"ri={best_params['refit_interval']}, nr={best_params['n_rounds']}")
    print("=" * 70)

    results = []
    for lam in lambdas:
        scores = []
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, va_idx in splits:
                m = GeoXGBClassifier(**best_params, e3_target_lambda=lam)
                m.fit(X_hpo[tr_idx], y_hpo[tr_idx].astype(float))
                proba = m.predict_proba(X_hpo[va_idx])[:, 1]
                scores.append(roc_auc_score(y_hpo[va_idx], proba))
        elapsed = time.perf_counter() - t0
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        results.append((lam, mean_auc, std_auc, elapsed))
        print(f"  λ={lam:4.1f}  CV AUC={mean_auc:.5f} ± {std_auc:.5f}  ({elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 70)
    baseline = results[0][1]
    print(f"{'λ':>5}  {'CV AUC':>10}  {'Δ vs λ=0':>10}")
    print("-" * 30)
    for lam, auc, std, _ in results:
        delta = auc - baseline
        marker = " ★" if delta > 0 and lam > 0 else ""
        print(f"{lam:5.1f}  {auc:10.5f}  {delta:+10.5f}{marker}")

    # Run best λ on full training set → test set
    best_lam = max(results, key=lambda r: r[1])[0]
    print(f"\nBest λ={best_lam:.1f}")

    print(f"\nRefitting on full train ({len(X_train)} samples) with λ={best_lam}...")
    t0 = time.perf_counter()
    m_final = GeoXGBClassifier(**best_params, e3_target_lambda=best_lam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_final.fit(X_train, y_train.astype(float))
    fit_time = time.perf_counter() - t0

    proba_test = m_final.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba_test)
    print(f"Test AUC (λ={best_lam}): {test_auc:.5f}  ({fit_time:.1f}s)")

    # Also run λ=0 baseline on full train for fair comparison
    print(f"\nRefitting baseline λ=0 on full train...")
    t0 = time.perf_counter()
    m_base = GeoXGBClassifier(**best_params, e3_target_lambda=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_base.fit(X_train, y_train.astype(float))
    fit_time_base = time.perf_counter() - t0

    proba_base = m_base.predict_proba(X_test)[:, 1]
    test_auc_base = roc_auc_score(y_test, proba_base)
    print(f"Test AUC (λ=0):   {test_auc_base:.5f}  ({fit_time_base:.1f}s)")
    print(f"Test Δ:           {test_auc - test_auc_base:+.5f}")
