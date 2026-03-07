"""
Benchmark: blended T + λ·e₃ HVRT target on Kaggle Churn — grid over λ × refit_interval.

The e₃ blend only matters at refits, so shorter refit_interval → more leverage.
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    MAX_SAMPLES = 10_000
    X_hpo, _, y_hpo, _ = train_test_split(
        X_train, y_train, train_size=MAX_SAMPLES,
        stratify=y_train, random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_hpo, y_hpo))

    base_params = dict(
        n_rounds=5000,
        learning_rate=0.046,
        max_depth=7,
        min_samples_leaf=20,
        reduce_ratio=0.54,
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
        sample_block_n=None,  # disable block cycling for cleaner e3 signal
    )

    lambdas = [0.0, 0.5, 1.0, 2.0]
    refit_intervals = [25, 50, 100, 300]

    print(f"Grid: λ × refit_interval, {MAX_SAMPLES} subsample, 3-fold CV")
    print(f"Block cycling DISABLED (sample_block_n=None)")
    print("=" * 70)

    results = []
    for ri in refit_intervals:
        for lam in lambdas:
            scores = []
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, va_idx in splits:
                    m = GeoXGBClassifier(**base_params, refit_interval=ri,
                                         e3_target_lambda=lam)
                    m.fit(X_hpo[tr_idx], y_hpo[tr_idx].astype(float))
                    proba = m.predict_proba(X_hpo[va_idx])[:, 1]
                    scores.append(roc_auc_score(y_hpo[va_idx], proba))
            elapsed = time.perf_counter() - t0
            mean_auc = float(np.mean(scores))
            results.append((ri, lam, mean_auc, elapsed))
            print(f"  ri={ri:3d}  λ={lam:4.1f}  CV AUC={mean_auc:.5f}  ({elapsed:.1f}s)")
        print()

    # Summary table
    print("=" * 70)
    print(f"{'ri':>5}  {'λ=0':>10}  {'λ=0.5':>10}  {'λ=1.0':>10}  {'λ=2.0':>10}")
    print("-" * 50)
    for ri in refit_intervals:
        vals = [r[2] for r in results if r[0] == ri]
        row = f"{ri:5d}"
        base = vals[0]
        for v in vals:
            delta = v - base
            if delta > 0.001:
                row += f"  {v:.5f}★"
            elif delta < -0.001:
                row += f"  {v:.5f}↓"
            else:
                row += f"  {v:.5f} "
        print(row)

    # Best config
    best = max(results, key=lambda r: r[2])
    print(f"\nBest: ri={best[0]}, λ={best[1]}, CV AUC={best[2]:.5f}")
