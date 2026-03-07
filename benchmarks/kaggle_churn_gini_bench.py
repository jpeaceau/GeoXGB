"""
Benchmark: GeoXGBGiniClassifier vs GeoXGBClassifier (log-loss) on Kaggle Churn.

Tests whether Gini-impurity loss gradients improve classification performance
compared to the standard log-loss gradients.

Gini loss: L = 2*p*(1-p), gradient = (y-p) * |1-2p|
Log-loss:  L = -y*log(p) - (1-y)*log(1-p), gradient = y-p
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
    from geoxgb import GeoXGBClassifier, GeoXGBGiniClassifier

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

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

    # Parameter configs to test (from yw0_hpo best + variants)
    configs = [
        {
            "name": "yw0_hpo best",
            "params": dict(
                n_rounds=3000, learning_rate=0.15, max_depth=3,
                min_samples_leaf=13, reduce_ratio=0.92, refit_interval=500,
                expand_ratio=0.49, y_weight=0.0, hvrt_min_samples_leaf=5,
                noise_guard=True, auto_noise=False, auto_expand=True,
                generation_strategy="laplace", method="orthant_stratified",
                variance_weighted=True, partitioner="hvrt",
                convergence_tol=0.01, random_state=42, sample_block_n="auto",
            ),
        },
        {
            "name": "defaults",
            "params": dict(
                n_rounds=1000, learning_rate=0.02, max_depth=3,
                y_weight=0.0, convergence_tol=0.01, random_state=42,
                sample_block_n="auto",
            ),
        },
        {
            "name": "high lr shallow",
            "params": dict(
                n_rounds=5000, learning_rate=0.05, max_depth=3,
                min_samples_leaf=10, reduce_ratio=0.90, refit_interval=300,
                expand_ratio=0.3, y_weight=0.0, noise_guard=False,
                auto_noise=False, auto_expand=True,
                generation_strategy="laplace", method="orthant_stratified",
                convergence_tol=0.01, random_state=42, sample_block_n="auto",
            ),
        },
    ]

    print(f"\n{'='*80}")
    print(f"Gini vs Log-Loss classifier comparison")
    print(f"10k subsample, 3-fold CV, y_weight=0")
    print(f"{'='*80}")

    for cfg in configs:
        name = cfg["name"]
        params = cfg["params"]
        print(f"\n--- Config: {name} ---")

        for cls_name, Cls in [("LogLoss", GeoXGBClassifier),
                               ("Gini",    GeoXGBGiniClassifier)]:
            scores = []
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, va_idx in splits:
                    m = Cls(**params)
                    m.fit(X_hpo[tr_idx], y_hpo[tr_idx].astype(float))
                    proba = m.predict_proba(X_hpo[va_idx])[:, 1]
                    scores.append(roc_auc_score(y_hpo[va_idx], proba))
            elapsed = time.perf_counter() - t0
            mean_auc = float(np.mean(scores))
            std_auc = float(np.std(scores))
            print(f"  {cls_name:8s}  CV AUC={mean_auc:.5f} +/- {std_auc:.5f}  ({elapsed:.1f}s)")

    # Full train/test evaluation with best config
    print(f"\n{'='*80}")
    print(f"Full train/test evaluation (yw0_hpo best config)")
    print(f"{'='*80}")

    best_params = configs[0]["params"]

    for cls_name, Cls in [("LogLoss", GeoXGBClassifier),
                           ("Gini",    GeoXGBGiniClassifier)]:
        t0 = time.perf_counter()
        m = Cls(**best_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train, y_train.astype(float))
        fit_time = time.perf_counter() - t0
        proba = m.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        cr = m._cpp_model.convergence_round()
        print(f"  {cls_name:8s}  Test AUC={test_auc:.5f}  conv@{cr}  ({fit_time:.1f}s)")

    # XGBoost reference
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"  XGBoost   Test AUC={xgb_auc:.5f}")
