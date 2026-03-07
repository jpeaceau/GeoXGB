"""
Kaggle Churn HPO — expanded search space, 10k subsample, many trials.

Uses GeoXGBOptimizer with _MAX_HPO_SAMPLES=10k so each trial is fast.
Final refit on the full 594k dataset, evaluated via single train/test split.
"""
import sys, time, os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    feat_names = X.columns.tolist()
    return X.values.astype(np.float64), y, feat_names


if __name__ == "__main__":
    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    X, y, feat_names = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    # Hold out 20% for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── HPO ──────────────────────────────────────────────────────────────
    from geoxgb import GeoXGBOptimizer

    print(f"\nRunning HPO: {N_TRIALS} trials, 10k subsample, cv=3")
    print("=" * 70)

    t0 = time.perf_counter()
    opt = GeoXGBOptimizer(
        task="classification",
        n_trials=N_TRIALS,
        cv=3,
        random_state=42,
        verbose=False,
    )
    opt.fit(X_train, y_train.astype(float))
    hpo_time = time.perf_counter() - t0

    print(f"\nHPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s per trial)")
    print(f"Best CV AUC (10k subsample): {opt.best_score_:.4f}")
    print(f"Best params: {opt.best_params_}")

    # Top 5 trials
    df = opt.study_.trials_dataframe()
    df = df.sort_values("value", ascending=False).head(5)
    print(f"\nTop 5 trials:")
    for _, row in df.iterrows():
        params = {k.replace("params_", ""): row[k]
                  for k in row.index if k.startswith("params_")}
        print(f"  AUC={row['value']:.4f} | {params}")

    # ── Evaluate on held-out test set ────────────────────────────────────
    print(f"\nEvaluating best model on held-out test set ({len(X_test)} samples)...")
    proba = opt.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)

    # Feature importance
    fi = opt.best_model_._cpp_model.feature_importances()
    n_used = sum(1 for v in fi if v > 0)

    print(f"Test AUC: {test_auc:.4f} | {n_used}/{len(feat_names)} feats")

    # ── XGBoost baseline for comparison ──────────────────────────────────
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"XGBoost default test AUC: {xgb_auc:.4f}")
    print(f"Delta: {test_auc - xgb_auc:+.4f}")

    # Feature importance ranking
    print(f"\nFeature importances (GeoXGB best):")
    for i in np.argsort(fi)[::-1]:
        if fi[i] > 0:
            print(f"  {feat_names[i]:<20} {fi[i]:.4f}")
