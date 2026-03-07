"""
Kaggle Churn Competition — GeoXGB investigation.

Goal: understand where GeoXGB loses to XGBoost/CatBoost and close the gap.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── Data loading & preprocessing ─────────────────────────────────────────
DATA_DIR = "data"

def load_data():
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")

    y = (tr["Churn"] == "Yes").astype(int).values
    drop = ["id", "Churn"]
    X_tr = tr.drop(columns=drop)
    X_te = te.drop(columns=["id"])

    # Label-encode categoricals
    cat_cols = X_tr.select_dtypes("object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr[col], X_te[col]]))
        X_tr[col] = le.transform(X_tr[col])
        X_te[col] = le.transform(X_te[col])
        encoders[col] = le

    X_tr = X_tr.values.astype(np.float64)
    X_te = X_te.values.astype(np.float64)
    return X_tr, y, X_te, te["id"].values, cat_cols, encoders


def cv_auc(model_cls, X, y, n_splits=3, seed=42, **kw):
    """Stratified K-fold AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    t0 = time.time()
    for tr_idx, va_idx in skf.split(X, y):
        m = model_cls(**kw)
        m.fit(X[tr_idx], y[tr_idx])
        proba = m.predict_proba(X[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], proba))
    elapsed = time.time() - t0
    return np.mean(scores), np.std(scores), elapsed


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, X_test, test_ids, cat_cols, encoders = load_data()
    print(f"Train: {X.shape}, Churn rate: {y.mean():.4f}")
    print(f"Test:  {X_test.shape}")
    print(f"Features: {X.shape[1]}, Categoricals: {len(cat_cols)}")
    print()

    # ── 1. XGBoost baseline ──────────────────────────────────────────────
    print("=" * 60)
    print("XGBoost baseline (defaults + scale_pos_weight)")
    print("=" * 60)
    xgb_kw = dict(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        tree_method="hist", verbosity=0, random_state=42,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        enable_categorical=False,
    )
    auc, std, t = cv_auc(XGBClassifier, X, y, **xgb_kw)
    print(f"  AUC = {auc:.6f} +/- {std:.6f}  ({t:.1f}s)")
    print()

    # ── 2. GeoXGB default ────────────────────────────────────────────────
    from geoxgb import GeoXGBClassifier
    print("=" * 60)
    print("GeoXGB default")
    print("=" * 60)
    geo_kw = dict(
        n_rounds=1000, random_state=42,
        class_weight="balanced",
    )
    auc_g, std_g, t_g = cv_auc(GeoXGBClassifier, X, y, **geo_kw)
    print(f"  AUC = {auc_g:.6f} +/- {std_g:.6f}  ({t_g:.1f}s)")
    print()

    # ── 3. GeoXGB tuned (manual) ─────────────────────────────────────────
    print("=" * 60)
    print("GeoXGB tuned (manual params)")
    print("=" * 60)
    geo_tuned_kw = dict(
        n_rounds=2000,
        learning_rate=0.03,
        max_depth=5,
        reduce_ratio=0.9,
        refit_interval=100,
        expand_ratio=0.05,
        y_weight=0.3,
        class_weight="balanced",
        random_state=42,
    )
    auc_gt, std_gt, t_gt = cv_auc(GeoXGBClassifier, X, y, **geo_tuned_kw)
    print(f"  AUC = {auc_gt:.6f} +/- {std_gt:.6f}  ({t_gt:.1f}s)")
    print()

    # ── 4. Feature importance analysis ───────────────────────────────────
    print("=" * 60)
    print("Feature importance (XGBoost)")
    print("=" * 60)
    feat_names = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    xgb_m = XGBClassifier(**xgb_kw)
    xgb_m.fit(X, y)
    imp = xgb_m.feature_importances_
    for i in np.argsort(imp)[::-1]:
        print(f"  {feat_names[i]:<20} {imp[i]:.4f}")
    print()

    # ── 5. Correlation analysis ──────────────────────────────────────────
    print("=" * 60)
    print("Feature-target correlations (point-biserial)")
    print("=" * 60)
    for i, fn in enumerate(feat_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {fn:<20} {corr:+.4f}")
