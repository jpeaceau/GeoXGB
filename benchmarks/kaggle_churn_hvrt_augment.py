"""
Kaggle Churn — HVRT-augmented training strategies.

Strategy 1: HVRT-reduce 594k synthetic → N, concat with 7k original
Strategy 2: HVRT-expand 7k original → N, concat with synthetic
Strategy 3: HVRT-reduce synthetic + HVRT-expand original (combined)
Strategy 4: GeoXGB on combined data
Strategy 5: Random reduce comparison (ablation)
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from geoxgb import GeoXGBClassifier

DATA_DIR = "data"


def load_all():
    """Load and encode train, test, and original datasets."""
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")
    orig = pd.read_csv(f"{DATA_DIR}/original_data.csv")

    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(0.0)

    y_tr = (tr["Churn"] == "Yes").astype(int).values
    y_orig = (orig["Churn"] == "Yes").astype(int).values

    feat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    X_tr = tr[feat_cols].copy()
    X_te = te[feat_cols].copy()
    X_orig = orig[feat_cols].copy()

    cat_cols = X_tr.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr[col], X_te[col], X_orig[col]]))
        X_tr[col] = le.transform(X_tr[col])
        X_te[col] = le.transform(X_te[col])
        X_orig[col] = le.transform(X_orig[col])

    return (X_tr.values.astype(np.float64), y_tr,
            X_te.values.astype(np.float64), te["id"].values,
            X_orig.values.astype(np.float64), y_orig,
            feat_cols)


def eval_fold(X_train, y_train, X_val, y_val, label, model="xgb"):
    """Train and evaluate on a single fold."""
    t0 = time.time()
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if model == "xgb":
        m = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            tree_method="hist", verbosity=0, random_state=42,
            scale_pos_weight=spw,
        )
    elif model == "geo":
        m = GeoXGBClassifier(
            n_rounds=1000, max_depth=6, learning_rate=0.10,
            class_weight="balanced", random_state=42,
        )
    else:
        raise ValueError(model)

    m.fit(X_train, y_train)
    proba = m.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    elapsed = time.time() - t0
    print(f"  {label:<55} AUC={auc:.6f}  ({elapsed:.1f}s)  n={len(X_train)}")
    return auc


def hvrt_reduce(X, y, n_keep, seed=42):
    """HVRT geometry-aware reduction preserving class labels."""
    from hvrt import HVRT
    h = HVRT(
        n_partitions=None,
        min_samples_leaf=max(5, len(X) // 200),
        random_state=seed,
    )
    h.fit(X, y.astype(np.float64))
    X_red, idx = h.reduce(n=n_keep, return_indices=True)
    return X_red, y[idx]


def hvrt_expand(X, y, n_generate, seed=42):
    """HVRT geometry-aware expansion with k-NN label assignment."""
    from hvrt import HVRT
    h = HVRT(
        n_partitions=None,
        min_samples_leaf=max(5, len(X) // 50),
        random_state=seed,
    )
    h.fit(X, y.astype(np.float64))
    X_syn = h.expand(n_generate)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X, y)
    y_syn = knn.predict(X_syn)
    return X_syn, y_syn


def random_reduce(X, y, n_keep, seed=42):
    """Stratified random subsample (ablation baseline)."""
    rng = np.random.default_rng(seed)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    frac = n_keep / len(y)
    n0 = int(len(idx_0) * frac)
    n1 = n_keep - n0
    sel = np.concatenate([
        rng.choice(idx_0, size=min(n0, len(idx_0)), replace=False),
        rng.choice(idx_1, size=min(n1, len(idx_1)), replace=False),
    ])
    return X[sel], y[sel]


if __name__ == "__main__":
    X_synth, y_synth, X_test, test_ids, X_orig, y_orig, feat_names = load_all()
    print(f"Synthetic train: {X_synth.shape}, churn={y_synth.mean():.4f}")
    print(f"Original data:   {X_orig.shape}, churn={y_orig.mean():.4f}")
    print(f"Test:            {X_test.shape}")
    print()

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    synth_tr_idx, synth_va_idx = next(iter(skf.split(X_synth, y_synth)))
    X_synth_tr, y_synth_tr = X_synth[synth_tr_idx], y_synth[synth_tr_idx]
    X_val, y_val = X_synth[synth_va_idx], y_synth[synth_va_idx]

    # ── Baselines ────────────────────────────────────────────────────
    print("=" * 75)
    print("BASELINES")
    print("=" * 75)
    eval_fold(X_synth_tr, y_synth_tr, X_val, y_val,
              "XGB full synthetic (396k)", model="xgb")
    eval_fold(X_synth_tr, y_synth_tr, X_val, y_val,
              "GEO full synthetic (396k)", model="geo")
    eval_fold(X_orig, y_orig, X_val, y_val,
              "XGB original only (7k)", model="xgb")

    # ── Strategy 1: HVRT-reduce synthetic + original ─────────────────
    print()
    print("=" * 75)
    print("STRATEGY 1: HVRT-reduce synthetic + original")
    print("=" * 75)
    for n_keep in [10_000, 30_000, 50_000, 100_000]:
        t0 = time.time()
        X_red, y_red = hvrt_reduce(X_synth_tr, y_synth_tr, n_keep)
        print(f"  [HVRT reduce to {n_keep}: {time.time()-t0:.1f}s, churn={y_red.mean():.4f}]")
        X_combo = np.vstack([X_orig, X_red])
        y_combo = np.concatenate([y_orig, y_red])
        eval_fold(X_combo, y_combo, X_val, y_val,
                  f"XGB: orig(7k) + HVRT_red({n_keep//1000}k)", model="xgb")

    # ── Ablation: random reduce vs HVRT reduce ───────────────────────
    print()
    print("=" * 75)
    print("ABLATION: HVRT reduce vs random reduce (50k)")
    print("=" * 75)
    X_rr, y_rr = random_reduce(X_synth_tr, y_synth_tr, 50_000)
    X_combo_rr = np.vstack([X_orig, X_rr])
    y_combo_rr = np.concatenate([y_orig, y_rr])
    eval_fold(X_combo_rr, y_combo_rr, X_val, y_val,
              "XGB: orig(7k) + RANDOM_red(50k)", model="xgb")

    X_hr, y_hr = hvrt_reduce(X_synth_tr, y_synth_tr, 50_000)
    X_combo_hr = np.vstack([X_orig, X_hr])
    y_combo_hr = np.concatenate([y_orig, y_hr])
    eval_fold(X_combo_hr, y_combo_hr, X_val, y_val,
              "XGB: orig(7k) + HVRT_red(50k)", model="xgb")

    # ── Strategy 2: HVRT-expand original ─────────────────────────────
    print()
    print("=" * 75)
    print("STRATEGY 2: HVRT-expand original")
    print("=" * 75)
    for n_gen in [10_000, 30_000, 50_000]:
        t0 = time.time()
        X_exp, y_exp = hvrt_expand(X_orig, y_orig, n_gen)
        print(f"  [HVRT expand {n_gen}: {time.time()-t0:.1f}s, churn={y_exp.mean():.4f}]")
        X_combo = np.vstack([X_orig, X_exp])
        y_combo = np.concatenate([y_orig, y_exp])
        eval_fold(X_combo, y_combo, X_val, y_val,
                  f"XGB: orig(7k) + HVRT_exp({n_gen//1000}k)", model="xgb")

    # ── Strategy 3: Combined ─────────────────────────────────────────
    print()
    print("=" * 75)
    print("STRATEGY 3: HVRT-expand original + HVRT-reduce synthetic")
    print("=" * 75)
    X_exp_50, y_exp_50 = hvrt_expand(X_orig, y_orig, 50_000)
    X_red_50, y_red_50 = hvrt_reduce(X_synth_tr, y_synth_tr, 50_000)
    X_combo = np.vstack([X_orig, X_exp_50, X_red_50])
    y_combo = np.concatenate([y_orig, y_exp_50, y_red_50])
    eval_fold(X_combo, y_combo, X_val, y_val,
              "XGB: orig + HVRT_exp(50k) + HVRT_red(50k)", model="xgb")

    # ── Strategy 4: GeoXGB on best combos ────────────────────────────
    print()
    print("=" * 75)
    print("STRATEGY 4: GeoXGB on combined data")
    print("=" * 75)
    for n_keep in [30_000, 50_000]:
        X_red, y_red = hvrt_reduce(X_synth_tr, y_synth_tr, n_keep)
        X_combo = np.vstack([X_orig, X_red])
        y_combo = np.concatenate([y_orig, y_red])
        eval_fold(X_combo, y_combo, X_val, y_val,
                  f"GEO: orig(7k) + HVRT_red({n_keep//1000}k)", model="geo")

    # Also try GeoXGB with expanded original
    X_exp_30, y_exp_30 = hvrt_expand(X_orig, y_orig, 30_000)
    X_combo = np.vstack([X_orig, X_exp_30])
    y_combo = np.concatenate([y_orig, y_exp_30])
    eval_fold(X_combo, y_combo, X_val, y_val,
              "GEO: orig(7k) + HVRT_exp(30k)", model="geo")
