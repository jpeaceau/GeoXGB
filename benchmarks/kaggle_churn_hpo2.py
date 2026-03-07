"""
Kaggle Churn HPO round 2 — more diverse exploration.

Changes from round 1:
- Use Optuna's suggest_float/suggest_int for continuous ranges instead of
  categorical, giving TPE finer-grained exploration
- Add noise_guard and auto_noise to search space
- Warm-start with both the default AND the best from round 1
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
    feat_names = X.columns.tolist()
    return X.values.astype(np.float64), y, feat_names


def objective(trial, X, y, splits, feat_names):
    from geoxgb import GeoXGBClassifier

    params = {
        "n_rounds":       trial.suggest_int("n_rounds", 500, 5000, step=500),
        "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "max_depth":      trial.suggest_int("max_depth", 2, 7),
        "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.3, 0.95),
        "refit_interval": trial.suggest_categorical("refit_interval", [10, 25, 50, 100, 200, 300, 500]),
        "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.5),
        "y_weight":       trial.suggest_float("y_weight", 0.05, 0.8),
        "class_weight":   trial.suggest_categorical("class_weight", [None, "balanced"]),
        "hvrt_min_samples_leaf": trial.suggest_categorical("hvrt_min_samples_leaf", [None, 5, 10, 20, 30, 50]),
        "noise_guard":    trial.suggest_categorical("noise_guard", [True, False]),
        "convergence_tol": 0.01,
        "random_state":   42,
        "sample_block_n": "auto",
    }

    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, va_idx in splits:
            m = GeoXGBClassifier(**params)
            m.fit(X[tr_idx], y[tr_idx].astype(float))
            proba = m.predict_proba(X[va_idx])[:, 1]
            scores.append(roc_auc_score(y[va_idx], proba))
    return float(np.mean(scores))


if __name__ == "__main__":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    MAX_SAMPLES = 10_000

    X, y, feat_names = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    # Hold out 20% for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Subsample for HPO
    X_hpo, _, y_hpo, _ = train_test_split(
        X_train, y_train, train_size=MAX_SAMPLES,
        stratify=y_train, random_state=42
    )
    print(f"HPO subsample: {len(X_hpo)}, Test: {len(X_test)}")

    # Fixed CV splits on HPO subsample
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_hpo, y_hpo))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=30)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start: default + best from round 1
    study.enqueue_trial({
        "n_rounds": 1000, "learning_rate": 0.02, "max_depth": 3,
        "reduce_ratio": 0.8, "refit_interval": 50, "expand_ratio": 0.1,
        "y_weight": 0.25, "class_weight": None,
        "hvrt_min_samples_leaf": None, "noise_guard": True,
    })
    study.enqueue_trial({
        "n_rounds": 4000, "learning_rate": 0.1, "max_depth": 5,
        "reduce_ratio": 0.95, "refit_interval": 300, "expand_ratio": 0.0,
        "y_weight": 0.15, "class_weight": None,
        "hvrt_min_samples_leaf": 30, "noise_guard": True,
    })
    # Previous best from partition_feature HPO (adapted — no pf/gg)
    study.enqueue_trial({
        "n_rounds": 4000, "learning_rate": 0.015, "max_depth": 6,
        "reduce_ratio": 0.90, "refit_interval": 25, "expand_ratio": 0.05,
        "y_weight": 0.30, "class_weight": None,
        "hvrt_min_samples_leaf": None, "noise_guard": True,
    })

    print(f"\nRunning HPO: {N_TRIALS} trials, {MAX_SAMPLES} subsample, cv=3")
    print("=" * 70)

    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_hpo, y_hpo, splits, feat_names),
        n_trials=N_TRIALS,
    )
    hpo_time = time.perf_counter() - t0

    print(f"\nHPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s per trial)")
    print(f"Best CV AUC (10k subsample): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Top 10 unique trials
    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False).drop_duplicates(subset="value").head(10)
    print(f"\nTop 10 distinct trials:")
    for _, row in df.iterrows():
        p = {k.replace("params_", ""): row[k]
             for k in row.index if k.startswith("params_")}
        lr = p.get("learning_rate", 0)
        md = p.get("max_depth", 0)
        ri = p.get("refit_interval", 0)
        nr = p.get("n_rounds", 0)
        rr = p.get("reduce_ratio", 0)
        yw = p.get("y_weight", 0)
        msl = p.get("hvrt_min_samples_leaf", None)
        ng = p.get("noise_guard", True)
        cw = p.get("class_weight", None)
        print(f"  AUC={row['value']:.4f} | lr={lr:.4f} d={md} ri={ri} nr={nr} "
              f"rr={rr:.2f} yw={yw:.2f} msl={msl} ng={ng} cw={cw}")

    # ── Evaluate top config on full data ─────────────────────────────────
    from geoxgb import GeoXGBClassifier

    best = dict(study.best_params)
    best["convergence_tol"] = 0.01
    best["random_state"] = 42
    best["sample_block_n"] = "auto"

    print(f"\nRefitting best on full train ({len(X_train)} samples)...")
    t0 = time.perf_counter()
    m = GeoXGBClassifier(**best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_train, y_train.astype(float))
    fit_time = time.perf_counter() - t0

    proba = m.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    fi = m._cpp_model.feature_importances()
    n_used = sum(1 for v in fi if v > 0)

    print(f"Test AUC: {test_auc:.4f} | {n_used}/{len(feat_names)} feats | {fit_time:.1f}s")

    # XGBoost comparison
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"XGBoost default test AUC: {xgb_auc:.4f}")
    print(f"Delta: {test_auc - xgb_auc:+.4f}")

    print(f"\nFeature importances (GeoXGB best):")
    for i in np.argsort(fi)[::-1]:
        if fi[i] > 0:
            print(f"  {feat_names[i]:<20} {fi[i]:.4f}")
