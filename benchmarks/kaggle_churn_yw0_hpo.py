"""
Kaggle Churn HPO — y_weight=0 (pure geometry partitions).

500 trials, 10k subsample, same expanded space as hpo3 but y_weight fixed at 0.
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


def objective(trial, X, y, splits):
    from geoxgb import GeoXGBClassifier

    params = {
        "n_rounds":       trial.suggest_int("n_rounds", 500, 5000, step=500),
        "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "max_depth":      trial.suggest_int("max_depth", 2, 7),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),

        "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.3, 0.95),
        "refit_interval": trial.suggest_categorical("refit_interval",
                              [10, 25, 50, 100, 200, 300, 500]),
        "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.5),

        # FIXED: y_weight=0 — pure geometry partitions
        "y_weight":       0.0,

        "hvrt_min_samples_leaf": trial.suggest_categorical(
            "hvrt_min_samples_leaf", [None, 5, 10, 20, 30, 50]),

        "noise_guard":    trial.suggest_categorical("noise_guard", [True, False]),
        "auto_noise":     trial.suggest_categorical("auto_noise", [True, False]),
        "auto_expand":    trial.suggest_categorical("auto_expand", [True, False]),

        "generation_strategy": trial.suggest_categorical(
            "generation_strategy",
            ["epanechnikov", "simplex_mixup", "laplace", "bootstrap"]),

        "method": trial.suggest_categorical(
            "method",
            ["variance_ordered", "orthant_stratified"]),

        "variance_weighted": trial.suggest_categorical(
            "variance_weighted", [True, False]),

        "class_weight":   trial.suggest_categorical("class_weight",
                              [None, "balanced"]),

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

    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    MAX_SAMPLES = 10_000

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_hpo, _, y_hpo, _ = train_test_split(
        X_train, y_train, train_size=MAX_SAMPLES,
        stratify=y_train, random_state=42
    )
    print(f"HPO subsample: {len(X_hpo)}, Test: {len(X_test)}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_hpo, y_hpo))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start seeds (all with y_weight=0)
    warm_starts = [
        # Best from hpo3 but with y_weight=0
        {"n_rounds": 4500, "learning_rate": 0.052, "max_depth": 5,
         "min_samples_leaf": 5, "reduce_ratio": 0.54, "refit_interval": 300,
         "expand_ratio": 0.475,
         "hvrt_min_samples_leaf": 30, "noise_guard": False,
         "auto_noise": False, "auto_expand": True,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "class_weight": None},
        # GeoXGB defaults but y_weight=0
        {"n_rounds": 1000, "learning_rate": 0.02, "max_depth": 3,
         "min_samples_leaf": 5, "reduce_ratio": 0.8, "refit_interval": 50,
         "expand_ratio": 0.1,
         "hvrt_min_samples_leaf": None, "noise_guard": True,
         "auto_noise": True, "auto_expand": True,
         "generation_strategy": "simplex_mixup", "method": "variance_ordered",
         "variance_weighted": False, "class_weight": None},
        # High lr, high depth
        {"n_rounds": 3000, "learning_rate": 0.08, "max_depth": 7,
         "min_samples_leaf": 10, "reduce_ratio": 0.7, "refit_interval": 100,
         "expand_ratio": 0.2,
         "hvrt_min_samples_leaf": 20, "noise_guard": False,
         "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": False, "class_weight": "balanced"},
        # Very low lr, many rounds
        {"n_rounds": 5000, "learning_rate": 0.005, "max_depth": 4,
         "min_samples_leaf": 15, "reduce_ratio": 0.85, "refit_interval": 200,
         "expand_ratio": 0.05,
         "hvrt_min_samples_leaf": 10, "noise_guard": True,
         "auto_noise": True, "auto_expand": True,
         "generation_strategy": "epanechnikov", "method": "variance_ordered",
         "variance_weighted": True, "class_weight": None},
    ]
    for ws in warm_starts:
        study.enqueue_trial(ws)

    print(f"\nRunning HPO: {N_TRIALS} trials, y_weight=0 (pure geometry)")
    print(f"Search space: 15 parameters (5 continuous, 10 categorical)")
    print("=" * 80)

    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_hpo, y_hpo, splits),
        n_trials=N_TRIALS,
    )
    hpo_time = time.perf_counter() - t0

    print(f"\nHPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s per trial)")
    print(f"Best CV AUC (10k subsample, y_weight=0): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Top 15 unique trials
    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False).drop_duplicates(subset="value").head(15)
    print(f"\nTop 15 distinct trials:")
    for _, row in df.iterrows():
        p = {k.replace("params_", ""): row[k]
             for k in row.index if k.startswith("params_")}
        print(f"  AUC={row['value']:.4f} | "
              f"lr={p.get('learning_rate',0):.4f} "
              f"d={p.get('max_depth',0)} "
              f"msl={p.get('min_samples_leaf',0)} "
              f"ri={p.get('refit_interval',0)} "
              f"nr={p.get('n_rounds',0)} "
              f"rr={p.get('reduce_ratio',0):.2f} "
              f"er={p.get('expand_ratio',0):.2f} "
              f"hmsl={p.get('hvrt_min_samples_leaf',None)} "
              f"ng={p.get('noise_guard','')} "
              f"an={p.get('auto_noise','')} "
              f"ae={p.get('auto_expand','')} "
              f"gs={p.get('generation_strategy','')} "
              f"rm={p.get('method','')} "
              f"vw={p.get('variance_weighted','')} "
              f"cw={p.get('class_weight','')}")

    # ── Evaluate top config on full data ─────────────────────────────────
    from geoxgb import GeoXGBClassifier

    best = dict(study.best_params)
    best["y_weight"] = 0.0
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

    print(f"Test AUC (y_weight=0): {test_auc:.4f} | {n_used}/{X.shape[1]} feats | {fit_time:.1f}s")

    # Compare with best y_weight>0 result from hpo3
    print(f"\nReference: best hpo3 (y_weight=0.31) test AUC = 0.8861")
    print(f"Delta vs hpo3: {test_auc - 0.8861:+.4f}")

    # XGBoost comparison
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"XGBoost default test AUC: {xgb_auc:.4f}")
    print(f"Delta vs XGBoost: {test_auc - xgb_auc:+.4f}")
