"""
Kaggle Churn HPO — PCA-transformed features.

Hypothesis: PCA decorrelates the feature space, giving HVRT cleaner geometry
to partition. The whitening step already normalizes, but PCA can reduce
dimensionality and remove noise dimensions.

500 trials, 10k subsample, y_weight=0, same expanded space as hpo4.
Tests PCA with n_components in [5, 8, 10, 12, 15, 19] (19=all, just rotation).
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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

    n_components = trial.suggest_categorical("n_components",
                       [5, 8, 10, 12, 15, 19])

    params = {
        "n_rounds":       trial.suggest_int("n_rounds", 500, 8000, step=500),
        "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "max_depth":      trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.3, 0.99),
        "refit_interval": trial.suggest_categorical("refit_interval",
                              [25, 50, 100, 200, 300, 500, 1000]),
        "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.8),
        "y_weight":       0.0,
        "hvrt_min_samples_leaf": trial.suggest_categorical(
            "hvrt_min_samples_leaf", [None, 3, 5, 10, 20, 30, 50, 100]),
        "noise_guard":    trial.suggest_categorical("noise_guard", [True, False]),
        "auto_noise":     trial.suggest_categorical("auto_noise", [True, False]),
        "auto_expand":    trial.suggest_categorical("auto_expand", [True, False]),
        "generation_strategy": trial.suggest_categorical(
            "generation_strategy",
            ["epanechnikov", "simplex_mixup", "laplace", "bootstrap"]),
        "method": trial.suggest_categorical(
            "method", ["variance_ordered", "orthant_stratified"]),
        "variance_weighted": trial.suggest_categorical(
            "variance_weighted", [True, False]),
        "partitioner": trial.suggest_categorical(
            "partitioner", ["hvrt", "hart", "pyramid_hart"]),
        "class_weight":   trial.suggest_categorical("class_weight",
                              [None, "balanced"]),
        "n_bins": trial.suggest_categorical("n_bins", [32, 64, 128]),
        "convergence_tol": 0.01,
        "random_state":   42,
        "sample_block_n": "auto",
    }

    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, va_idx in splits:
            # Apply PCA per fold (fit on train, transform both)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_va = scaler.transform(X[va_idx])

            pca = PCA(n_components=n_components, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_va_pca = pca.transform(X_va)

            m = GeoXGBClassifier(**params)
            m.fit(X_tr_pca, y[tr_idx].astype(float))
            proba = m.predict_proba(X_va_pca)[:, 1]
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

    # Warm-start with hpo4 best params + PCA variants
    warm_starts = [
        # hpo4 best with all 19 components (just PCA rotation)
        {"n_components": 19, "n_rounds": 5000, "learning_rate": 0.015,
         "max_depth": 6, "min_samples_leaf": 6, "reduce_ratio": 0.59,
         "refit_interval": 500, "expand_ratio": 0.64,
         "hvrt_min_samples_leaf": 50, "noise_guard": False,
         "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
        # Same but fewer components
        {"n_components": 12, "n_rounds": 5000, "learning_rate": 0.015,
         "max_depth": 6, "min_samples_leaf": 6, "reduce_ratio": 0.59,
         "refit_interval": 500, "expand_ratio": 0.64,
         "hvrt_min_samples_leaf": 50, "noise_guard": False,
         "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
        {"n_components": 8, "n_rounds": 5000, "learning_rate": 0.015,
         "max_depth": 6, "min_samples_leaf": 6, "reduce_ratio": 0.59,
         "refit_interval": 500, "expand_ratio": 0.64,
         "hvrt_min_samples_leaf": 50, "noise_guard": False,
         "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
    ]

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for ws in warm_starts:
        study.enqueue_trial(ws)

    print(f"\nRunning PCA HPO: {N_TRIALS} trials, y_weight=0")
    print(f"n_components in [5, 8, 10, 12, 15, 19]")
    print(f"Same expanded search space as hpo4 + StandardScaler + PCA")
    print("=" * 80)

    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_hpo, y_hpo, splits),
        n_trials=N_TRIALS,
    )
    hpo_time = time.perf_counter() - t0

    print(f"\nHPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s per trial)")
    print(f"Best CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    # Top 15 unique trials
    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False).drop_duplicates(subset="value").head(15)
    print(f"\nTop 15 distinct trials:")
    for _, row in df.iterrows():
        p = {k.replace("params_", ""): row[k]
             for k in row.index if k.startswith("params_")}
        print(f"  AUC={row['value']:.5f} | "
              f"nc={p.get('n_components',0)} "
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
              f"pt={p.get('partitioner','')} "
              f"nb={p.get('n_bins','')} "
              f"cw={p.get('class_weight','')}")

    # ── Evaluate on full data ────────────────────────────────────────────
    from geoxgb import GeoXGBClassifier

    best = dict(study.best_params)
    n_comp = int(best.pop("n_components"))
    best["y_weight"] = 0.0
    best["convergence_tol"] = 0.01
    best["random_state"] = 42
    best["sample_block_n"] = "auto"

    # PCA transform full train/test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"\nRefitting best on full train ({len(X_train)} samples), PCA n_components={n_comp}...")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    t0 = time.perf_counter()
    m = GeoXGBClassifier(**best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_train_pca, y_train.astype(float))
    fit_time = time.perf_counter() - t0

    proba = m.predict_proba(X_test_pca)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    cr = m._cpp_model.convergence_round()
    fi = m._cpp_model.feature_importances()
    n_used = sum(1 for v in fi if v > 0)
    print(f"Test AUC (PCA): {test_auc:.5f} | {n_used}/{n_comp} PCs used | conv@{cr} | {fit_time:.1f}s")

    # Compare with non-PCA results
    print(f"\nReference results (no PCA):")
    print(f"  hpo4 test AUC: 0.9010")
    print(f"  yw0_hpo test AUC: 0.8954")
    print(f"  hpo3 test AUC: 0.8861")
    print(f"  Delta vs hpo4: {test_auc - 0.9010:+.5f}")

    # XGBoost comparison (also with PCA, for fair comparison)
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train_pca, y_train)
    xgb_auc_pca = roc_auc_score(y_test, xgb.predict_proba(X_test_pca)[:, 1])
    print(f"\nXGBoost + PCA({n_comp}) test AUC: {xgb_auc_pca:.5f}")
    print(f"XGBoost raw test AUC: 0.91610")
    print(f"Delta vs XGBoost+PCA: {test_auc - xgb_auc_pca:+.5f}")
