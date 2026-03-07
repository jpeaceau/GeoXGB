"""
Kaggle Churn HPO — SqHinge loss, PCA(19), y_weight=0.
Trials at HPO subsample sizes: 5k, 10k, 15k.

Fixed: y_weight=0, n_components=19 (PCA rotation), SqHinge loss.
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
    from geoxgb import GeoXGBHingeClassifier

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
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_va = scaler.transform(X[va_idx])
            pca = PCA(n_components=19, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_va_pca = pca.transform(X_va)

            m = GeoXGBHingeClassifier(**params)
            m.fit(X_tr_pca, y[tr_idx].astype(float))
            proba = m.predict_proba(X_va_pca)[:, 1]
            scores.append(roc_auc_score(y[va_idx], proba))
    return float(np.mean(scores))


if __name__ == "__main__":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    BATCH_SIZES = [5_000, 10_000, 15_000]

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Warm-start seeds from prior best results
    warm_starts = [
        # SqHinge best from loss HPO
        {"n_rounds": 3500, "learning_rate": 0.005, "max_depth": 2,
         "min_samples_leaf": 11, "reduce_ratio": 0.93, "refit_interval": 300,
         "expand_ratio": 0.47, "hvrt_min_samples_leaf": None,
         "noise_guard": True, "auto_noise": False, "auto_expand": False,
         "generation_strategy": "epanechnikov", "method": "variance_ordered",
         "variance_weighted": False, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 64},
        # PCA HPO best adapted for SqHinge
        {"n_rounds": 4000, "learning_rate": 0.017, "max_depth": 3,
         "min_samples_leaf": 24, "reduce_ratio": 0.54, "refit_interval": 300,
         "expand_ratio": 0.74, "hvrt_min_samples_leaf": 30,
         "noise_guard": True, "auto_noise": True, "auto_expand": False,
         "generation_strategy": "bootstrap", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
        # HPO4 best adapted
        {"n_rounds": 5000, "learning_rate": 0.015, "max_depth": 6,
         "min_samples_leaf": 6, "reduce_ratio": 0.59, "refit_interval": 500,
         "expand_ratio": 0.64, "hvrt_min_samples_leaf": 50,
         "noise_guard": False, "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
    ]

    from geoxgb import GeoXGBHingeClassifier

    for batch_size in BATCH_SIZES:
        print(f"\n{'='*80}")
        print(f"SqHinge + PCA(19) HPO — batch_size={batch_size}, {N_TRIALS} trials")
        print(f"{'='*80}")

        X_hpo, _, y_hpo, _ = train_test_split(
            X_train, y_train, train_size=batch_size,
            stratify=y_train, random_state=42
        )
        print(f"HPO subsample: {len(X_hpo)}, Test: {len(X_test)}")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(skf.split(X_hpo, y_hpo))

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for ws in warm_starts:
            study.enqueue_trial(ws)

        t0 = time.perf_counter()
        study.optimize(
            lambda trial: objective(trial, X_hpo, y_hpo, splits),
            n_trials=N_TRIALS,
        )
        hpo_time = time.perf_counter() - t0

        print(f"HPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s/trial)")
        print(f"Best CV AUC: {study.best_value:.5f}")
        print(f"Best params: {study.best_params}")

        # Top 10
        df = study.trials_dataframe()
        df = df.sort_values("value", ascending=False).drop_duplicates(subset="value").head(10)
        print(f"\nTop 10 distinct trials:")
        for _, row in df.iterrows():
            p = {k.replace("params_", ""): row[k]
                 for k in row.index if k.startswith("params_")}
            print(f"  AUC={row['value']:.5f} | "
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

        # Full train/test evaluation
        best = dict(study.best_params)
        best["y_weight"] = 0.0
        best["convergence_tol"] = 0.01
        best["random_state"] = 42
        best["sample_block_n"] = "auto"

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        pca = PCA(n_components=19, random_state=42)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)

        print(f"\nRefitting best on full train ({len(X_train)} samples)...")
        t0 = time.perf_counter()
        m = GeoXGBHingeClassifier(**best)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train_pca, y_train.astype(float))
        fit_time = time.perf_counter() - t0
        proba = m.predict_proba(X_test_pca)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        cr = m._cpp_model.convergence_round()
        fi = m._cpp_model.feature_importances()
        n_used = sum(1 for v in fi if v > 0)
        print(f"Test AUC: {test_auc:.5f} | {n_used}/19 PCs | conv@{cr} | {fit_time:.1f}s")

    # XGBoost references
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=19, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train_pca, y_train)
    xgb_pca_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_pca)[:, 1])

    xgb_raw = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb_raw.fit(X_train, y_train)
    xgb_raw_auc = roc_auc_score(y_test, xgb_raw.predict_proba(X_test)[:, 1])

    print(f"\n{'='*80}")
    print(f"References:")
    print(f"  XGBoost raw:    {xgb_raw_auc:.5f}")
    print(f"  XGBoost+PCA:    {xgb_pca_auc:.5f}")
    print(f"  Prior best (PCA+LogLoss 10k): 0.90493")
