"""
Kaggle Churn — HVRT-reduce 475k → 50k, then HPO on the reduced set.

Uses HVRT's variance_ordered reduce to intelligently downsample the training
data to 50k samples that preserve the geometric structure. Then runs HPO
on that 50k set without further subsampling.

PCA(19) rotation, y_weight=0, LogLoss (best loss for this dataset).
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from hvrt import HVRT

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


def hvrt_reduce(X, y, target_n=50_000, random_state=42):
    """Use HVRT to reduce X down to target_n samples preserving geometry."""
    n = len(X)
    if n <= target_n:
        return X, y

    h = HVRT(random_state=random_state)
    h.fit(X, y.astype(np.float64))

    _, idx = h.reduce(n=target_n, method="variance_ordered",
                      variance_weighted=True, return_indices=True)
    idx = np.array(idx)

    return X[idx], y[idx]


def objective(trial, X, y, splits):
    from geoxgb import GeoXGBClassifier

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
            # PCA per fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_va = scaler.transform(X[va_idx])
            pca = PCA(n_components=19, random_state=42)
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
    TARGET_N = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # HVRT-reduce the training set
    print(f"\nHVRT-reducing {len(X_train)} → {TARGET_N}...")
    t0 = time.perf_counter()
    X_reduced, y_reduced = hvrt_reduce(X_train, y_train, target_n=TARGET_N)
    reduce_time = time.perf_counter() - t0
    print(f"Reduced to {len(X_reduced)} samples in {reduce_time:.1f}s")
    print(f"Reduced churn rate: {y_reduced.mean():.4f} (original: {y_train.mean():.4f})")

    # CV splits on the reduced data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_reduced, y_reduced))

    # Warm-start seeds
    warm_starts = [
        # PCA HPO best
        {"n_rounds": 4000, "learning_rate": 0.017, "max_depth": 3,
         "min_samples_leaf": 24, "reduce_ratio": 0.54, "refit_interval": 300,
         "expand_ratio": 0.74, "hvrt_min_samples_leaf": 30,
         "noise_guard": True, "auto_noise": True, "auto_expand": False,
         "generation_strategy": "bootstrap", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
        # HPO4 best
        {"n_rounds": 5000, "learning_rate": 0.015, "max_depth": 6,
         "min_samples_leaf": 6, "reduce_ratio": 0.59, "refit_interval": 500,
         "expand_ratio": 0.64, "hvrt_min_samples_leaf": 50,
         "noise_guard": False, "auto_noise": False, "auto_expand": False,
         "generation_strategy": "laplace", "method": "orthant_stratified",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": None, "n_bins": 128},
        # SqHinge 15k best (adapted for LogLoss)
        {"n_rounds": 500, "learning_rate": 0.003, "max_depth": 4,
         "min_samples_leaf": 21, "reduce_ratio": 0.93, "refit_interval": 500,
         "expand_ratio": 0.06, "hvrt_min_samples_leaf": 50,
         "noise_guard": False, "auto_noise": True, "auto_expand": True,
         "generation_strategy": "simplex_mixup", "method": "variance_ordered",
         "variance_weighted": True, "partitioner": "hvrt",
         "class_weight": "balanced", "n_bins": 128},
    ]

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=50)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for ws in warm_starts:
        study.enqueue_trial(ws)

    print(f"\nRunning HPO on HVRT-reduced {len(X_reduced)} samples, {N_TRIALS} trials")
    print(f"PCA(19) rotation, y_weight=0, LogLoss")
    print("=" * 80)

    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_reduced, y_reduced, splits),
        n_trials=N_TRIALS,
    )
    hpo_time = time.perf_counter() - t0

    print(f"\nHPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s/trial)")
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

    # ── Full train/test evaluation ──────────────────────────────────────
    from geoxgb import GeoXGBClassifier

    best = dict(study.best_params)
    best["y_weight"] = 0.0
    best["convergence_tol"] = 0.01
    best["random_state"] = 42
    best["sample_block_n"] = "auto"

    # PCA on full train/test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=19, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # Fit on full 475k train
    print(f"\nRefitting best on FULL train ({len(X_train)} samples)...")
    t0 = time.perf_counter()
    m_full = GeoXGBClassifier(**best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_full.fit(X_train_pca, y_train.astype(float))
    fit_time = time.perf_counter() - t0
    proba_full = m_full.predict_proba(X_test_pca)[:, 1]
    test_auc_full = roc_auc_score(y_test, proba_full)
    cr_full = m_full._cpp_model.convergence_round()
    print(f"Test AUC (full train): {test_auc_full:.5f} | conv@{cr_full} | {fit_time:.1f}s")

    # Also fit on just the HVRT-reduced 50k
    # PCA on reduced data
    scaler_r = StandardScaler()
    X_red_s = scaler_r.fit_transform(X_reduced)
    X_test_rs = scaler_r.transform(X_test)
    pca_r = PCA(n_components=19, random_state=42)
    X_red_pca = pca_r.fit_transform(X_red_s)
    X_test_rpca = pca_r.transform(X_test_rs)

    print(f"Refitting best on REDUCED train ({len(X_reduced)} samples)...")
    t0 = time.perf_counter()
    m_red = GeoXGBClassifier(**best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_red.fit(X_red_pca, y_reduced.astype(float))
    fit_time_r = time.perf_counter() - t0
    proba_red = m_red.predict_proba(X_test_rpca)[:, 1]
    test_auc_red = roc_auc_score(y_test, proba_red)
    cr_red = m_red._cpp_model.convergence_round()
    print(f"Test AUC (reduced train): {test_auc_red:.5f} | conv@{cr_red} | {fit_time_r:.1f}s")

    # References
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train_pca, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_pca)[:, 1])

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  HVRT-reduced {TARGET_N} → full train test AUC: {test_auc_full:.5f}")
    print(f"  HVRT-reduced {TARGET_N} → reduced train test AUC: {test_auc_red:.5f}")
    print(f"  Prior best (PCA+LogLoss 10k sub): 0.90493")
    print(f"  XGBoost+PCA full train:            {xgb_auc:.5f}")
    print(f"  XGBoost raw full train:            0.91610")
