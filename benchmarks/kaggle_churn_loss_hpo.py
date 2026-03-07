"""
HPO comparison of classification loss functions on Kaggle Churn.

Runs 200-trial Optuna HPO for each of:
  1. Log-Loss (standard cross-entropy)       — GeoXGBClassifier
  2. Gini impurity loss                       — GeoXGBGiniClassifier
  3. Focal Loss (gamma=1,2,3 searched)        — GeoXGBFocalClassifier
  4. Exponential Loss (AdaBoost)              — GeoXGBExpClassifier
  5. Squared Hinge Loss (margin-based)        — GeoXGBHingeClassifier

All use y_weight=0, HVRT bins=32, same search space.
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


def make_objective(ClsClass, X, y, splits, extra_params=None):
    """Create an Optuna objective for a given classifier class."""
    def objective(trial):
        params = {
            "n_rounds":       trial.suggest_int("n_rounds", 500, 5000, step=500),
            "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            "max_depth":      trial.suggest_int("max_depth", 2, 7),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.3, 0.99),
            "refit_interval": trial.suggest_categorical("refit_interval",
                                  [25, 50, 100, 200, 300, 500]),
            "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.6),
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
                "method", ["variance_ordered", "orthant_stratified"]),
            "variance_weighted": trial.suggest_categorical(
                "variance_weighted", [True, False]),
            "class_weight":   trial.suggest_categorical("class_weight",
                                  [None, "balanced"]),
            "convergence_tol": 0.01,
            "random_state":   42,
            "sample_block_n": "auto",
        }
        if extra_params:
            params.update(extra_params)

        scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, va_idx in splits:
                m = ClsClass(**params)
                m.fit(X[tr_idx], y[tr_idx].astype(float))
                proba = m.predict_proba(X[va_idx])[:, 1]
                scores.append(roc_auc_score(y[va_idx], proba))
        return float(np.mean(scores))
    return objective


def make_focal_objective(X, y, splits):
    """Focal loss objective with gamma as a hyperparameter."""
    from geoxgb import GeoXGBFocalClassifier

    def objective(trial):
        gamma = trial.suggest_categorical("gamma", [0.5, 1.0, 2.0, 3.0, 5.0])
        params = {
            "gamma":          gamma,
            "n_rounds":       trial.suggest_int("n_rounds", 500, 5000, step=500),
            "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            "max_depth":      trial.suggest_int("max_depth", 2, 7),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.3, 0.99),
            "refit_interval": trial.suggest_categorical("refit_interval",
                                  [25, 50, 100, 200, 300, 500]),
            "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.6),
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
                "method", ["variance_ordered", "orthant_stratified"]),
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
                m = GeoXGBFocalClassifier(**params)
                m.fit(X[tr_idx], y[tr_idx].astype(float))
                proba = m.predict_proba(X[va_idx])[:, 1]
                scores.append(roc_auc_score(y[va_idx], proba))
        return float(np.mean(scores))
    return objective


if __name__ == "__main__":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
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

    from geoxgb import (
        GeoXGBClassifier, GeoXGBGiniClassifier, GeoXGBFocalClassifier,
        GeoXGBExpClassifier, GeoXGBHingeClassifier,
    )

    # ── Run HPO for each loss ──────────────────────────────────────────────
    results = {}

    loss_configs = [
        ("LogLoss",  GeoXGBClassifier,     None,                    False),
        ("Gini",     GeoXGBGiniClassifier,  None,                    False),
        ("Focal",    GeoXGBFocalClassifier, None,                    True),   # special: gamma searched
        ("ExpLoss",  GeoXGBExpClassifier,   None,                    False),
        ("SqHinge",  GeoXGBHingeClassifier, None,                    False),
    ]

    for loss_name, Cls, extra, is_focal in loss_configs:
        print(f"\n{'='*60}")
        print(f"HPO: {loss_name} ({N_TRIALS} trials)")
        print(f"{'='*60}")

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=30)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        if is_focal:
            obj = make_focal_objective(X_hpo, y_hpo, splits)
        else:
            obj = make_objective(Cls, X_hpo, y_hpo, splits, extra)

        t0 = time.perf_counter()
        study.optimize(obj, n_trials=N_TRIALS)
        hpo_time = time.perf_counter() - t0

        print(f"  HPO time: {hpo_time:.0f}s ({hpo_time/N_TRIALS:.1f}s/trial)")
        print(f"  Best CV AUC: {study.best_value:.5f}")

        bp = dict(study.best_params)
        print(f"  Best params: {bp}")

        # Evaluate on test set
        bp["y_weight"] = 0.0
        bp["convergence_tol"] = 0.01
        bp["random_state"] = 42
        bp["sample_block_n"] = "auto"

        t0 = time.perf_counter()
        m = Cls(**bp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train, y_train.astype(float))
        fit_time = time.perf_counter() - t0
        proba = m.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
        cr = m._cpp_model.convergence_round()
        print(f"  Test AUC: {test_auc:.5f}  conv@{cr}  ({fit_time:.1f}s)")

        results[loss_name] = {
            "cv_auc": study.best_value,
            "test_auc": test_auc,
            "best_params": bp,
            "hpo_time": hpo_time,
            "conv_round": cr,
        }

    # ── XGBoost reference ────────────────────────────────────────────────
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        tree_method="hist", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY: Loss Function Comparison (Kaggle Churn)")
    print(f"{'='*60}")
    print(f"{'Loss':12s} {'CV AUC':>10s} {'Test AUC':>10s} {'Conv':>6s} {'Time':>6s}")
    print("-" * 50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auc"]):
        cr_str = str(r["conv_round"]) if r["conv_round"] >= 0 else "full"
        print(f"{name:12s} {r['cv_auc']:10.5f} {r['test_auc']:10.5f} {cr_str:>6s} {r['hpo_time']:5.0f}s")
    print(f"{'XGBoost':12s} {'—':>10s} {xgb_auc:10.5f} {'—':>6s}")
