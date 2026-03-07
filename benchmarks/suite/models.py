"""
Model wrappers for GeoXGB and XGBoost.

Each wrapper provides a uniform interface:
    .fit(X_train, y_train) -> self
    .predict(X_test) -> y_pred
    .predict_proba(X_test) -> y_proba  (classification only)
    .name -> str
    .save(path) -> None  (serialize fitted model for later inspection)
"""
from __future__ import annotations
import time
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# GeoXGB wrappers
# ---------------------------------------------------------------------------

class GeoXGBDefaultModel:
    name = "geoxgb_default"

    def __init__(self, task: str):
        self.task = task
        self.model_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        from geoxgb import GeoXGBRegressor, GeoXGBClassifier
        if self.task == "regression":
            self.model_ = GeoXGBRegressor(random_state=42)
        else:
            self.model_ = GeoXGBClassifier(random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            self.model_.fit(X, y)
            self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(path))


class GeoXGBHPOModel:
    name = "geoxgb_hpo"

    def __init__(self, task: str, n_trials: int = 50):
        self.task = task
        self.n_trials = n_trials
        self.model_ = None
        self.optimizer_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        from geoxgb import GeoXGBOptimizer
        opt_task = "regression" if self.task == "regression" else "classification"
        self.optimizer_ = GeoXGBOptimizer(
            task=opt_task, n_trials=self.n_trials, cv=3,
            random_state=42, verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            self.optimizer_.fit(X, y)
            self.fit_time_ = time.perf_counter() - t0
        self.model_ = self.optimizer_.best_model_
        return self

    def predict(self, X):
        return self.optimizer_.predict(X)

    def predict_proba(self, X):
        return self.optimizer_.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(path))
        # Save HPO metadata alongside
        import json
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_params": self.optimizer_.best_params_,
                        "best_score": self.optimizer_.best_score_}, f, indent=2,
                       default=str)


# ---------------------------------------------------------------------------
# XGBoost wrappers
# ---------------------------------------------------------------------------

class XGBoostDefaultModel:
    name = "xgboost_default"

    def __init__(self, task: str):
        self.task = task
        self.model_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        import xgboost as xgb
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(
                random_state=42, n_jobs=1, verbosity=0,
            )
        elif self.task == "binary":
            self.model_ = xgb.XGBClassifier(
                random_state=42, n_jobs=1, verbosity=0,
                eval_metric="logloss",
            )
        else:
            self.model_ = xgb.XGBClassifier(
                random_state=42, n_jobs=1, verbosity=0,
                eval_metric="mlogloss",
            )
        t0 = time.perf_counter()
        self.model_.fit(X, y)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path.with_suffix(".json")))


class XGBoostHPOModel:
    name = "xgboost_hpo"

    _SEARCH_SPACE = {
        "n_estimators": [100, 300, 500, 1000, 2000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }

    def __init__(self, task: str, n_trials: int = 50):
        self.task = task
        self.n_trials = n_trials
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        import optuna
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        is_clf = self.task != "regression"

        # Subsample large datasets for HPO speed
        max_hpo = 10_000
        X_hpo, y_hpo = X, y
        if len(X) > max_hpo:
            rng = np.random.RandomState(42)
            if is_clf:
                from sklearn.model_selection import train_test_split
                X_hpo, _, y_hpo, _ = train_test_split(
                    X, y, train_size=max_hpo, stratify=y, random_state=42,
                )
            else:
                idx = rng.choice(len(X), max_hpo, replace=False)
                X_hpo, y_hpo = X[idx], y[idx]

        if is_clf:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scoring = "roc_auc" if self.task == "binary" else "roc_auc_ovr"
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scoring = "r2"

        def objective(trial):
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in self._SEARCH_SPACE.items()
            }
            if self.task == "regression":
                m = xgb.XGBRegressor(**params, random_state=42,
                                      n_jobs=1, verbosity=0)
            elif self.task == "binary":
                m = xgb.XGBClassifier(**params, random_state=42,
                                       n_jobs=1, verbosity=0,
                                       eval_metric="logloss")
            else:
                m = xgb.XGBClassifier(**params, random_state=42,
                                       n_jobs=1, verbosity=0,
                                       eval_metric="mlogloss")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(m, X_hpo, y_hpo, cv=cv,
                                          scoring=scoring, error_score="raise")
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)

        # Refit on full data
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(**self.best_params_, random_state=42,
                                            n_jobs=1, verbosity=0)
        elif self.task == "binary":
            self.model_ = xgb.XGBClassifier(**self.best_params_, random_state=42,
                                             n_jobs=1, verbosity=0,
                                             eval_metric="logloss")
        else:
            self.model_ = xgb.XGBClassifier(**self.best_params_, random_state=42,
                                             n_jobs=1, verbosity=0,
                                             eval_metric="mlogloss")
        self.model_.fit(X, y)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path.with_suffix(".json")))
        # Also save best params alongside
        import json
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_params": self.best_params_,
                        "best_score": self.best_score_}, f, indent=2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_models(mode: str, task: str, n_trials: int = 50) -> list:
    """Return model wrappers for the given mode and task."""
    if mode == "default":
        return [GeoXGBDefaultModel(task), XGBoostDefaultModel(task)]
    elif mode == "hpo":
        return [GeoXGBHPOModel(task, n_trials), XGBoostHPOModel(task, n_trials)]
    else:
        return [
            GeoXGBDefaultModel(task), XGBoostDefaultModel(task),
            GeoXGBHPOModel(task, n_trials), XGBoostHPOModel(task, n_trials),
        ]
