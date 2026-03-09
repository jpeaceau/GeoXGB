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
import os
import time
import warnings
from pathlib import Path

import numpy as np


def _get_njobs() -> int:
    """Get thread budget set by the runner (or all cores if unset)."""
    return int(os.environ.get("GEOXGB_BENCH_NJOBS", os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# GeoXGB wrappers
# ---------------------------------------------------------------------------

class GeoXGBDefaultModel:
    name = "geoxgb_default"

    def __init__(self, task: str, random_state: int = 42):
        self.task = task
        self.random_state = random_state
        self.model_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        from geoxgb import GeoXGBRegressor, GeoXGBClassifier
        if self.task == "regression":
            self.model_ = GeoXGBRegressor(random_state=self.random_state)
        else:
            self.model_ = GeoXGBClassifier(random_state=self.random_state)
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


# ---------------------------------------------------------------------------
# GeoXGB grid-search HPO (random sampling from categorical grid)
# ---------------------------------------------------------------------------

# Shared search space — categorical lists for random grid search.
_GEOXGB_SEARCH_SPACE_REG = {
    "n_rounds":       [300, 500, 1000, 1500, 2000],
    "learning_rate":  [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
    "max_depth":      [2, 3, 4, 5, 6],
    "reduce_ratio":   [0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
    "refit_interval": [5, 10, 20, 50, 100],
    "expand_ratio":   [0.0, 0.05, 0.1, 0.2, 0.3],
    "y_weight":       [0.5, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "hvrt_min_samples_leaf": [None, 5, 10, 20],
    "boost_optimizer": ["standard", "momentum", "adam", "partition_adaptive"],
}

_GEOXGB_SEARCH_SPACE_CLF = {
    **_GEOXGB_SEARCH_SPACE_REG,
    "boost_optimizer": ["standard", "momentum", "adam", "newton", "partition_adaptive"],
}

# Defaults for warm-start trial 0 (match constructor defaults).
_GEOXGB_DEFAULTS_REG = {
    "n_rounds": 500, "learning_rate": 0.05, "max_depth": 3,
    "reduce_ratio": 0.9, "refit_interval": 10, "expand_ratio": 0.1,
    "y_weight": 0.9, "colsample_bytree": 1.0, "hvrt_min_samples_leaf": None,
    "boost_optimizer": "partition_adaptive",
}

_GEOXGB_DEFAULTS_CLF = {
    "n_rounds": 1000, "learning_rate": 0.2, "max_depth": 5,
    "reduce_ratio": 0.9, "refit_interval": 10, "expand_ratio": 0.1,
    "y_weight": 0.7, "colsample_bytree": 0.8, "hvrt_min_samples_leaf": None,
    "boost_optimizer": "partition_adaptive",
}


def _grid_search_cv(model_cls, search_space, defaults, X, y, n_trials,
                    random_state, scorer, is_clf, fixed_params=None):
    """Random grid search with 3-fold CV. Returns (best_params, best_score, all trials)."""
    from sklearn.model_selection import ParameterSampler, KFold, StratifiedKFold

    fixed_params = fixed_params or {}

    # Subsample large datasets for HPO speed
    max_hpo = 10_000
    X_hpo, y_hpo = X, y
    if len(X) > max_hpo:
        rng = np.random.RandomState(random_state)
        if is_clf:
            from sklearn.model_selection import train_test_split
            X_hpo, _, y_hpo, _ = train_test_split(
                X, y, train_size=max_hpo, stratify=y, random_state=random_state,
            )
        else:
            idx = rng.choice(len(X), max_hpo, replace=False)
            X_hpo, y_hpo = X[idx], y[idx]

    # Build CV splits
    if is_clf:
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_hpo, y_hpo))
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_hpo))

    # Generate trial configurations: trial 0 = defaults, rest = random samples
    # ParameterSampler draws n_iter random combos from the grid.
    sampled = list(ParameterSampler(search_space, n_iter=n_trials - 1,
                                     random_state=random_state))
    configs = [defaults] + sampled

    best_score = -np.inf
    best_params = None

    for cfg in configs:
        run_params = {
            **cfg,
            "random_state": random_state,
            "convergence_tol": 0.01,
            **fixed_params,
        }
        fold_scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, val_idx in splits:
                m = model_cls(**run_params)
                m.fit(X_hpo[tr_idx], y_hpo[tr_idx])
                fold_scores.append(scorer(m, X_hpo[val_idx], y_hpo[val_idx]))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(cfg)

    return best_params, best_score


def _r2_scorer(model, X, y):
    from sklearn.metrics import r2_score
    return float(r2_score(y, model.predict(X)))


def _auc_scorer(model, X, y):
    from sklearn.metrics import roc_auc_score
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


class GeoXGBHPOModel:
    """Grid-search HPO for GeoXGB (random sampling from categorical grid)."""
    name = "geoxgb_hpo"

    def __init__(self, task: str, n_trials: int = 50, random_state: int = 42):
        self.task = task
        self.n_trials = n_trials
        self.random_state = random_state
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.fit_time_ = 0.0
        self.extended_model_ = None
        self.extended_fit_time_ = 0.0

    def fit(self, X, y):
        from geoxgb import GeoXGBRegressor, GeoXGBClassifier

        is_clf = self.task != "regression"
        model_cls = GeoXGBClassifier if is_clf else GeoXGBRegressor
        scorer = _auc_scorer if is_clf else _r2_scorer
        space = _GEOXGB_SEARCH_SPACE_CLF if is_clf else _GEOXGB_SEARCH_SPACE_REG
        defaults = _GEOXGB_DEFAULTS_CLF if is_clf else _GEOXGB_DEFAULTS_REG

        # Block cycling for large datasets
        fixed = {}
        if len(X) >= 5_000:
            fixed["sample_block_n"] = "auto"

        t0 = time.perf_counter()
        self.best_params_, self.best_score_ = _grid_search_cv(
            model_cls, space, defaults, X, y, self.n_trials,
            self.random_state, scorer, is_clf, fixed_params=fixed,
        )

        # Refit best on full data
        best_run = {**self.best_params_, "random_state": self.random_state, **fixed}
        self.model_ = model_cls(**best_run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(X, y)
        self.fit_time_ = time.perf_counter() - t0

        # Extended refit: n_rounds=5000, no early stopping
        ext_params = {**self.best_params_, "n_rounds": 5000,
                      "convergence_tol": None, "random_state": self.random_state,
                      **fixed}
        self.extended_model_ = model_cls(**ext_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            self.extended_model_.fit(X, y)
            self.extended_fit_time_ = time.perf_counter() - t0

        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save(str(path))
        import json
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_params": self.best_params_,
                        "best_score": self.best_score_}, f, indent=2,
                       default=str)


class GeoXGBSinglePassHPOModel:
    """Grid-search HPO for GeoXGB with single_pass=True (forward-pass mode)."""
    name = "geoxgb_hpo_single_pass"

    # Single-pass search space: reduce_ratio is derived, so not searched.
    # refit_interval is critical (controls batch size).
    _SEARCH_SPACE_REG = {
        "n_rounds":       [300, 500, 1000, 1500, 2000],
        "learning_rate":  [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
        "max_depth":      [2, 3, 4, 5, 6],
        "refit_interval": [5, 10, 20, 50, 100],
        "expand_ratio":   [0.0, 0.05, 0.1, 0.2, 0.3],
        "y_weight":       [0.5, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "hvrt_min_samples_leaf": [None, 5, 10, 20],
        "boost_optimizer": ["standard", "momentum", "adam", "partition_adaptive"],
    }

    _SEARCH_SPACE_CLF = {
        **_SEARCH_SPACE_REG,
        "boost_optimizer": ["standard", "momentum", "adam", "newton", "partition_adaptive"],
    }

    _DEFAULTS_REG = {
        "n_rounds": 500, "learning_rate": 0.05, "max_depth": 3,
        "refit_interval": 10, "expand_ratio": 0.1,
        "y_weight": 0.9, "colsample_bytree": 1.0, "hvrt_min_samples_leaf": None,
        "boost_optimizer": "partition_adaptive",
    }

    _DEFAULTS_CLF = {
        "n_rounds": 1000, "learning_rate": 0.2, "max_depth": 5,
        "refit_interval": 10, "expand_ratio": 0.1,
        "y_weight": 0.7, "colsample_bytree": 0.8, "hvrt_min_samples_leaf": None,
        "boost_optimizer": "partition_adaptive",
    }

    def __init__(self, task: str, n_trials: int = 50, random_state: int = 42):
        self.task = task
        self.n_trials = n_trials
        self.random_state = random_state
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        from geoxgb import GeoXGBRegressor, GeoXGBClassifier

        is_clf = self.task != "regression"
        model_cls = GeoXGBClassifier if is_clf else GeoXGBRegressor
        scorer = _auc_scorer if is_clf else _r2_scorer
        space = self._SEARCH_SPACE_CLF if is_clf else self._SEARCH_SPACE_REG
        defaults = self._DEFAULTS_CLF if is_clf else self._DEFAULTS_REG

        # single_pass is always on; reduce_ratio derived automatically
        fixed = {"single_pass": True}

        t0 = time.perf_counter()
        self.best_params_, self.best_score_ = _grid_search_cv(
            model_cls, space, defaults, X, y, self.n_trials,
            self.random_state, scorer, is_clf, fixed_params=fixed,
        )

        # Refit best on full data
        best_run = {**self.best_params_, "random_state": self.random_state, **fixed}
        self.model_ = model_cls(**best_run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        import json
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_params": self.best_params_,
                        "best_score": self.best_score_}, f, indent=2,
                       default=str)


# ---------------------------------------------------------------------------
# XGBoost wrappers
# ---------------------------------------------------------------------------

class XGBoostDefaultModel:
    name = "xgboost_default"

    def __init__(self, task: str, random_state: int = 42):
        self.task = task
        self.random_state = random_state
        self.model_ = None
        self.fit_time_ = 0.0

    def fit(self, X, y):
        import xgboost as xgb
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(
                random_state=self.random_state, n_jobs=_get_njobs(), verbosity=0,
            )
        elif self.task == "binary":
            self.model_ = xgb.XGBClassifier(
                random_state=self.random_state, n_jobs=_get_njobs(), verbosity=0,
                eval_metric="logloss",
            )
        else:
            self.model_ = xgb.XGBClassifier(
                random_state=self.random_state, n_jobs=_get_njobs(), verbosity=0,
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
        "n_estimators": [100, 300, 500, 1000],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }

    def __init__(self, task: str, n_trials: int = 50, random_state: int = 42):
        self.task = task
        self.n_trials = n_trials
        self.random_state = random_state
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
        rs = self.random_state

        # Subsample large datasets for HPO speed
        max_hpo = 10_000
        X_hpo, y_hpo = X, y
        if len(X) > max_hpo:
            rng = np.random.RandomState(rs)
            if is_clf:
                from sklearn.model_selection import train_test_split
                X_hpo, _, y_hpo, _ = train_test_split(
                    X, y, train_size=max_hpo, stratify=y, random_state=rs,
                )
            else:
                idx = rng.choice(len(X), max_hpo, replace=False)
                X_hpo, y_hpo = X[idx], y[idx]

        if is_clf:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
            scoring = "roc_auc" if self.task == "binary" else "roc_auc_ovr"
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=rs)
            scoring = "r2"

        def objective(trial):
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in self._SEARCH_SPACE.items()
            }
            if self.task == "regression":
                m = xgb.XGBRegressor(**params, random_state=rs,
                                      n_jobs=_get_njobs(), verbosity=0)
            elif self.task == "binary":
                m = xgb.XGBClassifier(**params, random_state=rs,
                                       n_jobs=_get_njobs(), verbosity=0,
                                       eval_metric="logloss")
            else:
                m = xgb.XGBClassifier(**params, random_state=rs,
                                       n_jobs=_get_njobs(), verbosity=0,
                                       eval_metric="mlogloss")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(m, X_hpo, y_hpo, cv=cv,
                                          scoring=scoring, error_score="raise")
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=rs)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=self.n_trials)

        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)

        # Refit on full data
        if self.task == "regression":
            self.model_ = xgb.XGBRegressor(**self.best_params_, random_state=rs,
                                            n_jobs=_get_njobs(), verbosity=0)
        elif self.task == "binary":
            self.model_ = xgb.XGBClassifier(**self.best_params_, random_state=rs,
                                             n_jobs=_get_njobs(), verbosity=0,
                                             eval_metric="logloss")
        else:
            self.model_ = xgb.XGBClassifier(**self.best_params_, random_state=rs,
                                             n_jobs=_get_njobs(), verbosity=0,
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
        import json
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_params": self.best_params_,
                        "best_score": self.best_score_}, f, indent=2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_models(mode: str, task: str, n_trials: int = 50,
               random_state: int = 42) -> list:
    """Return model wrappers for the given mode and task."""
    if mode == "default":
        return [GeoXGBDefaultModel(task, random_state),
                XGBoostDefaultModel(task, random_state)]
    elif mode == "hpo":
        return [GeoXGBHPOModel(task, n_trials, random_state),
                GeoXGBSinglePassHPOModel(task, n_trials, random_state),
                XGBoostHPOModel(task, n_trials, random_state)]
    else:
        return [
            GeoXGBDefaultModel(task, random_state),
            XGBoostDefaultModel(task, random_state),
            GeoXGBHPOModel(task, n_trials, random_state),
            GeoXGBSinglePassHPOModel(task, n_trials, random_state),
            XGBoostHPOModel(task, n_trials, random_state),
        ]
