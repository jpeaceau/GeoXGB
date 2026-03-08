"""
GeoXGBOptimizer
===============
Optuna TPE hyperparameter search for GeoXGBRegressor / GeoXGBClassifier.

Mirrors the HVRTOptimizer API and design philosophy:
  - Categorical search space (discrete option lists, not continuous ranges)
  - Warm-start trial 0 = task-specific defaults (guarantees HPO >= baseline)
  - Optional dependency guard: optuna only required inside .fit()
  - Exposes best_params_, best_score_, best_model_, study_ after fitting

Task-specific search spaces
----------------------------
Regression (task='regression'):
  n_rounds, learning_rate, max_depth, refit_interval, y_weight

Classification (task='classification'):
  n_rounds, learning_rate, max_depth, refit_interval, y_weight,
  class_weight, expand_ratio

  ``class_weight`` ('balanced' or None) corrects gradient signal for
  imbalanced classes; ``expand_ratio`` controls how much extra synthetic
  data is generated per HVRT resampling step.  Both are irrelevant for
  regression and excluded from that search space.

All other GeoXGB parameters remain at their defaults unless overridden
via keyword arguments to .fit().

Install
-------
    pip install optuna          # base
    pip install geoxgb[optimizer]  # if installed from PyPI
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


def _require_optuna():
    """Deferred import so missing optuna does not break 'from geoxgb import ...'."""
    try:
        import optuna
        return optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for GeoXGBOptimizer. "
            "Install it with:  pip install optuna"
        ) from exc


class GeoXGBOptimizer:
    """
    Optuna TPE hyperparameter optimizer for GeoXGBRegressor / GeoXGBClassifier.

    Runs an Optuna study that searches over core boosting parameters via
    Tree-structured Parzen Estimator (TPE) sampling.  The search space is
    task-specific: classification adds ``class_weight`` and ``expand_ratio``
    which are irrelevant for regression.

    All other GeoXGB parameters (HVRT geometry, noise settings, etc.) remain
    at their defaults unless overridden through ``fit(**fixed_params)``.

    Parameters
    ----------
    task : 'auto' | 'regression' | 'classification', default='auto'
        Task type.  'auto' detects from y: <= 20 unique values ->
        classification, otherwise regression.
    n_trials : int, default=50
        Number of Optuna trials (including the warm-start trial 0).
    cv : int, default=3
        Cross-validation folds used to evaluate each trial.
    n_jobs : int, default=1
        Parallel Optuna workers.  Values > 1 require the geoxgb package to
        be importable from subprocess workers (i.e. installed via pip, not
        only on sys.path).  Keep at 1 when running from an editable install
        on Windows.
    random_state : int, default=42
        Seed for both Optuna's TPE sampler and GeoXGB's random_state.
    verbose : bool, default=False
        If True, show Optuna trial-level progress logs.

    Attributes (set after .fit())
    ------------------------------
    best_params_  : dict  -- best found parameter values
    best_score_   : float -- best mean CV score (AUC for clf, R^2 for reg)
    best_model_   : GeoXGBRegressor | GeoXGBClassifier
                            refit on full training data with best params
    study_        : optuna.Study -- raw study for diagnostic plots
    task_         : str   -- 'regression' or 'classification'

    Examples
    --------
    >>> from geoxgb import GeoXGBOptimizer
    >>> opt = GeoXGBOptimizer(n_trials=50, cv=3)
    >>> opt.fit(X_train, y_train)
    >>> y_pred = opt.predict(X_test)               # regression
    >>> y_prob = opt.predict_proba(X_test)[:, 1]   # classification
    >>> print(opt.best_params_, opt.best_score_)
    >>> opt.study_.trials_dataframe()              # Optuna diagnostics
    """

    # ------------------------------------------------------------------
    # Regression: warm-start defaults + search space
    #
    # Warm-start defaults match GeoXGBRegressor constructor defaults so
    # trial 0 is guaranteed to be at least as good as the out-of-box model.
    #
    # Search space covers the empirically-validated ranges from 2 000+
    # Optuna TPE trials across diabetes, Friedman-1, California Housing,
    # and Kaggle churn benchmarks:
    #   - learning_rate: log-spaced 0.003–0.1 (wide range for diverse datasets)
    #   - max_depth: 2–7 (deeper trees help high-d and large-n)
    #   - reduce_ratio: 0.3–0.95 (large dataset-dependent variation)
    #   - refit_interval: 10–500 (diabetes optimal=200, Friedman-1=10)
    #   - expand_ratio: 0.0–0.5 (secondary; most impactful on small n)
    #   - y_weight: 0.1–0.8 (classifier benefits from higher values)
    #   - hvrt_min_samples_leaf: controls partition granularity
    # ------------------------------------------------------------------

    _DEFAULTS_REGRESSION = {
        "n_rounds":       1000,
        "learning_rate":  0.02,
        "max_depth":      3,
        "reduce_ratio":   0.8,
        "refit_interval": 50,
        "expand_ratio":   0.1,
        "y_weight":       0.25,
        "colsample_bytree": 1.0,
        "hvrt_min_samples_leaf": None,
    }

    _SEARCH_SPACE_REGRESSION = {
        "n_rounds":       [500, 1000, 1500, 2000, 3000, 4000],
        "learning_rate":  [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth":      [2, 3, 4, 5, 6, 7],
        "reduce_ratio":   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "refit_interval": [10, 20, 50, 100, 200, 300, 500],
        "expand_ratio":   [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "y_weight":       [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "hvrt_min_samples_leaf": [None, 5, 10, 20, 30],
    }

    # ------------------------------------------------------------------
    # Classification: warm-start defaults + search space
    # Adds class_weight and expand_ratio (irrelevant for regression).
    # ------------------------------------------------------------------

    _DEFAULTS_CLASSIFICATION = {
        "n_rounds":       1000,
        "learning_rate":  0.02,
        "max_depth":      3,
        "reduce_ratio":   0.8,
        "refit_interval": 50,
        "expand_ratio":   0.1,
        "y_weight":       0.25,
        "colsample_bytree": 0.8,
        "class_weight":   None,
        "hvrt_min_samples_leaf": None,
    }

    _SEARCH_SPACE_CLASSIFICATION = {
        "n_rounds":       [500, 1000, 1500, 2000, 3000, 4000],
        "learning_rate":  [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth":      [2, 3, 4, 5, 6, 7],
        "reduce_ratio":   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "refit_interval": [10, 20, 50, 100, 200, 300, 500],
        "expand_ratio":   [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "y_weight":       [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "class_weight":   [None, "balanced"],
        "hvrt_min_samples_leaf": [None, 5, 10, 20, 30],
    }

    # Applied to every trial: convergence_tol enables early stopping.
    _TRIAL_DEFAULTS = {
        "convergence_tol": 0.01,
    }

    # Maximum samples used for HPO CV evaluation.  Datasets larger than
    # this are subsampled (stratified for classification) to keep each
    # trial fast.  The best params are then refit on the full dataset.
    _MAX_HPO_SAMPLES = 10_000

    def __init__(
        self,
        task="auto",
        n_trials=50,
        cv=3,
        n_jobs=1,
        random_state=42,
        verbose=False,
    ):
        self.task = task
        self.n_trials = n_trials
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y, **fixed_params):
        """
        Run Optuna TPE search on (X, y).

        Parameters
        ----------
        X : array-like, shape (n, p)
        y : array-like, shape (n,)
        **fixed_params
            Extra GeoXGB parameters passed verbatim to every trial
            (not searched).  These override defaults.
            Example: ``opt.fit(X, y, n_jobs=4)`` runs each trial with
            4 OvR parallel workers (classification only).

        Returns
        -------
        self
        """
        optuna = _require_optuna()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        task = self._detect_task(y) if self.task == "auto" else self.task
        self.task_ = task

        # Select task-specific defaults and search space
        if task == "classification":
            defaults     = dict(self._DEFAULTS_CLASSIFICATION)
            search_space = dict(self._SEARCH_SPACE_CLASSIFICATION)
        else:
            defaults     = dict(self._DEFAULTS_REGRESSION)
            search_space = dict(self._SEARCH_SPACE_REGRESSION)

        # Subsample to _MAX_HPO_SAMPLES for speed.  Stratified for
        # classification to preserve class distribution.
        n_samples = len(X)
        X_full, y_full = X, y  # keep originals for final refit
        if n_samples > self._MAX_HPO_SAMPLES:
            rng = np.random.RandomState(self.random_state)
            if task == "classification":
                from sklearn.model_selection import train_test_split
                X, _, y, _ = train_test_split(
                    X, y, train_size=self._MAX_HPO_SAMPLES,
                    stratify=y, random_state=self.random_state,
                )
            else:
                idx = rng.choice(n_samples, self._MAX_HPO_SAMPLES, replace=False)
                X, y = X[idx], y[idx]
            n_samples = len(X)

        # Always enable block cycling at n >= 5000.  sample_block_n='auto'
        # is injected as a fixed param so every single trial uses it —
        # running on the full dataset is never an option above this threshold.
        if n_samples >= 5_000 and 'sample_block_n' not in fixed_params:
            fixed_params = {**fixed_params, 'sample_block_n': 'auto'}

        from geoxgb import GeoXGBClassifier, GeoXGBRegressor
        model_cls = GeoXGBClassifier if task == "classification" else GeoXGBRegressor
        scorer    = self._auc_scorer   if task == "classification" else self._r2_scorer

        # Build fixed CV splits (same across all trials for fair comparison)
        if task == "classification":
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                 random_state=self.random_state)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=self.cv, shuffle=True,
                       random_state=self.random_state)
            splits = list(kf.split(X))

        # Capture for closure
        _X, _y       = X, y
        _splits      = splits
        _fixed       = fixed_params
        _rs          = self.random_state
        _space       = search_space

        def objective(trial):
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in _space.items()
            }
            run_params = {
                **params,
                "random_state": _rs,
                **self._TRIAL_DEFAULTS,  # convergence_tol always enabled
                **_fixed,                # user-supplied overrides last
            }
            fold_scores = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr_idx, val_idx in _splits:
                    m = model_cls(**run_params)
                    m.fit(_X[tr_idx], _y[tr_idx])
                    fold_scores.append(scorer(m, _X[val_idx], _y[val_idx]))
            return float(np.mean(fold_scores))

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(direction="maximize", sampler=sampler)

        # Warm start: trial 0 = task-specific defaults
        study.enqueue_trial(
            {name: defaults[name] for name in search_space}
        )

        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        self.study_       = study
        self.best_params_ = dict(study.best_params)
        self.best_score_  = float(study.best_value)

        # Refit best config on full training data (not the HPO subsample)
        best_run = {
            **self.best_params_,
            "random_state": self.random_state,
            **fixed_params,
        }
        self.best_model_ = model_cls(**best_run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.best_model_.fit(X_full, y_full)

        return self

    def predict(self, X):
        """Predict using the best model (regression: values; classifier: class labels)."""
        self._check_fitted()
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        """Return class probabilities (GeoXGBClassifier only)."""
        self._check_fitted()
        if self.task_ != "classification":
            raise RuntimeError("predict_proba is only available for classification tasks.")
        return self.best_model_.predict_proba(X)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_task(y):
        return "classification" if len(np.unique(y)) <= 20 else "regression"

    @staticmethod
    def _r2_scorer(model, X, y):
        return float(r2_score(y, model.predict(X)))

    @staticmethod
    def _auc_scorer(model, X, y):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))

    def _check_fitted(self):
        if not hasattr(self, "best_model_"):
            raise RuntimeError("GeoXGBOptimizer has not been fitted. Call .fit() first.")

    def __repr__(self):
        fitted = hasattr(self, "best_model_")
        s = f"fitted, best_score={self.best_score_:.4f}" if fitted else "unfitted"
        return (
            f"GeoXGBOptimizer({s}, n_trials={self.n_trials}, "
            f"cv={self.cv}, task={getattr(self, 'task_', self.task)})"
        )
