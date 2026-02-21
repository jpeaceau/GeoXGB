import numpy as np

from geoxgb._base import _GeoXGBBase


class GeoXGBRegressor(_GeoXGBBase):
    """
    Geometry-aware gradient boosting regressor.

    Optimises squared-error loss with HVRT-based sample curation.

    Parameters
    ----------
    n_rounds : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage per tree.
    max_depth : int, default=6
        Maximum depth of each weak learner.
    min_samples_leaf : int, default=5
        Minimum samples per leaf in each weak learner (DecisionTree).
    hvrt_min_samples_leaf : int or None, default=None
        Minimum samples per leaf in the HVRT partition tree. None = HVRT
        auto-tunes using its default formula. Set explicitly to control the
        partitioning granularity: smaller values (20-30) give finer partitions
        for richer interpretability; larger values (50+) produce more stable
        KDE fits for synthetic expansion.
    n_partitions : int or None, default=None
        Target number of HVRT partitions. None = auto-tuned based on dataset
        size and feature count. When set explicitly, HVRT will attempt to
        produce this many partitions, subject to min_samples_leaf constraints.
        Increase for finer partition-level reporting; decrease for more stable
        KDE expansion.
    reduce_ratio : float, default=0.7
        Fraction of samples to keep via FPS (before noise modulation).
    expand_ratio : float, default=0.0
        Fraction of n to generate as synthetic samples. 0 = disabled.
    y_weight : float, default=0.5
        HVRT blend: 0 = unsupervised, 1 = fully y-driven partitioning.
    method : str, default='fps'
        HVRT selection method: 'fps', 'medoid_fps', 'variance_ordered',
        'stratified'.
    variance_weighted : bool, default=True
        Allocate budgets by partition variance (tail-preserving).
    bandwidth : float, default=0.5
        KDE bandwidth for expansion.
    refit_interval : int or None, default=10
        Refit partitions on residuals every N rounds. None disables.
    auto_noise : bool, default=True
        Auto-detect noise and modulate resampling aggressiveness.
    random_state : int, default=42
    """

    def fit(self, X, y, feature_types=None):
        """
        Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_types : list of str, optional
            'continuous' or 'categorical' per column.
            None means all continuous.

        Returns
        -------
        self
        """
        self._feature_types = feature_types
        self._resample_history = []
        self._fit_boosting(X, y)
        return self

    def predict(self, X):
        """
        Predict continuous target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return self._raw_predict(X)

    def _compute_init_prediction(self, y):
        return float(np.mean(y))

    def _compute_gradients(self, y, predictions):
        # Negative gradient of squared-error = residual
        return y - predictions
