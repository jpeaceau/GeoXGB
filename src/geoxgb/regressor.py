import numpy as np

from geoxgb._base import _GeoXGBBase, _resolve_auto_block

_REFIT_NOISE_FLOOR = 0.05  # kept for external importers


class GeoXGBRegressor(_GeoXGBBase):
    """
    Geometry-aware gradient boosting regressor.

    Supports two loss functions via the ``loss`` parameter:

    * ``'squared_error'`` (default) — standard L2 gradient boosting.
      Gradients are residuals ``y − ŷ``; leaf values are means.
      Optimal defaults: ``max_depth=3``, ``y_weight=0.9``,
      ``method='variance_ordered'``.

    * ``'absolute_error'`` — L1 gradient boosting.
      Gradients are signs ``sign(y − ŷ) ∈ {−1, 0, +1}``; trees minimise
      MAE directly.  For best results, also set:
      ``max_depth=4``, ``y_weight=0.5``,
      ``method='orthant_stratified'``,
      ``adaptive_reduce_ratio=True``.

    .. note:: **HPO is strongly recommended.**
        ``learning_rate`` and ``max_depth`` are the two most sensitive
        parameters and interact strongly.  Use ``GeoXGBOptimizer`` or any
        Optuna/sklearn HPO tool for production models.

    Parameters
    ----------
    loss : 'squared_error' | 'absolute_error', default='squared_error'
        Loss function.  ``'absolute_error'`` uses sign gradients (L1
        boosting) and always runs on the Python backend.  When switching
        to ``'absolute_error'``, consider also setting
        ``method='orthant_stratified'``, ``max_depth=4``,
        ``y_weight=0.5``, and ``adaptive_reduce_ratio=True`` for best
        results.
    n_rounds : int, default=500
        Number of boosting rounds.
    learning_rate : float, default=0.05
        Shrinkage per tree.
    max_depth : int, default=3
        Maximum depth of each weak learner.
    min_samples_leaf : int, default=3
        Minimum samples per leaf in each weak learner.
    n_partitions : int or None, default=None
        Target number of partitions (None = auto).
    reduce_ratio : float, default=0.9
        Fraction of samples retained per resampling round.
    expand_ratio : float, default=0.1
        Fraction of n to generate as synthetic samples.
    y_weight : float, default=0.9
        Partition blend: 0 = geometry-only, 1 = fully gradient-driven.
    method : str, default='variance_ordered'
        Reduction strategy.  Use ``'orthant_stratified'`` with
        ``loss='absolute_error'``.
    refit_interval : int or None, default=10
        Re-fit partition tree every N rounds.
    auto_noise : bool, default=False
        Enable noise-modulation gating.  Disabled for regression
        (noise injection hurts regression accuracy).  When enabled,
        ``noise_guard`` and ``refit_noise_floor`` are automatically
        activated.
    partitioner : str, default='hvrt'
        Partition geometry.  See parameters reference for options.
    adaptive_reduce_ratio : bool, default=False
        Dynamically increase reduce_ratio for heavy-tailed gradients.
        Recommended with ``loss='absolute_error'``.
    single_pass : bool, default=False
        When True, each training sample is seen at most once across the
        entire boosting run.  The ``reduce_ratio`` is automatically set to
        ``refit_interval / n_rounds`` so that the total samples consumed
        across all refit cycles equals the training set size.  This makes
        memorization physically impossible and provides strong robustness
        to label noise and data drift.  Requires tuning ``refit_interval``
        to control batch size: fewer refits → larger batches per cycle.
    sample_block_n : int, 'auto', or None, default='auto'
        Block size for epoch-based data cycling.  When active and n > block size,
        the full dataset is divided into non-overlapping blocks (deterministic
        permutation seeded by random_state).  At each refit_interval the boosting
        loop advances to the next block, cycling through epochs, giving progressive
        exposure to the full dataset.  All per-round costs scale with block size.
        ``'auto'`` (default): ``max(2000, int(sqrt(n) * 15 * ri_scale))`` when
        n > 5 000, else disabled.  ``ri_scale = clamp(refit_interval / 50,
        0.5, 2.0)`` — fewer refits → proportionally larger blocks to maintain
        data coverage; more refits → smaller blocks for speed.
        ``None``: disabled.  Integer: explicit block size.
        Included in HPO via ``GeoXGBOptimizer`` when n is large enough.
    random_state : int, default=42
    """

    def __init__(
        self,
        loss='squared_error',
        n_rounds=500,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        n_partitions=None,
        hvrt_min_samples_leaf=None,
        reduce_ratio=0.9,
        expand_ratio=0.1,
        y_weight=0.9,
        method="variance_ordered",
        refit_interval=10,
        auto_noise=False,
        auto_expand=True,
        min_train_samples=5000,
        random_state=42,
        n_jobs=1,
        generation_strategy="simplex_mixup",
        convergence_tol=None,
        feature_weights=None,
        tree_splitter="random",
        hvrt_params=None,
        partitioner='hvrt',
        adaptive_reduce_ratio=False,
        single_pass=False,
        sample_block_n='auto',
        n_bins=64,
        max_resample_n=None,
        sample_without_replacement=True,
        colsample_bytree=1.0,
        predict_stride=1,
        track_partition_trajectory=True,
        boost_optimizer='partition_adaptive',
        momentum_beta=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    ):
        if loss not in ('squared_error', 'absolute_error'):
            raise ValueError(
                f"loss must be 'squared_error' or 'absolute_error', got {loss!r}"
            )
        self.loss = loss
        super().__init__(
            n_rounds=n_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_partitions=n_partitions,
            hvrt_min_samples_leaf=hvrt_min_samples_leaf,
            reduce_ratio=reduce_ratio,
            expand_ratio=expand_ratio,
            y_weight=y_weight,
            method=method,
            refit_interval=refit_interval,
            auto_noise=auto_noise,
            auto_expand=auto_expand,
            min_train_samples=min_train_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            generation_strategy=generation_strategy,
            convergence_tol=convergence_tol,
            feature_weights=feature_weights,
            tree_splitter=tree_splitter,
            hvrt_params=hvrt_params,
            partitioner=partitioner,
            adaptive_reduce_ratio=adaptive_reduce_ratio,
            single_pass=single_pass,
            sample_block_n=sample_block_n,
            n_bins=n_bins,
            max_resample_n=max_resample_n,
            sample_without_replacement=sample_without_replacement,
            colsample_bytree=colsample_bytree,
            predict_stride=predict_stride,
            track_partition_trajectory=track_partition_trajectory,
            boost_optimizer=boost_optimizer,
            momentum_beta=momentum_beta,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
        )

    def fit(self, X, y, feature_types=None):
        """
        Fit the regressor using the C++ backend.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_types : list of str, optional
            'continuous' or 'categorical' per column.  Categorical columns are
            label-encoded to integers before passing to the C++ backend.

        Returns
        -------
        self
        """
        from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor as _CppReg
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._n_features = X.shape[1]
        self._feature_types = feature_types
        X = self._encode_features(X, feature_types, fitting=True)

        params = self._resolve_params(len(X))
        params['loss'] = self.loss
        self._cpp_model = _CppReg(make_cpp_config(**params))
        self._cpp_model.fit(X, y)
        self._X_train = X
        self._is_fitted = True
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr
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
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)
        return self._cpp_model.predict(X)


def GeoXGBMAERegressor(
    loss='absolute_error',
    max_depth=4,
    y_weight=0.5,
    method='orthant_stratified',
    adaptive_reduce_ratio=True,
    **kwargs,
):
    """
    Backward-compatible alias for ``GeoXGBRegressor(loss='absolute_error')``.

    Returns a ``GeoXGBRegressor`` pre-configured with MAE-optimal defaults:
    ``max_depth=4``, ``y_weight=0.5``, ``method='orthant_stratified'``,
    ``adaptive_reduce_ratio=True``.  All parameters can be overridden.

    Prefer constructing ``GeoXGBRegressor(loss='absolute_error', ...)``
    directly in new code.
    """
    return GeoXGBRegressor(
        loss=loss,
        max_depth=max_depth,
        y_weight=y_weight,
        method=method,
        adaptive_reduce_ratio=adaptive_reduce_ratio,
        **kwargs,
    )
