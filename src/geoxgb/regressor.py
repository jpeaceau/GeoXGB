import numpy as np

from geoxgb._base import _GeoXGBBase, _resolve_auto_block

_REFIT_NOISE_FLOOR = 0.05  # kept for external importers


class GeoXGBRegressor(_GeoXGBBase):
    """
    Geometry-aware gradient boosting regressor.

    Supports two loss functions via the ``loss`` parameter:

    * ``'squared_error'`` (default) — standard L2 gradient boosting.
      Gradients are residuals ``y − ŷ``; leaf values are means.
      Optimal defaults: ``max_depth=3``, ``y_weight=0.25``,
      ``method='variance_ordered'``.

    * ``'absolute_error'`` — L1 gradient boosting.
      Gradients are signs ``sign(y − ŷ) ∈ {−1, 0, +1}``; trees minimise
      MAE directly.  For best results, also set:
      ``max_depth=4``, ``y_weight=0.5``,
      ``method='orthant_stratified'``,
      ``adaptive_reduce_ratio=True``.

    .. note:: **HPO is strongly recommended.**
        ``learning_rate`` and ``max_depth`` are the two most sensitive
        parameters and interact strongly.  The optimal regime is low
        learning_rate (0.010–0.020) paired with high ``n_rounds``
        (1 000–5 000) and shallow ``max_depth`` (2–3 for L2; 3–4 for L1).
        Use ``GeoXGBOptimizer`` or any Optuna/sklearn HPO tool for
        production models.

    Parameters
    ----------
    loss : 'squared_error' | 'absolute_error', default='squared_error'
        Loss function.  ``'absolute_error'`` uses sign gradients (L1
        boosting) and always runs on the Python backend.  When switching
        to ``'absolute_error'``, consider also setting
        ``method='orthant_stratified'``, ``max_depth=4``,
        ``y_weight=0.5``, and ``adaptive_reduce_ratio=True`` for best
        results.
    n_rounds : int, default=1000
        Number of boosting rounds.
    learning_rate : float, default=0.02
        Shrinkage per tree.  Optimal range 0.010–0.020.
    max_depth : int, default=3
        Maximum depth of each weak learner.
    min_samples_leaf : int, default=5
        Minimum samples per leaf in each weak learner.
    n_partitions : int or None, default=None
        Target number of partitions (None = auto).
    reduce_ratio : float, default=0.8
        Fraction of samples retained per resampling round.
    expand_ratio : float, default=0.1
        Fraction of n to generate as synthetic samples.
    y_weight : float, default=0.25
        Partition blend: 0 = geometry-only, 1 = fully gradient-driven.
    method : str, default='variance_ordered'
        Reduction strategy.  Use ``'orthant_stratified'`` with
        ``loss='absolute_error'``.
    refit_interval : int or None, default=50
        Re-fit partition tree every N rounds.
    auto_noise : bool, default=True
        Enable noise-modulation gating.
    noise_guard : bool, default=True
        Enable noise-guard veto on resampling.
    partitioner : str, default='hvrt'
        Partition geometry.  See parameters reference for options.
    adaptive_reduce_ratio : bool, default=False
        Dynamically increase reduce_ratio for heavy-tailed gradients.
        Recommended with ``loss='absolute_error'``.
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
    leave_last_block_out : bool, default=False
        When True, the last block of each epoch is never trained on and is
        stored as ``model.held_out_X_`` / ``model.held_out_y_`` for external
        validation or convergence monitoring.
    random_state : int, default=42
    """

    def __init__(
        self,
        loss='squared_error',
        n_rounds=1000,
        learning_rate=0.02,
        max_depth=3,
        min_samples_leaf=5,
        n_partitions=None,
        hvrt_min_samples_leaf=None,
        hvrt_max_samples_leaf=None,
        reduce_ratio=0.8,
        expand_ratio=0.1,
        y_weight=0.25,
        method="variance_ordered",
        variance_weighted=False,
        bandwidth="auto",
        refit_interval=50,
        auto_noise=True,
        auto_expand=True,
        min_train_samples=5000,
        random_state=42,
        lr_schedule=None,
        tree_criterion="squared_error",
        n_jobs=1,
        generation_strategy="simplex_mixup",
        adaptive_bandwidth=False,
        convergence_tol=None,
        feature_weights=None,
        assignment_strategy="auto",
        tree_splitter="random",
        refit_noise_floor=_REFIT_NOISE_FLOOR,
        noise_guard=True,
        hvrt_params=None,
        hvrt_tree_splitter=None,
        hvrt_auto_reduce_threshold=None,
        partitioner='hvrt',
        adaptive_reduce_ratio=False,
        sample_block_n='auto',
        leave_last_block_out=False,
        n_bins=64,
        max_resample_n=None,
        sample_without_replacement=True,
        colsample_bytree=1.0,
        predict_stride=1,
        grad_budget_weight=0.0,
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
            hvrt_max_samples_leaf=hvrt_max_samples_leaf,
            reduce_ratio=reduce_ratio,
            expand_ratio=expand_ratio,
            y_weight=y_weight,
            method=method,
            variance_weighted=variance_weighted,
            bandwidth=bandwidth,
            refit_interval=refit_interval,
            auto_noise=auto_noise,
            auto_expand=auto_expand,
            min_train_samples=min_train_samples,
            random_state=random_state,
            lr_schedule=lr_schedule,
            tree_criterion=tree_criterion,
            n_jobs=n_jobs,
            generation_strategy=generation_strategy,
            adaptive_bandwidth=adaptive_bandwidth,
            convergence_tol=convergence_tol,
            feature_weights=feature_weights,
            assignment_strategy=assignment_strategy,
            tree_splitter=tree_splitter,
            refit_noise_floor=refit_noise_floor,
            noise_guard=noise_guard,
            hvrt_params=hvrt_params,
            hvrt_tree_splitter=hvrt_tree_splitter,
            hvrt_auto_reduce_threshold=hvrt_auto_reduce_threshold,
            partitioner=partitioner,
            adaptive_reduce_ratio=adaptive_reduce_ratio,
            sample_block_n=sample_block_n,
            leave_last_block_out=leave_last_block_out,
            n_bins=n_bins,
            max_resample_n=max_resample_n,
            sample_without_replacement=sample_without_replacement,
            colsample_bytree=colsample_bytree,
            predict_stride=predict_stride,
            grad_budget_weight=grad_budget_weight,
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

        params = {k: getattr(self, k) for k in self._PARAM_NAMES if hasattr(self, k)}
        params['loss'] = self.loss
        if params.get('sample_block_n') == 'auto':
            params['sample_block_n'] = _resolve_auto_block(
                len(X), self.refit_interval, self.n_rounds,
            )
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
