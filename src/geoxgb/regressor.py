import numpy as np

from geoxgb._base import _GeoXGBBase, _REFIT_NOISE_FLOOR


class GeoXGBRegressor(_GeoXGBBase):
    """
    Geometry-aware gradient boosting regressor.

    Optimises squared-error loss with PyramidHART-based sample curation.
    Defaults are tuned via OAT + pairwise sweep (meta-regression, 5 seeds ×
    3 folds, 4 datasets) followed by 2 000+ trial Optuna TPE search on
    diabetes and Friedman-1/2 benchmarks.

    .. note:: **HPO is strongly recommended.**
        ``learning_rate`` and ``max_depth`` are the two most sensitive
        parameters and interact strongly: the optimal regime is low
        learning_rate (0.010–0.020) paired with high ``n_rounds``
        (1 000–5 000), while shallower ``max_depth`` (2–3) consistently
        outperforms deeper trees once the PyramidHART geometry is well-tuned.
        These optima shift substantially across datasets — what works for
        diabetes can differ 5× in ``learning_rate`` from what works for
        Friedman-1.  Use ``GeoXGBOptimizer`` or any Optuna/sklearn HPO tool
        for production models.

    Parameters
    ----------
    n_rounds : int, default=1000
        Number of boosting rounds.
    learning_rate : float, default=0.02
        Shrinkage per tree.  Optimal range is 0.010–0.020 (Optuna, 2 000+
        trials); lower values require more rounds.  **Highly dataset-sensitive
        — HPO recommended.**
    max_depth : int, default=3
        Maximum depth of each weak learner.  Optuna finds depth 2–3 optimal
        across regression benchmarks (PyramidHART OAT rank 1 parameter).
        **Highly dataset-sensitive — HPO recommended.**
    min_samples_leaf : int, default=5
        Minimum samples per leaf in each weak learner (DecisionTree).
    hvrt_min_samples_leaf : int or None, default=None
        Minimum samples per leaf in the partition tree.
    n_partitions : int or None, default=None
        Target number of partitions.
    reduce_ratio : float, default=0.8
        Fraction of samples to keep per boosting round.
    expand_ratio : float, default=0.1
        Fraction of n to generate as synthetic samples.
    y_weight : float, default=0.25
        Partition blend: 0 = unsupervised geometry, 1 = fully y-driven.
        Optuna consistently finds 0.21–0.28 optimal across regression tasks.
    variance_weighted : bool, default=False
        Uniform FPS budgets across partitions.
    refit_interval : int or None, default=50
        Refit partition tree on residuals every N rounds.  PyramidHART OAT
        rank 2; polyhedral boundaries are stable once established.
    auto_noise : bool, default=True
        Enable automatic noise-modulation gating.  Recommended for
        PyramidHART: compensates for the loss of T-orthogonality (Theorem 3)
        under isotropic Gaussian feature noise.
    noise_guard : bool, default=True
        Enable noise-guard veto on resampling.  Works in tandem with
        auto_noise to protect PyramidHART partitions when gradient signal
        is structureless.
    random_state : int, default=42
    """

    def __init__(
        self,
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
        cache_geometry=False,
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
        partitioner='pyramid_hart',
        adaptive_reduce_ratio=False,
    ):
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
            cache_geometry=cache_geometry,
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
        )

    def fit(self, X, y, feature_types=None):
        """
        Fit the regressor.

        Uses the compiled C++ backend by default (faster, no Numba required).
        Falls back to the pure-Python path when the C++ extension is not
        available or when ``feature_types`` is specified (categorical columns
        are not yet supported in the C++ backend).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_types : list of str, optional
            'continuous' or 'categorical' per column.  When provided the
            Python backend is used automatically.

        Returns
        -------
        self
        """
        from geoxgb._cpp_backend import _CPP_AVAILABLE
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if _CPP_AVAILABLE and feature_types is None and self.convergence_tol is None:
            from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor as _CppReg
            params = {k: getattr(self, k) for k in self._PARAM_NAMES if hasattr(self, k)}
            self._cpp_model = _CppReg(make_cpp_config(**params))
            self._cpp_model.fit(X, y)
            self._is_fitted = True
            cr = self._cpp_model.convergence_round()
            self.convergence_round_ = None if cr < 0 else cr
        else:
            self._cpp_model = None
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
        if getattr(self, '_cpp_model', None) is not None:
            return self._cpp_model.predict(np.asarray(X, dtype=np.float64))
        self._check_fitted()
        return self._raw_predict(X)

    @property
    def n_trees(self):
        if getattr(self, '_cpp_model', None) is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return len(self._trees)

    def _compute_init_prediction(self, y):
        return float(np.mean(y))

    def _compute_gradients(self, y, predictions):
        # Negative gradient of squared-error = residual
        return y - predictions


class GeoXGBMAERegressor(GeoXGBRegressor):
    """
    GeoXGB regressor optimised for Mean Absolute Error.

    Uses L1 (sign) gradients so trees minimise MAE directly, combined with
    PyramidHART partitioning (``A = |S| - ||z||_1``, MAD-normalised) whose
    polyhedral level sets are exactly representable by axis-aligned decision
    tree splits — eliminating the approximation mismatch of smooth quadric
    boundaries (HVRT/HART) with axis-aligned weak learners.

    L1 gradient: ``sign(y - pred)`` in {-1, 0, +1}.  Trees target the median
    of residuals in each leaf (optimal under MAE).

    Default hyperparameters selected for MAE performance:

    - PyramidHART partitioner: exact tree-level-set alignment + outlier
      cancellation; MAD normalisation robust to heavy-tailed residuals
    - orthant_stratified FPS: covers all 2^d sign-consistent facets of the
      cross-polytope, ensuring no cooperative region is under-sampled
    - simplex_mixup expansion: in-orthant interpolation, parameter-free,
      stays in the convex hull; outperforms Laplace kernel empirically
    - adaptive_reduce_ratio: keeps more samples when gradient tail is heavy
    - depth=4, refit_interval=50: PyramidHART OAT + pairwise optimum
    - auto_noise + noise_guard: compensate for PyramidHART's loss of the
      T-orthogonality noise-invariance guarantee (Theorem 3)

    Parameters
    ----------
    n_rounds : int, default=1000
    learning_rate : float, default=0.02
    max_depth : int, default=4
    min_samples_leaf : int, default=5
    reduce_ratio : float, default=0.8
    expand_ratio : float, default=0.1
    y_weight : float, default=0.5
    variance_weighted : bool, default=False
    refit_interval : int, default=50
    auto_noise : bool, default=True
    noise_guard : bool, default=True
    partitioner : str, default='pyramid_hart'
    adaptive_reduce_ratio : bool, default=True
    random_state : int, default=42
    **kwargs : forwarded to GeoXGBRegressor
    """

    def __init__(
        self,
        n_rounds=1000,
        learning_rate=0.02,
        max_depth=4,
        min_samples_leaf=5,
        reduce_ratio=0.8,
        expand_ratio=0.1,
        y_weight=0.5,
        variance_weighted=False,
        refit_interval=50,
        auto_noise=True,
        noise_guard=True,
        partitioner='pyramid_hart',
        method='orthant_stratified',
        generation_strategy='simplex_mixup',
        adaptive_reduce_ratio=True,
        random_state=42,
        **kwargs,
    ):
        super().__init__(
            n_rounds=n_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            reduce_ratio=reduce_ratio,
            expand_ratio=expand_ratio,
            y_weight=y_weight,
            variance_weighted=variance_weighted,
            refit_interval=refit_interval,
            auto_noise=auto_noise,
            noise_guard=noise_guard,
            partitioner=partitioner,
            method=method,
            generation_strategy=generation_strategy,
            adaptive_reduce_ratio=adaptive_reduce_ratio,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y, feature_types=None):
        # MAE requires L1 (sign) gradients — always uses the Python path.
        self._cpp_model = None
        self._feature_types = feature_types
        self._resample_history = []
        self._fit_boosting(X, y)
        return self

    def _compute_init_prediction(self, y):
        # Median is the optimal constant predictor under MAE
        return float(np.median(y))

    def _compute_gradients(self, y, predictions):
        # L1 gradient: sign of residual (clipped to {-1, 0, +1})
        # Trees minimise mean absolute gradient residual → leaf values = median
        return np.sign(y - predictions)
