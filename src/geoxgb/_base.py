import numpy as np
from hvrt import HVRT
from sklearn.tree import DecisionTreeRegressor, export_text

from geoxgb._resampling import hvrt_resample

# When auto_noise=True, skip a scheduled refit if the previous resample's
# noise_modulation fell below this threshold.  Near-zero noise_mod means
# gradients are structureless (converged); refitting HVRT on them produces
# poor geometry and floods the training set with near-zero-gradient synthetic
# samples that dilute the real signal.  The frozen Xr/yr from the last
# meaningful resample is kept; only preds is re-synced to stay current.
_REFIT_NOISE_FLOOR = 0.05


class _GeoXGBBase:
    """
    Shared infrastructure for GeoXGBRegressor and GeoXGBClassifier.

    Subclasses must implement:
        _compute_init_prediction(y) -> float
        _compute_gradients(y, predictions) -> ndarray

    Subclasses may override:
        _targets_from_gradients(gradients, predictions) -> ndarray
    """

    _PARAM_NAMES = (
        "n_rounds", "learning_rate", "max_depth", "min_samples_leaf",
        "n_partitions", "hvrt_min_samples_leaf", "hvrt_max_samples_leaf",
        "reduce_ratio", "expand_ratio",
        "y_weight", "method", "variance_weighted", "bandwidth", "refit_interval",
        "auto_noise", "auto_expand", "min_train_samples", "random_state",
        "lr_schedule", "tree_criterion", "n_jobs",
        "generation_strategy", "adaptive_bandwidth", "convergence_tol",
        "feature_weights", "assignment_strategy", "tree_splitter",
        "refit_noise_floor", "noise_guard", "hvrt_params", "hvrt_tree_splitter",
        "hvrt_auto_reduce_threshold", "partitioner", "adaptive_reduce_ratio",
        "loss", "sample_block_n", "leave_last_block_out",
    )

    # Subclasses set this to True to enable class-conditional noise estimation
    _is_classifier = False

    def __init__(
        self,
        n_rounds=1000,
        learning_rate=0.2,
        max_depth=4,
        min_samples_leaf=5,
        n_partitions=None,
        hvrt_min_samples_leaf=None,
        hvrt_max_samples_leaf=None,
        reduce_ratio=0.7,
        expand_ratio=0.0,
        y_weight=0.5,
        method="variance_ordered",
        variance_weighted=True,
        bandwidth="auto",
        refit_interval=20,
        auto_noise=True,
        auto_expand=True,
        min_train_samples=5000,
        random_state=42,
        lr_schedule=None,
        tree_criterion="squared_error",
        n_jobs=1,
        generation_strategy="epanechnikov",
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
    ):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_partitions = n_partitions
        self.hvrt_min_samples_leaf = hvrt_min_samples_leaf
        self.hvrt_max_samples_leaf = hvrt_max_samples_leaf
        self.reduce_ratio = reduce_ratio
        self.expand_ratio = expand_ratio
        self.y_weight = y_weight
        self.method = method
        self.variance_weighted = variance_weighted
        self.bandwidth = bandwidth
        self.refit_interval = refit_interval
        self.auto_noise = auto_noise
        self.auto_expand = auto_expand
        self.min_train_samples = min_train_samples
        self.random_state = random_state
        self.lr_schedule = lr_schedule
        self.tree_criterion = tree_criterion
        self.n_jobs = n_jobs
        self.generation_strategy = generation_strategy
        self.adaptive_bandwidth = adaptive_bandwidth
        self.convergence_tol = convergence_tol
        self.feature_weights = feature_weights
        self.assignment_strategy = assignment_strategy
        self.tree_splitter = tree_splitter
        self.refit_noise_floor = refit_noise_floor
        self.noise_guard = noise_guard
        self.hvrt_params = hvrt_params
        self.hvrt_tree_splitter = hvrt_tree_splitter
        self.hvrt_auto_reduce_threshold = hvrt_auto_reduce_threshold
        self.partitioner = partitioner
        self.adaptive_reduce_ratio = adaptive_reduce_ratio
        self.sample_block_n = sample_block_n
        self.leave_last_block_out = leave_last_block_out

        # Fitted state
        self._trees = []
        self._lr_values = []   # per-tree effective learning rates (populated during fit)
        self._init_pred = None
        self._is_fitted = False
        self._n_features = None
        self._feature_types = None
        self._resample_history = []
        self._train_n_original = None
        self._train_n_resampled = None
        self._y_cls_orig = None   # original integer class labels (classifier only)
        self._hvrt_cache = None   # cached HVRT geometry (reused across refits)
        self._convergence_losses = []   # mean-abs-gradient at each refit (convergence tracking)
        self.convergence_round_ = None  # round at which convergence_tol fired (None = ran to completion)

    # ------------------------------------------------------------------
    # Resample delegate
    # ------------------------------------------------------------------

    def _do_resample(self, X, y, hvrt_cache=None, _overrides=None):
        _ov = _overrides or {}
        return hvrt_resample(
            X, y,
            reduce_ratio=_ov.get("reduce_ratio", self.reduce_ratio),
            expand_ratio=_ov.get("expand_ratio", self.expand_ratio),
            y_weight=_ov.get("y_weight", self.y_weight),
            n_partitions=_ov.get("n_partitions", self.n_partitions),
            method=_ov.get("method", self.method),
            variance_weighted=self.variance_weighted,
            bandwidth=self.bandwidth,
            auto_noise=self.auto_noise,
            feature_types=self._feature_types,
            random_state=self.random_state,
            auto_expand=self.auto_expand,
            min_train_samples=self.min_train_samples,
            is_classifier=self._is_classifier,
            y_cls=self._y_cls_orig,
            hvrt_cache=hvrt_cache,
            min_samples_leaf=self.hvrt_min_samples_leaf,
            max_samples_leaf=self.hvrt_max_samples_leaf,
            generation_strategy=self.generation_strategy,
            adaptive_bandwidth=self.adaptive_bandwidth,
            feature_weights=self.feature_weights,
            assignment_strategy=self.assignment_strategy,
            hvrt_params=self.hvrt_params,
            hvrt_tree_splitter=self.hvrt_tree_splitter,
            partitioner=_ov.get("partitioner", self.partitioner),
        )

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _compute_init_prediction(self, y):
        raise NotImplementedError

    def _compute_gradients(self, y, predictions):
        raise NotImplementedError

    def _targets_from_gradients(self, gradients, predictions):
        """
        Reconstruct training targets from gradients and current predictions.

        Default (regression): y = pred + (y - pred) = pred + gradient.

        Overridden by GeoXGBClassifier to recover class probabilities from
        log-loss gradients: y_true = sigmoid(pred) + gradient.
        """
        return predictions + gradients

    # ------------------------------------------------------------------
    # Raw prediction (log-odds for classifier, values for regressor)
    # ------------------------------------------------------------------

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        p = np.full(X.shape[0], self._init_pred)
        if self._lr_values:
            for t, lr in zip(self._trees, self._lr_values):
                p += lr * t.predict(X)
        else:
            for t in self._trees:
                p += self.learning_rate * t.predict(X)
        return p

    # ------------------------------------------------------------------
    # Core boosting loop
    # ------------------------------------------------------------------

    def _fit_boosting(self, X, y):
        """
        Core gradient boosting with HVRT-based resampling.

        ``y`` must be the appropriate gradient target: continuous for
        regression, float-encoded for binary classification.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Block cycling for scalability.
        # When sample_block_n is set (or 'auto') and n exceeds the block size,
        # the full dataset is divided into non-overlapping blocks.  At each
        # refit_interval the boosting loop advances to the next block, cycling
        # through epochs.  This gives full data coverage over time while keeping
        # all per-round costs proportional to sample_block_n rather than n.
        # 'auto': block_n = 500 + (n - 5000) // 50
        #   n=5k->500, n=10k->600, n=50k->1400, n=100k->2400
        #   Grows slowly so more blocks fit per epoch at large n. Disabled for n <= 5000.
        _eff_block_n = self.sample_block_n
        if _eff_block_n == 'auto':
            n_ = len(X)
            _eff_block_n = None if n_ <= 5000 else 500 + (n_ - 5000) // 50
        _block_cycle = _eff_block_n is not None and len(X) > _eff_block_n
        if _block_cycle:
            _X_full = X
            _y_full = y
            _n_full = len(X)
            _n_total_blocks = _n_full // _eff_block_n
            # Usable blocks: optionally reserve last block as held-out set.
            _n_usable = max(1, _n_total_blocks - (1 if self.leave_last_block_out else 0))
            _blk_epoch = 0
            _blk_ctr = 0
            _blk_perm = np.random.RandomState(self.random_state).permutation(_n_full)

            def _get_block(ctr, perm):
                s = ctr * _eff_block_n
                return perm[s : s + _eff_block_n]

            _blk_idx = _get_block(0, _blk_perm)
            X = _X_full[_blk_idx]
            y = _y_full[_blk_idx]
            if self.leave_last_block_out and _n_total_blocks > 1:
                _lo_idx = _blk_perm[(_n_total_blocks - 1) * _eff_block_n:]
                self._held_out_X_ = _X_full[_lo_idx]
                self._held_out_y_ = _y_full[_lo_idx]

        # Auto-reduce: when n exceeds the threshold, use HVRT (with its own
        # auto_tune) to select a representative subset before boosting begins.
        # HVRT's auto_tune selects n_partitions and min_samples_leaf from the
        # data — "HVRT's own optimizer" doing the geometry work.
        if (
            self.hvrt_auto_reduce_threshold is not None
            and len(X) > self.hvrt_auto_reduce_threshold
        ):
            _ar_kwargs = dict(self.hvrt_params) if self.hvrt_params else {}
            _ar_kwargs.update(
                y_weight=self.y_weight,
                bandwidth=self.bandwidth,
                random_state=self.random_state,
                min_samples_leaf=self.hvrt_min_samples_leaf,
                auto_tune=True,
            )
            if self.hvrt_tree_splitter is not None:
                _ar_kwargs["tree_splitter"] = self.hvrt_tree_splitter
            _ar_hvrt = HVRT(**_ar_kwargs)
            _ar_hvrt.fit(X, y)
            _, _ar_idx = _ar_hvrt.reduce(
                n=self.hvrt_auto_reduce_threshold,
                method=self.method,
                variance_weighted=self.variance_weighted,
                return_indices=True,
            )
            X = X[_ar_idx]
            y = y[_ar_idx]

        self._n_features = X.shape[1]
        self._train_n_original = _n_full if _block_cycle else len(X)

        # Store integer class labels for class-conditional noise estimation
        # (classifier sets _is_classifier=True; regressor leaves it False)
        if self._is_classifier:
            self._y_cls_orig = y.astype(int)

        # Look-ahead guard fires only when synthetic expansion is active AND n is
        # small enough that expanded samples would dominate the training set.  At
        # large n, auto_expand never triggers (n >= min_train_samples), so noisy
        # geometry produces a suboptimal FPS selection of REAL samples — far less
        # harmful than a synthetic-contaminated training set.  Disabling the guard
        # at large n also prevents stale geometry accumulating when the distribution
        # is genuinely evolving (e.g. live time-series).
        _expansion_risk = self.auto_expand and (len(X) < self.min_train_samples)

        res = self._do_resample(X, y)
        Xr, yr = res.X, res.y
        _last_valid_res = res   # most recent meaningful resample (for skip path)
        _last_refit_noise = res.noise_modulation  # track noise separately from history
        self._train_n_resampled = len(Xr)
        self._record_resample(0, res)

        self._hvrt_cache = None  # geometry always refreshed at each refit interval

        self._init_pred = self._compute_init_prediction(yr)
        preds = np.full(len(Xr), self._init_pred)
        preds_on_X = np.full(len(X), self._init_pred)  # incremental predictions on full X
        self._trees = []
        self._lr_values = []
        self._convergence_losses = []
        self.convergence_round_ = None

        # Record initial gradient magnitude as the baseline for convergence tracking
        if self.convergence_tol is not None:
            _init_grads = self._compute_gradients(y, preds_on_X)
            self._convergence_losses.append(float(np.mean(np.abs(_init_grads))))

        for i in range(self.n_rounds):
            lr_i = (
                float(self.lr_schedule(i, self.n_rounds, self.learning_rate))
                if self.lr_schedule is not None
                else self.learning_rate
            )
            if (
                self.refit_interval is not None
                and i > 0
                and i % self.refit_interval == 0
            ):
                # Block cycling: advance to next non-overlapping block.
                # Resync preds_on_X for the new block before computing gradients.
                _block_just_advanced = False
                if _block_cycle:
                    _blk_ctr += 1
                    if _blk_ctr >= _n_usable:
                        _blk_ctr = 0
                        _blk_epoch += 1
                        _blk_perm = np.random.RandomState(
                            self.random_state + _blk_epoch
                        ).permutation(_n_full)
                    _blk_idx = _get_block(_blk_ctr, _blk_perm)
                    X = _X_full[_blk_idx]
                    y = _y_full[_blk_idx]
                    preds_on_X = self._raw_predict(X)
                    _expansion_risk = self.auto_expand and (len(X) < self.min_train_samples)
                    _block_just_advanced = True
                    # Reset so noise guard cannot fire with stale red_idx on first
                    # refit of a new block.
                    _last_refit_noise = 1.0

                # Use incrementally-tracked preds_on_X — avoids O(i × n) _raw_predict(X)
                grads_orig = self._compute_gradients(y, preds_on_X)

                # Convergence check: stop if gradient improvement over the last
                # 2 refit cycles is below convergence_tol (done before the
                # resample to avoid wasted compute on a converged model).
                if self.convergence_tol is not None:
                    loss_now = float(np.mean(np.abs(grads_orig)))
                    self._convergence_losses.append(loss_now)
                    if len(self._convergence_losses) >= 3:
                        prev = self._convergence_losses[-3]
                        rel_improvement = (prev - loss_now) / (prev + 1e-12)
                        if rel_improvement < self.convergence_tol:
                            self.convergence_round_ = i
                            self.convergence_reason_ = "tol"
                            break

                # Skip refit when auto_noise is active, expansion risk is present,
                # and the PREVIOUS resample detected near-zero noise_modulation.
                # _last_refit_noise is updated on every _do_resample() call —
                # including look-ahead discards — so once a low-noise refit is
                # detected the skip fires cheaply on all subsequent intervals.
                # _expansion_risk gates the guard: at large n (n >= min_train_samples)
                # auto_expand never synthesises samples, so noisy geometry is far
                # less harmful and stale geometry from repeated skips is worse.
                _skip_refit = (
                    not _block_just_advanced
                    and self.noise_guard
                    and self.auto_noise
                    and _expansion_risk
                    and (_last_refit_noise < self.refit_noise_floor)
                )

                if _skip_refit:
                    if _last_valid_res.n_expanded == 0:
                        preds = preds_on_X[_last_valid_res.red_idx]
                    else:
                        preds = self._raw_predict(Xr)
                    # Xr, yr, self._train_n_resampled remain from last valid resample
                else:
                    # Adaptive reduce ratio: heavier gradient tail → keep more samples
                    _resample_overrides = None
                    if self.adaptive_reduce_ratio and len(grads_orig) > 10:
                        abs_g = np.abs(grads_orig)
                        tail_ratio = np.percentile(abs_g, 90) / (np.median(abs_g) + 1e-12)
                        adapt_delta = float(np.clip((tail_ratio - 1.5) / 20.0, 0.0, 0.15))
                        _eff_reduce = min(self.reduce_ratio + adapt_delta, 1.0)
                        if _eff_reduce != self.reduce_ratio:
                            _resample_overrides = {"reduce_ratio": _eff_reduce}
                    res = self._do_resample(X, grads_orig, hvrt_cache=self._hvrt_cache,
                                            _overrides=_resample_overrides)
                    # Always update the noise tracker (even if we discard the geometry)
                    # so the next interval's skip-check uses the current signal level.
                    _last_refit_noise = res.noise_modulation

                    # Look-ahead: if the freshly-fitted HVRT reports collapsed SNR
                    # AND expansion risk is present, discard the new geometry —
                    # unreliable partitions would flood the training set with
                    # synthetic samples carrying no signal.  At large n the guard
                    # is skipped: noisy geometry on real-only FPS is acceptable,
                    # and keeping geometry fresh matters more for evolving data.
                    if (
                        self.noise_guard
                        and self.auto_noise
                        and _expansion_risk
                        and res.noise_modulation < self.refit_noise_floor
                    ):
                        if not _block_just_advanced and _last_valid_res.n_expanded == 0:
                            preds = preds_on_X[_last_valid_res.red_idx]
                        else:
                            preds = self._raw_predict(Xr)
                        # Xr, yr, self._train_n_resampled stay from _last_valid_res
                    else:
                        _last_valid_res = res
                        Xr = res.X
                        # Fast path: when Xr contains only real samples (no synthetic
                        # expansion), index into the incrementally-maintained preds_on_X
                        # tracker.  Fancy indexing returns a copy so preds_on_X is safe.
                        # This replaces an O(i × |Xr|) loop over all trees with O(|Xr|),
                        # converting the quadratic refit cost to linear.
                        #
                        # When synthetic samples are present (n_expanded > 0), fall back
                        # to _raw_predict on the full Xr: batching real + synthetic in one
                        # contiguous predict call is faster than splitting into two loops.
                        if res.n_expanded == 0:
                            preds_on_Xr = preds_on_X[res.red_idx]
                        else:
                            preds_on_Xr = self._raw_predict(Xr)
                        # Reconstruct targets: regression uses additive residuals;
                        # classifier recovers class probabilities via sigmoid (UPDATE-005)
                        yr = self._targets_from_gradients(res.y, preds_on_Xr)
                        preds = preds_on_Xr
                        self._train_n_resampled = len(Xr)
                        self._record_resample(i, res)

            grads = self._compute_gradients(yr, preds)
            tree = DecisionTreeRegressor(
                criterion=self.tree_criterion,
                splitter=self.tree_splitter,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i,
            )
            tree.fit(Xr, grads)
            leaf_preds_Xr = tree.predict(Xr)
            leaf_preds_X  = tree.predict(X)
            preds        += lr_i * leaf_preds_Xr
            preds_on_X   += lr_i * leaf_preds_X
            self._trees.append(tree)
            self._lr_values.append(lr_i)

        self._is_fitted = True

    def _record_resample(self, round_idx, res):
        self._resample_history.append({
            "round": round_idx,
            "trace": res.trace,
            "hvrt_model": res.hvrt_model,
            "noise_modulation": res.noise_modulation,
            "n_samples": len(res.X),
            "n_reduced": res.n_reduced,
            "n_expanded": res.n_expanded,
        })

    # ==================================================================
    # Interpretability API
    # ==================================================================

    def feature_importances(self, feature_names=None):
        """
        Aggregate impurity-based feature importance from all weak learners.

        Parameters
        ----------
        feature_names : list of str, optional

        Returns
        -------
        dict {name: importance} sorted descending
        """
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self._n_features)]
        total = np.zeros(self._n_features)
        for t in self._all_trees():
            total += t.feature_importances_
        total /= total.sum() + 1e-12
        return dict(sorted(zip(feature_names, total), key=lambda x: -x[1]))

    def partition_feature_importances(self, feature_names=None):
        """
        Feature importance from the HVRT partition tree — which features
        drive the geometry, not the boosting predictions.

        Returns list of ``{round, importances}`` for each resample event.
        """
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self._n_features)]
        results = []
        for entry in self._resample_history:
            ptree = entry["hvrt_model"].tree_
            imp = ptree.feature_importances_
            n_imp = min(len(imp), len(feature_names))
            named = {feature_names[i]: float(imp[i]) for i in range(n_imp)}
            named = dict(sorted(named.items(), key=lambda x: -x[1]))
            results.append({"round": entry["round"], "importances": named})
        return results

    def partition_trace(self):
        """
        Full trace of partition decisions across all resample events.

        Each entry contains: ``round``, ``noise_modulation``, ``n_samples``,
        ``n_reduced``, ``n_expanded``, ``partitions`` (list of per-partition
        metadata dicts with keys ``id``, ``size``, ``mean_abs_z``,
        ``variance``).
        """
        self._check_fitted()
        return [
            {
                "round": e["round"],
                "noise_modulation": e["noise_modulation"],
                "n_samples": e["n_samples"],
                "n_reduced": e["n_reduced"],
                "n_expanded": e["n_expanded"],
                "partitions": e["trace"],
            }
            for e in self._resample_history
        ]

    def partition_tree_rules(self, round_idx=0):
        """
        Human-readable decision rules from a partition tree.

        Parameters
        ----------
        round_idx : int
            Index into resample history (0 = initial fit).

        Returns
        -------
        str — sklearn ``export_text`` format
        """
        self._check_fitted()
        if round_idx >= len(self._resample_history):
            raise IndexError(
                f"round_idx={round_idx}, only "
                f"{len(self._resample_history)} resamples recorded"
            )
        ptree = self._resample_history[round_idx]["hvrt_model"].tree_
        return export_text(ptree, max_depth=5)

    def sample_provenance(self):
        """
        Summary of sample provenance from the most recent resampling.

        Returns
        -------
        dict with keys:
            original_n, reduced_n, expanded_n, total_training,
            reduction_ratio
        """
        self._check_fitted()
        last = self._resample_history[-1]
        total = last["n_samples"]
        return {
            "original_n": self._train_n_original,
            "reduced_n": last["n_reduced"],
            "expanded_n": last["n_expanded"],
            "total_training": total,
            "reduction_ratio": round(last["n_reduced"] / self._train_n_original, 3),
        }

    def noise_estimate(self):
        """
        Initial noise modulation factor from the first resample event.

        Returns
        -------
        float in [0, 1]: 1.0 = clean, 0.0 = pure noise
        """
        self._check_fitted()
        return self._resample_history[0]["noise_modulation"]

    def cooperation_matrix(self, X, feature_names=None):
        """
        Per-prediction local feature cooperation matrix.

        For each row of X, identifies which training partition it belongs to
        (using the most recent partition geometry) and computes the Pearson
        correlation of z-scores between every pair of features within that
        partition.

        A positive entry C[s, i, j] means features *i* and *j* tend to move
        in the same direction in z-space around sample *s* (cooperative). A
        negative entry means they tend to move in opposite directions
        (competitive). The diagonal is always 1.

        This is a strictly *local* quantity — the same pair of features may
        cooperate in one region of feature space and compete in another.
        SHAP interaction values and EBM pairwise terms are global averages;
        this matrix varies per prediction.

        Requires a Python-path fit: pass ``feature_types`` to ``fit()``.  The
        model must not have been loaded via ``load()`` (which strips ``X_z_``
        to reduce file size).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points at which to evaluate local cooperation.
        feature_names : list of str, optional
            Defaults to ``['x0', 'x1', ...]``.

        Returns
        -------
        dict with keys:

        ``'matrices'`` : ndarray, shape (n_samples, n_features, n_features)
            C[s, i, j] = Pearson corr(z_i, z_j) within sample *s*'s
            partition. Positive = cooperative. Negative = competitive.

        ``'global_matrix'`` : ndarray, shape (n_features, n_features)
            Partition-size-weighted average across all training partitions.
            Summarises the cooperation geometry over the full training set.

        ``'feature_names'`` : list of str

        ``'partitioner'`` : str — partitioner used to build the geometry.

        Raises
        ------
        RuntimeError
            If fitted via C++ path or if the model was saved/loaded (``X_z_``
            stripped).
        """
        self._check_fitted()
        if not self._resample_history:
            raise RuntimeError(
                "cooperation_matrix() requires a Python-path fit. "
                "Pass feature_types=[...] to fit() to use the Python backend."
            )
        last = self._resample_history[-1]
        hvrt_model = last["hvrt_model"]
        if (hvrt_model is None
                or not hasattr(hvrt_model, 'X_z_')
                or hvrt_model.X_z_ is None):
            raise RuntimeError(
                "cooperation_matrix() requires a Python-path fit with geometry "
                "intact. Pass feature_types=[...] to fit(), and do not call "
                "save()/load() first (save() strips X_z_ to reduce file size)."
            )

        X = np.asarray(X, dtype=np.float64)
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        X_z_train = hvrt_model.X_z_          # (n_train, d)
        part_ids  = hvrt_model.partition_ids_ # (n_train,)

        # Map test points to z-space and find their partition leaf
        X_test_z  = hvrt_model._to_z(X)      # (n_test, d)
        test_leaves = hvrt_model.tree_.apply(
            X_test_z.astype(np.float32)
        )                                     # (n_test,) — sklearn leaf node IDs

        def _pearson_corr(Z_p):
            """Pearson correlation matrix of rows of Z_p."""
            m = len(Z_p)
            if m < 2:
                return np.eye(d)
            Z_c = Z_p - Z_p.mean(axis=0, keepdims=True)
            std = Z_c.std(axis=0)
            std[std < 1e-10] = 1.0
            Z_n = Z_c / std
            return (Z_n.T @ Z_n) / m

        # Cache per-leaf result (many test points may share a partition)
        unique_test_leaves = np.unique(test_leaves)
        leaf_to_corr = {}
        for leaf in unique_test_leaves:
            mask = part_ids == leaf
            leaf_to_corr[leaf] = _pearson_corr(X_z_train[mask])

        n_test = len(X)
        matrices = np.empty((n_test, d, d))
        for s in range(n_test):
            matrices[s] = leaf_to_corr[test_leaves[s]]

        # Global: training-partition-size-weighted average
        global_C  = np.zeros((d, d))
        total_w   = 0.0
        for leaf in hvrt_model.unique_partitions_:
            mask = part_ids == leaf
            w    = float(mask.sum())
            if w < 2:
                continue
            global_C += w * _pearson_corr(X_z_train[mask])
            total_w  += w
        if total_w > 0:
            global_C /= total_w

        return {
            "matrices":      matrices,
            "global_matrix": global_C,
            "feature_names": list(feature_names),
            "partitioner":   self.partitioner,
        }

    def cooperation_score(self, X):
        """
        Per-prediction partitioner cooperation score.

        Applies the architecture-native cooperation formula to each row of X
        in the z-score space of the most recent partition geometry.

        +-------------------+--------------------------------------------+---------+
        | Partitioner       | Formula                                    | Range   |
        +===================+============================================+=========+
        | ``hvrt``          | T = 0.5·(S² − ‖z‖₂²) = Σᵢ<ⱼ zᵢzⱼ       | (−∞,+∞) |
        +-------------------+--------------------------------------------+---------+
        | ``hart``          | T = 0.5·(‖z‖₁² − ‖z‖₂²) = Σᵢ<ⱼ|zᵢ|·|zⱼ|| [0,+∞)  |
        +-------------------+--------------------------------------------+---------+
        | ``fasthvrt``      | T = ‖z‖₁ (L1 radius)                      | [0,+∞)  |
        +-------------------+--------------------------------------------+---------+
        | ``pyramid_hart``  | A = |S| − ‖z‖₁ ≤ 0                        |[−r√d, 0]|
        +-------------------+--------------------------------------------+---------+

        where S = Σᵢ zᵢ.  Higher magnitude generally indicates the sample
        sits in a region of higher geometric complexity (features cooperating
        or competing strongly).

        Requires a Python-path fit: pass ``feature_types`` to ``fit()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,)
            Cooperation score per prediction.
        """
        self._check_fitted()
        if not self._resample_history:
            raise RuntimeError(
                "cooperation_score() requires a Python-path fit. "
                "Pass feature_types=[...] to fit() to use the Python backend."
            )
        last = self._resample_history[-1]
        hvrt_model = last["hvrt_model"]
        if hvrt_model is None or not hasattr(hvrt_model, '_to_z'):
            raise RuntimeError(
                "cooperation_score() requires a Python-path fit. "
                "Pass feature_types=[...] to fit()."
            )

        X   = np.asarray(X, dtype=np.float64)
        X_z = hvrt_model._to_z(X)     # (n, d)

        p = self.partitioner
        if p == 'pyramid_hart':
            S  = X_z.sum(axis=1)
            l1 = np.abs(X_z).sum(axis=1)
            return np.abs(S) - l1              # ≤ 0; 0 iff all features same sign
        if p == 'hart':
            abs_z = np.abs(X_z)
            l1    = abs_z.sum(axis=1)
            l2sq  = (X_z * X_z).sum(axis=1)
            return 0.5 * (l1 * l1 - l2sq)     # = Σᵢ<ⱼ |zᵢ||zⱼ| ≥ 0
        if p in ('fasthvrt', 'fasthart'):
            return np.abs(X_z).sum(axis=1)    # L1 radius ≥ 0
        # Default: HVRT signed cooperation
        S    = X_z.sum(axis=1)
        l2sq = (X_z * X_z).sum(axis=1)
        return 0.5 * (S * S - l2sq)           # = Σᵢ<ⱼ zᵢzⱼ

    def cooperation_tensor(self, X, feature_names=None):
        """
        Per-prediction three-way feature cooperation tensor.

        For each row of X and every feature triple (i, j, k), computes the
        mean product of z-scores within the prediction's local partition:

            T[s, i, j, k] = E[z_i · z_j · z_k]  (within partition of sample s)

        Interpretation
        --------------
        * **T[s, i, j, k]** with i ≠ j ≠ k: how does feature *k* modulate the
          cooperation of features *i* and *j*?  Positive means that when *k*
          is above its local median, *i* and *j* tend to cooperate; negative
          means *k*'s presence pushes *i* and *j* to compete.
        * **T[s, i, j, i]**: how does feature *i* itself modulate the pairwise
          cooperation of *i* and *j* — skewness contribution.
        * **T[s, i, i, i]**: third central moment (skewness) of feature *i*
          in the local partition.

        This is the natural three-way extension of the cooperation matrix.
        Neither SHAP interaction values nor EBM pairwise terms expose
        three-way structure; this tensor does.

        Requires a Python-path fit: pass ``feature_types`` to ``fit()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional
            Defaults to ``['x0', 'x1', ...]``.

        Returns
        -------
        dict with keys:

        ``'tensor'`` : ndarray, shape (n_samples, d, d, d)
            T[s, i, j, k] = E[z_i · z_j · z_k] within sample s's partition.

        ``'global_tensor'`` : ndarray, shape (d, d, d)
            Partition-size-weighted average over all training partitions.

        ``'feature_names'`` : list of str

        ``'partitioner'`` : str
        """
        self._check_fitted()
        if not self._resample_history:
            raise RuntimeError(
                "cooperation_tensor() requires a Python-path fit. "
                "Pass feature_types=[...] to fit() to use the Python backend."
            )
        last = self._resample_history[-1]
        hvrt_model = last["hvrt_model"]
        if (hvrt_model is None
                or not hasattr(hvrt_model, 'X_z_')
                or hvrt_model.X_z_ is None):
            raise RuntimeError(
                "cooperation_tensor() requires a Python-path fit with geometry "
                "intact. Pass feature_types=[...] to fit(), and do not use "
                "save()/load()."
            )

        X = np.asarray(X, dtype=np.float64)
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        X_z_train = hvrt_model.X_z_          # (n_train, d)
        part_ids  = hvrt_model.partition_ids_ # (n_train,)

        X_test_z  = hvrt_model._to_z(X)
        test_leaves = hvrt_model.tree_.apply(
            X_test_z.astype(np.float32)
        )

        def _triple_moment(Z_p):
            """E[z_i * z_j * z_k] — shape (d, d, d)."""
            m = len(Z_p)
            if m < 2:
                return np.zeros((d, d, d))
            # np.einsum 'si,sj,sk->ijk' / m
            return np.einsum('si,sj,sk->ijk', Z_p, Z_p, Z_p) / m

        unique_test_leaves = np.unique(test_leaves)
        leaf_to_tensor = {}
        for leaf in unique_test_leaves:
            mask = part_ids == leaf
            leaf_to_tensor[leaf] = _triple_moment(X_z_train[mask])

        n_test = len(X)
        tensors = np.empty((n_test, d, d, d))
        for s in range(n_test):
            tensors[s] = leaf_to_tensor[test_leaves[s]]

        # Global: training-partition-size-weighted average
        global_T = np.zeros((d, d, d))
        total_w  = 0.0
        for leaf in hvrt_model.unique_partitions_:
            mask = part_ids == leaf
            w    = float(mask.sum())
            if w < 2:
                continue
            global_T += w * _triple_moment(X_z_train[mask])
            total_w  += w
        if total_w > 0:
            global_T /= total_w

        return {
            "tensor":        tensors,
            "global_tensor": global_T,
            "feature_names": list(feature_names),
            "partitioner":   self.partitioner,
        }

    def local_model(self, x, feature_names=None, min_pair_coop=0.10, alpha=1e-3):
        """
        Fit a local additive + multiplicative polynomial at sample *x*.

        Identifies the training partition that *x* belongs to, then fits a
        ridge-regularised polynomial on those training points:

            f̂(z|P) ≈ a₀
                     + Σᵢ        αᵢ · zᵢ
                     + Σᵢ<ⱼ      βᵢⱼ · zᵢ · zⱼ   (only pairs |Cᵢⱼ| ≥ min_pair_coop)

        where z = z-scores (the partition's natural coordinate system) and the
        regression targets are the **ensemble predictions** at the training
        points in partition P.

        This provides per-sample, locally-valid additive + multiplicative
        coefficients grounded in the cooperation geometry — something neither
        SHAP (additive only) nor EBM (global only) provides.

        Requires a Python-path fit: pass ``feature_types`` to ``fit()``.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            A single sample.
        feature_names : list of str, optional
        min_pair_coop : float, default=0.10
            Include pairwise term (i, j) only if |Cᵢⱼ(P)| ≥ this threshold.
            Higher values = sparser, more interpretable model.
        alpha : float, default=1e-3
            Ridge regularisation strength.  Used to stabilise small partitions.

        Returns
        -------
        dict with keys:

        ``'intercept'`` : float
            Constant term (partition mean of ensemble predictions).

        ``'additive'`` : ndarray, shape (n_features,)
            αᵢ — linear coefficient for feature i in z-space.

        ``'pairwise'`` : dict {(i, j): float}
            βᵢⱼ — multiplicative coefficient for each cooperation-active pair.
            Key is (i, j) with i < j.

        ``'prediction'`` : float
            Local polynomial evaluated at x's z-score (should approximate
            ``model.predict([x])``).

        ``'partition_size'`` : int
            Number of training points in the local partition.

        ``'local_r2'`` : float
            R² of the polynomial on the partition training points.
            Near 1.0 means the partition is well approximated by this polynomial.

        ``'feature_names'`` : list of str

        Raises
        ------
        RuntimeError
            If fitted via C++ path or if the model was saved/loaded.
        """
        self._check_fitted()
        if not self._resample_history:
            raise RuntimeError(
                "local_model() requires a Python-path fit. "
                "Pass feature_types=[...] to fit() to use the Python backend."
            )
        last = self._resample_history[-1]
        hvrt_model = last["hvrt_model"]
        if (hvrt_model is None
                or not hasattr(hvrt_model, 'X_z_')
                or hvrt_model.X_z_ is None):
            raise RuntimeError(
                "local_model() requires a Python-path fit with geometry intact. "
                "Pass feature_types=[...] to fit(), and do not call save()/load()."
            )

        x = np.asarray(x, dtype=np.float64).ravel()
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        # ── locate partition ──────────────────────────────────────────────
        X_z_train = hvrt_model.X_z_          # (n_train, d)
        X_orig    = hvrt_model.X_             # (n_train, d)
        part_ids  = hvrt_model.partition_ids_ # (n_train,)

        x_z      = hvrt_model._to_z(x.reshape(1, -1))[0]   # (d,)
        leaf      = int(hvrt_model.tree_.apply(
            x_z.reshape(1, -1).astype(np.float32)
        )[0])
        mask      = part_ids == leaf
        n_p       = int(mask.sum())

        Z_p   = X_z_train[mask]    # (n_p, d) — z-scores in partition
        X_p   = X_orig[mask]       # (n_p, d) — original X in partition

        # ── ensemble predictions as regression targets ────────────────────
        y_hat_p = self._raw_predict(X_p)    # (n_p,)

        # ── cooperation matrix for pair selection ─────────────────────────
        if n_p >= 2:
            Z_c  = Z_p - Z_p.mean(axis=0, keepdims=True)
            std  = Z_c.std(axis=0)
            std[std < 1e-10] = 1.0
            Z_n  = Z_c / std
            C_p  = (Z_n.T @ Z_n) / n_p       # (d, d) Pearson corr
        else:
            C_p  = np.eye(d)

        # active pairs (upper triangle, |coop| >= threshold)
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)
                 if abs(C_p[i, j]) >= min_pair_coop]

        # ── design matrix in z-space ──────────────────────────────────────
        # columns: [1, z_0, ..., z_{d-1}, z_i*z_j for active pairs]
        n_cols = 1 + d + len(pairs)
        M      = np.empty((n_p, n_cols))
        M[:, 0] = 1.0
        M[:, 1:1 + d] = Z_p
        for col, (i, j) in enumerate(pairs, start=1 + d):
            M[:, col] = Z_p[:, i] * Z_p[:, j]

        # ── ridge regression: θ = (MᵀM + αI)⁻¹ Mᵀ y_hat ─────────────────
        A   = M.T @ M + alpha * np.eye(n_cols)
        b   = M.T @ y_hat_p
        theta = np.linalg.solve(A, b)

        intercept   = float(theta[0])
        additive    = theta[1:1 + d]
        pairwise    = {(i, j): float(theta[1 + d + col])
                       for col, (i, j) in enumerate(pairs)}

        # ── evaluate at x's z-score ───────────────────────────────────────
        m_x     = np.empty(n_cols)
        m_x[0]  = 1.0
        m_x[1:1 + d] = x_z
        for col, (i, j) in enumerate(pairs, start=1 + d):
            m_x[col] = x_z[i] * x_z[j]
        prediction = float(m_x @ theta)

        # ── local R² (polynomial quality on partition) ────────────────────
        y_pred_p = M @ theta
        ss_res   = float(np.sum((y_hat_p - y_pred_p) ** 2))
        ss_tot   = float(np.sum((y_hat_p - y_hat_p.mean()) ** 2))
        local_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        return {
            "intercept":      intercept,
            "additive":       additive,
            "pairwise":       pairwise,
            "prediction":     prediction,
            "partition_size": n_p,
            "local_r2":       local_r2,
            "feature_names":  list(feature_names),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_trees(self):
        """Total number of weak learners."""
        return len(self._trees)

    @property
    def n_resamples(self):
        """Number of resample events (initial + refits)."""
        return len(self._resample_history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_trees(self):
        """Iterate over every weak learner (overridden by multiclass)."""
        return iter(self._trees)

    def save(self, path):
        """
        Save the fitted model to disk.

        Uses joblib serialisation (pickle-compatible).  All trees and fitted
        state are preserved.  HVRT raw-data arrays (``X_``, ``X_z_``) are
        stripped from resample history before serialisation to keep file sizes
        manageable — the partition tree structure used for interpretability is
        retained.  The model remains fully functional for prediction after
        loading; only ``expand()`` on the HVRT instances is unavailable.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"heart_disease.pkl"``).

        Examples
        --------
        >>> model.save("heart_disease.pkl")
        >>> from geoxgb import load_model
        >>> model = load_model("heart_disease.pkl")
        """
        import copy
        import joblib

        # Strip heavy HVRT raw-data arrays from each resample-history entry
        # before pickling.  This can reduce file size by 10–100× on large
        # datasets (each HVRT instance stores a copy of its full training X).
        # The sklearn decision tree inside each HVRT is kept intact so that
        # partition_feature_importances() and partition_tree_rules() continue
        # to work on the loaded model.
        slim = copy.copy(self)          # shallow copy of the model
        slim_history = []
        for entry in self._resample_history:
            e2 = dict(entry)            # shallow copy of the entry dict
            if "hvrt_model" in e2 and e2["hvrt_model"] is not None:
                hm = copy.copy(e2["hvrt_model"])  # shallow copy of HVRT
                # Null out the large data arrays; keep tree_ and metadata
                for attr in ("X_", "X_z_", "partition_ids_"):
                    if hasattr(hm, attr):
                        object.__setattr__(hm, attr, None)
                e2["hvrt_model"] = hm
            slim_history.append(e2)
        slim._resample_history = slim_history

        joblib.dump(slim, path)

    @classmethod
    def load(cls, path):
        """
        Load a model previously saved with ``.save()``.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        GeoXGBRegressor | GeoXGBClassifier
        """
        import joblib
        return joblib.load(path)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")

    def __repr__(self):
        if not self._is_fitted:
            s = "unfitted"
        elif self.convergence_round_ is not None:
            s = f"fitted, converged at round {self.convergence_round_}/{self.n_rounds}"
        else:
            s = "fitted"
        return (
            f"{self.__class__.__name__}({s}, n_rounds={self.n_rounds}, "
            f"refit_interval={self.refit_interval}, "
            f"auto_noise={self.auto_noise}, auto_expand={self.auto_expand})"
        )
