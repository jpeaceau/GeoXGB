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
        "cache_geometry", "lr_schedule", "tree_criterion", "n_jobs",
        "generation_strategy", "adaptive_bandwidth", "convergence_tol",
        "feature_weights", "assignment_strategy", "tree_splitter",
        "refit_noise_floor", "noise_guard", "hvrt_params", "hvrt_tree_splitter",
        "hvrt_auto_reduce_threshold",
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
        cache_geometry=False,
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
        self.cache_geometry = cache_geometry
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
        self.convergence_round_ = None  # round at which early stopping fired (None = ran to completion)

    # ------------------------------------------------------------------
    # Resample delegate
    # ------------------------------------------------------------------

    def _do_resample(self, X, y, hvrt_cache=None):
        return hvrt_resample(
            X, y,
            reduce_ratio=self.reduce_ratio,
            expand_ratio=self.expand_ratio,
            y_weight=self.y_weight,
            n_partitions=self.n_partitions,
            method=self.method,
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
        self._train_n_original = len(X)

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

        # Cache HVRT geometry after the initial fit (reused at every refit interval)
        self._hvrt_cache = res.hvrt_model if self.cache_geometry else None

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
                    self.noise_guard
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
                    res = self._do_resample(X, grads_orig, hvrt_cache=self._hvrt_cache)
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
                        if _last_valid_res.n_expanded == 0:
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
