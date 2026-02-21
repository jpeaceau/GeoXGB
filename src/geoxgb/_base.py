import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from geoxgb._resampling import hvrt_resample


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
        "n_partitions", "reduce_ratio", "expand_ratio", "y_weight",
        "method", "variance_weighted", "bandwidth", "refit_interval",
        "auto_noise", "auto_expand", "min_train_samples", "random_state",
        "cache_geometry",
    )

    # Subclasses set this to True to enable class-conditional noise estimation
    _is_classifier = False

    def __init__(
        self,
        n_rounds=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=5,
        n_partitions=None,
        reduce_ratio=0.7,
        expand_ratio=0.0,
        y_weight=0.5,
        method="fps",
        variance_weighted=True,
        bandwidth=0.5,
        refit_interval=10,
        auto_noise=True,
        auto_expand=True,
        min_train_samples=5000,
        random_state=42,
        cache_geometry=False,
    ):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_partitions = n_partitions
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

        # Fitted state
        self._trees = []
        self._init_pred = None
        self._is_fitted = False
        self._n_features = None
        self._feature_types = None
        self._resample_history = []
        self._train_n_original = None
        self._train_n_resampled = None
        self._y_cls_orig = None   # original integer class labels (classifier only)
        self._hvrt_cache = None   # cached HVRT geometry (reused across refits)

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
        self._n_features = X.shape[1]
        self._train_n_original = len(X)

        # Store integer class labels for class-conditional noise estimation
        # (classifier sets _is_classifier=True; regressor leaves it False)
        if self._is_classifier:
            self._y_cls_orig = y.astype(int)

        res = self._do_resample(X, y)
        Xr, yr = res.X, res.y
        self._train_n_resampled = len(Xr)
        self._record_resample(0, res)

        # Cache HVRT geometry after the initial fit (reused at every refit interval)
        self._hvrt_cache = res.hvrt_model if self.cache_geometry else None

        self._init_pred = self._compute_init_prediction(yr)
        preds = np.full(len(Xr), self._init_pred)
        preds_on_X = np.full(len(X), self._init_pred)  # incremental predictions on full X
        self._trees = []

        for i in range(self.n_rounds):
            if (
                self.refit_interval is not None
                and i > 0
                and i % self.refit_interval == 0
            ):
                # Use incrementally-tracked preds_on_X — avoids O(i × n) _raw_predict(X)
                grads_orig = self._compute_gradients(y, preds_on_X)
                res = self._do_resample(X, grads_orig, hvrt_cache=self._hvrt_cache)
                Xr = res.X
                preds_on_Xr = self._raw_predict(Xr)
                # Reconstruct targets: regression uses additive residuals;
                # classifier recovers class probabilities via sigmoid (UPDATE-005)
                yr = self._targets_from_gradients(res.y, preds_on_Xr)
                preds = preds_on_Xr
                self._train_n_resampled = len(Xr)
                self._record_resample(i, res)

            grads = self._compute_gradients(yr, preds)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i,
            )
            tree.fit(Xr, grads)
            preds += self.learning_rate * tree.predict(Xr)
            preds_on_X += self.learning_rate * tree.predict(X)  # keep X predictions current
            self._trees.append(tree)

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
            "reduction_ratio": round(total / self._train_n_original, 3),
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

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit(X, y) first.")

    def __repr__(self):
        s = "fitted" if self._is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}({s}, n_rounds={self.n_rounds}, "
            f"refit_interval={self.refit_interval}, "
            f"auto_noise={self.auto_noise}, auto_expand={self.auto_expand})"
        )
