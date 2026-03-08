import math

import numpy as np
from sklearn.tree import export_text


def _resolve_auto_block(n, refit_interval=50, n_rounds=500, n_features=None):
    """Resolve sample_block_n='auto' to a concrete int or None.

    Formula: ``max(2000, int(sqrt(n) * 15 * ri_scale))``

    Where ``ri_scale = clamp(refit_interval / 50, 0.5, 2.0)``:

    * Fewer refits → proportionally larger blocks (more data per block
      compensates for fewer geometric re-surveys).
    * More refits → smaller blocks (each block is cheaper; cycling
      provides regularisation through geometric diversity).

    The sqrt(n) scaling was empirically validated across 5 datasets,
    6 block coefficients, and n up to 500k (Studies 1–4).  A pure
    dimensionality-based formula (540d) was tested but loses overall
    because the regularisation benefit of smaller blocks outweighs
    the partition-quality gain of larger blocks at high d.

    Returns None when n <= 5000 (block cycling disabled).
    """
    if n <= 5000:
        return None
    ri = max(1, refit_interval if refit_interval else 50)
    ri_scale = max(0.5, min(2.0, ri / 50))
    return max(2000, int(math.sqrt(n) * 15 * ri_scale))


class _GeoXGBBase:
    """
    Shared infrastructure for GeoXGBRegressor and GeoXGBClassifier.

    C++ is the sole fitting engine.  Python handles label encoding,
    multiclass orchestration, and interpretability APIs.
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
        "n_bins", "max_resample_n",
        "sample_without_replacement",
        "colsample_bytree", "predict_stride",
        "grad_budget_weight",
    )

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
        refit_noise_floor=0.05,
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
        self.n_bins = n_bins
        self.max_resample_n = max_resample_n
        self.sample_without_replacement = sample_without_replacement
        self.colsample_bytree = colsample_bytree
        self.predict_stride = predict_stride
        self.grad_budget_weight = grad_budget_weight

        # Fitted state
        self._is_fitted = False
        self._n_features = None
        self._feature_types = None
        self._cat_encoders = {}       # column index → LabelEncoder for categorical cols
        self._X_train = None          # encoded training X (for interpretability)
        self._cpp_model = None        # CppGeoXGBRegressor / CppGeoXGBClassifier
        self.convergence_round_ = None

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------

    def _encode_features(self, X, feature_types=None, fitting=False):
        """Label-encode categorical columns; treat integers as continuous for C++."""
        if feature_types is None:
            return X
        X = X.copy()
        if fitting:
            self._cat_encoders = {}
        for i, ft in enumerate(feature_types):
            if ft == 'categorical':
                if fitting:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[:, i] = le.fit_transform(X[:, i].astype(str)).astype(np.float64)
                    self._cat_encoders[i] = le
                else:
                    le = self._cat_encoders.get(i)
                    if le is not None:
                        X[:, i] = le.transform(X[:, i].astype(str)).astype(np.float64)
        return X

    # ------------------------------------------------------------------
    # Geometry accessor (used by all interpretability methods)
    # ------------------------------------------------------------------

    def _get_geometry(self):
        """Return (cpp_model, X_train) for interpretability methods."""
        self._check_fitted()
        cpp = getattr(self, '_cpp_model', None)
        if cpp is None:
            cpp = getattr(self, '_mc_cpp_model', None)
        if cpp is None:
            raise RuntimeError(
                "Interpretability APIs require a fitted C++ model. "
                "Call .fit() first."
            )
        X_z = np.asarray(cpp.X_z())
        if X_z.size == 0:
            raise RuntimeError(
                "Geometry state not available (model may have used d_geom_threshold fast-path). "
                "Re-fit without d_geom_threshold."
            )
        return cpp, self._X_train

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
        cpp = getattr(self, '_cpp_model', None)
        if cpp is None:
            cpp = getattr(self, '_mc_cpp_model', None)
        if cpp is not None:
            fi = list(cpp.feature_importances())
        else:
            raise RuntimeError("feature_importances() requires a fitted C++ model.")
        n = min(len(fi), len(feature_names))
        return dict(sorted(
            ((feature_names[i], fi[i]) for i in range(n)),
            key=lambda x: -x[1]
        ))

    def partition_feature_importances(self, feature_names=None):
        """Not available for C++ path models."""
        raise RuntimeError(
            "partition_feature_importances() is not available for C++ path models. "
            "This API has been removed in favour of feature_importances()."
        )

    def partition_trace(self):
        """Not available for C++ path models."""
        raise RuntimeError(
            "partition_trace() is not available for C++ path models. "
            "This API has been removed as part of the C++-only architecture."
        )

    def partition_tree_rules(self, round_idx=0):
        """Not available for C++ path models."""
        raise RuntimeError(
            "partition_tree_rules() is not available for C++ path models. "
            "This API has been removed as part of the C++-only architecture."
        )

    def sample_provenance(self):
        """
        Summary of sample provenance from the initial HVRT resampling.

        Returns
        -------
        dict with keys:
            original_n, reduced_n, expanded_n (always 0), total_training,
            reduction_ratio
        """
        self._check_fitted()
        cpp = getattr(self, '_cpp_model', None)
        if cpp is None:
            cpp = getattr(self, '_mc_cpp_model', None)
        if cpp is None:
            raise RuntimeError("sample_provenance() requires a fitted C++ model.")
        orig = cpp.n_train()
        reduced = cpp.n_init_reduced()
        return {
            "original_n":     orig,
            "reduced_n":      reduced,
            "expanded_n":     0,
            "total_training": reduced,
            "reduction_ratio": round(reduced / orig, 3) if orig > 0 else 0.0,
        }

    def noise_estimate(self):
        """
        Noise modulation factor from the initial resample event.

        Returns
        -------
        float in [0, 1]: 1.0 = clean, 0.0 = pure noise
        """
        self._check_fitted()
        cpp = getattr(self, '_cpp_model', None)
        if cpp is None:
            cpp = getattr(self, '_mc_cpp_model', None)
        if cpp is None:
            raise RuntimeError("noise_estimate() requires a fitted C++ model.")
        return cpp.init_noise_modulation()

    def cooperation_matrix(self, X, feature_names=None):
        """
        Per-prediction local feature cooperation matrix.

        For each row of X, identifies which training partition it belongs to
        and computes the Pearson correlation of z-scores between every pair of
        features within that partition.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional

        Returns
        -------
        dict with keys: ``'matrices'``, ``'global_matrix'``, ``'feature_names'``,
        ``'partitioner'``
        """
        cpp, _ = self._get_geometry()
        X = np.asarray(X, dtype=np.float64)
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        X_z_train = np.asarray(cpp.X_z())          # (n_train, d)
        part_ids  = np.asarray(cpp.partition_ids()) # (n_train,)
        X_z_query = np.asarray(cpp.to_z(X))         # (n_test, d)
        query_leaf = np.asarray(cpp.apply(X))        # (n_test,)

        def _pearson_corr(Z_p):
            m = len(Z_p)
            if m < 2:
                return np.eye(d)
            Z_c = Z_p - Z_p.mean(axis=0, keepdims=True)
            std = Z_c.std(axis=0)
            std[std < 1e-10] = 1.0
            Z_n = Z_c / std
            return (Z_n.T @ Z_n) / m

        unique_test_leaves = np.unique(query_leaf)
        leaf_to_corr = {}
        for leaf in unique_test_leaves:
            mask = part_ids == leaf
            leaf_to_corr[leaf] = _pearson_corr(X_z_train[mask])

        n_test = len(X)
        matrices = np.empty((n_test, d, d))
        for s in range(n_test):
            matrices[s] = leaf_to_corr[query_leaf[s]]

        # Global: training-partition-size-weighted average
        global_C = np.zeros((d, d))
        total_w  = 0.0
        for leaf in np.unique(part_ids):
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        cpp, _ = self._get_geometry()
        X   = np.asarray(X, dtype=np.float64)
        X_z = np.asarray(cpp.to_z(X))   # (n, d)

        p = self.partitioner
        if p == 'pyramid_hart':
            S  = X_z.sum(axis=1)
            l1 = np.abs(X_z).sum(axis=1)
            return np.abs(S) - l1
        if p == 'hart':
            abs_z = np.abs(X_z)
            l1    = abs_z.sum(axis=1)
            l2sq  = (X_z * X_z).sum(axis=1)
            return 0.5 * (l1 * l1 - l2sq)
        if p in ('fasthvrt', 'fasthart'):
            return np.abs(X_z).sum(axis=1)
        S    = X_z.sum(axis=1)
        l2sq = (X_z * X_z).sum(axis=1)
        return 0.5 * (S * S - l2sq)

    def cooperation_tensor(self, X, feature_names=None):
        """
        Per-prediction three-way feature cooperation tensor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional

        Returns
        -------
        dict with keys: ``'tensor'``, ``'global_tensor'``, ``'feature_names'``,
        ``'partitioner'``
        """
        cpp, _ = self._get_geometry()
        X = np.asarray(X, dtype=np.float64)
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        X_z_train = np.asarray(cpp.X_z())
        part_ids  = np.asarray(cpp.partition_ids())
        X_z_query = np.asarray(cpp.to_z(X))
        query_leaf = np.asarray(cpp.apply(X))

        def _triple_moment(Z_p):
            m = len(Z_p)
            if m < 2:
                return np.zeros((d, d, d))
            return np.einsum('si,sj,sk->ijk', Z_p, Z_p, Z_p) / m

        unique_test_leaves = np.unique(query_leaf)
        leaf_to_tensor = {}
        for leaf in unique_test_leaves:
            mask = part_ids == leaf
            leaf_to_tensor[leaf] = _triple_moment(X_z_train[mask])

        n_test = len(X)
        tensors = np.empty((n_test, d, d, d))
        for s in range(n_test):
            tensors[s] = leaf_to_tensor[query_leaf[s]]

        global_T = np.zeros((d, d, d))
        total_w  = 0.0
        for leaf in np.unique(part_ids):
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

    def local_model(self, x, feature_names=None, min_pair_coop=0.10, alpha=1e-3,
                    target_class=None):
        """
        Fit a local additive + multiplicative polynomial at sample *x*.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        feature_names : list of str, optional
        min_pair_coop : float, default=0.10
        alpha : float, default=1e-3
        target_class : int, optional
            For multiclass classifiers, which class logit to explain.
            Required when the model is multiclass.

        Returns
        -------
        dict with keys: ``'intercept'``, ``'additive'``, ``'pairwise'``,
        ``'prediction'``, ``'partition_size'``, ``'local_r2'``, ``'feature_names'``
        """
        cpp, X_orig = self._get_geometry()
        x = np.asarray(x, dtype=np.float64).ravel()
        d = self._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        X_z_train  = np.asarray(cpp.X_z())
        part_ids   = np.asarray(cpp.partition_ids())

        # Get per-sample training predictions (scalar or per-class)
        mc = getattr(self, '_mc_cpp_model', None)
        if mc is not None:
            if target_class is None:
                raise ValueError(
                    "target_class is required for multiclass models. "
                    f"Pass target_class in 0..{self._n_classes - 1}."
                )
            preds_train = np.asarray(mc.train_predictions_multi())[:, target_class]
        else:
            preds_train = np.asarray(cpp.train_predictions())

        x_z   = np.asarray(cpp.to_z(x.reshape(1, -1)))[0]
        leaf  = int(np.asarray(cpp.apply(x.reshape(1, -1)))[0])
        mask  = part_ids == leaf
        n_p   = int(mask.sum())

        Z_p     = X_z_train[mask]     # (n_p, d)
        y_hat_p = preds_train[mask]   # (n_p,)

        if n_p >= 2:
            Z_c  = Z_p - Z_p.mean(axis=0, keepdims=True)
            std  = Z_c.std(axis=0)
            std[std < 1e-10] = 1.0
            Z_n  = Z_c / std
            C_p  = (Z_n.T @ Z_n) / n_p
        else:
            C_p  = np.eye(d)

        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)
                 if abs(C_p[i, j]) >= min_pair_coop]

        n_cols = 1 + d + len(pairs)
        M      = np.empty((n_p, n_cols))
        M[:, 0] = 1.0
        M[:, 1:1 + d] = Z_p
        for col, (i, j) in enumerate(pairs, start=1 + d):
            M[:, col] = Z_p[:, i] * Z_p[:, j]

        A     = M.T @ M + alpha * np.eye(n_cols)
        b     = M.T @ y_hat_p
        theta = np.linalg.solve(A, b)

        intercept = float(theta[0])
        additive  = theta[1:1 + d]
        pairwise  = {(i, j): float(theta[1 + d + col])
                     for col, (i, j) in enumerate(pairs)}

        m_x     = np.empty(n_cols)
        m_x[0]  = 1.0
        m_x[1:1 + d] = x_z
        for col, (i, j) in enumerate(pairs, start=1 + d):
            m_x[col] = x_z[i] * x_z[j]
        prediction = float(m_x @ theta)

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

    def contributions(self, X, feature_names=None, min_pair_coop=0.10, alpha=1e-3,
                      target_class=None):
        """
        Compute per-sample EBM-style feature contributions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : list of str, optional
        min_pair_coop : float, default=0.10
        alpha : float, default=1e-3
        target_class : int, optional
            For multiclass classifiers, which class logit to explain.
            Required when the model is multiclass.

        Returns
        -------
        ContributionFrame
        """
        from geoxgb.contributions import compute_contributions
        X = np.asarray(X, dtype=np.float64)
        X = self._encode_features(X, self._feature_types)
        return compute_contributions(self, X, feature_names=feature_names,
                                     min_pair_coop=min_pair_coop, alpha=alpha,
                                     target_class=target_class)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_trees(self):
        """Total number of weak learners."""
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0

    @property
    def n_resamples(self):
        """Approximate number of HVRT resample events (initial + refits) during fit."""
        if not getattr(self, '_is_fitted', False):
            return 0
        if getattr(self, '_cpp_model', None) is None:
            return 0
        cr = getattr(self, 'convergence_round_', None)
        rounds = cr if cr is not None else self.n_rounds
        ri = self.refit_interval
        if ri is None or ri == 0:
            return 1
        return max(1, rounds // ri)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the fitted model to disk using joblib.

        Parameters
        ----------
        path : str or Path
        """
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a model previously saved with ``.save()``."""
        import joblib
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
