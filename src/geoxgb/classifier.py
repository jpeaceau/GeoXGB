import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from geoxgb._base import _GeoXGBBase, _resolve_auto_block


# ---------------------------------------------------------------------------
# Module-level worker for parallel multiclass training (C++ backend).
# Must be at module level for Windows process-based pickling.
# ---------------------------------------------------------------------------

def _fit_class_worker(k, X, y_enc, params):
    """
    Train one one-vs-rest binary C++ ensemble for class k.

    Returns the fitted CppGeoXGBClassifier instance.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as np
    from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBClassifier as _CppClf

    y_k = (y_enc == k).astype(np.float64)

    cpp_params = dict(params)
    cpp_params.pop('class_weights', None)
    if params.get('class_weights') is not None:
        cpp_params['pos_class_weight'] = float(params['class_weights'][k])
    if cpp_params.get('sample_block_n') == 'auto':
        cpp_params['sample_block_n'] = _resolve_auto_block(
            len(X),
            cpp_params.get('refit_interval', 50),
            cpp_params.get('n_rounds', 500),
        )

    cpp_model = _CppClf(make_cpp_config(**cpp_params))
    cpp_model.fit(X, y_k)
    return cpp_model


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class GeoXGBGiniClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier using Gini-impurity loss.

    Uses Gini-weighted pseudo-residuals instead of log-loss gradients:
        g_i = (y_i - p_i) * |1 - 2*p_i|

    Compared to log-loss (GeoXGBClassifier):
    - Bounded gradient magnitudes (no log-divergence near 0 or 1)
    - Smoother loss surface near the decision boundary
    - May be more robust to label noise

    Only supports binary classification. For multiclass, use GeoXGBClassifier.

    Parameters
    ----------
    class_weight : None | 'balanced' | dict, optional
        Class weight scheme applied to gradient updates.
    All other parameters inherited from _GeoXGBBase.
    """

    _is_classifier = True

    def __init__(self, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        super().__init__(**kwargs)
        self._class_weights = None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y).ravel()
        self._feature_types = feature_types
        self._n_features = X.shape[1]
        X = self._encode_features(X, feature_types, fitting=True)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes != 2:
            raise ValueError("GeoXGBGiniClassifier only supports binary classification.")

        n_s, n_c = len(y_enc), self._n_classes
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_c).astype(float)
            self._class_weights = n_s / (n_c * counts)
        elif isinstance(self.class_weight, dict):
            cls_list = list(self._label_encoder.classes_)
            self._class_weights = np.array(
                [self.class_weight.get(cls_list[k], 1.0) for k in range(n_c)],
                dtype=float,
            )
        else:
            self._class_weights = None

        self._fit_binary(X, y_enc.astype(np.float64))
        self._X_train = X
        self._is_fitted = True
        return self

    def _fit_binary(self, X, y):
        from geoxgb._cpp_backend import make_cpp_config
        from geoxgb._cpp_backend import CppGeoXGBGiniClassifier as _CppGini

        params = self._resolve_params(len(X))
        if self._class_weights is not None:
            params['pos_class_weight'] = float(self._class_weights[1])
        self._cpp_model = _CppGini(make_cpp_config(**params))
        self._cpp_model.fit(X, y)
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    def predict(self, X):
        self._check_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X):
        self._check_fitted()
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)
        return self._cpp_model.predict_proba(X)

    @property
    def n_trees(self):
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0


class GeoXGBFocalClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier using Focal Loss.

    Focal loss (Lin et al., 2017): L = -(1-p_t)^gamma * log(p_t)
    where p_t is the probability assigned to the true class.

    gamma=0 recovers standard log-loss. gamma>0 downweights easy examples
    and focuses learning on hard, misclassified examples near the boundary.

    Only supports binary classification.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Higher values = more focus on hard examples.
    class_weight : None | 'balanced' | dict, optional
    """

    _is_classifier = True

    def __init__(self, gamma=2.0, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        self.gamma = gamma
        super().__init__(**kwargs)
        self._class_weights = None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y).ravel()
        self._feature_types = feature_types
        self._n_features = X.shape[1]
        X = self._encode_features(X, feature_types, fitting=True)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes != 2:
            raise ValueError("GeoXGBFocalClassifier only supports binary classification.")

        n_s, n_c = len(y_enc), self._n_classes
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_c).astype(float)
            self._class_weights = n_s / (n_c * counts)
        elif isinstance(self.class_weight, dict):
            cls_list = list(self._label_encoder.classes_)
            self._class_weights = np.array(
                [self.class_weight.get(cls_list[k], 1.0) for k in range(n_c)],
                dtype=float,
            )
        else:
            self._class_weights = None

        self._fit_binary(X, y_enc.astype(np.float64))
        self._X_train = X
        self._is_fitted = True
        return self

    def _fit_binary(self, X, y):
        from geoxgb._cpp_backend import make_cpp_config
        from geoxgb._cpp_backend import CppGeoXGBFocalClassifier as _CppFocal

        params = self._resolve_params(len(X))
        if self._class_weights is not None:
            params['pos_class_weight'] = float(self._class_weights[1])
        self._cpp_model = _CppFocal(make_cpp_config(**params), self.gamma)
        self._cpp_model.fit(X, y)
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    def predict(self, X):
        self._check_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X):
        self._check_fitted()
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)
        return self._cpp_model.predict_proba(X)

    @property
    def n_trees(self):
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0


class GeoXGBExpClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier using Exponential Loss (AdaBoost).

    Exponential loss: L = exp(-y_signed * F) where y_signed in {-1, +1}.
    Exponentially penalizes misclassified samples. More aggressive than log-loss.

    Only supports binary classification.

    Parameters
    ----------
    class_weight : None | 'balanced' | dict, optional
    """

    _is_classifier = True

    def __init__(self, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        super().__init__(**kwargs)
        self._class_weights = None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y).ravel()
        self._feature_types = feature_types
        self._n_features = X.shape[1]
        X = self._encode_features(X, feature_types, fitting=True)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes != 2:
            raise ValueError("GeoXGBExpClassifier only supports binary classification.")

        n_s, n_c = len(y_enc), self._n_classes
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_c).astype(float)
            self._class_weights = n_s / (n_c * counts)
        elif isinstance(self.class_weight, dict):
            cls_list = list(self._label_encoder.classes_)
            self._class_weights = np.array(
                [self.class_weight.get(cls_list[k], 1.0) for k in range(n_c)],
                dtype=float,
            )
        else:
            self._class_weights = None

        self._fit_binary(X, y_enc.astype(np.float64))
        self._X_train = X
        self._is_fitted = True
        return self

    def _fit_binary(self, X, y):
        from geoxgb._cpp_backend import make_cpp_config
        from geoxgb._cpp_backend import CppGeoXGBExpClassifier as _CppExp

        params = self._resolve_params(len(X))
        if self._class_weights is not None:
            params['pos_class_weight'] = float(self._class_weights[1])
        self._cpp_model = _CppExp(make_cpp_config(**params))
        self._cpp_model.fit(X, y)
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    def predict(self, X):
        self._check_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X):
        self._check_fitted()
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)
        return self._cpp_model.predict_proba(X)

    @property
    def n_trees(self):
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0


class GeoXGBHingeClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier using Squared Hinge Loss.

    Squared hinge loss: L = max(0, 1 - y_signed * F)^2.
    Margin-based: once correctly classified with margin >= 1, zero gradient.
    Bounded gradient magnitude. Geometrically interpretable as margin maximization.

    Only supports binary classification.

    Parameters
    ----------
    class_weight : None | 'balanced' | dict, optional
    """

    _is_classifier = True

    def __init__(self, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        super().__init__(**kwargs)
        self._class_weights = None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y).ravel()
        self._feature_types = feature_types
        self._n_features = X.shape[1]
        X = self._encode_features(X, feature_types, fitting=True)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes != 2:
            raise ValueError("GeoXGBHingeClassifier only supports binary classification.")

        n_s, n_c = len(y_enc), self._n_classes
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_c).astype(float)
            self._class_weights = n_s / (n_c * counts)
        elif isinstance(self.class_weight, dict):
            cls_list = list(self._label_encoder.classes_)
            self._class_weights = np.array(
                [self.class_weight.get(cls_list[k], 1.0) for k in range(n_c)],
                dtype=float,
            )
        else:
            self._class_weights = None

        self._fit_binary(X, y_enc.astype(np.float64))
        self._X_train = X
        self._is_fitted = True
        return self

    def _fit_binary(self, X, y):
        from geoxgb._cpp_backend import make_cpp_config
        from geoxgb._cpp_backend import CppGeoXGBHingeClassifier as _CppHinge

        params = self._resolve_params(len(X))
        if self._class_weights is not None:
            params['pos_class_weight'] = float(self._class_weights[1])
        self._cpp_model = _CppHinge(make_cpp_config(**params))
        self._cpp_model.fit(X, y)
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    def predict(self, X):
        self._check_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X):
        self._check_fitted()
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)
        return self._cpp_model.predict_proba(X)

    @property
    def n_trees(self):
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0


class GeoXGBClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier.

    Supports binary and multiclass classification via log-loss.

    * **Binary** (K = 2): single C++ tree ensemble on log-odds.
    * **Multiclass** (K > 2): K one-vs-rest C++ ensembles with softmax output.

    Parameters
    ----------
    class_weight : None | 'balanced' | dict, optional
        Class weight scheme applied to gradient updates.

        - ``None``        : no reweighting (default)
        - ``'balanced'``  : weights inversely proportional to class frequencies
        - ``dict``        : ``{class_label: weight}`` mapping

    n_jobs : int, default 1
        Number of parallel jobs for multiclass training.
        ``1`` = sequential, ``-1`` = all CPUs.

    All other parameters inherited from _GeoXGBBase.
    """

    _is_classifier = True

    def __init__(self, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        super().__init__(**kwargs)
        self._class_weights = None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None
        self._mc_cpp_models = None  # list of K CppGeoXGBClassifier (multiclass)

    def fit(self, X, y, feature_types=None):
        """
        Fit the classifier using the C++ backend.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_types : list of str, optional
            'continuous' or 'categorical' per column.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y_raw = np.asarray(y).ravel()
        self._feature_types = feature_types
        self._n_features = X.shape[1]
        X = self._encode_features(X, feature_types, fitting=True)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        # Compute class weights
        n_s, n_c = len(y_enc), self._n_classes
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_c).astype(float)
            self._class_weights = n_s / (n_c * counts)
        elif isinstance(self.class_weight, dict):
            cls_list = list(self._label_encoder.classes_)
            self._class_weights = np.array(
                [self.class_weight.get(cls_list[k], 1.0) for k in range(n_c)],
                dtype=float,
            )
        else:
            self._class_weights = None

        if self._n_classes == 2:
            self._fit_binary(X, y_enc.astype(np.float64))
        else:
            self._fit_multiclass(X, y_enc)

        self._X_train = X
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    def _fit_binary(self, X, y):
        """Single log-odds C++ ensemble for binary classification."""
        from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBClassifier as _CppClf
        params = self._resolve_params(len(X))
        if self._class_weights is not None:
            params['pos_class_weight'] = float(self._class_weights[1])
        self._cpp_model = _CppClf(make_cpp_config(**params))
        self._cpp_model.fit(X, y)
        cr = self._cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    # ------------------------------------------------------------------
    # Multiclass
    # ------------------------------------------------------------------

    def _fit_multiclass(self, X, y_enc):
        """Shared-geometry multiclass: single C++ model for all K classes."""
        from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBMulticlassClassifier as _CppMC

        params = self._resolve_params(len(X))

        # Build one-hot Y matrix (n × K)
        K = self._n_classes
        Y_onehot = np.zeros((len(y_enc), K), dtype=np.float64)
        for k in range(K):
            Y_onehot[:, k] = (y_enc == k).astype(np.float64)

        # Class weights vector (K,)
        if self._class_weights is not None:
            cw = np.asarray(self._class_weights, dtype=np.float64)
        else:
            cw = np.ones(K, dtype=np.float64)

        self._mc_cpp_model = _CppMC(make_cpp_config(**params))
        self._mc_cpp_model.fit(X, Y_onehot, cw)
        self._mc_cpp_models = None  # no longer using K separate models
        self._cpp_model = None
        cr = self._mc_cpp_model.convergence_round()
        self.convergence_round_ = None if cr < 0 else cr

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,) with original label types
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._encode_features(np.asarray(X, dtype=np.float64), self._feature_types)

        if self._n_classes == 2:
            return self._cpp_model.predict_proba(X)

        # Shared-geometry multiclass: single C++ model handles softmax
        return self._mc_cpp_model.predict_proba(X)

    # ------------------------------------------------------------------
    # n_trees
    # ------------------------------------------------------------------

    @property
    def n_trees(self):
        """Total number of weak learners across all classes."""
        mc = getattr(self, '_mc_cpp_model', None)
        if mc is not None:
            cr = self.convergence_round_
            per_class = cr if cr is not None else self.n_rounds
            return per_class * self._n_classes
        cpp = getattr(self, '_cpp_model', None)
        if cpp is not None:
            cr = self.convergence_round_
            return cr if cr is not None else self.n_rounds
        return 0
