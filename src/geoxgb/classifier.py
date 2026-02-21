import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from geoxgb._base import _GeoXGBBase
from geoxgb._utils import _sigmoid


class GeoXGBClassifier(_GeoXGBBase):
    """
    Geometry-aware gradient boosting classifier.

    Supports binary and multiclass classification via log-loss.

    * **Binary** (K = 2): single tree ensemble on log-odds.
    * **Multiclass** (K > 2): K one-vs-rest ensembles with softmax output.

    Parameters
    ----------
    class_weight : None | 'balanced' | dict, optional
        Class weight scheme applied to gradient updates (UPDATE-002).

        - ``None``        : no reweighting (default, current behaviour)
        - ``'balanced'``  : weights inversely proportional to class frequencies,
                            matching sklearn's ``compute_class_weight('balanced')``
        - ``dict``        : ``{class_label: weight}`` mapping; missing labels
                            default to 1.0

        Interacts with HVRT curation: HVRT preserves minority-class geometric
        diversity; ``class_weight`` amplifies the gradient signal from those
        partitions.  Neither substitutes for the other on heavily imbalanced data.

    All other parameters are inherited from GeoXGBRegressor / _GeoXGBBase.
    """

    # Enable class-conditional noise estimation (UPDATE-003)
    _is_classifier = True

    def __init__(self, **kwargs):
        self.class_weight = kwargs.pop("class_weight", None)
        super().__init__(**kwargs)
        self._class_weights = None   # ndarray (n_classes,) or None
        self._classes = None
        self._label_encoder = None
        self._n_classes = None
        # Multiclass state
        self._mc_trees = None       # list of K lists of trees
        self._mc_init_preds = None  # list of K init predictions

    def fit(self, X, y, feature_types=None):
        """
        Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels. Any type (int, str, …). LabelEncoder applied
            internally.
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
        self._train_n_original = len(X)
        self._resample_history = []

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_raw)
        self._classes = self._label_encoder.classes_
        self._n_classes = len(self._classes)

        if self._n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        # Compute class weights (UPDATE-002)
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

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------------

    def _fit_binary(self, X, y):
        """Single log-odds ensemble for binary classification."""
        # _y_cls_orig is set inside _fit_boosting for binary
        self._fit_boosting(X, y)

    def _compute_init_prediction(self, y):
        """Log-odds of positive-class proportion."""
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    def _compute_gradients(self, y, predictions):
        """
        Negative gradient of binary log-loss, with optional class weighting.
        grads = (y - sigmoid(pred)) * class_weight
        """
        grads = y - _sigmoid(predictions)
        if self._class_weights is not None:
            w = np.where(y > 0.5, self._class_weights[1], self._class_weights[0])
            grads *= w
        return grads

    def _targets_from_gradients(self, gradients, predictions):
        """
        Recover class probabilities from log-loss gradients (UPDATE-005).

        gradient = y_true - sigmoid(pred)
        => y_true = gradient + sigmoid(pred)
        """
        return np.clip(gradients + _sigmoid(predictions), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Multiclass
    # ------------------------------------------------------------------

    def _fit_multiclass(self, X, y_enc):
        """K one-vs-rest ensembles with per-class gradient boosting."""
        self._mc_trees = []
        self._mc_init_preds = []

        for k in range(self._n_classes):
            y_k = (y_enc == k).astype(np.float64)

            # Set class labels for class-conditional noise estimation (UPDATE-003)
            self._y_cls_orig = y_k.astype(int)

            res = self._do_resample(X, y_k)
            Xr, yr = res.X, np.clip(res.y, 0, 1)
            self._record_resample(0, res)

            # Cache HVRT geometry for class k (reused at every refit interval)
            hvrt_cache_k = res.hvrt_model if self.cache_geometry else None

            p_k = np.clip(np.mean(yr), 1e-6, 1 - 1e-6)
            init_pred = float(np.log(p_k / (1 - p_k)))
            self._mc_init_preds.append(init_pred)

            preds = np.full(len(Xr), init_pred)
            preds_on_X = np.full(len(X), init_pred)  # incremental predictions on full X
            trees_k = []

            for i in range(self.n_rounds):
                if (
                    self.refit_interval is not None
                    and i > 0
                    and i % self.refit_interval == 0
                ):
                    # Use incrementally-tracked preds_on_X — eliminates O(i × n) loop over trees_k
                    grads_orig = (y_enc == k).astype(float) - _sigmoid(preds_on_X)
                    if self._class_weights is not None:
                        grads_orig *= np.where(
                            y_enc == k, self._class_weights[k], 1.0
                        )

                    # Reset class labels before refit resample
                    self._y_cls_orig = y_k.astype(int)
                    res = self._do_resample(X, grads_orig, hvrt_cache=hvrt_cache_k)
                    Xr = res.X

                    preds_on_Xr = np.full(len(Xr), init_pred)
                    for t in trees_k:
                        preds_on_Xr += self.learning_rate * t.predict(Xr)
                    # Correct target reconstruction (UPDATE-005)
                    yr = self._targets_from_gradients(res.y, preds_on_Xr)
                    preds = preds_on_Xr
                    self._record_resample(i, res)

                probs = _sigmoid(preds)
                grads = np.clip(yr, 0, 1) - probs
                if self._class_weights is not None:
                    grads *= np.where(yr > 0.5, self._class_weights[k], 1.0)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state + i + k * self.n_rounds,
                )
                tree.fit(Xr, grads)
                preds += self.learning_rate * tree.predict(Xr)
                preds_on_X += self.learning_rate * tree.predict(X)  # keep X predictions current
                trees_k.append(tree)

            self._mc_trees.append(trees_k)

        self._train_n_resampled = len(Xr)

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
            Rows sum to 1.0. Column order follows ``self._classes``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        if self._n_classes == 2:
            raw = self._raw_predict(X)
            p1 = _sigmoid(raw)
            return np.column_stack([1 - p1, p1])

        # Multiclass: softmax over K log-odds
        logits = np.zeros((len(X), self._n_classes))
        for k in range(self._n_classes):
            lo = np.full(len(X), self._mc_init_preds[k])
            for t in self._mc_trees[k]:
                lo += self.learning_rate * t.predict(X)
            logits[:, k] = lo

        logits -= logits.max(axis=1, keepdims=True)
        exp_lo = np.exp(logits)
        return exp_lo / exp_lo.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Override _all_trees and n_trees for interpretability
    # ------------------------------------------------------------------

    def _all_trees(self):
        if self._n_classes == 2:
            return iter(self._trees)
        trees = []
        for trees_k in (self._mc_trees or []):
            trees.extend(trees_k)
        return iter(trees)

    @property
    def n_trees(self):
        """Total number of weak learners across all classes."""
        if self._mc_trees is None:
            return len(self._trees)
        return sum(len(trees_k) for trees_k in self._mc_trees)
