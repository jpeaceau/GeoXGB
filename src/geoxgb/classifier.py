import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from geoxgb._base import _GeoXGBBase
from geoxgb._utils import _sigmoid


# ---------------------------------------------------------------------------
# Module-level worker for parallel multiclass training
# Must be at module level for Windows process-based pickling.
# ---------------------------------------------------------------------------

def _fit_class_worker(k, X, y_enc, params):
    """
    Train one one-vs-rest binary ensemble for class k.

    Returns (trees_k, init_pred, lr_vals_k, resample_history_k, n_resampled).
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    from geoxgb._resampling import hvrt_resample
    from geoxgb._utils import _sigmoid as _sig
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np

    y_k   = (y_enc == k).astype(np.float64)
    y_cls = y_k.astype(int)

    n_rounds        = params["n_rounds"]
    learning_rate   = params["learning_rate"]
    refit_interval  = params["refit_interval"]
    max_depth       = params["max_depth"]
    min_samples_leaf = params["min_samples_leaf"]
    tree_criterion  = params["tree_criterion"]
    random_state    = params["random_state"]
    lr_schedule     = params["lr_schedule"]
    class_weights   = params["class_weights"]   # ndarray (n_classes,) or None
    cache_geometry  = params["cache_geometry"]

    hvrt_kw = dict(
        reduce_ratio     = params["reduce_ratio"],
        expand_ratio     = params["expand_ratio"],
        y_weight         = params["y_weight"],
        n_partitions     = params["n_partitions"],
        method           = params["method"],
        variance_weighted= params["variance_weighted"],
        bandwidth        = params["bandwidth"],
        auto_noise       = params["auto_noise"],
        feature_types    = params["feature_types"],
        random_state     = random_state,
        auto_expand      = params["auto_expand"],
        min_train_samples= params["min_train_samples"],
        is_classifier    = True,
        min_samples_leaf = params["hvrt_min_samples_leaf"],
    )

    # Initial resample
    res = hvrt_resample(X, y_k, y_cls=y_cls, **hvrt_kw)
    Xr  = res.X
    yr  = np.clip(res.y, 0, 1)

    resample_history = [{
        "round":            0,
        "trace":            res.trace,
        "hvrt_model":       res.hvrt_model,
        "noise_modulation": res.noise_modulation,
        "n_samples":        len(res.X),
        "n_reduced":        res.n_reduced,
        "n_expanded":       res.n_expanded,
    }]

    hvrt_cache_k = res.hvrt_model if cache_geometry else None

    p_k       = np.clip(np.mean(yr), 1e-6, 1 - 1e-6)
    init_pred = float(np.log(p_k / (1 - p_k)))

    preds      = np.full(len(Xr), init_pred)
    preds_on_X = np.full(len(X),  init_pred)
    trees_k    = []
    lr_vals_k  = []

    for i in range(n_rounds):
        lr_i = (
            float(lr_schedule(i, n_rounds, learning_rate))
            if lr_schedule is not None
            else learning_rate
        )

        if refit_interval is not None and i > 0 and i % refit_interval == 0:
            grads_orig = (y_enc == k).astype(float) - _sig(preds_on_X)
            if class_weights is not None:
                grads_orig *= np.where(y_enc == k, class_weights[k], 1.0)

            res = hvrt_resample(
                X, grads_orig, y_cls=y_cls, hvrt_cache=hvrt_cache_k, **hvrt_kw
            )
            Xr = res.X

            if res.n_expanded == 0:
                preds_on_Xr = preds_on_X[res.red_idx]
            else:
                preds_on_Xr = np.full(len(Xr), init_pred)
                for t, lr in zip(trees_k, lr_vals_k):
                    preds_on_Xr += lr * t.predict(Xr)

            yr    = np.clip(res.y + _sig(preds_on_Xr), 0.0, 1.0)
            preds = preds_on_Xr
            resample_history.append({
                "round":            i,
                "trace":            res.trace,
                "hvrt_model":       res.hvrt_model,
                "noise_modulation": res.noise_modulation,
                "n_samples":        len(res.X),
                "n_reduced":        res.n_reduced,
                "n_expanded":       res.n_expanded,
            })

        probs = _sig(preds)
        grads = np.clip(yr, 0, 1) - probs
        if class_weights is not None:
            grads *= np.where(yr > 0.5, class_weights[k], 1.0)

        tree = DecisionTreeRegressor(
            criterion        = tree_criterion,
            max_depth        = max_depth,
            min_samples_leaf = min_samples_leaf,
            random_state     = random_state + i + k * n_rounds,
        )
        tree.fit(Xr, grads)
        leaf_preds_Xr = tree.predict(Xr)
        leaf_preds_X  = tree.predict(X)
        preds        += lr_i * leaf_preds_Xr
        preds_on_X   += lr_i * leaf_preds_X
        trees_k.append(tree)
        lr_vals_k.append(lr_i)

    return trees_k, init_pred, lr_vals_k, resample_history, len(Xr)


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

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

    n_jobs : int, default 1
        Number of parallel jobs for multiclass training. Each class ensemble
        (one-vs-rest) is trained independently and can run in parallel.
        Follows the sklearn convention: ``1`` = sequential, ``-1`` = all CPUs.
        Has no effect on binary classification or regression.

        Note: ``lr_schedule``, if provided, must be a picklable callable
        (module-level function) when ``n_jobs != 1``.

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
        """K one-vs-rest ensembles, optionally trained in parallel (n_jobs)."""
        params = dict(
            n_rounds         = self.n_rounds,
            learning_rate    = self.learning_rate,
            refit_interval   = self.refit_interval,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            tree_criterion   = self.tree_criterion,
            random_state     = self.random_state,
            lr_schedule      = self.lr_schedule,
            class_weights    = self._class_weights,
            cache_geometry   = self.cache_geometry,
            reduce_ratio     = self.reduce_ratio,
            expand_ratio     = self.expand_ratio,
            y_weight         = self.y_weight,
            n_partitions     = self.n_partitions,
            method           = self.method,
            variance_weighted= self.variance_weighted,
            bandwidth        = self.bandwidth,
            auto_noise       = self.auto_noise,
            feature_types    = self._feature_types,
            auto_expand      = self.auto_expand,
            min_train_samples= self.min_train_samples,
            hvrt_min_samples_leaf = self.hvrt_min_samples_leaf,
        )

        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_fit_class_worker)(k, X, y_enc, params)
            for k in range(self._n_classes)
        )

        self._mc_trees      = []
        self._mc_init_preds = []
        self._resample_history = []

        for k, (trees_k, init_pred, lr_vals_k, history_k, n_resampled) in enumerate(results):
            self._mc_trees.append(trees_k)
            self._mc_init_preds.append(init_pred)
            self._resample_history.extend(history_k)
            if k == 0:
                self._lr_values = lr_vals_k

        self._train_n_resampled = n_resampled

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
            if self._lr_values:
                for t, lr in zip(self._mc_trees[k], self._lr_values):
                    lo += lr * t.predict(X)
            else:
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
