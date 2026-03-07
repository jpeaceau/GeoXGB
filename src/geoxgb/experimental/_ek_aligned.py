"""
Experimental: Multi-degree-aligned reduction + e_k augmentation.

Tests the hypothesis that HVRT's T-only (degree-2) reduction discards samples
critical for e₃/e₄ structure, and that aligning reduction with a joint
e₂/e₃/e₄ importance criterion makes higher-order augmentation effective.

Uses sklearn HistGradientBoostingRegressor as the GBT backend to isolate the
reduction variable from GeoXGB's T-locked HVRT geometry.

Classes
-------
EkAlignedRegressor
    Joint multi-degree reduction + e_k augmented GBT.
    Reduction preserves samples spanning the full range of a combined
    e₂/e₃/e₄ importance score, not just T (= 2·e₂).

TReducedEkRegressor
    T-only reduction (HVRT-style) + e_k augmentation.
    Control arm: same augmentation, but reduction aligned only with degree-2.

PlainEkRegressor
    No reduction, just e_k augmentation on full data.
    Tests whether e_k features help at all when reduction isn't a confound.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

from geoxgb.experimental._e3_augment import (
    _compute_e2_scalar,
    _compute_e3_scalar,
    _compute_e4_scalar,
    _compute_e3_partial,
    _compute_e4_partial,
)


# ── Whitening ────────────────────────────────────────────────────────────────

def _robust_whiten(X):
    """Median/MAD whitening → Z with unit MAD per feature."""
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0)
    mad[mad < 1e-12] = 1.0
    scale = mad * 1.4826  # MAD → std estimate for normal
    return (X - center) / scale, center, scale


# ── Joint importance score ───────────────────────────────────────────────────

def _joint_importance(Z, max_degree=4):
    """
    Per-sample importance combining e₂, e₃, e₄ contributions.

    Each e_k is normalized to unit variance, then combined as:
        importance = sqrt(sum_k (e_k_normalized)²)

    This ensures samples at extremes of ANY degree are considered important.
    """
    components = []
    if max_degree >= 2 and Z.shape[1] >= 2:
        e2 = _compute_e2_scalar(Z)
        std2 = np.std(e2)
        if std2 > 1e-12:
            components.append(e2 / std2)
    if max_degree >= 3 and Z.shape[1] >= 3:
        e3 = _compute_e3_scalar(Z)
        std3 = np.std(e3)
        if std3 > 1e-12:
            components.append(e3 / std3)
    if max_degree >= 4 and Z.shape[1] >= 4:
        e4 = _compute_e4_scalar(Z)
        std4 = np.std(e4)
        if std4 > 1e-12:
            components.append(e4 / std4)

    if not components:
        return np.ones(len(Z))

    stacked = np.column_stack(components)
    return np.sqrt((stacked ** 2).sum(axis=1))


# ── Variance-ordered reduction ───────────────────────────────────────────────

def _variance_ordered_reduce(scores, n_keep, random_state=42):
    """
    Variance-ordered reduction: keep samples spanning the full range of `scores`.

    Selects n_keep samples at evenly-spaced quantile positions along the
    score-sorted order, covering extremes and interior equally.
    """
    n = len(scores)
    if n_keep >= n:
        return np.arange(n)
    order = np.argsort(scores)
    positions = np.round(np.linspace(0, n - 1, n_keep)).astype(int)
    return order[positions]


# ── Feature augmentation ────────────────────────────────────────────────────

def _augment_features(X, Z, max_degree=4, include_partials=False):
    """Append e_k scalar aggregates (and optional partials) to X."""
    d = Z.shape[1]
    parts = [X]

    if max_degree >= 2 and d >= 2:
        parts.append(_compute_e2_scalar(Z).reshape(-1, 1))
    if max_degree >= 3 and d >= 3:
        parts.append(_compute_e3_scalar(Z).reshape(-1, 1))
    if max_degree >= 4 and d >= 4:
        parts.append(_compute_e4_scalar(Z).reshape(-1, 1))

    if include_partials:
        if max_degree >= 3 and d >= 3:
            parts.append(_compute_e3_partial(Z))
        if max_degree >= 4 and d >= 4:
            parts.append(_compute_e4_partial(Z))

    return np.hstack(parts)


# ── Regressors ───────────────────────────────────────────────────────────────

class EkAlignedRegressor:
    """
    Joint multi-degree reduction + e_k augmented HistGBT.

    Pipeline:
    1. Whiten X → Z (robust median/MAD)
    2. Compute joint importance = sqrt(e₂² + e₃² + e₄²) (each normalized)
    3. Variance-ordered reduce: keep samples spanning the full importance range
    4. Augment with e_k scalar aggregates
    5. Fit HistGradientBoostingRegressor on reduced+augmented data

    Parameters
    ----------
    max_degree : int
        Maximum e_k degree (2, 3, or 4).
    reduce_ratio : float
        Fraction of samples to keep (1.0 = no reduction).
    include_partials : bool
        Include per-feature partial e_k contributions.
    gbt_params : dict or None
        Extra params for HistGradientBoostingRegressor.
    """

    def __init__(self, max_degree=4, reduce_ratio=0.8, include_partials=False,
                 random_state=42, gbt_params=None):
        self.max_degree = max_degree
        self.reduce_ratio = reduce_ratio
        self.include_partials = include_partials
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._center = None
        self._scale = None
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, self._center, self._scale = _robust_whiten(X)

        # Joint-criterion reduction
        importance = _joint_importance(Z, self.max_degree)
        n_keep = max(10, int(len(X) * self.reduce_ratio))
        red_idx = _variance_ordered_reduce(importance, n_keep, self.random_state)

        X_red, Z_red, y_red = X[red_idx], Z[red_idx], y[red_idx]

        # Augment
        X_aug = _augment_features(X_red, Z_red, self.max_degree, self.include_partials)

        # Fit GBT
        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X_aug, y_red)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self._center) / self._scale
        X_aug = _augment_features(X, Z, self.max_degree, self.include_partials)
        return self._model.predict(X_aug)


class TReducedEkRegressor:
    """
    T-only reduction (degree-2 aligned) + e_k augmented HistGBT.

    Control arm: same augmentation as EkAlignedRegressor, but reduction
    uses only T = 2·e₂ as the importance criterion.
    """

    def __init__(self, max_degree=4, reduce_ratio=0.8, include_partials=False,
                 random_state=42, gbt_params=None):
        self.max_degree = max_degree
        self.reduce_ratio = reduce_ratio
        self.include_partials = include_partials
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._center = None
        self._scale = None
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, self._center, self._scale = _robust_whiten(X)

        # T-only reduction (degree-2 only)
        T = _compute_e2_scalar(Z)
        n_keep = max(10, int(len(X) * self.reduce_ratio))
        red_idx = _variance_ordered_reduce(T, n_keep, self.random_state)

        X_red, Z_red, y_red = X[red_idx], Z[red_idx], y[red_idx]

        # Augment with full e_k hierarchy
        X_aug = _augment_features(X_red, Z_red, self.max_degree, self.include_partials)

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X_aug, y_red)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self._center) / self._scale
        X_aug = _augment_features(X, Z, self.max_degree, self.include_partials)
        return self._model.predict(X_aug)


class PlainEkRegressor:
    """
    No reduction, just e_k augmentation on full data with HistGBT.

    Tests whether e_k features help at all when reduction isn't a confound.
    """

    def __init__(self, max_degree=4, include_partials=False,
                 random_state=42, gbt_params=None):
        self.max_degree = max_degree
        self.include_partials = include_partials
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._center = None
        self._scale = None
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, self._center, self._scale = _robust_whiten(X)

        X_aug = _augment_features(X, Z, self.max_degree, self.include_partials)

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X_aug, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self._center) / self._scale
        X_aug = _augment_features(X, Z, self.max_degree, self.include_partials)
        return self._model.predict(X_aug)


class PlainHistGBT:
    """
    Bare HistGradientBoostingRegressor baseline. No reduction, no augmentation.
    """

    def __init__(self, random_state=42, gbt_params=None):
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._model = None

    def fit(self, X, y):
        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float64))


class TReducedHistGBT:
    """
    T-only reduction + bare HistGBT (no e_k augmentation).
    Control: does reduction alone hurt?
    """

    def __init__(self, reduce_ratio=0.8, random_state=42, gbt_params=None):
        self.reduce_ratio = reduce_ratio
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, _, _ = _robust_whiten(X)
        T = _compute_e2_scalar(Z)
        n_keep = max(10, int(len(X) * self.reduce_ratio))
        red_idx = _variance_ordered_reduce(T, n_keep, self.random_state)

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X[red_idx], y[red_idx])
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float64))
