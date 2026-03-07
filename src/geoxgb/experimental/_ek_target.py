"""
Experimental: Multiplicative multi-degree HVRT target statistics.

Instead of augmenting features (which dilutes signal), this module changes
what statistic HVRT uses for partitioning and reduction. The idea:

    target = |T| * |e₃|^w₃ * |e₄|^w₄

A sample only scores high if it has strong structure at ALL included degrees.
HVRT then partitions by this composite, so partition boundaries naturally
align with multi-degree interaction boundaries.

Three target statistics:
- T (= 2·e₂): current HVRT default. Degree-2 only.
- T·e₃: multiplicative. Partitions align with joint degree-2 AND degree-3.
- T·e₃·e₄: full hierarchy. Partitions align with degrees 2, 3, AND 4.

The experiment uses Python HVRT + HistGBT to test whether the composite
target produces better partitions for datasets with higher-order interactions.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from hvrt import HVRT

from geoxgb.experimental._e3_augment import (
    _compute_e2_scalar,
    _compute_e3_scalar,
    _compute_e4_scalar,
)


def _robust_whiten(X):
    """Median/MAD whitening."""
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0)
    mad[mad < 1e-12] = 1.0
    scale = mad * 1.4826
    return (X - center) / scale, center, scale


def composite_target_t(Z):
    """Standard HVRT target: T = S² - Q (= 2·e₂)."""
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    return S ** 2 - Q


def composite_target_t_e3(Z):
    """Multiplicative: |T| · |e₃|. Zero when either is zero."""
    T = composite_target_t(Z)
    e3 = _compute_e3_scalar(Z)
    # Signed product preserves direction info for both
    return np.sign(T) * np.abs(T) * np.sign(e3) * np.abs(e3)


def composite_target_t_e3_e4(Z):
    """Full hierarchy: |T| · |e₃| · |e₄|."""
    T = composite_target_t(Z)
    e3 = _compute_e3_scalar(Z)
    e4 = _compute_e4_scalar(Z)
    # Three-way signed product
    signs = np.sign(T) * np.sign(e3) * np.sign(e4)
    mags = np.abs(T) * np.abs(e3) * np.abs(e4)
    return signs * mags


def composite_target_additive(Z, max_degree=4):
    """Additive: T + e₃_norm + e₄_norm (each normalized to unit variance)."""
    T = composite_target_t(Z)
    parts = [T / max(np.std(T), 1e-12)]

    if max_degree >= 3 and Z.shape[1] >= 3:
        e3 = _compute_e3_scalar(Z)
        s3 = np.std(e3)
        if s3 > 1e-12:
            parts.append(e3 / s3)

    if max_degree >= 4 and Z.shape[1] >= 4:
        e4 = _compute_e4_scalar(Z)
        s4 = np.std(e4)
        if s4 > 1e-12:
            parts.append(e4 / s4)

    return sum(parts)


def composite_target_product_normalized(Z, max_degree=4):
    """
    Normalized multiplicative: product of rank-normalized |e_k| values.

    Each |e_k| is mapped to its rank percentile [0, 1], then multiplied.
    This avoids scale issues (e₄ values can be huge) while preserving
    the multiplicative gate: a sample only scores high if it ranks high
    in ALL included degrees.
    """
    d = Z.shape[1]
    n = len(Z)

    def rank_normalize(v):
        order = np.argsort(np.abs(v))
        ranks = np.empty(n)
        ranks[order] = np.linspace(0, 1, n)
        return ranks

    T = composite_target_t(Z)
    product = rank_normalize(T)

    if max_degree >= 3 and d >= 3:
        e3 = _compute_e3_scalar(Z)
        product *= rank_normalize(e3)

    if max_degree >= 4 and d >= 4:
        e4 = _compute_e4_scalar(Z)
        product *= rank_normalize(e4)

    return product


# ── HVRT with custom target ──────────────────────────────────────────────────

class CompositeHVRTRegressor:
    """
    HistGBT with HVRT reduction using a custom composite target statistic.

    The composite target drives HVRT's partitioning. Reduction then preserves
    samples spanning the full range of the composite, and GBT trees train
    within partitions already aligned with multi-degree structure.

    Parameters
    ----------
    target_fn : callable(Z) -> ndarray(n,)
        Function mapping whitened data to the target statistic for HVRT.
    reduce_ratio : float
        Fraction of samples to keep.
    y_weight : float
        HVRT y_weight (blend of geometry and gradient signal).
    n_partitions : int or None
        Number of HVRT partitions.
    augment_ek : bool
        Also augment features with e_k scalars (combines both approaches).
    max_degree : int
        Max degree for augmentation (if augment_ek=True).
    random_state : int
    gbt_params : dict or None
    """

    def __init__(self, target_fn=None, reduce_ratio=0.8, y_weight=0.25,
                 n_partitions=None, augment_ek=False, max_degree=4,
                 random_state=42, gbt_params=None):
        self.target_fn = target_fn or composite_target_t
        self.reduce_ratio = reduce_ratio
        self.y_weight = y_weight
        self.n_partitions = n_partitions
        self.augment_ek = augment_ek
        self.max_degree = max_degree
        self.random_state = random_state
        self.gbt_params = gbt_params or {}
        self._center = None
        self._scale = None
        self._model = None

    def _whiten(self, X):
        return (X - self._center) / self._scale

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, self._center, self._scale = _robust_whiten(X)

        # Compute composite target for HVRT
        composite = self.target_fn(Z)

        # Fit HVRT using the composite as the y-signal
        hvrt = HVRT(
            y_weight=self.y_weight,
            n_partitions=self.n_partitions,
            random_state=self.random_state,
        )
        hvrt.fit(X, y=composite)

        # Reduce using HVRT's variance-ordered method
        n_keep = max(10, int(len(X) * self.reduce_ratio))
        _, red_idx = hvrt.reduce(
            n=n_keep,
            method='variance_ordered',
            variance_weighted=True,
            return_indices=True,
        )

        X_red = X[red_idx]
        y_red = y[red_idx]
        Z_red = Z[red_idx]

        # Optionally augment
        if self.augment_ek:
            X_train = self._augment(X_red, Z_red)
        else:
            X_train = X_red

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state, **self.gbt_params
        )
        self._model.fit(X_train, y_red)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.augment_ek:
            Z = self._whiten(X)
            X_pred = self._augment(X, Z)
        else:
            X_pred = X
        return self._model.predict(X_pred)

    def _augment(self, X, Z):
        d = Z.shape[1]
        parts = [X]
        if self.max_degree >= 2 and d >= 2:
            parts.append(_compute_e2_scalar(Z).reshape(-1, 1))
        if self.max_degree >= 3 and d >= 3:
            parts.append(_compute_e3_scalar(Z).reshape(-1, 1))
        if self.max_degree >= 4 and d >= 4:
            parts.append(_compute_e4_scalar(Z).reshape(-1, 1))
        return np.hstack(parts)
