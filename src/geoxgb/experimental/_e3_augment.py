"""
Experimental: Noise-invariant elementary symmetric polynomial augmentation.

Theory
------
The k-th elementary symmetric polynomial e_k(z) = sum_{i1<...<ik} z_i1 * ... * z_ik
is exactly noise-invariant: E[e_k(z + eps)] = e_k(z) for zero-mean independent eps.

This holds because e_k involves only products of DISTINCT indices. When any index
appears only once in a product, and noise at each index is independent zero-mean,
expectation kills every noise-containing term.

However, computing e_k directly from the definition is O(C(d,k)) per sample.
Newton's identities give us e_k in terms of power sums p_j = sum(z_i^j):

    e_1 = p_1 = S
    e_2 = (S^2 - Q) / 2              = T / 2
    e_3 = (S^3 + 2*p_3 - 3*Q*S) / 6
    e_4 = (S^4 - 6*S^2*Q + 3*Q^2 + 8*S*p_3 - 6*p_4) / 24

Each e_k involves only power sums up to p_k, all computable in O(n*d).

The noise cancellation at each degree works because the noise biases in each
power sum p_j(z~) are polynomial in sigma^2 and the Newton's identity
coefficients are precisely those that null out the bias.

This module provides:
- Scalar aggregates e_2, e_3, e_4 as features (noise-invariant, O(n*d) each)
- Per-feature partial contributions at each degree
- Selective k-tuples gated by noise_estimate()
- A unified AdaptiveEkGeoXGBRegressor that stacks the full hierarchy

Usage
-----
    from geoxgb.experimental._e3_augment import AdaptiveEkGeoXGBRegressor

    model = AdaptiveEkGeoXGBRegressor(max_degree=4, n_rounds=1000)
    model.fit(X, y)
    preds = model.predict(X_test)
"""

from itertools import combinations

import numpy as np

from geoxgb.regressor import GeoXGBRegressor


# ── Noise-invariant scalar aggregates ─────────────────────────────────────────

def _compute_e2_scalar(Z):
    """e_2 = (S^2 - Q) / 2 = T / 2.  Shape (n,)."""
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    return (S ** 2 - Q) / 2.0


def _compute_e3_scalar(Z):
    """e_3 = (S^3 + 2*p_3 - 3*Q*S) / 6.  Shape (n,)."""
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    p3 = (Z ** 3).sum(axis=1)
    return (S ** 3 + 2 * p3 - 3 * Q * S) / 6.0


def _compute_e4_scalar(Z):
    """e_4 = (S^4 - 6*S^2*Q + 3*Q^2 + 8*S*p_3 - 6*p_4) / 24.  Shape (n,)."""
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    p3 = (Z ** 3).sum(axis=1)
    p4 = (Z ** 4).sum(axis=1)
    return (S**4 - 6*S**2*Q + 3*Q**2 + 8*S*p3 - 6*p4) / 24.0


# ── Per-feature partial contributions ─────────────────────────────────────────

def _compute_e3_partial(Z):
    """Per-feature partial e_3: e_3^{(j)} = z_j * e_2(z_{-j}).  Shape (n, d)."""
    S = Z.sum(axis=1, keepdims=True)
    Q = (Z ** 2).sum(axis=1, keepdims=True)
    S_j = S - Z
    Q_j = Q - Z ** 2
    e2_without_j = (S_j ** 2 - Q_j) / 2.0
    return Z * e2_without_j


def _compute_e4_partial(Z):
    """Per-feature partial e_4: e_4^{(j)} = z_j * e_3(z_{-j}).  Shape (n, d).

    e_3(z_{-j}) = (S_j^3 + 2*p3_j - 3*Q_j*S_j) / 6
    where S_j = S - z_j, Q_j = Q - z_j^2, p3_j = p_3 - z_j^3.
    """
    S = Z.sum(axis=1, keepdims=True)
    Q = (Z ** 2).sum(axis=1, keepdims=True)
    p3 = (Z ** 3).sum(axis=1, keepdims=True)
    S_j = S - Z
    Q_j = Q - Z ** 2
    p3_j = p3 - Z ** 3
    e3_without_j = (S_j**3 + 2*p3_j - 3*Q_j*S_j) / 6.0
    return Z * e3_without_j


# ── Selective k-tuple ranking ─────────────────────────────────────────────────

def _rank_ktuples(Z, residuals, k, top_n):
    """Rank all C(d,k) products by |corr with residual|.

    Returns list of tuples and their correlations, sorted descending.
    """
    d = Z.shape[1]
    if d < k or top_n <= 0:
        return [], np.array([])

    all_tuples = list(combinations(range(d), k))
    n_cand = len(all_tuples)
    if n_cand == 0:
        return [], np.array([])

    res_c = residuals - residuals.mean()
    res_std = np.std(residuals)
    if res_std < 1e-12:
        return [], np.array([])

    correlations = np.empty(n_cand)
    for idx, tup in enumerate(all_tuples):
        prod = np.ones(len(Z))
        for j in tup:
            prod *= Z[:, j]
        pc = prod - prod.mean()
        ps = np.std(prod)
        if ps < 1e-12:
            correlations[idx] = 0.0
        else:
            correlations[idx] = abs(
                np.dot(res_c, pc) / (len(residuals) * res_std * ps)
            )

    n = min(top_n, n_cand)
    top_idx = np.argpartition(correlations, -n)[-n:]
    top_idx = top_idx[np.argsort(correlations[top_idx])[::-1]]
    return [all_tuples[i] for i in top_idx], correlations[top_idx]


# ── Whitening param extraction ────────────────────────────────────────────────

def _extract_whitening_params(X, Z):
    """Recover per-feature affine params: z = (x - center) / scale."""
    d = X.shape[1]
    center = np.empty(d)
    scale = np.empty(d)
    for j in range(d):
        z_std = np.std(Z[:, j])
        if z_std < 1e-15:
            center[j] = np.median(X[:, j])
            scale[j] = 1.0
        else:
            A = np.vstack([Z[:, j], np.ones(len(Z))]).T
            result = np.linalg.lstsq(A, X[:, j], rcond=None)
            s, c = result[0]
            scale[j] = s if abs(s) > 1e-15 else 1.0
            center[j] = c
    return center, scale


# ── Main class: AdaptiveEkGeoXGBRegressor ─────────────────────────────────────

class AdaptiveEkGeoXGBRegressor(GeoXGBRegressor):
    """
    GeoXGBRegressor with noise-gated elementary symmetric polynomial hierarchy.

    Augments the feature matrix with noise-invariant e_k aggregates and
    noise-gated selective k-tuples up to degree max_degree.

    The hierarchy at each degree k:
    - e_k scalar aggregate (always included if k <= max_degree)
    - Per-feature partial e_k (included if include_partials=True)
    - Selective k-tuples ranked by residual correlation (noise-gated)

    Noise gating for degree-k selective tuples:
        budget_k = floor(top_k_max * noise_estimate ^ (noise_alpha * (k - 1)))

    Higher-degree tuples are gated more aggressively because their variance
    under noise grows as O(sigma^{2k-2}).

    Parameters
    ----------
    max_degree : int, default=4
        Maximum polynomial degree. Supports 2, 3, or 4.
    top_k_max : int, default=10
        Maximum selective k-tuples per degree when data is perfectly clean.
    noise_alpha : float, default=1.5
        Base exponent for noise gating. Actual exponent at degree k is
        noise_alpha * (k - 1).
    noise_floor : float, default=0.3
        Below this noise_estimate, no selective tuples at any degree.
    include_partials : bool, default=False
        Whether to include per-feature partial e_k contributions.
    """

    def __init__(self, max_degree=4, top_k_max=10, noise_alpha=1.5,
                 noise_floor=0.3, include_partials=False, **kwargs):
        self.max_degree = max_degree
        self.top_k_max = top_k_max
        self.noise_alpha = noise_alpha
        self.noise_floor = noise_floor
        self.include_partials = include_partials
        self._orig_n_features = None
        self._z_center = None
        self._z_scale = None
        self._selective = {}  # degree -> (tuples, correlations, effective_k)
        self._noise_estimate = None
        super().__init__(**kwargs)

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._orig_n_features = X.shape[1]
        d = self._orig_n_features

        if d < 3:
            return super().fit(X, y, feature_types=feature_types)

        # Stage 1: initial fit on original features
        super().fit(X, y, feature_types=feature_types)
        cpp = self._cpp_model

        noise_est = cpp.init_noise_modulation()
        self._noise_estimate = noise_est

        Z = np.asarray(cpp.to_z(X))
        self._z_center, self._z_scale = _extract_whitening_params(X, Z)

        # Compute residuals from initial fit
        preds_full = self.predict(X)
        residuals = y - preds_full

        # Compute noise-gated selective tuples at each degree
        self._selective = {}
        for k in range(2, self.max_degree + 1):
            if d < k:
                self._selective[k] = ([], np.array([]), 0)
                continue

            # Noise gating: higher degrees get exponentially more aggressive cutoff
            alpha_k = self.noise_alpha * (k - 1)
            if noise_est < self.noise_floor:
                eff_k = 0
            else:
                eff_k = max(0, int(self.top_k_max * (noise_est ** alpha_k)))

            if eff_k > 0:
                tuples, corrs = _rank_ktuples(Z, residuals, k, eff_k)
            else:
                tuples, corrs = [], np.array([])

            self._selective[k] = (tuples, corrs, eff_k)

        # Stage 2: re-fit with full augmented features
        X_aug = self._augment(X)
        self._is_fitted = False
        self._cpp_model = None
        super().fit(X_aug, y, feature_types=None)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._orig_n_features is not None and self._z_center is not None:
            X = self._augment(X)
        return super().predict(X)

    def _to_z(self, X):
        return (X - self._z_center) / self._z_scale

    def _augment(self, X):
        Z = self._to_z(X)
        d = self._orig_n_features
        parts = [X]

        # Scalar aggregates (always included, noise-invariant)
        if self.max_degree >= 2 and d >= 2:
            parts.append(_compute_e2_scalar(Z).reshape(-1, 1))
        if self.max_degree >= 3 and d >= 3:
            parts.append(_compute_e3_scalar(Z).reshape(-1, 1))
        if self.max_degree >= 4 and d >= 4:
            parts.append(_compute_e4_scalar(Z).reshape(-1, 1))

        # Per-feature partials (noise-invariant, optional)
        if self.include_partials:
            if self.max_degree >= 3 and d >= 3:
                parts.append(_compute_e3_partial(Z))
            if self.max_degree >= 4 and d >= 4:
                parts.append(_compute_e4_partial(Z))

        # Selective k-tuples (noise-gated)
        for k in range(2, self.max_degree + 1):
            tuples, _, _ = self._selective.get(k, ([], None, 0))
            if tuples:
                n = X.shape[0]
                extra = np.empty((n, len(tuples)))
                for col, tup in enumerate(tuples):
                    prod = np.ones(n)
                    for j in tup:
                        prod *= Z[:, j]
                    extra[:, col] = prod
                parts.append(extra)

        return np.hstack(parts)

    def ek_summary(self, feature_names=None):
        """Summary of the augmentation hierarchy."""
        d = self._orig_n_features or 0
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]

        info = {
            'noise_estimate': self._noise_estimate,
            'max_degree': self.max_degree,
            'noise_alpha': self.noise_alpha,
            'noise_floor': self.noise_floor,
            'include_partials': self.include_partials,
            'degrees': {},
        }

        for k in range(2, self.max_degree + 1):
            tuples, corrs, eff_k = self._selective.get(k, ([], np.array([]), 0))
            alpha_k = self.noise_alpha * (k - 1)
            degree_info = {
                'effective_k': eff_k,
                'alpha': alpha_k,
                'tuples': [],
            }
            for idx, tup in enumerate(tuples):
                corr = float(corrs[idx]) if idx < len(corrs) else None
                names = tuple(feature_names[j] for j in tup)
                degree_info['tuples'].append({
                    'indices': tup,
                    'names': names,
                    'residual_corr': corr,
                })
            info['degrees'][k] = degree_info

        return info


# ── Legacy aliases ────────────────────────────────────────────────────────────

class E3ScalarGeoXGBRegressor(GeoXGBRegressor):
    """Appends scalar e_3 only. Legacy — prefer AdaptiveEkGeoXGBRegressor."""

    def __init__(self, **kwargs):
        self._orig_n_features = None
        self._z_center = None
        self._z_scale = None
        super().__init__(**kwargs)

    def fit(self, X, y, feature_types=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._orig_n_features = X.shape[1]
        if X.shape[1] < 3:
            return super().fit(X, y, feature_types=feature_types)
        super().fit(X, y, feature_types=feature_types)
        Z = np.asarray(self._cpp_model.to_z(X))
        self._z_center, self._z_scale = _extract_whitening_params(X, Z)
        X_aug = self._augment(X)
        self._is_fitted = False
        self._cpp_model = None
        super().fit(X_aug, y, feature_types=None)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._orig_n_features is not None and self._z_center is not None:
            X = self._augment(X)
        return super().predict(X)

    def _augment(self, X):
        Z = (X - self._z_center) / self._z_scale
        return np.column_stack([X, _compute_e3_scalar(Z)])


class E3HybridGeoXGBRegressor(AdaptiveEkGeoXGBRegressor):
    """Legacy alias: AdaptiveEk with max_degree=3."""

    def __init__(self, top_k_max=10, noise_alpha=2.0, noise_floor=0.3, **kwargs):
        super().__init__(
            max_degree=3, top_k_max=top_k_max,
            noise_alpha=noise_alpha, noise_floor=noise_floor,
            include_partials=False, **kwargs
        )

    def e3_summary(self, feature_names=None):
        return self.ek_summary(feature_names)
