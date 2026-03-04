"""
HART: Hyperplane Absolute-error Regression Tree partitioner.

A local GeoXGB variant of HVRT that replaces variance-based splits and
z-score y-normalization with MAD-based equivalents, making the geometry
more robust to outliers and better aligned with L1/MAE objectives.

Key differences from HVRT:
  - _compute_synthetic_target: MAD normalization of y instead of std
  - _fit_tree: uses criterion='absolute_error' (median-minimising splits)
  - _compute_x_component: identical to HVRT (pairwise z-score product sums)

FastHART uses the z-score sum x-component (O(d) vs O(d^2) for HART)
with the same MAD-based internals.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from hvrt._base import _HVRTBase
from hvrt._partitioning import resolve_tree_params


class HART(_HVRTBase):
    """
    MAD-based HVRT partitioner for MAE-optimised geometry.

    Uses pairwise z-score product sums for the x-component (same as HVRT)
    and MAD normalization for the y-extremeness component.  The partition
    tree uses criterion='absolute_error' (median-minimising splits) instead
    of the default 'squared_error', making it robust to outlier gradients.

    Parameters
    ----------
    All parameters are inherited from ``hvrt._base._HVRTBase`` — see that
    class for full documentation.  HART adds no new constructor parameters.
    """

    def _compute_x_component(self, X_z):
        """
        Absolute pairwise cooperation: Σ_{i<j} |z_i|·|z_j| = (‖z‖₁² − ‖z‖₂²) / 2.

        This is the L1 analog of HVRT's signed cooperation Σ z_i·z_j.
        Level sets are cross-polytopes (bipyramids in 3D) rather than the
        quadric cone of HVRT — each face of the cross-polytope corresponds to
        one of the 2^d orthants, separating "above/below median" combinations.

        Sign-invariant: T(−z) = T(z) per individual feature flip.  This
        aligns with MAD normalization (direction-agnostic) and with sign
        gradients (±1, no magnitude), which also carry only direction.

        O(n·d) — no pairwise loop needed via the algebraic identity.
        """
        abs_z = np.abs(X_z)
        l1    = abs_z.sum(axis=1)           # (n,)  ‖z‖₁ per sample
        l2_sq = (X_z * X_z).sum(axis=1)    # (n,)  ‖z‖₂² per sample
        return (l1 * l1 - l2_sq) * 0.5     # = Σ_{i<j} |z_i|·|z_j|

    def _compute_synthetic_target(self, X_z, y=None):
        """
        Blend pairwise x-component with MAD-normalised y-extremeness.

        Replaces HVRT's std/mean normalization with MAD/median for robustness
        to outliers in the gradient signal.
        """
        x_component = self._compute_x_component(X_z)

        if y is None or self.y_weight == 0.0:
            return x_component

        # MAD-based y normalization (robust to outliers)
        y_med = np.median(y)
        y_mad = np.median(np.abs(y - y_med)) * 1.4826 + 1e-10
        y_norm = (y - y_med) / y_mad

        # y-extremeness: absolute deviation from centre (already centred at 0)
        y_extremeness = np.abs(y_norm)
        ye_med = np.median(y_extremeness)
        ye_mad = np.median(np.abs(y_extremeness - ye_med)) * 1.4826 + 1e-10
        y_component = (y_extremeness - ye_med) / ye_mad

        mode = self.y_weight_mode

        if mode == 'fixed':
            return (1.0 - self.y_weight) * x_component + self.y_weight * y_component

        if mode == 'adaptive':
            r = float(np.corrcoef(x_component, y_component)[0, 1])
            eff_weight = self.y_weight * (1.0 - abs(r))
            return (1.0 - eff_weight) * x_component + eff_weight * y_component

        if mode == 'linear':
            y_ext_norm = y_extremeness / (y_extremeness.max() + 1e-10)
            per_w = self.y_weight * y_ext_norm
            return (1.0 - per_w) * x_component + per_w * y_component

        if mode == 'adaptive_linear':
            r = float(np.corrcoef(x_component, y_component)[0, 1])
            attenuation = 1.0 - abs(r)
            y_ext_norm = y_extremeness / (y_extremeness.max() + 1e-10)
            per_w = self.y_weight * attenuation * y_ext_norm
            return (1.0 - per_w) * x_component + per_w * y_component

        raise ValueError(
            f"Unknown y_weight_mode {mode!r}. "
            "Expected 'fixed', 'adaptive', 'linear', or 'adaptive_linear'."
        )

    def _fit_tree(self, X_z, target, n_partitions_override=None, is_reduction=False):
        """
        (Re-)fit the partitioning tree with absolute_error criterion.

        Overrides the base _fit_tree to use criterion='absolute_error'
        (median-minimising splits) instead of 'squared_error', aligning
        the partition geometry with MAE objectives.

        All state updates match the parent implementation exactly.
        """
        n_samples, n_features = X_z.shape
        max_leaf, min_leaf = resolve_tree_params(
            n_samples, n_features,
            n_partitions_override=n_partitions_override,
            n_partitions=self.n_partitions,
            min_samples_leaf=self.min_samples_leaf,
            auto_tune=self.auto_tune,
            is_reduction=is_reduction,
        )

        # Reuse existing tree if params haven't changed
        if (
            hasattr(self, 'tree_')
            and getattr(self, '_tree_max_leaf_', None) == max_leaf
            and getattr(self, '_tree_min_leaf_', None) == min_leaf
        ):
            return

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=max_leaf,
            min_samples_leaf=min_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=0.0,
            random_state=self.random_state,
            splitter=self.tree_splitter,
            criterion='absolute_error',   # KEY: median-minimising splits
        ).fit(X_z, self._last_target_)

        self._tree_max_leaf_ = max_leaf
        self._tree_min_leaf_ = min_leaf
        self.partition_ids_ = self.tree_.apply(X_z)
        self.unique_partitions_ = np.unique(self.partition_ids_)
        self.n_partitions_ = len(self.unique_partitions_)
        self._kdes_ = None
        self._cat_partition_freqs_ = None
        self._strategy_context_cache_ = {}


class FastHART(HART):
    """
    FastHART: L1-norm x-component + MAD-based HART internals.

    Uses ‖z‖₁ = Σ_i |z_i| as the x-component — the L1 radius itself rather
    than the absolute pairwise cooperation.  Level sets are exact L1 balls
    (cross-polytopes), making this the purest expression of L1 geometry.

    O(n·d) — identical complexity to FastHVRT, but sign-invariant where
    FastHVRT's signed sum is not.  Recommended when n > 5000 or d > 20.
    """

    def _compute_x_component(self, X_z):
        """L1 norm: ‖z‖₁ = Σ_i |z_i| — radius in L1 geometry."""
        return np.abs(X_z).sum(axis=1)


class PyramidHART(HART):
    """
    PyramidHART: ℓ1-cooperation statistic A = |S| − ‖z‖₁.

    A := |Σ_i z_i| − Σ_i |z_i| ≤ 0 always (triangle inequality), with
    A = 0 if and only if all components share a sign.  The zero-boundary
    {A = 0} is the union of coordinate hyperplanes {z_i = 0} — a double
    pyramid in d = 3, eight triangular sign-consistent faces on the unit
    sphere in general.

    The critical advantage over HVRT and HART: level sets {A = c} are
    axis-aligned piecewise-linear surfaces, representable **exactly** by
    a decision tree's axis-aligned splits.  HVRT and HART have smooth
    quadric level sets that axis-aligned trees can only approximate.
    PyramidHART eliminates that structural mismatch.

    Algebraic properties (Proposition 1, Peace 2026):
      1. Bounded range: A ∈ [−r√d, 0].  HART's absolute cooperation grows
         as (d−1)r²; PyramidHART's range grows only as √d.
      2. Exact minority interpretation: −A/2 = total magnitude of the
         minority-sign components (whichever sign has smaller ‖·‖₁).
      3. Single-feature outlier cancellation: if |z_k| ≫ Σ_{i≠k} |z_i|,
         |z_k| contributes equally to |S| and ‖z‖₁ and cancels exactly.
         A 50σ spike leaves A virtually unchanged; it shifts HART's
         absolute cooperation by O(50√d).
      4. Degree-1 homogeneity: A(λz) = λA(z) for λ ≥ 0.

    Trade-off: Cov(A, Q) ≠ 0 in general, so PyramidHART does NOT satisfy
    the T ⊥ Q orthogonality of Theorem 1.  E[A] is a nonlinear function of
    the covariance structure (not the clean linear sum of Theorem 2), and
    E[A] is not preserved under isotropic noise (Theorem 3 fails).
    PyramidHART is a complement to HART, preferred when exact tree-level-set
    alignment and single-feature outlier immunity are the priority.
    """

    def _compute_x_component(self, X_z):
        """
        A = |S| − ‖z‖₁ ∈ [−r√d, 0].

        The zero-boundary is the coordinate hyperplane union {z_i = 0};
        level sets are axis-aligned piecewise-linear surfaces that a
        decision tree represents exactly via axis-aligned splits.
        """
        S  = X_z.sum(axis=1)           # (n,)  signed sum S = Σ z_i
        l1 = np.abs(X_z).sum(axis=1)   # (n,)  L1 norm ‖z‖₁
        return np.abs(S) - l1          # ≤ 0 by triangle inequality
