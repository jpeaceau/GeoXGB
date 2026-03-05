"""
experimental/_synergy.py
========================
Synergy strategies combining PyramidHART and HVRT.

From main.pdf Section 5 ("Synergy between PyramidHART and HVRT"):

  A is invariant to single-feature spikes (ell-infty corruption) while
  E[T] is invariant to isotropic additive noise (ell-2 corruption).
  Real data commonly contains both simultaneously, and neither statistic
  alone covers the full corruption model.

Strategy 1 -- Sequential spike isolation
-----------------------------------------
PyramidHART computes A = |S| - ||z||_1 for every sample.  Spike samples
(ell-infty corruption) have A determined solely by remaining non-spike
components (Proposition 1 item 3).  With a large spike of magnitude M,
||z||_1 ~ M while |A| ~ O(d * r) where r is the typical non-spike scale,
so |A|/||z||_1 is suppressed well below the ambient expectation ~(1 - 1/sqrt(d)).

detect_spikes_pyramid() flags samples satisfying BOTH:
  (a) ||z||_1 > median + sigma_threshold * MAD  (spike inflates L1 norm), AND
  (b) |A|/||z||_1 < 0.5 * ambient_median       (A-cancellation suppresses ratio).
HVRT is then fitted on the spike-free bulk, where Theorem 3
(E[T] invariant to isotropic noise) holds without spike interference.

Strategy 2 -- Hierarchical augmentation
----------------------------------------
PyramidHART partitions are outlier-robust, sign-consistent regions with
axis-aligned boundaries exactly representable by decision trees.

_pyramid_hvrt_expand() fits a local HVRT on each partition's samples and
calls HVRT.expand() within the cell.  Synthetic samples inherit:
  - Polyhedral outer structure  (spike robustness from PyramidHART)
  - Hyperboloidal inner T-structure (Theorem 3 per-partition)

Because Theorem 3 is an algebraic identity it holds per-partition as well
as globally, so the noise invariance of E[T] is inherited by the generated
distribution inside each PyramidHART cell.

Activate with generation_strategy='pyramid_hvrt' in GeoXGBRegressor
(requires partitioner='pyramid_hart' for the full synergy guarantee).
"""

import numpy as np

try:
    from hvrt import HVRT
except ImportError:  # pragma: no cover
    HVRT = None


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

def _a_ratio(X_z):
    """
    Compute |A|/||z||_1 = 1 - |S|/||z||_1 in [0, 1] for every sample.

    Near 0: all features share a sign (or spike cancels in A, suppressing ratio).
    Near 1: features are maximally sign-mixed with no dominant direction.

    Parameters
    ----------
    X_z : ndarray (n, d) -- whitened z-score coordinates

    Returns
    -------
    ratio : ndarray (n,) in [0, 1]
    """
    S  = X_z.sum(axis=1)
    l1 = np.abs(X_z).sum(axis=1)
    return 1.0 - np.abs(S) / (l1 + 1e-12)


# ---------------------------------------------------------------------------
# Strategy 1: spike detection
# ---------------------------------------------------------------------------

def detect_spikes_pyramid(X_z, sigma_threshold=2.5):
    """
    Detect ell-infty spike samples via the joint (ratio, ||z||_1) criterion.

    Theory (Proposition 1 item 3): a spike sample with |z_k| >> sum_{i!=k}|z_i|
    has A determined solely by the remaining d-1 components, so:
      |A| ~ O(d * r)   (non-spike scale)
      ||z||_1 ~ |z_k|  (dominated by the spike)
    => |A|/||z||_1 is suppressed far below the ambient value ~(1 - 1/sqrt(d)).

    Detection requires BOTH signals simultaneously:
      1. ||z||_1 significantly exceeds ambient median (outlier in L1 distance)
      2. |A|/||z||_1 < half the ambient median (A-cancellation active)

    Criterion 1 uses the standard robust MAD outlier rule.
    Criterion 2 uses a FIXED relative threshold (ratio < r_med * 0.5) rather
    than a MAD-based one.  The reason: in moderate dimensions (d~10), the
    ratio distribution has high spread across the full [0,1] range (MAD~0.28),
    so a MAD-based lower threshold (r_med - k*MAD) goes to zero or negative and
    never flags anything.  By contrast, spike samples have ratio ≈ 0 while the
    ambient is 1 - 1/sqrt(d); "below half the ambient" captures this suppression
    cleanly and scales correctly with d.

    A natural sign-aligned sample has low ratio but normal ||z||_1, so it
    passes criterion 2 but fails criterion 1 -- avoiding false positives.

    Parameters
    ----------
    X_z : ndarray (n, d)
        Whitened z-score coordinates.  Use hvrt_model._to_z(X) to obtain.
    sigma_threshold : float, default 2.5
        Robust-sigma multiplier for criterion 1 (||z||_1 outlier test).

    Returns
    -------
    is_spike : bool ndarray (n,)
        True for samples identified as ell-infty spike-corrupted.
    ratio : float ndarray (n,)
        The per-sample |A|/||z||_1 values (useful for diagnostics).
    expected : float
        Ambient median |A|/||z||_1 ratio (reference for plotting).
    """
    ratio = _a_ratio(X_z)
    l1    = np.abs(X_z).sum(axis=1)

    # Criterion 1: ||z||_1 is an outlier (spike inflates L1 norm)
    l1_med = np.median(l1)
    l1_mad = np.median(np.abs(l1 - l1_med)) * 1.4826 + 1e-12
    high_l1 = l1 > (l1_med + sigma_threshold * l1_mad)

    # Criterion 2: |A|/||z||_1 is suppressed below half the ambient median.
    # Theory: spike ratio ≈ (d-1)*r_typical / z_spike → 0 for large z_spike,
    # while ambient r_med ≈ 1 - 1/sqrt(d).  A MAD-based lower bound is
    # unreliable here because the ratio has wide spread across [0,1]; use a
    # fixed relative cutoff instead.
    r_med = np.median(ratio)
    low_ratio = ratio < r_med * 0.5

    is_spike = high_l1 & low_ratio
    return is_spike, ratio, float(r_med)


# ---------------------------------------------------------------------------
# Strategy 2: hierarchical augmentation
# ---------------------------------------------------------------------------

def _pyramid_hvrt_expand(pyramid_model, y, n_expand,
                          variance_weighted=True,
                          hvrt_params=None,
                          random_state=42):
    """
    Hierarchical augmentation: HVRT expand() within each PyramidHART cell.

    Algorithm
    ---------
    1. Read existing partition assignments from the fitted PyramidHART model.
    2. Allocate a synthetic-sample budget per partition (Var(y)-weighted if
       variance_weighted=True, otherwise proportional to partition size).
    3. For each partition: fit a local HVRT on that partition's samples,
       then call HVRT.expand() to generate the budgeted synthetic points.
    4. Aggregate synthetic samples across all partitions.

    The per-partition HVRT is geometry-only (y_weight=0) so it captures the
    inner hyperboloidal cooperative structure of that sign-consistent region.
    Theorem 3 (E[T] invariant to isotropic noise) holds per-partition as an
    algebraic identity, so the noise invariance guarantee is inherited by
    every generated point.

    Parameters
    ----------
    pyramid_model : fitted PyramidHART (or HART/HVRT) instance
        Must have .X_, .partition_ids_, .unique_partitions_ attributes.
    y : ndarray (n,)
        Gradient signal used for budget allocation.
    n_expand : int
        Total number of synthetic samples to generate.
    variance_weighted : bool, default True
        Weight per-partition budget by Var(y_i) * partition_size.
    hvrt_params : dict | None
        Extra keyword arguments forwarded to each per-partition HVRT
        constructor.  y_weight and random_state are set automatically.
    random_state : int, default 42

    Returns
    -------
    X_syn : ndarray (n_expand, d) in pyramid_model.X_ space
    """
    if HVRT is None:  # pragma: no cover
        raise ImportError("hvrt package is required for _pyramid_hvrt_expand")

    rng = np.random.RandomState(random_state)

    X_raw      = pyramid_model.X_
    part_ids   = pyramid_model.partition_ids_
    unique_parts = pyramid_model.unique_partitions_
    n_total    = len(X_raw)

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------
    part_mask  = {pid: part_ids == pid for pid in unique_parts}
    part_sizes = np.array([part_mask[p].sum() for p in unique_parts], dtype=float)

    if variance_weighted:
        part_vars = np.array([
            float(y[part_mask[p]].var()) if part_sizes[i] > 1 else 0.0
            for i, p in enumerate(unique_parts)
        ])
        weights = part_sizes * (part_vars + 1e-12)
    else:
        weights = part_sizes.copy()

    total_w = weights.sum()
    if total_w < 1e-12:
        weights = part_sizes
        total_w  = weights.sum()

    budgets = np.round(weights / total_w * n_expand).astype(int)
    diff    = n_expand - budgets.sum()
    if diff != 0:
        budgets[np.argmax(budgets)] += diff

    # ------------------------------------------------------------------
    # Per-partition HVRT expand
    # ------------------------------------------------------------------
    _base_kw = dict(hvrt_params) if hvrt_params else {}
    _base_kw['y_weight']     = 0.0    # geometry-only within each cell
    _base_kw['random_state'] = random_state

    X_syn_parts = []

    for i, pid in enumerate(unique_parts):
        n_budget = int(budgets[i])
        if n_budget < 1:
            continue

        mask   = part_mask[pid]
        X_part = X_raw[mask]
        y_part = y[mask]
        n_part = len(X_part)

        if n_part < 2:
            # Single-sample partition: tile the sample
            X_syn_parts.append(np.tile(X_part, (n_budget, 1)))
            continue

        # Fit local HVRT; min_samples_leaf must be strictly < n_part
        kw = dict(_base_kw)
        kw['min_samples_leaf'] = min(
            kw.get('min_samples_leaf', 5),
            max(1, n_part // 2),
        )

        try:
            hvrt_local = HVRT(**kw)
            hvrt_local.fit(X_part, y=y_part)
            X_syn_i = hvrt_local.expand(n=n_budget, variance_weighted=False)
        except Exception:
            # Fallback: random draw with replacement from this partition
            idx = rng.randint(0, n_part, size=n_budget)
            X_syn_i = X_part[idx]

        X_syn_parts.append(X_syn_i)

    if not X_syn_parts:
        idx = rng.randint(0, n_total, size=n_expand)
        return X_raw[idx]

    X_syn = np.vstack(X_syn_parts)

    # Trim or pad to exactly n_expand
    if len(X_syn) > n_expand:
        X_syn = X_syn[:n_expand]
    elif len(X_syn) < n_expand:
        shortfall = n_expand - len(X_syn)
        idx = rng.randint(0, n_total, size=shortfall)
        X_syn = np.vstack([X_syn, X_raw[idx]])

    return X_syn
