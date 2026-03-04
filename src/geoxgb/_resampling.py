import numpy as np
from sklearn.neighbors import NearestNeighbors

from hvrt import HVRT

from geoxgb._noise import estimate_noise_modulation, estimate_noise_modulation_classifier


# ---------------------------------------------------------------------------
# KDE-stratified reduction
# ---------------------------------------------------------------------------

def _kde_stratified_reduce(hvrt_model, y, n_keep, variance_weighted=True,
                            budget_mode='variance', distance_mode='l2',
                            random_state=42):
    """
    Centroid-distance stratified reduction.

    Ranks samples within each partition by their distance from the partition
    anchor (L2 centroid or L1 geometric median) and selects at evenly-spaced
    quantile positions, covering every density band from peripheral to core.

    Algorithm (per partition)
    -------------------------
    1. Compute anchor: mean (L2, default) or component-wise median (L1).
    2. Compute distances: Euclidean (L2) or Manhattan (L1).
    3. Select k_i samples at evenly-spaced quantile positions along the
       distance-sorted order, covering outlier → core density bands.

    Parameters
    ----------
    hvrt_model : fitted HVRT/HART instance.
    y : ndarray (n,) — gradient signal for budget weighting.
    n_keep : int — total samples to select.
    variance_weighted : bool
    budget_mode : 'variance' | 'mad'
        'variance' — Var(y|partition) weights (L2-natural).
        'mad'      — MAD(y|partition) weights (L1-natural, default for HART).
    distance_mode : 'l2' | 'l1'
        'l2' — Euclidean distance from partition mean (default).
        'l1' — Manhattan distance from partition geometric median
               (component-wise median); aligned with cross-polytope geometry.
    random_state : int

    Returns
    -------
    red_idx : ndarray (n_keep,) — indices into the original training array.
    """
    X_z = hvrt_model.X_z_                        # (n, d) — already computed
    part_ids = hvrt_model.partition_ids_          # (n,)
    unique_parts = hvrt_model.unique_partitions_  # sorted unique partition IDs
    n, d = X_z.shape
    n_parts = len(unique_parts)

    # Map partition IDs to contiguous 0..n_parts-1
    part_pos = np.searchsorted(unique_parts, part_ids)   # (n,)
    part_sizes = np.bincount(part_pos, minlength=n_parts)

    # --- Variance-weighted budget allocation ---
    if variance_weighted:
        if budget_mode == 'mad':
            part_vars = np.array([
                float(np.median(np.abs(y[part_pos == i]
                                       - np.median(y[part_pos == i]))))
                if part_sizes[i] > 1 else 0.0
                for i in range(n_parts)
            ])
        else:  # 'variance' (default — unchanged behaviour)
            part_vars = np.array([
                float(y[part_pos == i].var()) if part_sizes[i] > 1 else 0.0
                for i in range(n_parts)
            ])
        weights = part_vars * part_sizes
        total_w = weights.sum()
        if total_w < 1e-12:
            weights = part_sizes.astype(float)
            total_w = weights.sum()
    else:
        weights = part_sizes.astype(float)
        total_w = weights.sum()

    raw_budgets = weights / total_w * n_keep

    # Largest-remainder rounding
    budgets = np.floor(raw_budgets).astype(int)
    budgets = np.minimum(budgets, part_sizes)
    remainders = raw_budgets - budgets
    deficit = n_keep - int(budgets.sum())
    if deficit > 0:
        top = np.argsort(-remainders)[:deficit]
        for idx in top:
            if budgets[idx] < part_sizes[idx]:
                budgets[idx] += 1

    # Ensure at least 1 sample from each non-empty partition
    budgets = np.maximum(budgets, np.where(part_sizes > 0, 1, 0))
    excess = int(budgets.sum()) - n_keep
    if excess > 0:
        order = np.argsort(-budgets)
        for idx in order:
            if excess <= 0:
                break
            trim = min(excess, budgets[idx] - 1)
            budgets[idx] -= trim
            excess -= trim

    # --- Per-partition centroid-distance selection ---
    rng = np.random.RandomState(random_state)
    selected = []

    for i in range(n_parts):
        k_i = int(budgets[i])
        if k_i <= 0:
            continue

        p_mask = part_pos == i
        p_idx = np.where(p_mask)[0]   # global indices into training array
        m = len(p_idx)

        if k_i >= m:
            selected.append(p_idx)
            continue

        if m <= 2:
            chosen = rng.choice(m, size=k_i, replace=False)
            selected.append(p_idx[chosen])
            continue

        X_part_z = X_z[p_idx]                         # (m, d)
        if distance_mode == 'l1':
            anchor = np.median(X_part_z, axis=0)      # geometric median (L1-optimal)
            diff   = X_part_z - anchor
            dist   = np.abs(diff).sum(axis=1)          # Manhattan distance
        else:
            anchor = X_part_z.mean(axis=0)             # centroid
            diff   = X_part_z - anchor
            dist   = np.sqrt((diff * diff).sum(axis=1))  # Euclidean distance

        # Sort peripheral→central (sparse→dense); select k_i samples at
        # evenly-spaced quantile positions to cover each density band.
        order = np.argsort(-dist)
        q_pos = np.round(np.linspace(0, m - 1, k_i)).astype(int)
        chosen = order[q_pos]
        selected.append(p_idx[chosen])

    red_idx = np.concatenate(selected) if selected else np.array([], dtype=int)

    if len(red_idx) > n_keep:
        red_idx = red_idx[:n_keep]

    return red_idx


def _residual_stratified_reduce(hvrt_model, y, n_keep, variance_weighted=True,
                                 random_state=42):
    """
    Residual-stratified FPS: joint ranking on geometric diversity + residual magnitude.

    Within each partition, samples are ranked jointly on:
      1. Centroid distance (geometric diversity, descending = sparse first)
      2. |y| magnitude (residual magnitude, descending = high gradient first)
    A joint score = 0.5 * centroid_rank_norm + 0.5 * residual_rank_norm is used,
    and k_i samples are selected at evenly-spaced quantile positions along it.

    Budget allocation: MAD-weighted (aligns with HART philosophy).

    Parameters
    ----------
    hvrt_model : fitted HVRT/HART instance
    y : ndarray (n,) — gradient signal (raw residuals or signs)
    n_keep : int
    variance_weighted : bool — if False, uniform partition budgets
    random_state : int

    Returns
    -------
    red_idx : ndarray (n_keep,)
    """
    X_z = hvrt_model.X_z_
    part_ids = hvrt_model.partition_ids_
    unique_parts = hvrt_model.unique_partitions_
    n_parts = len(unique_parts)

    part_pos = np.searchsorted(unique_parts, part_ids)
    part_sizes = np.bincount(part_pos, minlength=n_parts)

    # MAD-weighted budget (robust to outlier gradients)
    if variance_weighted:
        part_mads = np.array([
            float(np.median(np.abs(y[part_pos == i] - np.median(y[part_pos == i]))))
            if part_sizes[i] > 1 else 0.0
            for i in range(n_parts)
        ])
        weights = part_mads * part_sizes
        total_w = weights.sum()
        if total_w < 1e-12:
            weights = part_sizes.astype(float)
            total_w = weights.sum()
    else:
        weights = part_sizes.astype(float)
        total_w = weights.sum()

    raw_budgets = weights / total_w * n_keep

    # Largest-remainder rounding
    budgets = np.floor(raw_budgets).astype(int)
    budgets = np.minimum(budgets, part_sizes)
    remainders = raw_budgets - budgets
    deficit = n_keep - int(budgets.sum())
    if deficit > 0:
        top = np.argsort(-remainders)[:deficit]
        for idx in top:
            if budgets[idx] < part_sizes[idx]:
                budgets[idx] += 1

    # Ensure at least 1 sample from each non-empty partition
    budgets = np.maximum(budgets, np.where(part_sizes > 0, 1, 0))
    excess = int(budgets.sum()) - n_keep
    if excess > 0:
        order = np.argsort(-budgets)
        for idx in order:
            if excess <= 0:
                break
            trim = min(excess, budgets[idx] - 1)
            budgets[idx] -= trim
            excess -= trim

    rng = np.random.RandomState(random_state)
    abs_y = np.abs(y)
    selected = []

    for i in range(n_parts):
        k_i = int(budgets[i])
        if k_i <= 0:
            continue

        p_mask = part_pos == i
        p_idx = np.where(p_mask)[0]
        m = len(p_idx)

        if k_i >= m:
            selected.append(p_idx)
            continue

        if m <= 2:
            chosen = rng.choice(m, size=k_i, replace=False)
            selected.append(p_idx[chosen])
            continue

        X_part_z = X_z[p_idx]
        centroid = X_part_z.mean(axis=0)
        diff = X_part_z - centroid
        dist = np.sqrt((diff * diff).sum(axis=1))

        # Rank descending (0 = most peripheral / highest magnitude)
        # argsort of -dist gives ranks: rank 0 = largest dist
        centroid_rank = np.argsort(np.argsort(-dist)).astype(float)
        residual_rank = np.argsort(np.argsort(-abs_y[p_idx])).astype(float)

        # Normalize ranks to [0, 1]
        centroid_rank_norm = centroid_rank / max(m - 1, 1)
        residual_rank_norm = residual_rank / max(m - 1, 1)

        # Joint score: 0 = most peripheral AND highest residual (select first)
        joint_score = 0.5 * centroid_rank_norm + 0.5 * residual_rank_norm

        # Sort by joint score ascending; pick k_i at evenly-spaced quantile positions
        order = np.argsort(joint_score)
        q_pos = np.round(np.linspace(0, m - 1, k_i)).astype(int)
        chosen = order[q_pos]
        selected.append(p_idx[chosen])

    red_idx = np.concatenate(selected) if selected else np.array([], dtype=int)
    if len(red_idx) > n_keep:
        red_idx = red_idx[:n_keep]
    return red_idx


# ---------------------------------------------------------------------------
# Orthant-stratified FPS
# ---------------------------------------------------------------------------

def _orthant_stratified_reduce(hvrt_model, y, n_keep, variance_weighted=True,
                                random_state=42):
    """
    Orthant-stratified FPS — aligned with cross-polytope (pyramid) geometry.

    The cross-polytope has 2^d facets, each corresponding to one orthant
    (sign pattern s ∈ {±1}^d relative to the partition median).  This
    method ensures every occupied orthant is represented in the reduced set:
    the tree always sees both sides of every feature-median boundary.

    Algorithm (per partition)
    -------------------------
    1. Compute component-wise median of z-scores (geometric median in L1).
    2. Assign each sample to an orthant via sign(z − median_z).
    3. Allocate budget proportionally to orthant occupancy; minimum 1 per
       occupied orthant (up to the total per-partition budget k_i).
    4. Within each orthant, select samples at evenly-spaced quantile
       positions along L1-distance-from-median (most extreme → most central),
       covering the full depth of each facet.

    Budget allocation: MAD-weighted (consistent with L1/pyramid philosophy).

    Parameters
    ----------
    hvrt_model : fitted HART instance (works with HVRT too).
    y : ndarray (n,) — gradient signal for budget weighting.
    n_keep : int
    variance_weighted : bool
    random_state : int

    Returns
    -------
    red_idx : ndarray (n_keep,)
    """
    X_z = hvrt_model.X_z_
    part_ids = hvrt_model.partition_ids_
    unique_parts = hvrt_model.unique_partitions_
    n_parts = len(unique_parts)

    part_pos   = np.searchsorted(unique_parts, part_ids)
    part_sizes = np.bincount(part_pos, minlength=n_parts)

    # MAD-weighted partition budget
    if variance_weighted:
        part_mads = np.array([
            float(np.median(np.abs(y[part_pos == i] - np.median(y[part_pos == i]))))
            if part_sizes[i] > 1 else 0.0
            for i in range(n_parts)
        ])
        weights = part_mads * part_sizes
        total_w = weights.sum()
        if total_w < 1e-12:
            weights  = part_sizes.astype(float)
            total_w  = weights.sum()
    else:
        weights = part_sizes.astype(float)
        total_w = weights.sum()

    raw_budgets = weights / total_w * n_keep
    budgets     = np.floor(raw_budgets).astype(int)
    budgets     = np.minimum(budgets, part_sizes)
    remainders  = raw_budgets - budgets
    deficit     = n_keep - int(budgets.sum())
    if deficit > 0:
        top = np.argsort(-remainders)[:deficit]
        for idx in top:
            if budgets[idx] < part_sizes[idx]:
                budgets[idx] += 1

    budgets = np.maximum(budgets, np.where(part_sizes > 0, 1, 0))
    excess  = int(budgets.sum()) - n_keep
    if excess > 0:
        order = np.argsort(-budgets)
        for idx in order:
            if excess <= 0:
                break
            trim = min(excess, budgets[idx] - 1)
            budgets[idx] -= trim
            excess -= trim

    rng      = np.random.RandomState(random_state)
    selected = []

    for i in range(n_parts):
        k_i = int(budgets[i])
        if k_i <= 0:
            continue

        p_idx = np.where(part_pos == i)[0]
        m     = len(p_idx)

        if k_i >= m:
            selected.append(p_idx)
            continue

        X_part_z = X_z[p_idx]                        # (m, d)
        med_z    = np.median(X_part_z, axis=0)        # geometric median
        delta    = X_part_z - med_z                   # (m, d)
        l1_dist  = np.abs(delta).sum(axis=1)          # (m,) Manhattan dist

        # Determine orthant of each sample: sign(delta), resolving 0 randomly
        signs = np.sign(delta)                        # (m, d)  values {-1, 0, +1}
        zero_mask = signs == 0
        if zero_mask.any():
            signs[zero_mask] = rng.choice([-1, 1], size=int(zero_mask.sum()))

        # Group by orthant (sign pattern as tuple key)
        orthant_map = {}
        for j in range(m):
            key = tuple(signs[j].astype(np.int8))
            orthant_map.setdefault(key, []).append(j)

        # Sort orthants by descending occupancy
        orthant_list = sorted(orthant_map.items(), key=lambda kv: -len(kv[1]))
        n_occ        = len(orthant_list)
        occ_sizes    = np.array([len(v) for _, v in orthant_list])

        if k_i <= n_occ:
            # Fewer budget than orthants: one sample from each of the k_i
            # largest orthants — pick the most extreme (max L1 distance).
            chosen_local = []
            for _, local_idxs in orthant_list[:k_i]:
                best = local_idxs[int(np.argmax(l1_dist[local_idxs]))]
                chosen_local.append(best)
            selected.append(p_idx[chosen_local])
        else:
            # More budget than orthants: distribute proportionally.
            raw_ob   = occ_sizes / occ_sizes.sum() * k_i
            ob       = np.maximum(1, np.floor(raw_ob).astype(int))
            # Fix rounding
            while ob.sum() > k_i:
                ob[np.argmax(ob)] -= 1
            while ob.sum() < k_i:
                # Give remainder to the orthant with the most unallocated samples
                slack = occ_sizes - ob
                ob[np.argmax(slack)] += 1

            chosen_local = []
            for (_, local_idxs), ob_j in zip(orthant_list, ob):
                ob_j = min(int(ob_j), len(local_idxs))
                if ob_j <= 0:
                    continue
                # Within the orthant: quantile positions along L1-distance order
                # (most extreme → most central), so we cover the full facet depth
                dists  = l1_dist[local_idxs]
                order  = np.argsort(-dists)
                q_pos  = np.round(np.linspace(0, len(local_idxs) - 1, ob_j)).astype(int)
                chosen_local.extend([local_idxs[order[q]] for q in q_pos])
            selected.append(p_idx[chosen_local])

    red_idx = np.concatenate(selected) if selected else np.array([], dtype=int)
    if len(red_idx) > n_keep:
        red_idx = red_idx[:n_keep]
    return red_idx


# ---------------------------------------------------------------------------
# L1-native expansion: Laplace product kernel
# ---------------------------------------------------------------------------

def _laplace_expand(hvrt_model, y, n_expand, variance_weighted=True,
                    random_state=42):
    """
    Laplace product-kernel expansion — the natural KDE for L1/pyramid geometry.

    Generates synthetic samples by perturbing parent samples with independent
    Laplace noise on each feature.  The Laplace kernel's level sets are
    cross-polytopes, matching HART's pyramid geometry exactly.  The per-
    feature bandwidth h_j is the within-partition MAD in z-space (scale-
    invariant, consistent with the MAD-based partition target).

    The Laplace distribution is the maximum-entropy distribution for a given
    mean absolute deviation, and is the natural KDE kernel when the error
    distribution is assumed to be Laplace — i.e., when fitting under MAE.

    Algorithm (per partition)
    -------------------------
    1. Bandwidth: h_j = MAD(z_j | partition), floored at 1e-6.
    2. Sample parent uniformly from partition.
    3. Add Laplace(0, h_j) noise independently per feature in z-space.
    4. Convert synthetic z back to original feature space via _from_z().

    Parameters
    ----------
    hvrt_model : fitted HART instance (X_, X_z_, partition_ids_ required).
    y : ndarray (n,) — gradient signal for MAD-weighted budget.
    n_expand : int — total synthetic samples to generate.
    variance_weighted : bool
    random_state : int

    Returns
    -------
    X_syn : ndarray (n_expand, d) — synthetic samples in hvrt_model.X_ space.
        Divide by feature_weights before tree training if fw was applied.
    """
    X_z        = hvrt_model.X_z_
    part_ids   = hvrt_model.partition_ids_
    unique_parts = hvrt_model.unique_partitions_
    n_parts    = len(unique_parts)

    part_pos   = np.searchsorted(unique_parts, part_ids)
    part_sizes = np.bincount(part_pos, minlength=n_parts)

    # MAD-weighted expansion budget
    if variance_weighted:
        part_mads = np.array([
            float(np.median(np.abs(y[part_pos == i] - np.median(y[part_pos == i]))))
            if part_sizes[i] > 1 else 0.0
            for i in range(n_parts)
        ])
        weights = part_mads * part_sizes
        total_w = weights.sum()
        if total_w < 1e-12:
            weights = part_sizes.astype(float)
            total_w = weights.sum()
    else:
        weights = part_sizes.astype(float)
        total_w = weights.sum()

    raw_budgets = weights / total_w * n_expand
    budgets     = np.floor(raw_budgets).astype(int)
    remainders  = raw_budgets - budgets
    deficit     = n_expand - int(budgets.sum())
    if deficit > 0:
        top = np.argsort(-remainders)[:deficit]
        for idx in top:
            budgets[idx] += 1

    rng        = np.random.RandomState(random_state)
    synthetics = []

    for i in range(n_parts):
        n_i = int(budgets[i])
        if n_i <= 0:
            continue

        p_idx    = np.where(part_pos == i)[0]
        X_part_z = X_z[p_idx]                    # (m, d)

        # Per-feature bandwidth: MAD of z-scores within the partition
        med_z  = np.median(X_part_z, axis=0)     # (d,)
        h      = np.median(np.abs(X_part_z - med_z), axis=0)  # (d,) per-feature MAD
        h      = np.maximum(h, 1e-6)              # floor against degenerate partitions

        # Sample parents uniformly; add independent Laplace noise per feature
        parent_idx = rng.randint(0, len(p_idx), size=n_i)
        parents_z  = X_part_z[parent_idx]         # (n_i, d)
        noise_z    = rng.laplace(0.0, h[np.newaxis, :], size=(n_i, X_z.shape[1]))
        syn_z      = parents_z + noise_z           # (n_i, d) — in z-space

        # Convert from z-space back to X_hvrt space
        synthetics.append(hvrt_model._from_z(syn_z))

    if synthetics:
        return np.vstack(synthetics)
    return np.empty((0, hvrt_model.X_.shape[1]))


# ---------------------------------------------------------------------------
# L1-native expansion: in-orthant simplex interpolation (mixup)
# ---------------------------------------------------------------------------

def _simplex_mixup_expand(hvrt_model, y, n_expand, variance_weighted=True,
                           random_state=42):
    """
    In-orthant simplex interpolation for pyramid geometry.

    Synthetic samples are convex combinations of two training samples that
    share the same sign pattern (orthant) relative to the partition median.
    Because both parents are on the same simplex face of the cross-polytope,
    any convex combination of them lies on or inside that face — the
    synthetic sample inherits the same "above/below median" pattern for
    every feature, preserving the gradient-sign homogeneity of the orthant.

    This is a parameter-free method (no bandwidth tuning) that generates
    strictly in-distribution samples — every synthetic point lies inside
    the convex hull of the training data within its orthant.

    Algorithm (per partition)
    -------------------------
    1. Assign each sample to an orthant via sign(X_z − median(X_z)).
    2. Select an orthant proportional to occupancy.
    3. Pick two samples a, b from the orthant; interpolate X_a + λ(X_b − X_a),
       λ ~ Uniform(0, 1).
    4. Single-sample orthants contribute that sample unchanged.

    Interpolation is in original X_hvrt space (feature units, not z-space),
    so the result is always a realistic convex combination.

    Parameters
    ----------
    hvrt_model : fitted HART instance.
    y : ndarray (n,) — gradient signal for MAD-weighted budget.
    n_expand : int
    variance_weighted : bool
    random_state : int

    Returns
    -------
    X_syn : ndarray (n_expand, d) in hvrt_model.X_ space.
    """
    X_raw      = hvrt_model.X_                    # original (or fw-scaled) X
    X_z        = hvrt_model.X_z_
    part_ids   = hvrt_model.partition_ids_
    unique_parts = hvrt_model.unique_partitions_
    n_parts    = len(unique_parts)

    part_pos   = np.searchsorted(unique_parts, part_ids)
    part_sizes = np.bincount(part_pos, minlength=n_parts)

    # MAD-weighted expansion budget
    if variance_weighted:
        part_mads = np.array([
            float(np.median(np.abs(y[part_pos == i] - np.median(y[part_pos == i]))))
            if part_sizes[i] > 1 else 0.0
            for i in range(n_parts)
        ])
        weights = part_mads * part_sizes
        total_w = weights.sum()
        if total_w < 1e-12:
            weights = part_sizes.astype(float)
            total_w = weights.sum()
    else:
        weights = part_sizes.astype(float)
        total_w = weights.sum()

    raw_budgets = weights / total_w * n_expand
    budgets     = np.floor(raw_budgets).astype(int)
    remainders  = raw_budgets - budgets
    deficit     = n_expand - int(budgets.sum())
    if deficit > 0:
        top = np.argsort(-remainders)[:deficit]
        for idx in top:
            budgets[idx] += 1

    rng        = np.random.RandomState(random_state)
    synthetics = []

    for i in range(n_parts):
        n_i = int(budgets[i])
        if n_i <= 0:
            continue

        p_idx    = np.where(part_pos == i)[0]
        X_part   = X_raw[p_idx]                   # (m, d) in X_hvrt space
        X_part_z = X_z[p_idx]                     # (m, d) in z-space

        # Group by orthant in z-space
        med_z    = np.median(X_part_z, axis=0)
        delta    = X_part_z - med_z
        signs    = np.sign(delta)
        zero_mask = signs == 0
        if zero_mask.any():
            signs[zero_mask] = rng.choice([-1, 1], size=int(zero_mask.sum()))

        orthant_map = {}
        for j in range(len(p_idx)):
            key = tuple(signs[j].astype(np.int8))
            orthant_map.setdefault(key, []).append(j)

        orthant_list  = list(orthant_map.items())
        orthant_sizes = np.array([len(v) for _, v in orthant_list])
        orthant_probs = orthant_sizes / orthant_sizes.sum()

        # Generate n_i synthetic samples by in-orthant interpolation
        syn = np.empty((n_i, X_part.shape[1]))
        for k in range(n_i):
            # Choose orthant proportional to occupancy
            orth_idx   = rng.choice(len(orthant_list), p=orthant_probs)
            local_idxs = orthant_list[orth_idx][1]

            if len(local_idxs) == 1:
                syn[k] = X_part[local_idxs[0]]
            else:
                a, b = rng.choice(len(local_idxs), size=2, replace=False)
                lam  = rng.uniform(0.0, 1.0)
                syn[k] = X_part[local_idxs[a]] + lam * (
                    X_part[local_idxs[b]] - X_part[local_idxs[a]]
                )
        synthetics.append(syn)

    if synthetics:
        return np.vstack(synthetics)
    return np.empty((0, X_raw.shape[1]))


class _ResampleResult:
    """Container for one resample event."""

    __slots__ = (
        "X", "y", "hvrt_model", "trace", "noise_modulation",
        "n_reduced", "n_expanded", "n_original",
        "red_idx",   # indices into original X that produced the reduced real samples
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _snap_categoricals(X_syn, feature_types, fw=None):
    """Round categorical columns in synthetic samples back to nearest valid category.

    Works in hvrt-scaled space: when fw is provided, snaps to the nearest multiple of
    fw[j] (i.e. round(val / fw[j]) * fw[j]); otherwise rounds to nearest integer.
    """
    if feature_types is None:
        return X_syn
    cat_cols = [j for j, ft in enumerate(feature_types) if ft == 'categorical']
    if not cat_cols:
        return X_syn
    X_out = X_syn.copy()
    for j in cat_cols:
        scale = float(fw[j]) if fw is not None else 1.0
        X_out[:, j] = np.round(X_out[:, j] / scale) * scale
    return X_out


def _knn_assign_y(X_syn, X_red, y_red, hvrt_model, strategy="auto"):
    """
    Assign target values to synthetic samples.

    Parameters
    ----------
    X_syn : ndarray (m, p) — synthetic sample coordinates
    X_red : ndarray (r, p) — reduced real sample coordinates
    y_red : ndarray (r,)   — target values for reduced real samples
    hvrt_model : fitted HVRT instance (provides _to_z and tree_)
    strategy : str
        'knn'      — global inverse-distance weighted k-NN (k=3) in z-space
        'part-idw' — intra-partition IDW; k-NN fallback for empty partitions
        'auto'     — part-idw when X_red spans >= 50 unique HVRT partitions,
                     k-NN otherwise (safe for sparse auto-expand regimes)

    Returns
    -------
    y_syn : ndarray (m,)
    """
    k = min(3, len(X_red))
    X_red_z = hvrt_model._to_z(X_red)
    X_syn_z = hvrt_model._to_z(X_syn)

    # Resolve 'auto': count unique partitions represented in the reduced set.
    # Reuse the computed leaves for part-idw so we don't call apply() twice.
    _red_leaves = None
    if strategy == "auto":
        _red_leaves = hvrt_model.tree_.apply(X_red_z.astype(np.float32))
        strategy = "part-idw" if len(np.unique(_red_leaves)) >= 50 else "knn"

    if strategy == "knn":
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_red_z)
        dists, idxs = nn.kneighbors(X_syn_z)
        w = 1.0 / (dists + 1e-10)
        w /= w.sum(axis=1, keepdims=True)
        return np.sum(w * y_red[idxs], axis=1)

    # Intra-partition IDW path
    if _red_leaves is None:
        _red_leaves = hvrt_model.tree_.apply(X_red_z.astype(np.float32))
    syn_leaves = hvrt_model.tree_.apply(X_syn_z.astype(np.float32))

    leaf_idx = {}
    for i, leaf in enumerate(_red_leaves):
        leaf_idx.setdefault(leaf, []).append(i)

    # Build a global k-NN fallback for synthetic samples whose partition has
    # no reduced representatives (sparse edge case after aggressive FPS).
    nn_fb = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn_fb.fit(X_red_z)

    y_syn = np.empty(len(X_syn))
    for i, leaf in enumerate(syn_leaves):
        idxs = leaf_idx.get(leaf)
        if not idxs:
            fb_dists, fb_idxs = nn_fb.kneighbors(X_syn_z[i : i + 1])
            w = 1.0 / (fb_dists[0] + 1e-10)
            w /= w.sum()
            y_syn[i] = np.dot(w, y_red[fb_idxs[0]])
        elif len(idxs) == 1:
            y_syn[i] = y_red[idxs[0]]
        else:
            dists = np.linalg.norm(X_red_z[idxs] - X_syn_z[i], axis=1)
            w = 1.0 / (dists + 1e-10)
            w /= w.sum()
            y_syn[i] = np.dot(w, y_red[idxs])

    return y_syn


def hvrt_resample(
    X,
    y,
    *,
    reduce_ratio,
    expand_ratio,
    y_weight,
    n_partitions,
    method,
    variance_weighted,
    bandwidth,
    auto_noise,
    feature_types,
    random_state,
    auto_expand=False,
    min_train_samples=5000,
    is_classifier=False,
    y_cls=None,
    hvrt_cache=None,
    min_samples_leaf=None,
    max_samples_leaf=None,
    generation_strategy=None,
    adaptive_bandwidth=False,
    feature_weights=None,
    assignment_strategy="auto",
    hvrt_params=None,
    hvrt_tree_splitter=None,
    partitioner='hvrt',
):
    """
    Geometry-aware resample via HVRT.

    Steps
    -----
    1. Fit HVRT on X with y-blended synthetic target (skipped when
       ``hvrt_cache`` is provided — geometry is reused from prior fit).
    2. Estimate noise → modulate reduce/expand aggressiveness.
       - Regression: between/within variance of partition mean_abs_z.
       - Classification: partition purity vs random baseline (UPDATE-003).
    3. Reduce via FPS (variance-weighted budget allocation).
    4. Optionally expand sparse partitions via KDE:
       - Manual: when ``expand_ratio > 0``.
       - Auto:   when ``auto_expand=True``, ``expand_ratio == 0``, and
                 ``n_reduced < min_train_samples`` (UPDATE-001).
    5. Assign y to synthetic samples via ``assignment_strategy``
       (default 'auto': part-idw when >= 50 partitions, k-NN otherwise).

    Parameters
    ----------
    X : ndarray (n, p)
    y : ndarray (n,)  — gradient signal (residuals or raw targets)
    reduce_ratio : float   — base fraction to keep via FPS
    expand_ratio : float   — fraction of n to synthesize (0 = off)
    y_weight : float       — HVRT blend (0=unsupervised, 1=y-driven)
    n_partitions : int|None
    method : str           — FPS variant ('fps', 'medoid_fps', …)
    variance_weighted : bool
    bandwidth : float      — KDE bandwidth
    auto_noise : bool      — apply noise modulation
    feature_types : list|None
    random_state : int
    auto_expand : bool     — auto-expand small datasets to min_train_samples
    min_train_samples : int — target training set size for auto_expand
    is_classifier : bool   — use class-conditional noise estimator
    y_cls : ndarray of int | None — original class labels for classifier
                                    noise estimation (UPDATE-003)
    hvrt_cache : HVRT instance | None — if provided, skip fit() and reuse
                                        the cached model's geometry (tree_,
                                        partition_ids_, X_z_, KDEs).
    min_samples_leaf : int | None — passed to HVRT. None = HVRT auto-tunes.
    assignment_strategy : str — y-assignment method for synthetic samples.
        'auto' (default) — part-idw when X_red spans >= 50 unique HVRT
        partitions, global k-NN otherwise.  'knn' and 'part-idw' force the
        respective method unconditionally.
    generation_strategy : str | None — KDE sampling strategy passed to expand().
        None = multivariate KDE (default). 'epanechnikov', 'univariate_kde_copula',
        'bootstrap_noise', or a custom callable are also accepted.
    adaptive_bandwidth : bool — if True, scale per-partition KDE bandwidth with
        local expansion ratio. Only valid with the default multivariate KDE strategy.
    feature_weights : array-like of shape (p,) | None — per-feature scaling applied
        to X *before* HVRT sees it.  Features with weight > 1 dominate the geometry;
        weight < 1 de-emphasizes them.  Trees are always fitted on the original
        unscaled X.  Use ``Gardener.recommend_feature_weights()`` to derive weights
        from the boosting vs partition importance divergence.
    hvrt_params : dict | None — extra keyword arguments forwarded to the HVRT
        constructor.  Useful for parameters not explicitly exposed by GeoXGB,
        such as ``max_depth``, ``n_jobs``, or ``min_samples_per_partition``.
        Named GeoXGB parameters (y_weight, bandwidth, n_partitions,
        min_samples_leaf, random_state) take precedence over any overlapping
        keys in hvrt_params.
    partitioner : str, default 'hvrt'
        Which partitioner class to use when no ``hvrt_cache`` is provided.
        ``'hvrt'`` — standard HVRT (variance-based, squared_error splits).
        ``'hart'`` — HART (MAD-based y-normalization, absolute_error splits).
        ``'pyramid_hart'`` — PyramidHART (A = |S|−‖z‖₁; polyhedral level
            sets exactly representable by axis-aligned decision tree splits;
            exact single-feature outlier cancellation; degree-1 homogeneity).
            Trade-off: loses T⊥Q orthogonality and noise-invariance.

    Returns
    -------
    _ResampleResult
    """
    n_orig = len(X)

    # Scale X for HVRT geometry if feature_weights are provided.
    # Trees are always fitted on the original unscaled X (via red_idx).
    if feature_weights is not None:
        fw = np.asarray(feature_weights, dtype=np.float64)
        X_hvrt = X * fw[np.newaxis, :]
    else:
        fw = None
        X_hvrt = X

    if hvrt_cache is not None:
        hvrt_model = hvrt_cache
    else:
        _hvrt_kwargs = dict(hvrt_params) if hvrt_params else {}
        # Derive n_partitions from max_samples_leaf when n_partitions is not
        # set explicitly.  Ceiling division: -((-n) // M) avoids import math.
        # min_samples_leaf remains active as the floor constraint.
        if max_samples_leaf is not None and n_partitions is None:
            _eff_n_parts = -((-n_orig) // max_samples_leaf)
        else:
            _eff_n_parts = n_partitions
        # Named GeoXGB params always win over anything in hvrt_params.
        _hvrt_kwargs.update(
            y_weight=y_weight,
            bandwidth=bandwidth,
            random_state=random_state,
            n_partitions=_eff_n_parts,
            min_samples_leaf=min_samples_leaf,
        )
        if hvrt_tree_splitter is not None:
            _hvrt_kwargs["tree_splitter"] = hvrt_tree_splitter
        if partitioner == 'hart':
            from geoxgb._hart import HART as _HART
            hvrt_model = _HART(**_hvrt_kwargs)
        elif partitioner == 'pyramid_hart':
            from geoxgb._hart import PyramidHART as _PyramidHART
            hvrt_model = _PyramidHART(**_hvrt_kwargs)
        else:
            hvrt_model = HVRT(**_hvrt_kwargs)
        hvrt_model.fit(X_hvrt, y=y, feature_types=feature_types)

    # Noise estimation: class-conditional for classifiers, global for regressors.
    # Pass X_hvrt so _to_z() operates in the same space HVRT was fitted on.
    if auto_noise:
        if is_classifier and y_cls is not None:
            noise_mod = estimate_noise_modulation_classifier(hvrt_model, y_cls, X_hvrt)
        else:
            noise_mod = estimate_noise_modulation(hvrt_model, y, X_hvrt)
    else:
        noise_mod = 1.0

    # Noisier data → keep more samples (less aggressive reduction)
    eff_reduce = reduce_ratio + (1.0 - noise_mod) * (1.0 - reduce_ratio)
    eff_reduce = min(eff_reduce, 1.0)

    n_reduce = max(10, int(n_orig * eff_reduce))
    _is_hart  = partitioner in ('hart', 'pyramid_hart')
    _bud_mode = 'mad' if _is_hart else 'variance'

    if method == "kde_stratified":
        red_idx = _kde_stratified_reduce(
            hvrt_model, y, n_reduce,
            variance_weighted=variance_weighted,
            budget_mode=_bud_mode,
            distance_mode='l2',
            random_state=random_state,
        )
        X_red_hvrt = X_hvrt[red_idx]
    elif method == "kde_stratified_l1":
        red_idx = _kde_stratified_reduce(
            hvrt_model, y, n_reduce,
            variance_weighted=variance_weighted,
            budget_mode=_bud_mode,
            distance_mode='l1',
            random_state=random_state,
        )
        X_red_hvrt = X_hvrt[red_idx]
    elif method == "residual_stratified":
        red_idx = _residual_stratified_reduce(
            hvrt_model, y, n_reduce,
            variance_weighted=variance_weighted,
            random_state=random_state,
        )
        X_red_hvrt = X_hvrt[red_idx]
    elif method == "orthant_stratified":
        red_idx = _orthant_stratified_reduce(
            hvrt_model, y, n_reduce,
            variance_weighted=variance_weighted,
            random_state=random_state,
        )
        X_red_hvrt = X_hvrt[red_idx]
    else:
        X_red_hvrt, red_idx = hvrt_model.reduce(
            n=n_reduce,
            method=method,
            variance_weighted=variance_weighted,
            return_indices=True,
        )
    # Use unscaled X for tree training; keep X_red_hvrt for KDE y-assignment.
    X_red = X[red_idx] if fw is not None else X_red_hvrt
    y_red = y[red_idx]
    n_reduced = len(X_red)

    def _do_expand(n_exp):
        """Dispatch expansion to the appropriate strategy, return X_syn_hvrt."""
        if generation_strategy == 'laplace':
            return _laplace_expand(
                hvrt_model, y, n_exp,
                variance_weighted=variance_weighted,
                random_state=random_state,
            )
        if generation_strategy == 'simplex_mixup':
            return _simplex_mixup_expand(
                hvrt_model, y, n_exp,
                variance_weighted=variance_weighted,
                random_state=random_state,
            )
        if generation_strategy == 'pyramid_hvrt':
            from geoxgb.experimental._synergy import _pyramid_hvrt_expand
            return _pyramid_hvrt_expand(
                hvrt_model, y, n_exp,
                variance_weighted=variance_weighted,
                random_state=random_state,
            )
        # Default: delegate to HVRT's built-in expand()
        return hvrt_model.expand(
            n=n_exp,
            variance_weighted=variance_weighted,
            bandwidth=bandwidth,
            generation_strategy=generation_strategy,
            adaptive_bandwidth=adaptive_bandwidth,
        )

    n_expanded = 0
    if expand_ratio > 0:
        # Manual expansion.  Floor noise_mod at 0.1 so expansion always
        # happens when the user explicitly sets expand_ratio > 0, even on
        # noisy datasets where noise_mod ≈ 0.
        n_expand = max(0, int(n_orig * expand_ratio * max(noise_mod, 0.1)))
        if n_expand > 0:
            X_syn_hvrt = _do_expand(n_expand)
            X_syn_hvrt = _snap_categoricals(X_syn_hvrt, feature_types, fw=fw)
            y_syn = _knn_assign_y(X_syn_hvrt, X_red_hvrt, y_red, hvrt_model,
                                   strategy=assignment_strategy)
            X_syn = (X_syn_hvrt / fw[np.newaxis, :]) if fw is not None else X_syn_hvrt
            X_out = np.vstack([X_red, X_syn])
            y_out = np.concatenate([y_red, y_syn])
            n_expanded = n_expand
        else:
            X_out, y_out = X_red, y_red

    elif auto_expand and n_reduced < min_train_samples:
        # Auto-expand: bring training set toward min_train_samples (UPDATE-001).
        # Scale expansion by max(noise_mod, 0.1) — the same floor applied to
        # manual expand_ratio.  When noise_mod collapses to 0 at later refits
        # (converged gradients look structureless to the SNR estimator), this
        # prevents flooding the training set with near-zero synthetic gradients
        # that dilute the residual signal.  Floor at 0.1 ensures some expansion
        # always occurs for genuinely small datasets.
        #
        # Cap the expansion target at 5× n_orig so that small datasets (e.g.
        # n=200) are not swamped with synthetic samples.  Without this cap,
        # min_train_samples=5000 on a 200-sample dataset produces a 25× synthetic
        # expansion where the KDE samples dominate the real signal.  The cap is
        # inactive for n_orig >= 1000 (5×1000=5000 = min_train_samples default).
        _eff_min = min(min_train_samples, max(n_orig * 5, 1000))
        n_expand = max(0, int((_eff_min - n_reduced) * max(noise_mod, 0.1)))
        X_syn_hvrt = _do_expand(n_expand)
        X_syn_hvrt = _snap_categoricals(X_syn_hvrt, feature_types, fw=fw)
        y_syn = _knn_assign_y(X_syn_hvrt, X_red_hvrt, y_red, hvrt_model,
                               strategy=assignment_strategy)
        X_syn = (X_syn_hvrt / fw[np.newaxis, :]) if fw is not None else X_syn_hvrt
        X_out = np.vstack([X_red, X_syn])
        y_out = np.concatenate([y_red, y_syn])
        n_expanded = n_expand

    else:
        X_out, y_out = X_red, y_red

    return _ResampleResult(
        X=X_out,
        y=y_out,
        hvrt_model=hvrt_model,
        trace=hvrt_model.get_partitions(),
        noise_modulation=noise_mod,
        n_reduced=n_reduced,
        n_expanded=n_expanded,
        n_original=n_orig,
        red_idx=red_idx,
    )
