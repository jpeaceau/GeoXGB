import numpy as np
from sklearn.neighbors import NearestNeighbors

from hvrt import HVRT

from geoxgb._noise import estimate_noise_modulation, estimate_noise_modulation_classifier


# ---------------------------------------------------------------------------
# KDE-stratified reduction
# ---------------------------------------------------------------------------

def _kde_stratified_reduce(hvrt_model, y, n_keep, variance_weighted=True,
                            random_state=42):
    """
    KDE-stratified reduction using Scott's rule bandwidth as density proxy.

    Complexity: O(n * d) — no k-NN.  Uses centroid distance to rank samples
    by density within each partition (peripheral = sparse, central = dense),
    then selects at evenly-spaced quantile positions to cover every density
    band from outlier to core.  Scott's rule per-partition std serves as a
    cheap sanity check on partition spread.

    Algorithm (per partition)
    -------------------------
    1. Compute centroid of the partition in HVRT z-space — O(m * d).
    2. Rank samples by distance to centroid (descending = sparse-first).
    3. Select budget k_i samples at evenly-spaced quantile positions,
       giving one sample per density band.

    Budget allocation mirrors HVRT's variance_weighted FPS:
    weight_i = Var(y_i) * size_i.

    Parameters
    ----------
    hvrt_model : fitted HVRT instance
        Must have X_z_ and partition_ids_ set (always true after fit()).
    y : ndarray (n,) — gradient signal used for variance-weighted budget.
    n_keep : int — total samples to select.
    variance_weighted : bool
    random_state : int — used only for tiny partitions (< 3 samples).

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
        centroid = X_part_z.mean(axis=0)               # (d,)
        diff = X_part_z - centroid
        dist_to_centroid = np.sqrt((diff * diff).sum(axis=1))  # (m,)

        # Sort peripheral→central (sparse→dense); select k_i samples at
        # evenly-spaced quantile positions to cover each density band.
        order = np.argsort(-dist_to_centroid)
        q_pos = np.round(np.linspace(0, m - 1, k_i)).astype(int)
        chosen = order[q_pos]
        selected.append(p_idx[chosen])

    red_idx = np.concatenate(selected) if selected else np.array([], dtype=int)

    if len(red_idx) > n_keep:
        red_idx = red_idx[:n_keep]

    return red_idx


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
    if method == "kde_stratified":
        red_idx = _kde_stratified_reduce(
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

    n_expanded = 0
    if expand_ratio > 0:
        # Manual KDE expansion.  Floor noise_mod at 0.1 so expansion always
        # happens when the user explicitly sets expand_ratio > 0, even on
        # noisy datasets where noise_mod ≈ 0.
        n_expand = max(0, int(n_orig * expand_ratio * max(noise_mod, 0.1)))
        if n_expand > 0:
            X_syn_hvrt = hvrt_model.expand(
                n=n_expand,
                variance_weighted=variance_weighted,
                bandwidth=bandwidth,
                generation_strategy=generation_strategy,
                adaptive_bandwidth=adaptive_bandwidth,
            )
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
        X_syn_hvrt = hvrt_model.expand(
            n=n_expand,
            variance_weighted=variance_weighted,
            bandwidth=bandwidth,
            generation_strategy=generation_strategy,
            adaptive_bandwidth=adaptive_bandwidth,
        )
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
