import numpy as np
from sklearn.neighbors import NearestNeighbors

from hvrt import HVRT

from geoxgb._noise import estimate_noise_modulation, estimate_noise_modulation_classifier


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
    generation_strategy=None,
    adaptive_bandwidth=False,
    feature_weights=None,
    assignment_strategy="auto",
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
        hvrt_model = HVRT(
            y_weight=y_weight,
            bandwidth=bandwidth,
            random_state=random_state,
            n_partitions=n_partitions,
            min_samples_leaf=min_samples_leaf,
        )
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
        # Auto-expand: bring training set up to min_train_samples (UPDATE-001).
        # Expands from the full original distribution (HVRT fitted on all of X).
        n_expand = min_train_samples - n_reduced
        X_syn_hvrt = hvrt_model.expand(
            n=n_expand,
            variance_weighted=variance_weighted,
            bandwidth=bandwidth,
            generation_strategy=generation_strategy,
            adaptive_bandwidth=adaptive_bandwidth,
        )
        y_syn = _knn_assign_y(X_syn_hvrt, X_red_hvrt, y_red, hvrt_model)
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
