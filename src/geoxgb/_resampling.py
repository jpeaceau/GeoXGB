import numpy as np
from sklearn.neighbors import NearestNeighbors

from hvrt import HVRT

from geoxgb._noise import estimate_noise_modulation, estimate_noise_modulation_classifier


class _ResampleResult:
    """Container for one resample event."""

    __slots__ = (
        "X", "y", "hvrt_model", "trace", "noise_modulation",
        "n_reduced", "n_expanded", "n_original",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _knn_assign_y(X_syn, X_red, y_red, hvrt_model):
    """
    Assign target values to synthetic samples via k-NN weighted average
    (k=3, inverse-distance weights in z-score space).

    Parameters
    ----------
    X_syn : ndarray (m, p) — synthetic sample coordinates
    X_red : ndarray (r, p) — reduced real sample coordinates
    y_red : ndarray (r,)   — target values for reduced real samples
    hvrt_model : fitted HVRT instance (provides _to_z transformation)

    Returns
    -------
    y_syn : ndarray (m,)
    """
    k = min(3, len(X_red))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    X_red_z = hvrt_model._to_z(X_red)
    X_syn_z = hvrt_model._to_z(X_syn)
    nn.fit(X_red_z)
    dists, idxs = nn.kneighbors(X_syn_z)
    w = 1.0 / (dists + 1e-10)
    w /= w.sum(axis=1, keepdims=True)
    return np.sum(w * y_red[idxs], axis=1)


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
    5. Assign y to synthetic samples via k-NN weighted average (k=3,
       inverse-distance weights in z-score space).

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

    Returns
    -------
    _ResampleResult
    """
    n_orig = len(X)

    if hvrt_cache is not None:
        hvrt_model = hvrt_cache
    else:
        hvrt_model = HVRT(
            y_weight=y_weight,
            bandwidth=bandwidth,
            random_state=random_state,
            n_partitions=n_partitions,
        )
        hvrt_model.fit(X, y=y, feature_types=feature_types)

    # Noise estimation: class-conditional for classifiers, global for regressors
    if auto_noise:
        if is_classifier and y_cls is not None:
            noise_mod = estimate_noise_modulation_classifier(hvrt_model, y_cls, X)
        else:
            noise_mod = estimate_noise_modulation(hvrt_model)
    else:
        noise_mod = 1.0

    # Noisier data → keep more samples (less aggressive reduction)
    eff_reduce = reduce_ratio + (1.0 - noise_mod) * (1.0 - reduce_ratio)
    eff_reduce = min(eff_reduce, 1.0)

    n_reduce = max(10, int(n_orig * eff_reduce))
    X_red, red_idx = hvrt_model.reduce(
        n=n_reduce,
        method=method,
        variance_weighted=variance_weighted,
        return_indices=True,
    )
    y_red = y[red_idx]
    n_reduced = len(X_red)

    n_expanded = 0
    if expand_ratio > 0:
        # Manual KDE expansion
        n_expand = max(0, int(n_orig * expand_ratio * noise_mod))
        if n_expand > 0:
            X_syn = hvrt_model.expand(
                n=n_expand,
                variance_weighted=variance_weighted,
                bandwidth=bandwidth,
            )
            y_syn = _knn_assign_y(X_syn, X_red, y_red, hvrt_model)
            X_out = np.vstack([X_red, X_syn])
            y_out = np.concatenate([y_red, y_syn])
            n_expanded = n_expand
        else:
            X_out, y_out = X_red, y_red

    elif auto_expand and n_reduced < min_train_samples:
        # Auto-expand: bring training set up to min_train_samples (UPDATE-001).
        # Expands from the full original distribution (HVRT fitted on all of X).
        n_expand = min_train_samples - n_reduced
        X_syn = hvrt_model.expand(
            n=n_expand,
            variance_weighted=variance_weighted,
            bandwidth=bandwidth,
        )
        y_syn = _knn_assign_y(X_syn, X_red, y_red, hvrt_model)
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
    )
