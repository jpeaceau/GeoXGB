import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_noise_modulation(hvrt_model, y, X):
    """
    Estimate signal-to-noise via k-NN local-mean variance (regression path).

    Measures the fraction of y variance explained by local neighborhood means
    in HVRT z-score space.  Clean data → nearby samples (in X-space) have
    similar y values → local means vary widely → high signal-to-noise ratio →
    high modulation.  Noisy data → local means are dominated by noise and vary
    little relative to total y variance → low modulation → ``eff_reduce → 1.0``
    (keep all samples), which is the correct behaviour.

    This approach is robust regardless of the number of HVRT partitions: it
    operates in the same z-score space as HVRT but uses k-NN smoothing to
    directly measure y predictability from X, rather than relying on the
    partition count or structure.

    Formula
    -------
    local_mean_i = mean(y[k-nearest-neighbours(i)])    (excludes self)
    snr          = Var(local_means) / Var(y)
    modulation   = clip((snr − 0.15) / 0.45, 0, 1)

    This maps:
      - snr > 0.60 → modulation = 1.0  (clean, full resampling)
      - snr < 0.15 → modulation = 0.0  (noisy, pass-through)

    The lower threshold (0.15) comfortably exceeds the expected snr of pure
    white noise with k=10 neighbours (≈ 1/k = 0.10), ensuring pure-noise
    datasets are correctly identified as such.

    Parameters
    ----------
    hvrt_model : fitted HVRT instance
    y : ndarray of shape (n,)
        Gradient signal (residuals or raw targets) for each sample in X.
    X : ndarray of shape (n, p)
        Original feature matrix (z-scored internally via hvrt_model._to_z).

    Returns
    -------
    float in [0, 1] : 1.0 = clean signal, 0.0 = pure noise
    """
    y = np.asarray(y, dtype=float)
    y_var = y.var()
    if y_var < 1e-12:
        return 1.0

    X_z = hvrt_model._to_z(X)
    N = len(y)
    k = min(10, max(3, N // 30))

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X_z)
    _, indices = nn.kneighbors(X_z)

    # Local mean y for each sample, excluding self (column 0 is self)
    local_means = np.array([y[indices[i, 1:]].mean() for i in range(N)])

    snr = local_means.var() / y_var
    return float(np.clip((snr - 0.15) / 0.45, 0.0, 1.0))


def estimate_noise_modulation_classifier(hvrt_model, y_cls, X):
    """
    Class-conditional noise estimation for classification (UPDATE-003).

    Uses Gini impurity reduction: measures how much the HVRT partition
    structure reduces class mixing relative to the global class distribution.
    This correctly handles severe class imbalance (1-5% minority rate) where
    the spec's weighted-average-purity formula degenerates — weighted purity
    converges to the majority-class baseline even on perfectly separable data
    because pure-majority partitions dominate the average.

    Formula
    -------
    gini_reduction = (global_gini - weighted_partition_gini) / global_gini

    This maps:
      - gini_reduction → 1.0 : perfect class separation in partitions
      - gini_reduction → 0.0 : classes randomly distributed across partitions

    Parameters
    ----------
    hvrt_model : fitted HVRT instance
    y_cls : ndarray of int, shape (n,)
        Original integer class labels (0, 1, ..., k-1) for every sample in X.
    X : ndarray of shape (n, p)
        Original feature matrix (z-scored internally via hvrt_model._to_z).

    Returns
    -------
    float in [0, 1] : 1.0 = perfectly separated classes, 0.0 = random mix
    """
    # Assign each sample to a partition via the HVRT decision tree
    X_z = hvrt_model._to_z(X).astype(np.float32)
    partition_ids = hvrt_model.tree_.apply(X_z)

    y_cls = np.asarray(y_cls, dtype=int)
    n = len(y_cls)
    n_classes = int(y_cls.max()) + 1

    # Global Gini impurity
    global_counts = np.bincount(y_cls, minlength=n_classes).astype(float) / n
    global_gini = 1.0 - float(np.sum(global_counts ** 2))

    # Degenerate: all samples in one class → perfectly clean by definition
    if global_gini < 1e-10:
        return 1.0

    # Weighted Gini across partitions
    unique_parts = np.unique(partition_ids)
    weighted_gini = 0.0
    for pid in unique_parts:
        mask = partition_ids == pid
        n_p = float(mask.sum())
        if n_p < 2:
            continue
        part_counts = np.bincount(y_cls[mask], minlength=n_classes).astype(float) / n_p
        part_gini = 1.0 - float(np.sum(part_counts ** 2))
        weighted_gini += (n_p / n) * part_gini

    # Gini reduction: 1.0 = perfect separation, 0.0 = no improvement over prior
    gini_reduction = (global_gini - weighted_gini) / global_gini
    return float(np.clip(gini_reduction, 0.0, 1.0))
