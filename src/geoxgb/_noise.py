import numpy as np


def estimate_noise_modulation(hvrt_model):
    """
    Estimate signal-to-noise from HVRT partition structure (regression path).

    Uses between-partition variance of ``mean_abs_z`` (structural signal)
    relative to total variance (between + within).  Clean data → partitions
    capture real structure → high between-partition, low within-partition
    variance.  Noisy data → partitions capture noise → low between, high
    within.

    Formula
    -------
    explained = between / (between + within)
    modulation = clip((explained - 0.05) / 0.25, 0, 1)

    This maps:
      - explained > 0.30 → modulation = 1.0  (clean, full resampling)
      - explained < 0.05 → modulation = 0.0  (noisy, pass-through)

    Parameters
    ----------
    hvrt_model : fitted HVRT instance

    Returns
    -------
    float in [0, 1] : 1.0 = clean, 0.0 = pure noise
    """
    partitions = hvrt_model.get_partitions()
    if len(partitions) < 2:
        return 1.0

    sizes = np.array([p["size"] for p in partitions], dtype=float)
    variances = np.array([p["variance"] for p in partitions])
    mean_abs_z = np.array([p["mean_abs_z"] for p in partitions])

    total_n = sizes.sum()
    grand_mean = np.average(mean_abs_z, weights=sizes)
    between = np.sum(sizes * (mean_abs_z - grand_mean) ** 2) / total_n
    within = np.average(variances, weights=sizes)

    if (between + within) < 1e-12:
        return 1.0

    explained = between / (between + within)
    return float(np.clip((explained - 0.05) / 0.25, 0.0, 1.0))


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
