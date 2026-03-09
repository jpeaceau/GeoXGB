"""
C++ backend for GeoXGB.

Tries to import the compiled _geoxgb_cpp extension.  On success, exposes
``CppGeoXGBRegressor``, ``CppGeoXGBClassifier``, and ``make_cpp_config``.
On failure, sets ``_CPP_AVAILABLE = False`` so callers fall back to pure Python.

Usage
-----
from geoxgb._cpp_backend import _CPP_AVAILABLE, make_cpp_config, CppGeoXGBRegressor

if _CPP_AVAILABLE:
    cfg = make_cpp_config(n_rounds=500, learning_rate=0.2, max_depth=4)
    model = CppGeoXGBRegressor(cfg)
    model.fit(X, y)
    preds = model.predict(X_test)
"""

try:
    from geoxgb._geoxgb_cpp import (  # type: ignore[import]
        GeoXGBConfig,
        CppGeoXGBRegressor,
        CppGeoXGBClassifier,
        CppGeoXGBGiniClassifier,
        CppGeoXGBFocalClassifier,
        CppGeoXGBExpClassifier,
        CppGeoXGBHingeClassifier,
        CppGeoXGBMulticlassClassifier,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    GeoXGBConfig                    = None
    CppGeoXGBRegressor              = None
    CppGeoXGBClassifier             = None
    CppGeoXGBGiniClassifier         = None
    CppGeoXGBFocalClassifier        = None
    CppGeoXGBExpClassifier          = None
    CppGeoXGBHingeClassifier        = None
    CppGeoXGBMulticlassClassifier   = None


# Map Python _GeoXGBBase constructor kwargs → GeoXGBConfig fields.
# Only fields that exist in GeoXGBConfig are forwarded; extras are silently
# dropped (e.g. lr_schedule, tree_criterion, n_jobs — not exposed in C++).
_PYTHON_TO_CPP = {
    "n_rounds":              "n_rounds",
    "learning_rate":         "learning_rate",
    "max_depth":             "max_depth",
    "min_samples_leaf":      "min_samples_leaf",
    "reduce_ratio":          "reduce_ratio",
    "expand_ratio":          "expand_ratio",
    "y_weight":              "y_weight",
    "refit_interval":        "refit_interval",
    "auto_noise":            "auto_noise",
    "noise_guard":           "noise_guard",
    "refit_noise_floor":     "refit_noise_floor",
    "auto_expand":           "auto_expand",
    "min_train_samples":     "min_train_samples",
    "n_bins":                "n_bins",
    "random_state":          "random_state",
    "variance_weighted":     "variance_weighted",
    "adaptive_y_weight":     "adaptive_y_weight",
    "blend_cross_term":      "blend_cross_term",
    "syn_partition_correct": "syn_partition_correct",
    "y_geom_coupling":       "y_geom_coupling",
    "selective_target":      "selective_target",
    "selective_k_pairs":     "selective_k_pairs",
    "d_geom_threshold":          "d_geom_threshold",
    "residual_correct_lambda":   "residual_correct_lambda",
    "hvrt_min_samples_leaf": "hvrt_min_samples_leaf",
    "n_partitions":          "hvrt_n_partitions",
    "partitioner":           "partitioner",
    "method":                "reduce_method",
    "generation_strategy":   "generation_strategy",
    "adaptive_reduce_ratio":    "adaptive_reduce_ratio",
    "sample_block_n":        "sample_block_n",
    "leave_last_block_out":  "leave_last_block_out",
    "loss":                  "loss",
    "lazy_refit_tol":        "lazy_refit_tol",
    "fixed_geometry":        "fixed_geometry",
    "progressive_expand":    "progressive_expand",
    "sample_without_replacement": "sample_without_replacement",
    "colsample_bytree":      "colsample_bytree",
    "predict_stride":        "predict_stride",
    "grad_budget_weight":    "grad_budget_weight",
    "track_partition_trajectory": "track_partition_trajectory",
    "boost_optimizer":           "boost_optimizer",
    "momentum_beta":             "momentum_beta",
    "adam_beta2":                 "adam_beta2",
    "adam_epsilon":               "adam_epsilon",
}


def make_cpp_config(**kwargs) -> "GeoXGBConfig":
    """
    Build a ``GeoXGBConfig`` from Python parameter keyword arguments.

    ``bandwidth`` is handled separately: the Python API accepts either a float
    or the string ``"auto"``; the C++ config stores it as a double where
    ``-1.0`` means auto.

    Parameters not recognised by the C++ backend are silently ignored.
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError(
            "GeoXGB C++ backend is not available.  "
            "Rebuild with scikit-build-core: pip install -e . --no-build-isolation"
        )
    cfg = GeoXGBConfig()

    # Bandwidth: Python "auto" → C++ -1.0; numeric string or float → float
    bw = kwargs.get("bandwidth", "auto")
    if bw == "auto" or bw is None:
        cfg.bandwidth = -1.0
    else:
        try:
            cfg.bandwidth = float(bw)
        except (TypeError, ValueError):
            cfg.bandwidth = -1.0

    # convergence_tol: None = disabled (C++ uses 0.0 for disabled)
    conv_tol = kwargs.get("convergence_tol", None)
    if conv_tol is not None:
        cfg.convergence_tol = float(conv_tol)

    # pos_class_weight: passed separately by classifier (not in _PARAM_NAMES)
    if "pos_class_weight" in kwargs and kwargs["pos_class_weight"] is not None:
        cfg.pos_class_weight = float(kwargs["pos_class_weight"])

    for py_key, cpp_key in _PYTHON_TO_CPP.items():
        if py_key in kwargs:
            v = kwargs[py_key]
            if v is None:
                continue
            # 'auto' sample_block_n: caller should resolve before make_cpp_config;
            # but guard here in case it slips through — treat as disabled.
            if py_key == 'sample_block_n' and v == 'auto':
                continue
            # Guard against compiled extension missing newer config fields
            # (e.g. Docker image built from older C++ source).
            if not hasattr(cfg, cpp_key):
                continue
            setattr(cfg, cpp_key, v)

    return cfg
