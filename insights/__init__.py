"""
insights — Generic GeoXGB interpretability and visualization toolkit
====================================================================

Provides dataset-agnostic tools for generating interpretability reports
and visualizations from fitted GeoXGB models.  Dataset-specific code
(constants, data loading, orchestration) lives in the calling script.

Public API
----------
Fitting helpers
    FitResult       — dataclass holding model, proba, auc, elapsed
    fit_geoxgb      — fit GeoXGBClassifier, return FitResult
    fit_xgboost     — fit XGBClassifier, return FitResult

Report printing
    print_all_reports  — print all 8 geoxgb.report sections to stdout
    print_summary      — print narrative insights summary to stdout

Visualization (saved as PNG)
    save_all_figures         — generate and save all 7 figures
    plot_feature_importance  — boosting vs partition horizontal bars
    plot_rank_comparison     — rank scatter with divergent features
    plot_partition_evolution — noise / samples / partitions over refits
    plot_roc_curves          — overlay ROC curves for two models
    plot_sample_provenance   — donut chart of training-set composition
    plot_validation_checks   — PASS/FAIL grid for validation checks
    plot_partition_size_dist — histogram of partition sizes
"""

from insights._fit import FitResult, fit_geoxgb, fit_xgboost
from insights._reports import print_all_reports
from insights._summary import print_summary
from insights._visualize import (
    save_all_figures,
    plot_feature_importance,
    plot_rank_comparison,
    plot_partition_evolution,
    plot_roc_curves,
    plot_sample_provenance,
    plot_validation_checks,
    plot_partition_size_dist,
)

__all__ = [
    "FitResult",
    "fit_geoxgb",
    "fit_xgboost",
    "print_all_reports",
    "print_summary",
    "save_all_figures",
    "plot_feature_importance",
    "plot_rank_comparison",
    "plot_partition_evolution",
    "plot_roc_curves",
    "plot_sample_provenance",
    "plot_validation_checks",
    "plot_partition_size_dist",
]
