"""
Orchestrates all 8 geoxgb.report sections and prints them to stdout.
"""
from __future__ import annotations

import numpy as np

from geoxgb.report import (
    compare_report,
    evolution_report,
    importance_report,
    model_report,
    noise_report,
    partition_report,
    print_report,
    provenance_report,
    validation_report,
)


def print_all_reports(
    geo_clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_scores: dict,
    feature_names: list[str],
    ground_truth: dict,
) -> None:
    """
    Generate and pretty-print all 8 interpretability report sections.

    Parameters
    ----------
    geo_clf : fitted GeoXGBClassifier
    baseline_scores : dict with keys auc, geoxgb_auc, time, geoxgb_time, n_samples_used
    feature_names : list of feature name strings
    ground_truth : domain knowledge dict (signal_features, noise_features, mechanism)
    """
    # 1. Model overview
    print_report(
        model_report(geo_clf, X_test, y_test, feature_names, detail="standard"),
        title="1. MODEL OVERVIEW",
    )

    # 2. Noise assessment
    print_report(
        noise_report(geo_clf),
        title="2. NOISE ASSESSMENT",
    )

    # 3. Sample provenance
    print_report(
        provenance_report(geo_clf, detail="standard"),
        title="3. SAMPLE PROVENANCE",
    )

    # 4. Feature importance — boosting vs partition geometry
    print_report(
        importance_report(geo_clf, feature_names, ground_truth, detail="standard"),
        title="4. FEATURE IMPORTANCE (BOOSTING vs PARTITION GEOMETRY)",
    )

    # 5. Partition structure (initial fit)
    print_report(
        partition_report(geo_clf, round_idx=0, feature_names=feature_names, detail="standard"),
        title="5. PARTITION STRUCTURE (INITIAL FIT, round=0)",
    )

    # 6. Evolution across refits
    print_report(
        evolution_report(geo_clf, feature_names, detail="full"),
        title="6. PARTITION EVOLUTION ACROSS REFITS",
    )

    # 7. Validation against domain knowledge
    print_report(
        validation_report(geo_clf, X_train, y_train, feature_names, ground_truth),
        title="7. VALIDATION AGAINST DOMAIN KNOWLEDGE",
    )

    # 8. GeoXGB vs XGBoost comparison
    print_report(
        compare_report(geo_clf, baseline_scores, feature_names),
        title="8. GEOXGB vs XGBOOST COMPARISON",
    )
