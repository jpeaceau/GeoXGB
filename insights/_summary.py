"""
Prints the narrative INSIGHTS SUMMARY section.
"""
from __future__ import annotations

from geoxgb.report import (
    evolution_report,
    importance_report,
    noise_report,
    provenance_report,
)


def print_summary(
    geo_clf,
    geo_auc: float,
    geo_time: float,
    xgb_auc: float,
    xgb_time: float,
    feature_names: list[str],
) -> None:
    """
    Print the human-readable insights narrative to stdout.

    Parameters
    ----------
    geo_clf   : fitted GeoXGBClassifier
    geo_auc   : GeoXGB ROC-AUC on test set
    geo_time  : GeoXGB wall-clock fit time (seconds)
    xgb_auc   : XGBoost ROC-AUC on test set
    xgb_time  : XGBoost wall-clock fit time (seconds)
    feature_names : list of feature name strings
    """
    print("=" * 64)
    print("  INSIGHTS SUMMARY")
    print("=" * 64)

    nr   = noise_report(geo_clf)
    imp  = importance_report(geo_clf, feature_names, detail="standard")
    prov = provenance_report(geo_clf)
    evo  = evolution_report(geo_clf, feature_names, detail="standard")
    divs = imp.get("divergent_features", [])

    print(
        f"\n  Data quality    : {nr['assessment'].upper()}  "
        f"(initial noise modulation = {nr['initial_modulation']:.3f})"
    )
    print(
        f"  Training sample : {prov['total_training']:,} / {prov['original_n']:,}  "
        f"({prov['efficiency']})"
    )

    print(f"\n  Top-5 BOOSTING importance (what predicts heart disease):")
    for name, val in imp["top_boosting"]:
        print(f"    {name:<22s} {val:.4f}")

    print(f"\n  Top-5 PARTITION importance (what defines data geometry):")
    for name, val in imp["top_partition"]:
        print(f"    {name:<22s} {val:.4f}")

    print(f"\n  Spearman agreement (boosting vs geometry): {imp['agreement']:.4f}")
    print(f"  {imp['interpretation']}")

    if divs:
        print(f"\n  Structurally divergent features (rank diff > 3):")
        for d in divs[:3]:
            print(
                f"    {d['feature']:<22s}  "
                f"boosting_rank={d['boosting_rank']}  "
                f"partition_rank={d['partition_rank']}  "
                f"diff={d['rank_diff']}"
            )

    print(f"\n  Partition evolution : {evo.get('interpretation', 'n/a')}")
    print(f"  Noise trend         : {evo.get('noise_trend', {}).get('direction', 'n/a')}")

    print(
        f"\n  AUC   GeoXGB={geo_auc:.6f}  XGBoost={xgb_auc:.6f}  "
        f"delta={geo_auc - xgb_auc:+.6f}"
    )
    print(f"  Time  GeoXGB={geo_time:.1f}s  XGBoost={xgb_time:.1f}s")
    print()
