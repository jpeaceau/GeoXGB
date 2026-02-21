"""
Visualization functions for GeoXGB insights.

All figures are saved as PNG files (headless Agg backend).
No plt.show() is called; figures are closed after saving.

Public functions
----------------
    plot_feature_importance   — boosting vs partition horizontal bars
    plot_rank_comparison      — scatter of boosting rank vs partition rank
    plot_partition_evolution  — line chart of noise / samples / partitions over refits
    plot_roc_curves           — ROC curves for GeoXGB and XGBoost
    plot_sample_provenance    — donut chart of training-set composition
    plot_validation_checks    — PASS/FAIL grid for validation checks
    plot_partition_size_dist  — histogram of partition sizes
    save_all_figures          — calls all 7 and returns saved paths
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from geoxgb.report import (
    evolution_report,
    importance_report,
    partition_report,
    provenance_report,
    validation_report,
)

if TYPE_CHECKING:
    from insights._fit import FitResult

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_PALETTE = {
    "geo":       "#2196F3",   # blue  — GeoXGB
    "xgb":       "#FF5722",   # deep orange — XGBoost
    "boost":     "#1565C0",   # dark blue — boosting importance
    "partition": "#00897B",   # teal — partition importance
    "pass":      "#43A047",   # green
    "fail":      "#E53935",   # red
    "neutral":   "#90A4AE",   # grey
    "dropped":   "#CFD8DC",   # light grey
    "synthetic": "#A5D6A7",   # light green
    "kept":      "#90CAF9",   # light blue
}

_FIG_DPI = 150


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.0)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Feature importance — boosting vs partition
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_rep: dict,
    feature_names: list[str],
    out_path: str,
) -> None:
    """
    Side-by-side horizontal bar charts comparing boosting importance
    (what predicts heart disease) and partition importance (data geometry).

    Parameters
    ----------
    importance_rep : dict returned by geoxgb.report.importance_report()
    feature_names  : ordered list of feature name strings
    out_path       : file path to save the PNG
    """
    _setup()
    boost_imp = importance_rep["boosting_importance"]
    part_imp  = importance_rep["partition_importance"]

    # Sort features by boosting importance (descending)
    order = sorted(feature_names, key=lambda n: boost_imp.get(n, 0.0))
    boost_vals = [boost_imp.get(n, 0.0) for n in order]
    part_vals  = [part_imp.get(n, 0.0)  for n in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Boosting importance
    axes[0].barh(order, boost_vals, color=_PALETTE["boost"], alpha=0.85)
    axes[0].set_title("Boosting Importance\n(what predicts heart disease)", fontsize=11, pad=8)
    axes[0].set_xlabel("Importance")
    axes[0].invert_xaxis()
    axes[0].yaxis.set_tick_params(labelleft=False)

    # Partition importance
    axes[1].barh(order, part_vals, color=_PALETTE["partition"], alpha=0.85)
    axes[1].set_title("Partition Importance\n(data geometry structure)", fontsize=11, pad=8)
    axes[1].set_xlabel("Importance")

    # Feature labels in the centre
    for ax in axes:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([])

    fig.text(
        0.5, 0.5,
        "\n".join(order),
        ha="center", va="center",
        transform=fig.transFigure,
        fontsize=9,
        linespacing=2.38,
    )

    agreement = importance_rep.get("agreement", float("nan"))
    fig.suptitle(
        f"Feature Importance: Boosting vs Partition  "
        f"(Spearman agreement = {agreement:.3f})",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Rank comparison scatter
# ---------------------------------------------------------------------------

def plot_rank_comparison(
    importance_rep: dict,
    feature_names: list[str],
    out_path: str,
) -> None:
    """
    Scatter plot of boosting rank vs partition rank for each feature.
    Divergent features (|rank_diff| > 3) are highlighted in red.
    The diagonal represents perfect rank agreement.

    Parameters
    ----------
    importance_rep : dict returned by geoxgb.report.importance_report(detail='standard')
    feature_names  : ordered list of feature name strings
    out_path       : file path to save the PNG
    """
    _setup()
    boost_imp = importance_rep["boosting_importance"]
    part_imp  = importance_rep["partition_importance"]

    boost_sorted = sorted(feature_names, key=lambda n: -boost_imp.get(n, 0.0))
    part_sorted  = sorted(feature_names, key=lambda n: -part_imp.get(n, 0.0))
    boost_rank = {n: i + 1 for i, n in enumerate(boost_sorted)}
    part_rank  = {n: i + 1 for i, n in enumerate(part_sorted)}

    divergent_names = {
        d["feature"] for d in importance_rep.get("divergent_features", [])
    }

    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal (perfect agreement)
    ax.plot([1, n], [1, n], color="#B0BEC5", linewidth=1.2, linestyle="--", zorder=0)

    for fname in feature_names:
        br = boost_rank[fname]
        pr = part_rank[fname]
        color = _PALETTE["fail"] if fname in divergent_names else _PALETTE["geo"]
        ax.scatter(br, pr, color=color, s=70, zorder=2)
        ax.annotate(
            fname,
            xy=(br, pr),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
        )

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Boosting rank  (1 = highest importance)", fontsize=10)
    ax.set_ylabel("Partition rank  (1 = highest importance)", fontsize=10)
    ax.set_title(
        "Boosting vs Partition Feature Ranking\n"
        "Red = divergent (|rank diff| > 3)",
        fontsize=11,
    )

    legend_patches = [
        mpatches.Patch(color=_PALETTE["geo"],  label="Aligned"),
        mpatches.Patch(color=_PALETTE["fail"], label="Divergent (|diff| > 3)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Partition evolution
# ---------------------------------------------------------------------------

def plot_partition_evolution(
    evolution_rep: dict,
    out_path: str,
) -> None:
    """
    Three-panel line chart showing how noise modulation, training-set size,
    and partition count evolve across refit rounds.

    Parameters
    ----------
    evolution_rep : dict returned by geoxgb.report.evolution_report()
    out_path      : file path to save the PNG
    """
    _setup()
    rounds = evolution_rep["rounds"]
    if not rounds:
        return

    xs       = [r["round"]            for r in rounds]
    noise    = [r["noise_modulation"] for r in rounds]
    samples  = [r["n_samples"]        for r in rounds]
    n_parts  = [r["n_partitions"]     for r in rounds]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(xs, noise, marker="o", color=_PALETTE["geo"], linewidth=1.8)
    axes[0].set_ylabel("Noise modulation", fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0.7, color=_PALETTE["pass"], linewidth=0.8, linestyle=":", label="clean threshold")
    axes[0].axhline(0.3, color=_PALETTE["fail"], linewidth=0.8, linestyle=":", label="noisy threshold")
    axes[0].legend(fontsize=7, loc="upper right")

    axes[1].plot(xs, samples, marker="s", color=_PALETTE["partition"], linewidth=1.8)
    axes[1].set_ylabel("Training samples", fontsize=9)
    axes[1].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    axes[2].plot(xs, n_parts, marker="^", color=_PALETTE["xgb"], linewidth=1.8)
    axes[2].set_ylabel("Partitions", fontsize=9)
    axes[2].set_xlabel("Refit round", fontsize=9)

    fig.suptitle("Partition Evolution Across Refits", fontsize=12)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_test: np.ndarray,
    geo_proba: np.ndarray,
    xgb_proba: np.ndarray,
    geo_auc: float,
    xgb_auc: float,
    out_path: str,
) -> None:
    """
    Overlay ROC curves for GeoXGB and XGBoost on the same axes.

    Parameters
    ----------
    y_test     : true binary labels (0/1)
    geo_proba  : GeoXGB predict_proba output, shape (n, 2)
    xgb_proba  : XGBoost predict_proba output, shape (n, 2)
    geo_auc    : pre-computed GeoXGB AUC
    xgb_auc    : pre-computed XGBoost AUC
    out_path   : file path to save the PNG
    """
    _setup()
    fpr_geo, tpr_geo, _ = roc_curve(y_test, geo_proba[:, 1])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba[:, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr_geo, tpr_geo,
        color=_PALETTE["geo"], linewidth=2,
        label=f"GeoXGB  (AUC = {geo_auc:.4f})",
    )
    ax.plot(
        fpr_xgb, tpr_xgb,
        color=_PALETTE["xgb"], linewidth=2, linestyle="--",
        label=f"XGBoost (AUC = {xgb_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#B0BEC5", linewidth=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — GeoXGB vs XGBoost", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Sample provenance donut
# ---------------------------------------------------------------------------

def plot_sample_provenance(
    provenance_rep: dict,
    out_path: str,
) -> None:
    """
    Donut chart showing the composition of the training set:
    FPS-selected (real samples kept), KDE-synthetic (generated), and
    FPS-dropped (original samples not used).

    Parameters
    ----------
    provenance_rep : dict returned by geoxgb.report.provenance_report()
    out_path       : file path to save the PNG
    """
    _setup()
    orig_n    = provenance_rep["original_n"]
    reduced_n = provenance_rep["reduced_n"]
    expanded_n = provenance_rep["expanded_n"]
    dropped_n = max(orig_n - reduced_n, 0)

    labels = ["FPS-selected (kept)", "KDE-synthetic (added)", "FPS-dropped"]
    sizes  = [reduced_n, expanded_n, dropped_n]
    colors = [_PALETTE["kept"], _PALETTE["synthetic"], _PALETTE["dropped"]]

    # Remove zero slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not non_zero:
        return
    labels, sizes, colors = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)

    efficiency = provenance_rep.get("efficiency", "")
    ax.text(0, 0, efficiency, ha="center", va="center", fontsize=9, color="#37474F")

    ax.legend(
        wedges, labels,
        title="Sample origin",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
        ncol=2,
    )
    ax.set_title("Training-Set Composition (Sample Provenance)", fontsize=11, pad=14)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Validation checks grid
# ---------------------------------------------------------------------------

def plot_validation_checks(
    validation_rep: dict,
    out_path: str,
) -> None:
    """
    Horizontal grid showing PASS/FAIL status and detail text for each
    validation check run against domain knowledge.

    Parameters
    ----------
    validation_rep : dict returned by geoxgb.report.validation_report()
    out_path       : file path to save the PNG
    """
    _setup()
    checks = validation_rep.get("checks", [])
    if not checks:
        return

    n = len(checks)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.9 + 1)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    for i, check in enumerate(reversed(checks)):
        y = i
        passed = check.get("passed", False)
        color  = _PALETTE["pass"] if passed else _PALETTE["fail"]
        label  = "[PASS]" if passed else "[FAIL]"
        name   = check.get("name", "")
        detail = check.get("detail", "")

        # Coloured badge
        badge = mpatches.FancyBboxPatch(
            (0, y - 0.35), 0.9, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="none",
        )
        ax.add_patch(badge)
        ax.text(0.45, y, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

        # Check name
        ax.text(1.05, y + 0.12, name, ha="left", va="center",
                fontsize=9, fontweight="bold", color="#212121")

        # Detail text (wrapped at ~80 chars)
        if detail:
            ax.text(1.05, y - 0.17, detail[:120], ha="left", va="center",
                    fontsize=7.5, color="#546E7A", style="italic")

    overall = validation_rep.get("overall_pass", False)
    summary = validation_rep.get("summary", "")
    fig.suptitle(
        f"Validation Checks Against Domain Knowledge\n{summary}",
        fontsize=11,
        color=_PALETTE["pass"] if overall else _PALETTE["fail"],
    )
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Partition size distribution
# ---------------------------------------------------------------------------

def plot_partition_size_dist(
    partition_rep: dict,
    out_path: str,
) -> None:
    """
    Histogram of partition sizes with a vertical line at the median,
    annotated with imbalance ratio and size statistics.

    Parameters
    ----------
    partition_rep : dict returned by geoxgb.report.partition_report(detail='standard')
    out_path      : file path to save the PNG
    """
    _setup()
    parts = partition_rep.get("partitions", [])
    if not parts:
        return

    sizes = [p["size"] for p in parts]
    dist  = partition_rep.get("size_distribution", {})
    median_val = dist.get("median", float(np.median(sizes)))
    imbalance  = dist.get("imbalance_ratio", 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(sizes, bins=min(50, len(set(sizes))), color=_PALETTE["partition"],
                 alpha=0.8, ax=ax, edgecolor="white", linewidth=0.4)
    ax.axvline(median_val, color=_PALETTE["geo"], linewidth=1.8, linestyle="--",
               label=f"Median = {median_val:,.0f}")
    ax.set_xlabel("Partition size (samples)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"Partition Size Distribution  "
        f"(n={len(parts)} partitions, imbalance ratio={imbalance:.1f}x)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# save_all_figures — master entry point
# ---------------------------------------------------------------------------

def save_all_figures(
    geo_clf,
    geo_fit: "FitResult",
    xgb_fit: "FitResult",
    y_test: np.ndarray,
    feature_names: list[str],
    ground_truth: dict,
    figures_dir: str = "reports/figures",
) -> list[str]:
    """
    Generate all 7 insight figures and save them as PNG files.

    Parameters
    ----------
    geo_clf      : fitted GeoXGBClassifier
    geo_fit      : FitResult for GeoXGB (contains proba, auc, elapsed)
    xgb_fit      : FitResult for XGBoost (contains proba, auc, elapsed)
    y_test       : true test labels
    feature_names: ordered feature name list
    ground_truth : domain knowledge dict
    figures_dir  : directory to save PNGs (created if absent)

    Returns
    -------
    list of absolute paths to the saved PNG files
    """
    os.makedirs(figures_dir, exist_ok=True)

    def _path(name: str) -> str:
        return os.path.join(figures_dir, name)

    # Pre-compute report dicts (avoid redundant calls)
    imp_rep  = importance_report(geo_clf, feature_names, ground_truth, detail="standard")
    evo_rep  = evolution_report(geo_clf, feature_names, detail="standard")
    prov_rep = provenance_report(geo_clf, detail="standard")
    val_rep  = validation_report(geo_clf, None, None, feature_names, ground_truth)
    part_rep = partition_report(geo_clf, round_idx=0, feature_names=feature_names, detail="standard")

    saved: list[str] = []

    tasks = [
        (
            "01_feature_importance.png",
            lambda p: plot_feature_importance(imp_rep, feature_names, p),
            "Feature importance (boosting vs partition)",
        ),
        (
            "02_rank_comparison.png",
            lambda p: plot_rank_comparison(imp_rep, feature_names, p),
            "Boosting vs partition rank scatter",
        ),
        (
            "03_partition_evolution.png",
            lambda p: plot_partition_evolution(evo_rep, p),
            "Partition evolution across refits",
        ),
        (
            "04_roc_curves.png",
            lambda p: plot_roc_curves(
                y_test, geo_fit.proba, xgb_fit.proba,
                geo_fit.auc, xgb_fit.auc, p,
            ),
            "ROC curves (GeoXGB vs XGBoost)",
        ),
        (
            "05_sample_provenance.png",
            lambda p: plot_sample_provenance(prov_rep, p),
            "Sample provenance donut",
        ),
        (
            "06_validation_checks.png",
            lambda p: plot_validation_checks(val_rep, p),
            "Validation checks grid",
        ),
        (
            "07_partition_size_dist.png",
            lambda p: plot_partition_size_dist(part_rep, p),
            "Partition size distribution",
        ),
    ]

    print(f"\nSaving figures to: {os.path.abspath(figures_dir)}/")
    for filename, fn, description in tasks:
        full_path = _path(filename)
        try:
            fn(full_path)
            saved.append(full_path)
            print(f"  [OK] {filename}  ({description})")
        except Exception as exc:
            print(f"  [SKIP] {filename}  ({exc})")

    print(f"\n{len(saved)}/{len(tasks)} figures saved.\n")
    return saved
