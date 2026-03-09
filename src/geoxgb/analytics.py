"""
Analytics tools for ModelArtifact.

Decoupled visualization and reporting that consumes ModelArtifact objects.
Two categories:

1. **Matplotlib-based** — for users with their own tooling:
   plot_importances, plot_contributions, plot_cooperation, plot_residuals

2. **Interactive HTML** — JS-powered browser report:
   to_interactive_html_report

All functions accept a ModelArtifact and produce output without needing
the original model or training data.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact import ModelArtifact

from .artifact import _jsonify


# =========================================================================
# Matplotlib-based tools (require matplotlib)
# =========================================================================

def plot_importances(artifact: "ModelArtifact", top_n: int = 20, ax=None):
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    artifact : ModelArtifact
    top_n : int, show top N features
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    imp = artifact.feature_importances
    if not imp:
        raise ValueError("Artifact has no feature importances.")

    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*reversed(sorted_imp))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))
    else:
        fig = ax.figure

    ax.barh(range(len(names)), values, color="#4C72B0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances — {artifact.model_type}")
    fig.tight_layout()
    return fig


def plot_contributions(artifact: "ModelArtifact", top_n: int = 15, ax=None):
    """
    Bar chart of mean absolute contribution by feature and interaction.

    Parameters
    ----------
    artifact : ModelArtifact
    top_n : int, show top N terms
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if artifact.contributions is None:
        raise ValueError("Artifact has no contributions.")

    items = []
    for feat, stats in artifact.contributions["main"].items():
        items.append((feat, stats["abs_mean"]))
    for pair, stats in artifact.contributions.get("interaction", {}).items():
        items.append((pair, stats["abs_mean"]))

    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_n]
    names, values = zip(*reversed(items))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))
    else:
        fig = ax.figure

    colors = ["#DD8452" if " x " in str(n) else "#4C72B0" for n in reversed(list(names))]
    colors.reverse()
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean |Contribution|")
    ax.set_title("Feature & Interaction Contributions")
    fig.tight_layout()
    return fig


def plot_cooperation(artifact: "ModelArtifact", ax=None):
    """
    Heatmap of the global cooperation matrix.

    Parameters
    ----------
    artifact : ModelArtifact
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if artifact.cooperation is None:
        raise ValueError("Artifact has no cooperation data.")

    matrix = np.array(artifact.cooperation["matrix"])
    names = artifact.cooperation.get("feature_names", artifact.feature_names)
    d = len(names)

    if ax is None:
        size = max(5, d * 0.5)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=max(6, 10 - d // 5))
    ax.set_yticklabels(names, fontsize=max(6, 10 - d // 5))
    ax.set_title("Global Cooperation Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_residuals(artifact: "ModelArtifact", split: str = "test", ax=None):
    """
    Residual plot (predicted vs actual, or residual distribution).

    Parameters
    ----------
    artifact : ModelArtifact
    split : 'train' or 'test'
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if artifact.predictions is None or split not in artifact.predictions:
        raise ValueError(f"Artifact has no {split} predictions.")

    pred = artifact.predictions[split]
    y_true = np.array(pred["y_true"])
    y_pred = np.array(pred["y_pred"])

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.figure
        axes = [ax, ax]  # fallback: single axis

    if len(axes) >= 2:
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="#4C72B0")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title(f"Actual vs Predicted ({split})")

        # Residual distribution
        residuals = np.array(pred["residuals"])
        axes[1].hist(residuals, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
        axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Residual Distribution ({split})")

    fig.tight_layout()
    return fig


def _esc(s: str) -> str:
    return html.escape(str(s))


# =========================================================================
# Interactive HTML report (self-contained, JS-powered)
# =========================================================================

def to_interactive_html_report(
    artifact: "ModelArtifact",
    path: str | Path | None = None,
) -> str:
    """
    Generate an interactive standalone HTML report from a ModelArtifact.

    Features:
    - Feature selector dropdown with EBM-style main-effect curves + CI bands
    - Interaction pair selector with 2D heatmap
    - Best/worst/median sample inspector with local model decomposition
    - Leaflet map with cooperation overlay (when lat/lon detected)

    Parameters
    ----------
    artifact : ModelArtifact
    path : str or Path, optional
        If provided, write HTML to this file.

    Returns
    -------
    str : HTML content
    """
    import json as _json

    # Prepare JSON data bundles for JS consumption
    js_data = {
        "model_type": artifact.model_type,
        "task": artifact.task,
        "version": artifact.version,
        "timestamp": artifact.build_timestamp,
        "feature_names": artifact.feature_names,
        "feature_importances": artifact.feature_importances or {},
        "fit_stats": artifact.fit_stats or {},
        "noise": artifact.noise or {},
        "cooperation": artifact.cooperation,
        "contributions": artifact.contributions,
        "main_curves": artifact.main_curves,
        "partition_summary": artifact.partition_summary,
        "interaction_surfaces": artifact.interaction_surfaces,
        "exemplars": artifact.exemplars,
        "sample_lookup": artifact.sample_lookup,
        "irreducible_error": artifact.irreducible_error,
        "geo_data": artifact.geo_data,
        "predictions": artifact.predictions,
    }
    data_json = _json.dumps(_jsonify(js_data), default=str)

    has_map = artifact.geo_data is not None
    has_3d = artifact.partition_summary is not None
    leaflet_css = ('<link rel="stylesheet" href="https://unpkg.com/'
                   'leaflet@1.9.4/dist/leaflet.css"/>' if has_map else "")
    leaflet_js = ('<script src="https://unpkg.com/leaflet@1.9.4/'
                  'dist/leaflet.js"></script>' if has_map else "")
    three_js = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/'
                'three.js/r128/three.min.js"></script>'
                '<script src="https://cdn.jsdelivr.net/npm/'
                'three@0.128.0/examples/js/controls/OrbitControls.js">'
                '</script>'
                if has_3d else "")

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GeoXGB Interactive Report — {_esc(artifact.model_type)}</title>
{leaflet_css}
<style>
:root {{
    --bg-page: #f0f2f5;
    --bg-card: #ffffff;
    --bg-tabs: #ffffff;
    --bg-table-header: #f8f9fa;
    --bg-input: #ffffff;
    --text-primary: #333333;
    --text-secondary: #666666;
    --text-muted: #999999;
    --text-on-accent: #ffffff;
    --border-subtle: #e8ecf1;
    --border-input: #dddddd;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-legend: 0 1px 4px rgba(0,0,0,0.2);
    --header-from: #1a1a2e;
    --header-via: #16213e;
    --header-to: #0f3460;
    --accent: #0f3460;
    --accent-hover: #f0f2f5;
    --status-good: #28a745;
    --status-warn: #ffc107;
    --status-bad: #dc3545;
    --chart-primary: #4C72B0;
    --chart-primary-fill: rgba(76,114,176,0.25);
    --chart-primary-band: rgba(76,114,176,0.15);
    --chart-primary-edge: rgba(76,114,176,0.3);
    --chart-grid: #cccccc;
    --marker-outline: #333333;
    --div-pos: #2b7bba;
    --div-neg: #d94040;
    --div-pos-rgb: 43,123,186;
    --div-neg-rgb: 217,64,64;
    --div-mid-rgb: 230,230,230;
    --seq-lo-rgb: 240,240,248;
    --seq-mid-rgb: 60,120,210;
    --seq-hi-rgb: 120,40,170;
}}
[data-theme="dark"] {{
    --bg-page: #1a1a2e;
    --bg-card: #22223a;
    --bg-tabs: #22223a;
    --bg-table-header: #2a2a44;
    --bg-input: #2a2a44;
    --text-primary: #e0e0e0;
    --text-secondary: #aaaaaa;
    --text-muted: #777777;
    --text-on-accent: #ffffff;
    --border-subtle: #3a3a55;
    --border-input: #3a3a55;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-legend: 0 1px 4px rgba(0,0,0,0.5);
    --header-from: #0d0d1a;
    --header-via: #0f1525;
    --header-to: #0a2040;
    --accent: #3a6ea5;
    --accent-hover: #2a2a44;
    --status-good: #3dcc5f;
    --status-warn: #ffd54f;
    --status-bad: #ff6b6b;
    --chart-primary: #5ba3d9;
    --chart-primary-fill: rgba(91,163,217,0.3);
    --chart-primary-band: rgba(91,163,217,0.2);
    --chart-primary-edge: rgba(91,163,217,0.4);
    --chart-grid: #444466;
    --marker-outline: #aaaaaa;
    --div-pos: #5ba3d9;
    --div-neg: #e06060;
    --div-pos-rgb: 91,163,217;
    --div-neg-rgb: 224,96,96;
    --div-mid-rgb: 50,50,70;
    --seq-lo-rgb: 35,35,55;
    --seq-mid-rgb: 50,110,200;
    --seq-hi-rgb: 160,80,220;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg-page); color: var(--text-primary); line-height: 1.5;
    transition: background 0.3s, color 0.3s;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
.header {{
    background: linear-gradient(135deg, var(--header-from), var(--header-via), var(--header-to));
    color: white; padding: 30px; border-radius: 10px;
    margin-bottom: 20px; position: relative; overflow: hidden;
}}
.header::after {{
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 300px; height: 300px; border-radius: 50%;
    background: rgba(255,255,255,0.03);
}}
.header h1 {{ font-size: 26px; margin-bottom: 4px; }}
.header .sub {{ font-size: 13px; opacity: 0.8; }}
#theme-toggle {{
    position: absolute; top: 15px; right: 15px;
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
    color: white; padding: 8px 12px; border-radius: 8px; cursor: pointer;
    font-size: 20px; line-height: 1; transition: background 0.2s;
    z-index: 10;
}}
#theme-toggle:hover {{ background: rgba(255,255,255,0.3); }}
.tabs {{
    display: flex; gap: 4px; background: var(--bg-tabs); padding: 6px;
    border-radius: 10px; margin-bottom: 20px;
    box-shadow: var(--shadow-card); flex-wrap: wrap;
    transition: background 0.3s;
}}
.tab {{
    padding: 8px 18px; border-radius: 6px; cursor: pointer;
    font-size: 13px; font-weight: 500; color: var(--text-secondary);
    transition: all 0.2s; border: none; background: none;
}}
.tab:hover {{ background: var(--accent-hover); color: var(--text-primary); }}
.tab.active {{ background: var(--accent); color: var(--text-on-accent); }}
.panel {{ display: none; }}
.panel.active {{ display: block; }}
.card {{
    background: var(--bg-card); border-radius: 10px; padding: 24px;
    margin-bottom: 16px; box-shadow: var(--shadow-card);
    transition: background 0.3s;
}}
.card h2 {{
    font-size: 16px; color: var(--text-primary); margin-bottom: 14px;
    padding-bottom: 8px; border-bottom: 2px solid var(--border-subtle);
}}
.card h3 {{ font-size: 14px; color: var(--text-secondary); margin: 12px 0 6px; }}
table.dt {{
    width: 100%; border-collapse: collapse; font-size: 13px;
}}
table.dt td, table.dt th {{
    padding: 7px 10px; border-bottom: 1px solid var(--border-subtle); text-align: left;
}}
table.dt th {{ background: var(--bg-table-header); font-weight: 600; }}
select, button {{
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border-input);
    font-size: 13px; background: var(--bg-input); color: var(--text-primary);
    cursor: pointer; transition: background 0.3s, color 0.3s;
}}
select:focus {{ outline: 2px solid var(--accent); }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.row > * {{ flex: 1; min-width: 300px; }}
canvas {{ max-width: 100%; }}
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    color: white; font-size: 11px; font-weight: 600;
}}
.exemplar-tabs {{ display: flex; gap: 4px; margin-bottom: 12px; }}
.exemplar-tab {{
    padding: 5px 14px; border-radius: 5px; cursor: pointer;
    font-size: 12px; border: 1px solid var(--border-input); background: var(--bg-input);
    color: var(--text-primary);
}}
.exemplar-tab.active {{ background: var(--accent); color: var(--text-on-accent); border-color: var(--accent); }}
#map {{ height: 500px; border-radius: 8px; margin-top: 12px; }}
.mode-btn {{ background: var(--bg-input); color: var(--text-primary); }}
.mode-btn-active {{ background: var(--accent) !important; color: var(--text-on-accent) !important; border-color: var(--accent) !important; }}
.legend {{
    background: var(--bg-card); color: var(--text-primary);
    padding: 8px 12px; border-radius: 6px;
    font-size: 11px; line-height: 1.6; box-shadow: var(--shadow-legend);
}}
footer {{
    text-align: center; font-size: 11px; color: var(--text-muted);
    margin-top: 30px; padding-bottom: 20px;
}}
[data-theme="dark"] .leaflet-popup-content-wrapper {{
    background: var(--bg-card); color: var(--text-primary);
}}
[data-theme="dark"] .leaflet-popup-tip {{ background: var(--bg-card); }}
</style>
</head>
<body>
{leaflet_js}
{three_js}
<div class="container">

<div class="header">
    <h1>GeoXGB Interactive Report</h1>
    <div class="sub" id="header-sub"></div>
    <button id="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark theme">&#9790;</button>
</div>

<div class="tabs" id="main-tabs"></div>

<!-- Overview Panel -->
<div class="panel" id="panel-overview">
    <div class="row">
        <div class="card" id="card-fit"></div>
        <div class="card" id="card-metrics"></div>
    </div>
    <div class="card" id="card-importances">
        <h2>Feature Importances</h2>
        <canvas id="cv-imp" height="300"></canvas>
    </div>
    <div class="card" id="card-noise"></div>
</div>

<!-- Features Panel (feature-driven view) -->
<div class="panel" id="panel-features">
    <div class="card">
        <h2>Feature Effects</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Each dot is one sample's contribution, color-coded by partition.
            Each line is a partition's local linear effect.
            <b>Main</b>: direct effect only.
            <b>Net</b>: includes Shapley-attributed interaction share.</p>
        <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;">
            <select id="sel-feat"></select>
            <div id="ebm-mode-wrap" style="display:flex;gap:2px;">
                <button id="btn-ebm-main" class="mode-btn mode-btn-active"
                    onclick="setEbmMode('main')"
                    style="padding:4px 12px;font-size:12px;border:1px solid var(--border-input);border-radius:4px 0 0 4px;cursor:pointer;">Main</button>
                <button id="btn-ebm-net" class="mode-btn"
                    onclick="setEbmMode('net')"
                    style="padding:4px 12px;font-size:12px;border:1px solid var(--border-input);border-radius:0 4px 4px 0;cursor:pointer;">Net</button>
            </div>
        </div>
        <canvas id="cv-main" height="350"></canvas>
        <div id="feat-partition-table" style="margin-top:12px;"></div>
    </div>
</div>

<!-- Partitions Panel (partition-driven view) -->
<div class="panel" id="panel-partitions">
    <div class="card">
        <h2>Partition Explorer</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Each HVRT partition defines a local region where the model fits a
            separate linear model. Select a partition to see its details.</p>
        <div id="partition-overview"></div>
    </div>
    <div class="card" id="card-partition-detail" style="display:none;">
        <h2>Partition <span id="partition-detail-pid"></span></h2>
        <div id="partition-detail-stats"></div>
        <canvas id="cv-partition-slopes" height="300"></canvas>
    </div>
</div>

<!-- Residuals Panel -->
<div class="panel" id="panel-residuals">
    <div class="card">
        <h2>Residual Diagnostics</h2>
        <div class="row">
            <div><canvas id="cv-scatter" height="300"></canvas></div>
            <div><canvas id="cv-hist" height="300"></canvas></div>
        </div>
    </div>
</div>

<!-- Samples Panel -->
<div class="panel" id="panel-samples">
    <div class="card">
        <h2>Sample Inspector</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Examine predictions, contributions, and local feature relationships for any sample.</p>
        <div style="margin-bottom:10px;" id="irr-toggle-wrap"></div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
            <div class="exemplar-tabs" id="exemplar-tabs" style="display:flex;gap:4px;"></div>
            <div style="display:flex;gap:4px;align-items:center;margin-left:auto;">
                <input type="number" id="lookup-idx" min="0" placeholder="#"
                    style="width:70px;padding:5px 8px;border:1px solid var(--border-input);border-radius:6px;font-size:12px;"
                    onkeydown="if(event.key==='Enter')lookupSample()">
                <button onclick="lookupSample()" class="exemplar-tab"
                    style="padding:5px 10px;font-size:12px;">Lookup</button>
                <span id="lookup-range" style="font-size:10px;color:var(--text-muted);"></span>
            </div>
        </div>
        <div id="sample-detail-content"></div>
    </div>
    <div class="card" id="card-coop-detail" style="display:none;">
        <h2>Feature Relationships — <span id="coop-detail-title">Global</span></h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Compares the <b>global</b> cooperation matrix (weighted average across all partitions)
            with the <b>local</b> matrix for this sample's partition.
            Large deviations indicate heterogeneous or non-linear structure in this region.</p>
        <canvas id="cv-coop-trio" height="380"></canvas>
        <div id="coop-deviation-analysis" style="margin-top:12px;"></div>
    </div>
</div>

<!-- Geometry Panel (3D manifold visualization) -->
<div class="panel" id="panel-geometry">
    <div class="card">
        <h2>Feature Projection</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Each colored plane is a partition's local linear model in the selected feature subspace.
            Drag to rotate, scroll to zoom.</p>
        <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
            <label style="font-size:12px;">X:</label>
            <select id="sel-geo-x" style="font-size:12px;"></select>
            <label style="font-size:12px;">Y:</label>
            <select id="sel-geo-y" style="font-size:12px;"></select>
            <label style="font-size:12px;margin-left:12px;">Z:</label>
            <select id="sel-geo-z" style="font-size:12px;">
                <option value="pred">Prediction</option>
                <option value="true">True y</option>
            </select>
            <label style="font-size:12px;margin-left:12px;">
                <input type="checkbox" id="geo-show-planes" checked> Planes
            </label>
            <label style="font-size:12px;">
                <input type="checkbox" id="geo-show-points" checked> Points
            </label>
        </div>
        <div id="three-container" style="width:100%;height:550px;border-radius:8px;overflow:hidden;background:#1a1a2e;"></div>
    </div>
    <div class="card" style="margin-top:16px;padding:0;overflow:hidden;position:relative;background:#000;">
        <div id="three-global-container" style="width:100%;height:640px;cursor:grab;"></div>
        <!-- Overlay controls — glass-like floating UI -->
        <div id="gm-overlay" style="position:absolute;top:0;left:0;right:0;bottom:0;pointer-events:none;">
            <!-- Title -->
            <div style="position:absolute;top:20px;left:24px;">
                <div style="font-family:'Segoe UI',system-ui,sans-serif;font-size:22px;font-weight:300;
                    color:rgba(255,255,255,0.85);letter-spacing:2px;text-transform:uppercase;">
                    Geometric Fingerprint</div>
                <div id="gm-subtitle" style="font-family:'Segoe UI',system-ui,sans-serif;font-size:12px;
                    color:rgba(255,255,255,0.4);letter-spacing:1px;margin-top:4px;"></div>
            </div>
            <!-- PCA info -->
            <div id="global-pca-info" style="position:absolute;top:20px;right:24px;
                font-family:'Segoe UI Mono','SF Mono',monospace;font-size:11px;
                color:rgba(255,255,255,0.3);text-align:right;letter-spacing:0.5px;"></div>
            <!-- Controls — bottom-left -->
            <div style="position:absolute;bottom:20px;left:24px;display:flex;gap:8px;align-items:center;pointer-events:auto;">
                <label style="font-size:11px;color:rgba(255,255,255,0.5);cursor:pointer;
                    display:flex;align-items:center;gap:4px;">
                    <input type="checkbox" id="gm-show-surface" style="accent-color:#4e79a7;"> Surface
                </label>
                <label style="font-size:11px;color:rgba(255,255,255,0.5);cursor:pointer;
                    display:flex;align-items:center;gap:4px;">
                    <input type="checkbox" id="gm-hide-outliers" style="accent-color:#e08fad;"> Hide outliers
                </label>
                <button id="gm-pause" style="font-size:11px;padding:4px 10px;
                    background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.5);
                    border:1px solid rgba(255,255,255,0.12);border-radius:4px;
                    cursor:pointer;letter-spacing:0.5px;backdrop-filter:blur(8px);">Pause</button>
            </div>
            <!-- Branding — bottom-right -->
            <div style="position:absolute;bottom:20px;right:24px;
                font-family:'Segoe UI',system-ui,sans-serif;font-size:10px;
                color:rgba(255,255,255,0.2);letter-spacing:1px;">GeoXGB</div>
        </div>
    </div>
</div>

<!-- Map Panel (conditional) -->
<div class="panel" id="panel-map">
    <div class="card">
        <h2>Feature Influence Map</h2>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">
            Shows what percentage of each prediction&rsquo;s total adjustments come
            from a given feature. <b>Raw</b> = main effect only.
            <b>Net</b> = main + Shapley-attributed share of each interaction
            (1/k per feature in a k-way interaction).
            Net influences sum to 100% across all features.</p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:10px;align-items:center;">
            <select id="sel-map-feature" style="min-width:160px;"></select>
            <div id="mode-toggle-wrap" style="display:flex;gap:2px;">
                <button id="btn-raw" class="mode-btn mode-btn-active"
                    onclick="setInfluenceMode('raw')"
                    style="padding:4px 12px;font-size:12px;border:1px solid #ccc;border-radius:4px 0 0 4px;cursor:pointer;">Raw</button>
                <button id="btn-net" class="mode-btn"
                    onclick="setInfluenceMode('net')"
                    style="padding:4px 12px;font-size:12px;border:1px solid #ccc;border-radius:0 4px 4px 0;cursor:pointer;">Net</button>
            </div>
        </div>
        <div id="map"></div>
    </div>
</div>

</div><!-- container -->

<footer>Generated by GeoXGB v{_esc(artifact.version)}</footer>

<script>
// ========== DATA ==========
const D = {data_json};

// ========== THEME ==========
function toggleTheme() {{
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? '' : 'dark');
    document.getElementById('theme-toggle').innerHTML = isDark ? '&#9790;' : '&#9788;';
    try {{ localStorage.setItem('geoxgb-theme', isDark ? 'light' : 'dark'); }} catch(e) {{}}
    // Swap map tiles if map is initialised
    // isDark = was dark before toggle, so now switching TO light or dark
    if (window._tileLayer && window._map) {{
        window._tileLayer.setUrl(isDark
            ? 'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'
            : 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png');
    }}
    redrawAll();
}}
// Apply saved preference
(function() {{
    try {{
        const saved = localStorage.getItem('geoxgb-theme');
        if (saved === 'dark') {{
            document.documentElement.setAttribute('data-theme', 'dark');
            document.getElementById('theme-toggle').innerHTML = '&#9788;';
        }}
    }} catch(e) {{}}
}})();

// Theme-aware color helpers
function _css(v) {{ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }}
function _rgb(v) {{ return _css(v).split(',').map(Number); }}

function themeColors() {{
    return {{
        text: _css('--text-primary'),
        textSec: _css('--text-secondary'),
        textMuted: _css('--text-muted'),
        primary: _css('--chart-primary'),
        primaryFill: _css('--chart-primary-fill'),
        primaryBand: _css('--chart-primary-band'),
        primaryEdge: _css('--chart-primary-edge'),
        grid: _css('--chart-grid'),
        bgCard: _css('--bg-card'),
        bgTh: _css('--bg-table-header'),
        good: _css('--status-good'),
        warn: _css('--status-warn'),
        bad: _css('--status-bad'),
        divPos: _css('--div-pos'),
        divNeg: _css('--div-neg'),
        markerOutline: _css('--marker-outline'),
    }};
}}

// Sequential colormap for 0-1 values (used everywhere)
function seqColor(t) {{
    t = Math.max(0, Math.min(1, t));
    const lo = _rgb('--seq-lo-rgb'), mid = _rgb('--seq-mid-rgb'), hi = _rgb('--seq-hi-rgb');
    let r, g, b;
    if (t < 0.5) {{
        const u = t / 0.5;
        r = lo[0] + (mid[0]-lo[0])*u; g = lo[1] + (mid[1]-lo[1])*u; b = lo[2] + (mid[2]-lo[2])*u;
    }} else {{
        const u = (t - 0.5) / 0.5;
        r = mid[0] + (hi[0]-mid[0])*u; g = mid[1] + (hi[1]-mid[1])*u; b = mid[2] + (hi[2]-mid[2])*u;
    }}
    return [Math.round(r), Math.round(g), Math.round(b)];
}}
function seqRgb(t) {{ const c = seqColor(t); return 'rgb('+c[0]+','+c[1]+','+c[2]+')'; }}

// Fixed sequential colormap for map points (theme-independent)
function mapSeqColor(t) {{
    t = Math.max(0, Math.min(1, t));
    // Yellow → Orange → Red-Purple (good visibility on both light & dark tiles)
    const lo = [255,255,178], mid = [253,141,60], hi = [189,0,38];
    let r, g, b;
    if (t < 0.5) {{
        const u = t / 0.5;
        r = lo[0]+(mid[0]-lo[0])*u; g = lo[1]+(mid[1]-lo[1])*u; b = lo[2]+(mid[2]-lo[2])*u;
    }} else {{
        const u = (t-0.5)/0.5;
        r = mid[0]+(hi[0]-mid[0])*u; g = mid[1]+(hi[1]-mid[1])*u; b = mid[2]+(hi[2]-mid[2])*u;
    }}
    return [Math.round(r), Math.round(g), Math.round(b)];
}}
function mapSeqRgb(t) {{ const c = mapSeqColor(t); return 'rgb('+c[0]+','+c[1]+','+c[2]+')'; }}

// Diverging colormap for signed values [-1,1] → neg-mid-pos
function divColor(t) {{
    t = Math.max(-1, Math.min(1, t));
    const pos = _rgb('--div-pos-rgb'), neg = _rgb('--div-neg-rgb'), mid = _rgb('--div-mid-rgb');
    let r, g, b;
    if (t >= 0) {{
        r = mid[0] + (pos[0]-mid[0])*t; g = mid[1] + (pos[1]-mid[1])*t; b = mid[2] + (pos[2]-mid[2])*t;
    }} else {{
        const a = -t;
        r = mid[0] + (neg[0]-mid[0])*a; g = mid[1] + (neg[1]-mid[1])*a; b = mid[2] + (neg[2]-mid[2])*a;
    }}
    return [Math.round(r), Math.round(g), Math.round(b)];
}}
function divRgb(t) {{ const c = divColor(t); return 'rgb('+c[0]+','+c[1]+','+c[2]+')'; }}

// Redraw registry: canvas drawing functions register here for theme-change repaints
const _redraws = [];
function registerRedraw(fn) {{ _redraws.push(fn); }}
function redrawAll() {{ _redraws.forEach(fn => fn()); }}

// ========== TABS ==========
const tabDefs = [
    ["overview", "Overview"],
    ["features", "Features"],
    ["partitions", "Partitions"],
    ["samples", "Samples"],
    ["residuals", "Residuals"],
];
if (D.partition_summary) tabDefs.splice(3, 0, ["geometry", "Geometry"]);
if (D.geo_data) tabDefs.push(["map", "Map"]);

// Deferred draw registry: panels that need redraw when first shown
const _deferredDraw = {{}};
function onPanelShow(panelId, fn) {{
    if (!_deferredDraw[panelId]) _deferredDraw[panelId] = [];
    _deferredDraw[panelId].push(fn);
}}
// Persistent panel-show callbacks (fire every time panel is shown)
const _panelShowCallbacks = {{}};
function onPanelShowAlways(panelId, fn) {{
    if (!_panelShowCallbacks[panelId]) _panelShowCallbacks[panelId] = [];
    _panelShowCallbacks[panelId].push(fn);
}}

const tabsEl = document.getElementById("main-tabs");
tabDefs.forEach(([id, label], i) => {{
    const btn = document.createElement("button");
    btn.className = "tab" + (i === 0 ? " active" : "");
    btn.textContent = label;
    btn.onclick = () => {{
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("panel-" + id).classList.add("active");
        if (id === "map" && D.geo_data && !window._mapInit) initMap();
        // Fire deferred draws for this panel (one-shot)
        if (_deferredDraw[id]) {{
            _deferredDraw[id].forEach(fn => fn());
            delete _deferredDraw[id];
        }}
        // Fire persistent callbacks
        if (_panelShowCallbacks[id]) {{
            _panelShowCallbacks[id].forEach(fn => fn());
        }}
    }};
    tabsEl.appendChild(btn);
}});
document.getElementById("panel-overview").classList.add("active");

// ========== HEADER ==========
const _ts = D.timestamp ? new Date(D.timestamp).toLocaleDateString(undefined,
    {{year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit'}}) : D.timestamp;
document.getElementById("header-sub").innerHTML =
    D.model_type + " &mdash; " + D.task + " &mdash; v" + D.version +
    "<br>" + _ts;

// ========== OVERVIEW ==========
(function() {{
    const fs = D.fit_stats;
    let h = '<h2>Fit Summary</h2><table class="dt">';
    h += '<tr><td>Training samples</td><td>' + (fs.n_train||'?') + '</td></tr>';
    h += '<tr><td>Features</td><td>' + (fs.n_features||'?') + '</td></tr>';
    h += '<tr><td>Rounds (configured)</td><td>' + (fs.n_rounds||'?') + '</td></tr>';
    h += '<tr><td>Rounds (actual)</td><td>' + (fs.n_rounds_actual||'?') + '</td></tr>';
    h += '<tr><td>Convergence round</td><td>' + (fs.convergence_round||'n/a') + '</td></tr>';
    h += '</table>';
    document.getElementById("card-fit").innerHTML = h;

    // Test metrics
    let mh = '<h2>Test Performance</h2>';
    if (D.predictions && D.predictions.test && D.predictions.test.metrics) {{
        mh += '<table class="dt">';
        for (const [k,v] of Object.entries(D.predictions.test.metrics)) {{
            mh += '<tr><td>' + k + '</td><td>' +
                (typeof v === 'number' ? v.toFixed(4) : v) + '</td></tr>';
        }}
        mh += '</table>';
    }} else {{
        mh += '<p style="color:var(--text-muted)">No test data provided.</p>';
    }}
    document.getElementById("card-metrics").innerHTML = mh;

    // Noise + Irreducible Error
    const noise = D.noise;
    const irr = D.irreducible_error;
    let nh = '<h2>Noise & Irreducible Error</h2>';
    if (noise && noise.assessment) {{
        const colors = {{clean:"var(--status-good)", moderate:"var(--status-warn)", noisy:"var(--status-bad)"}};
        nh += '<span class="badge" style="background:' +
            (colors[noise.assessment]||"#6c757d") + '">' + noise.assessment + '</span>';
        nh += '<table class="dt" style="margin-top:8px">';
        if (noise.initial_modulation != null)
            nh += '<tr><td>Initial modulation</td><td>' +
                (typeof noise.initial_modulation === 'number'
                    ? (noise.initial_modulation * 100).toFixed(1) + '%'
                    : noise.initial_modulation) + '</td></tr>';
        if (noise.final_modulation != null)
            nh += '<tr><td>Final modulation</td><td>' +
                (typeof noise.final_modulation === 'number'
                    ? (noise.final_modulation * 100).toFixed(1) + '%'
                    : noise.final_modulation) + '</td></tr>';
        if (noise.interpretation)
            nh += '<tr><td>Interpretation</td><td>' + noise.interpretation + '</td></tr>';
        nh += '</table>';
    }} else {{
        nh += '<p style="color:var(--text-muted)">Noise assessment not available.</p>';
    }}
    if (irr) {{
        nh += '<h3 style="margin-top:14px">Irreducible Error Estimation</h3>';
        nh += '<table class="dt">';
        nh += '<tr><td>Combined noise floor (&sigma;)</td><td>' + irr.global_std.toFixed(4) + '</td></tr>';
        if (irr.knn_global_std != null)
            nh += '<tr><td>&nbsp;&nbsp;k-NN estimate (k=' + irr.k + ')</td><td>' + irr.knn_global_std.toFixed(4) + '</td></tr>';
        if (irr.partition_global_std != null)
            nh += '<tr><td>&nbsp;&nbsp;Partition estimate</td><td>' + irr.partition_global_std.toFixed(4) + '</td></tr>';
        nh += '<tr><td>Method</td><td>' + irr.method + '</td></tr>';
        nh += '</table>';
        const desc = irr.partition_global_std != null
            ? 'Two methods combined via geometric mean: (1) k-NN local y-variance estimates noise from nearby samples, ' +
              '(2) partition variance uses HVRT leaf structure. '
            : 'Per-sample local noise estimated from k nearest neighbor y-variance. ';
        nh += '<p style="font-size:11px;color:var(--text-muted);margin-top:6px;">' +
            desc +
            'The adjusted residual divides |residual| by local noise, identifying ' +
            'genuinely poor predictions vs inherently noisy regions.</p>';
    }}
    document.getElementById("card-noise").innerHTML = nh;
}})();

// ========== CANVAS HELPERS ==========
// Safe min/max for large arrays (avoids stack overflow from spread)
function safeMin(a) {{ let m = a[0]; for (let i = 1; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }}
function safeMax(a) {{ let m = a[0]; for (let i = 1; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }}

function getCtx(id) {{
    const cv = document.getElementById(id);
    const dpr = window.devicePixelRatio || 1;
    // Read logical height from data attribute (stable across redraws)
    if (!cv.dataset.logicalH) cv.dataset.logicalH = cv.getAttribute("height") || "300";
    const h = parseInt(cv.dataset.logicalH);
    const rect = cv.getBoundingClientRect();
    const w = rect.width || cv.parentElement.clientWidth || 700;
    cv.width = w * dpr;
    cv.height = h * dpr;
    cv.style.width = w + "px";
    cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    return [ctx, w, h];
}}

// rdbu now delegates to theme-aware divColor
function rdbu(t) {{ return divColor(t); }}
function clamp(v,lo,hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function rgb(r,g,b) {{ return 'rgb('+clamp(Math.round(r),0,255)+','+
    clamp(Math.round(g),0,255)+','+clamp(Math.round(b),0,255)+')'; }}

// ========== IMPORTANCES ==========
(function() {{
    const imp = D.feature_importances;
    if (!imp || Object.keys(imp).length === 0) return;
    const sorted = Object.entries(imp).sort((a,b) => b[1]-a[1]).slice(0, 20);
    const items = sorted.reverse();
    function drawImp() {{
        const tc = themeColors();
        const [ctx, W, H] = getCtx("cv-imp");
        const pad = {{l:120, r:60, t:10, b:20}};
        const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
        const maxV = Math.max(...items.map(x=>x[1])) || 1;
        const bh = Math.min(20, ch / items.length - 2);
        ctx.font = "11px -apple-system, sans-serif";
        ctx.textBaseline = "middle";
        items.forEach(([name, val], i) => {{
            const y = pad.t + (ch / items.length) * i + bh/2;
            const bw = (val/maxV) * cw;
            ctx.fillStyle = tc.primary;
            ctx.fillRect(pad.l, y - bh/2, bw, bh);
            ctx.fillStyle = tc.text;
            ctx.textAlign = "right";
            ctx.fillText(name, pad.l - 6, y);
            ctx.textAlign = "left";
            ctx.fillStyle = tc.textSec;
            ctx.fillText(val.toFixed(4), pad.l + bw + 4, y);
        }});
    }}
    drawImp();
    registerRedraw(drawImp);
}})();

// ========== MAIN EFFECTS ==========
let _ebmMode = "main";
function setEbmMode(mode) {{
    _ebmMode = mode;
    ["main","net"].forEach(m => {{
        const btn = document.getElementById("btn-ebm-" + m);
        if (btn) btn.classList.toggle("mode-btn-active", m === mode);
    }});
    const sel = document.getElementById("sel-feat");
    if (sel && sel.value) _ebmDraw(sel.value);
}}
let _ebmDraw = () => {{}};

(function() {{
    const curves = D.main_curves;
    if (!curves || Object.keys(curves).length === 0) {{
        document.getElementById("panel-features").innerHTML =
            '<div class="card"><p style="color:var(--text-muted)">No main effect curves available.</p></div>';
        return;
    }}
    const sel = document.getElementById("sel-feat");
    const feats = Object.keys(curves);

    // Detect geo lat/lon features — hide individual lat/lon when interaction exists
    const geoData = D.geo_data;
    const latFeat = geoData ? geoData.lat_feature : null;
    const lonFeat = geoData ? geoData.lon_feature : null;
    const hasLatLonInteraction = latFeat && lonFeat && feats.some(f => f === latFeat) && feats.some(f => f === lonFeat);

    feats.forEach(f => {{
        if (hasLatLonInteraction && (f === latFeat || f === lonFeat)) return;
        const opt = document.createElement("option");
        opt.value = f; opt.textContent = f;
        sel.appendChild(opt);
    }});

    // Distinct partition colors (categorical, high contrast)
    const partColors = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
        "#86bcb6","#8cd17d","#b6992d","#499894","#d37295",
        "#a0cbe8","#ffbe7d","#d4a6c8","#fabfd2","#d7b5a6"
    ];

    function draw(feat) {{
        const c = curves[feat];
        if (!c) return;
        const tc = themeColors();
        const [ctx, W, H] = getCtx("cv-main");
        const pad = {{l:65, r:130, t:20, b:40}};
        const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
        ctx.clearRect(0, 0, W, H);

        const useNet = _ebmMode === "net";
        const scatterY = useNet && c.scatter_net ? c.scatter_net : c.scatter_main;
        const scatterX = c.scatter_x;
        const pids = c.scatter_pid;
        const lines = useNet && c.partitions_net ? c.partitions_net : c.partitions;
        const yLabel = useNet ? "Contribution (net)" : "Contribution (main)";

        // Compute bounds from scatter points
        const xMin = safeMin(scatterX), xMax = safeMax(scatterX);
        const yMin = safeMin(scatterY), yMax = safeMax(scatterY);
        const yPad = (yMax - yMin) * 0.05 || 0.1;
        const yLo = yMin - yPad, yHi = yMax + yPad;
        const yRange = yHi - yLo;
        const xRange = (xMax - xMin) || 1;
        const sx = x => pad.l + (x - xMin) / xRange * cw;
        const sy = y => pad.t + ch - (y - yLo) / yRange * ch;

        // Scatter points color-coded by partition
        const dotR = Math.max(1.5, Math.min(3, 600 / scatterX.length));
        ctx.globalAlpha = 0.35;
        for (let i = 0; i < scatterX.length; i++) {{
            ctx.fillStyle = pids ? partColors[pids[i] % partColors.length] : tc.primary;
            ctx.beginPath();
            ctx.arc(sx(scatterX[i]), sy(scatterY[i]), dotR, 0, 2 * Math.PI);
            ctx.fill();
        }}
        ctx.globalAlpha = 1.0;

        // Per-partition regression lines
        if (lines) {{
            lines.forEach(ln => {{
                const col = partColors[ln.pid % partColors.length];
                const y0 = ln.slope * ln.x_min + ln.intercept;
                const y1 = ln.slope * ln.x_max + ln.intercept;
                ctx.strokeStyle = col;
                ctx.lineWidth = ln.n > 50 ? 2.5 : 1.5;
                ctx.beginPath();
                ctx.moveTo(sx(ln.x_min), sy(y0));
                ctx.lineTo(sx(ln.x_max), sy(y1));
                ctx.stroke();
            }});
        }}

        // Zero line
        if (yLo < 0 && yHi > 0) {{
            ctx.strokeStyle = tc.grid; ctx.lineWidth = 1;
            ctx.setLineDash([4,4]);
            ctx.beginPath(); ctx.moveTo(pad.l, sy(0)); ctx.lineTo(pad.l+cw, sy(0)); ctx.stroke();
            ctx.setLineDash([]);
        }}

        // Axes
        ctx.strokeStyle = tc.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ch);
        ctx.lineTo(pad.l+cw, pad.t+ch); ctx.stroke();

        // Labels
        ctx.fillStyle = tc.text; ctx.font = "12px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(feat, pad.l + cw/2, H - 8);
        ctx.save(); ctx.translate(14, pad.t + ch/2);
        ctx.rotate(-Math.PI/2); ctx.fillText(yLabel, 0, 0); ctx.restore();

        // Tick labels
        ctx.font = "10px -apple-system, sans-serif";
        ctx.fillStyle = tc.textMuted; ctx.textAlign = "center";
        const nxt = 5;
        for (let i=0; i<=nxt; i++) {{
            const v = xMin + (xRange * i / nxt);
            ctx.fillText(v.toPrecision(3), sx(v), pad.t + ch + 16);
        }}
        ctx.textAlign = "right";
        for (let i=0; i<=4; i++) {{
            const v = yLo + (yRange * i / 4);
            ctx.fillText(v.toPrecision(3), pad.l - 6, sy(v) + 3);
        }}

        // Legend: partition lines with slopes (right side)
        if (lines && lines.length > 0) {{
            const legendX = pad.l + cw + 12;
            let legendY = pad.t + 4;
            ctx.font = "bold 9px -apple-system, sans-serif";
            ctx.fillStyle = tc.text;
            ctx.textAlign = "left";
            ctx.fillText("Partitions", legendX, legendY);
            legendY += 12;
            ctx.font = "9px -apple-system, sans-serif";
            const maxShow = Math.min(lines.length, 12);
            lines.slice(0, maxShow).forEach((ln, i) => {{
                const col = partColors[ln.pid % partColors.length];
                // Color swatch
                ctx.fillStyle = col;
                ctx.fillRect(legendX, legendY - 6, 8, 8);
                // Label: n=X, slope=Y
                ctx.fillStyle = tc.text;
                const slopeStr = (ln.slope >= 0 ? "+" : "") + ln.slope.toFixed(3);
                ctx.fillText("n=" + ln.n + "  \u03B2=" + slopeStr, legendX + 12, legendY);
                legendY += 12;
            }});
            if (lines.length > maxShow) {{
                ctx.fillStyle = tc.textMuted;
                ctx.fillText("+" + (lines.length - maxShow) + " more", legendX + 12, legendY);
            }}
        }}
    }}
    // Partition table below the scatter plot
    function updateFeatTable(feat) {{
        const c = curves[feat];
        const tbl = document.getElementById("feat-partition-table");
        if (!tbl || !c) return;
        const useNet = _ebmMode === "net";
        const lines = useNet && c.partitions_net ? c.partitions_net : c.partitions;
        if (!lines || lines.length === 0) {{ tbl.innerHTML = ""; return; }}
        const tc = themeColors();
        let h = '<h3 style="font-size:13px;margin-bottom:6px;">Per-partition slopes (' +
            (useNet ? "net" : "main") + ')</h3>';
        h += '<table class="dt"><tr><th style="width:30px;"></th><th>Partition</th><th>Samples</th><th>Slope (&beta;)</th><th style="width:35%;">Bar</th></tr>';
        const maxSlope = Math.max(...lines.map(l => Math.abs(l.slope))) || 1;
        lines.forEach(ln => {{
            const col = partColors[ln.pid % partColors.length];
            const pct = Math.abs(ln.slope) / maxSlope * 100;
            const sign = ln.slope >= 0 ? "+" : "";
            h += '<tr><td><div style="width:12px;height:12px;border-radius:2px;background:' + col + ';"></div></td>';
            h += '<td>#' + ln.pid + '</td>';
            h += '<td>' + ln.n + ' (' + (ln.n / (c.scatter_x ? c.scatter_x.length : ln.n) * 100).toFixed(0) + '%)</td>';
            h += '<td style="font-family:monospace;">' + sign + ln.slope.toFixed(4) + '</td>';
            h += '<td><div style="background:' + col + ';height:10px;width:' + pct.toFixed(1) +
                '%;border-radius:3px;opacity:0.7;"></div></td></tr>';
        }});
        h += '</table>';
        tbl.innerHTML = h;
    }}

    const origDraw = draw;
    draw = function(feat) {{ origDraw(feat); updateFeatTable(feat); }};

    _ebmDraw = draw;
    sel.onchange = () => draw(sel.value);
    onPanelShow("features", () => draw(sel.value));
    registerRedraw(() => {{ if (document.getElementById("panel-features").classList.contains("active")) draw(sel.value); }});
}})();

// ========== PARTITIONS PANEL ==========
(function() {{
    const ps = D.partition_summary;
    const el = document.getElementById("partition-overview");
    if (!ps || !el) return;

    const parts = ps.partitions;
    const fnames = ps.feature_names;

    // Partition color palette (same as feature scatter)
    const partColors = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
        "#86bcb6","#8cd17d","#b6992d","#499894","#d37295",
        "#a0cbe8","#ffbe7d","#d4a6c8","#fabfd2","#d7b5a6"
    ];

    // Overview table
    let h = '<table class="dt"><tr><th style="width:30px;"></th><th>ID</th>' +
        '<th>Samples</th><th>%</th><th>Local R&sup2;</th>' +
        '<th>Mean |resid|</th><th>Mean y</th><th></th></tr>';
    parts.forEach(p => {{
        const col = partColors[p.pid % partColors.length];
        h += '<tr style="cursor:pointer;" onclick="showPartitionDetail(' + p.pid + ')">';
        h += '<td><div style="width:12px;height:12px;border-radius:2px;background:' + col + ';"></div></td>';
        h += '<td>#' + p.pid + '</td>';
        h += '<td>' + p.n + '</td>';
        h += '<td>' + p.pct + '%</td>';
        h += '<td>' + p.local_r2.toFixed(4) + '</td>';
        h += '<td>' + p.mean_abs_resid.toFixed(4) + '</td>';
        h += '<td>' + p.mean_y.toFixed(4) + '</td>';
        h += '<td style="font-size:11px;color:var(--accent);cursor:pointer;">Details &rarr;</td>';
        h += '</tr>';
    }});
    h += '</table>';
    h += '<p style="font-size:11px;color:var(--text-muted);margin-top:6px;">' +
        ps.n_partitions + ' partitions total. Click a row to explore.</p>';
    el.innerHTML = h;

    // Detail view
    window.showPartitionDetail = function(pid) {{
        const p = parts.find(pp => pp.pid === pid);
        if (!p) return;
        const tc = themeColors();
        const card = document.getElementById("card-partition-detail");
        card.style.display = "block";
        document.getElementById("partition-detail-pid").textContent = "#" + pid;

        const col = partColors[pid % partColors.length];
        let sh = '<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px;">';
        sh += '<div><b style="font-size:12px;">Samples</b><br>' + p.n + ' (' + p.pct + '%)</div>';
        sh += '<div><b style="font-size:12px;">Local R&sup2;</b><br>' + p.local_r2.toFixed(4) + '</div>';
        sh += '<div><b style="font-size:12px;">Mean |residual|</b><br>' + p.mean_abs_resid.toFixed(4) + '</div>';
        sh += '<div><b style="font-size:12px;">Residual &sigma;</b><br>' + p.std_resid.toFixed(4) + '</div>';
        sh += '<div><b style="font-size:12px;">Mean y</b><br>' + p.mean_y.toFixed(4) + '</div>';
        sh += '<div><b style="font-size:12px;">Mean pred</b><br>' + p.mean_pred.toFixed(4) + '</div>';
        sh += '</div>';

        // Feature means table
        sh += '<details><summary style="font-weight:600;font-size:13px;cursor:pointer;margin:8px 0 6px;">Feature Means</summary>';
        sh += '<table class="dt"><tr><th>Feature</th><th>Mean value</th></tr>';
        fnames.forEach(f => {{
            sh += '<tr><td>' + f + '</td><td style="font-family:monospace;">' +
                (p.feature_means[f] != null ? p.feature_means[f].toFixed(4) : '-') + '</td></tr>';
        }});
        sh += '</table></details>';
        document.getElementById("partition-detail-stats").innerHTML = sh;

        // Draw contribution slopes bar chart
        const slopes = p.contrib_slopes || {{}};
        const sEntries = fnames.map(f => ({{ name: f, slope: slopes[f] || 0 }}));
        sEntries.sort((a,b) => Math.abs(b.slope) - Math.abs(a.slope));

        const [ctx, W, H] = getCtx("cv-partition-slopes");
        const pad = {{l:90, r:20, t:30, b:10}};
        const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
        ctx.clearRect(0, 0, W, H);

        const maxS = Math.max(...sEntries.map(e => Math.abs(e.slope))) || 1;
        const barH = Math.min(22, ch / sEntries.length - 2);
        const gap = 2;

        ctx.fillStyle = tc.text; ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Contribution slopes (\u03B2) — Partition #" + pid, pad.l + cw/2, 16);

        ctx.font = "10px -apple-system, sans-serif";
        sEntries.forEach((e, i) => {{
            const y = pad.t + i * (barH + gap);
            const w = Math.abs(e.slope) / maxS * cw * 0.9;
            const x0 = pad.l + cw * 0.45;  // center point

            ctx.fillStyle = e.slope >= 0 ? tc.divPos : tc.divNeg;
            if (e.slope >= 0) {{
                ctx.fillRect(x0, y, w, barH);
            }} else {{
                ctx.fillRect(x0 - w, y, w, barH);
            }}

            // Label
            ctx.fillStyle = tc.text;
            ctx.textAlign = "right";
            ctx.fillText(e.name, pad.l - 4, y + barH * 0.7);

            // Value
            ctx.textAlign = "left";
            const valX = e.slope >= 0 ? x0 + w + 4 : x0 - w - 4;
            ctx.textAlign = e.slope >= 0 ? "left" : "right";
            ctx.fillStyle = tc.textMuted;
            ctx.fillText((e.slope >= 0 ? "+" : "") + e.slope.toFixed(4), valX, y + barH * 0.7);
        }});

        // Center zero line
        const x0 = pad.l + cw * 0.45;
        ctx.strokeStyle = tc.grid; ctx.lineWidth = 1;
        ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(x0, pad.t - 4); ctx.lineTo(x0, pad.t + sEntries.length * (barH + gap)); ctx.stroke();
        ctx.setLineDash([]);
    }};

    onPanelShow("partitions", () => {{}});
}})();

// ========== COOPERATION DETAIL (sample-level) ==========
let _lastCoopArgs = null;  // cache for theme redraw
// Draw all three cooperation heatmaps on a single canvas
function drawCoopTrio(globalMat, localMat, devMat, names) {{
    _lastCoopArgs = [globalMat, localMat, devMat, names];
    const cv = document.getElementById("cv-coop-trio");
    if (!cv) return;
    const d = names.length;
    const tc = themeColors();
    const dpr = window.devicePixelRatio || 1;

    // Compute layout based on container width
    const containerW = cv.parentElement.clientWidth || 900;
    const gap = 20;                     // gap between heatmaps
    const labelPadL = 80;              // left padding for row labels (first heatmap only)
    const labelPadT = 85;              // top padding for column labels + titles
    const padB = 10;

    // Each heatmap: cell grid of d×d. First one has left labels, others don't.
    // Available width for all three grids:
    const gridW = containerW - labelPadL - gap * 2;
    const cell = Math.max(8, Math.floor(gridW / (3 * d)));
    const gridPx = cell * d;
    const totalW = labelPadL + gridPx * 3 + gap * 2;
    const totalH = labelPadT + gridPx + padB;

    // Size the canvas — center via CSS auto margins
    cv.style.display = "block";
    cv.style.margin = "0 auto";
    cv.style.width = totalW + "px";
    cv.style.height = totalH + "px";
    cv.width = Math.round(totalW * dpr);
    cv.height = Math.round(totalH * dpr);
    cv.dataset.logicalH = String(totalH);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, totalW, totalH);

    const titles = ["Global", "Local (partition)", "Deviation (local \u2212 global)"];
    const mats = [globalMat, localMat, devMat];

    mats.forEach((mat, mi) => {{
        const ox = labelPadL + mi * (gridPx + gap);
        const oy = labelPadT;

        // Find max abs value for this matrix
        let vmax = 0;
        mat.forEach(row => row.forEach(v => {{ vmax = Math.max(vmax, Math.abs(v)); }}));
        vmax = vmax || 1;

        // Draw cells
        for (let i = 0; i < d; i++) {{
            for (let j = 0; j < d; j++) {{
                const t = mat[i][j] / vmax;
                const [r,g,b] = rdbu(t);
                ctx.fillStyle = rgb(r,g,b);
                ctx.fillRect(ox + j * cell, oy + i * cell, cell - 0.5, cell - 0.5);
            }}
        }}

        // Title above heatmap
        ctx.fillStyle = tc.text;
        ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(titles[mi], ox + gridPx / 2, 14);

        // Column labels (vertical)
        const fs = Math.max(7, Math.min(10, cell - 1));
        ctx.font = fs + "px -apple-system, sans-serif";
        ctx.fillStyle = tc.text;
        names.forEach((n, i) => {{
            ctx.save();
            ctx.translate(ox + i * cell + cell / 2 + fs * 0.35, oy - 4);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(n.length > 10 ? n.substring(0, 9) + "\u2026" : n, 0, 0);
            ctx.restore();
        }});

        // Row labels (only on first heatmap)
        if (mi === 0) {{
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.font = fs + "px -apple-system, sans-serif";
            ctx.fillStyle = tc.text;
            names.forEach((n, i) => {{
                ctx.fillText(n.length > 10 ? n.substring(0, 9) + "\u2026" : n, ox - 4, oy + i * cell + cell / 2);
            }});
        }}
    }});
}}

// Show cooperation detail panel for a specific sample index
function showCoopDetail(sampleIdx) {{
    const sl = D.sample_lookup;
    const card = document.getElementById("card-coop-detail");
    if (!sl || !sl.coop_partitions || !sl.global_coop_matrix) {{
        if (card) card.style.display = "none";
        return;
    }}

    card.style.display = "block";
    const names = sl.feature_names;
    const globalMat = sl.global_coop_matrix;
    const partIdx = sl.coop_partition_idx[sampleIdx];
    const localMat = sl.coop_partitions[partIdx];
    const d = names.length;

    document.getElementById("coop-detail-title").textContent = "Sample " + sampleIdx;

    // Deviation matrix
    const devMat = localMat.map((row, i) => row.map((v, j) => v - globalMat[i][j]));

    // Draw all three on one canvas
    drawCoopTrio(globalMat, localMat, devMat, names);

    // Deviation analysis: find top deviating pairs
    const deviations = [];
    for (let i = 0; i < d; i++) {{
        for (let j = i + 1; j < d; j++) {{
            const dev = devMat[i][j];
            deviations.push({{
                fa: names[i], fb: names[j],
                global_v: globalMat[i][j],
                local_v: localMat[i][j],
                dev: dev, abs_dev: Math.abs(dev)
            }});
        }}
    }}
    deviations.sort((a, b) => b.abs_dev - a.abs_dev);

    const tc = themeColors();
    let h = '<h3 style="font-size:13px;margin-bottom:8px;">Top deviating pairs (local vs global)</h3>';
    h += '<table class="dt"><tr><th>Pair</th><th>Global</th><th>Local</th><th>Deviation</th><th>Interpretation</th></tr>';
    deviations.slice(0, 6).forEach(p => {{
        const dc = p.dev > 0 ? tc.divPos : tc.divNeg;
        let interp = "";
        if (p.abs_dev < 0.05) {{
            interp = "Consistent";
        }} else if (Math.abs(p.local_v) > Math.abs(p.global_v) + 0.1) {{
            interp = p.local_v > 0
                ? "Stronger positive coupling locally"
                : "Stronger negative coupling locally";
        }} else if (Math.abs(p.local_v) < Math.abs(p.global_v) - 0.1) {{
            interp = "Weaker coupling locally — relationship is heterogeneous";
        }} else if (Math.sign(p.local_v) !== Math.sign(p.global_v) && p.abs_dev > 0.1) {{
            interp = "Sign reversal — highly heterogeneous";
        }} else {{
            interp = p.dev > 0 ? "More positive locally" : "More negative locally";
        }}
        h += '<tr><td>' + p.fa + ' &times; ' + p.fb + '</td>';
        h += '<td>' + p.global_v.toFixed(3) + '</td>';
        h += '<td>' + p.local_v.toFixed(3) + '</td>';
        h += '<td style="color:'+dc+';font-weight:600">' + (p.dev>0?"+":"") + p.dev.toFixed(3) + '</td>';
        h += '<td style="font-size:11px;color:var(--text-muted)">' + interp + '</td></tr>';
    }});
    h += '</table>';

    // Summary statistic
    const meanAbsDev = deviations.reduce((s,p) => s + p.abs_dev, 0) / (deviations.length || 1);
    const maxDev = deviations.length > 0 ? deviations[0].abs_dev : 0;
    let summary = "";
    if (maxDev < 0.05) {{
        summary = "This sample's partition has feature relationships very similar to the global average. " +
            "The model's learned structure here is largely <b>linear and homogeneous</b>.";
    }} else if (maxDev < 0.2) {{
        summary = "Some feature relationships differ from the global average, suggesting " +
            "<b>moderate heterogeneity</b> in this region of feature space.";
    }} else {{
        summary = "Feature relationships in this partition <b>diverge substantially</b> from the global average. " +
            "This region exhibits <b>non-linear or heterogeneous</b> structure that the model treats differently.";
    }}
    h += '<p style="font-size:12px;margin-top:10px;padding:8px 12px;background:var(--bg-body);border-radius:6px;border-left:3px solid var(--accent);">'
        + summary + ' (mean |dev| = ' + meanAbsDev.toFixed(3) + ', max |dev| = ' + maxDev.toFixed(3) + ')</p>';

    document.getElementById("coop-deviation-analysis").innerHTML = h;
}}
registerRedraw(() => {{ if (_lastCoopArgs) drawCoopTrio(..._lastCoopArgs); }});

// ========== SAMPLES (unified: exemplars + lookup) ==========
(function() {{
    const ex = D.exemplars;
    const sl = D.sample_lookup;
    const content = document.getElementById("sample-detail-content");
    const tabs = document.getElementById("exemplar-tabs");

    if (!ex && !sl) {{
        content.innerHTML = '<p style="color:var(--text-muted)">No sample data available.</p>';
        return;
    }}

    // Set up lookup range
    if (sl) {{
        const rangeEl = document.getElementById("lookup-range");
        if (rangeEl) rangeEl.textContent = "(0\u2013" + (sl.n - 1) + ")";
    }}

    const hasAdjusted = ex && ex.adjusted != null;
    let irrMode = "raw";  // "raw" or "adjusted"
    let currentKey = "best";  // "best", "worst", "median", or "lookup"

    // Build toggle if adjusted data exists
    const toggleWrap = document.getElementById("irr-toggle-wrap");
    if (hasAdjusted) {{
        toggleWrap.innerHTML =
            '<label style="font-size:12px;cursor:pointer;">' +
            '<input type="checkbox" id="irr-toggle" style="margin-right:6px;">' +
            '<b>Adjust for irreducible error</b> &mdash; rank by ' +
            '|residual|/local_noise.' +
            '</label>';
        if (D.irreducible_error) {{
            toggleWrap.innerHTML += '<span style="font-size:11px;color:var(--text-muted);margin-left:8px;">' +
                '(&sigma; = ' + D.irreducible_error.global_std.toFixed(4) + ')</span>';
        }}
    }}

    // ---- Unified sample rendering ----
    // Renders a sample with consistent rows regardless of source (exemplar or lookup).
    // s = {{ idx, y_true, y_pred, residual, local_noise_std, adjusted_residual,
    //        partition_size, local_r2, features:[{{name,val,contrib}}],
    //        pairwise:{{pair:coeff}}, interactions:[{{name,val}}] }}
    // tipHtml = optional italic tip above the table
    function renderSample(s, tipHtml) {{
        const tc = themeColors();
        let h = '';
        if (tipHtml) {{
            h += '<p style="font-size:11px;color:var(--text-muted);margin:0 0 6px 0;font-style:italic;">' + tipHtml + '</p>';
        }}
        h += '<table class="dt" style="table-layout:fixed;"><col style="width:180px;"><col>';
        h += '<tr><td>Sample index</td><td>' + s.idx + '</td></tr>';
        h += '<tr><td>y (true)</td><td>' + s.y_true.toFixed(4) + '</td></tr>';
        h += '<tr><td>y (predicted)</td><td>' + s.y_pred.toFixed(4) + '</td></tr>';
        const resColor = Math.abs(s.residual) > 0.5 ? tc.bad : tc.good;
        h += '<tr><td>Residual</td><td style="font-weight:600;color:' + resColor + ';">' + s.residual.toFixed(4) + '</td></tr>';
        if (s.local_noise_std != null)
            h += '<tr><td>Local noise &sigma;</td><td>' + s.local_noise_std.toFixed(4) + '</td></tr>';
        if (s.adjusted_residual != null) {{
            const adjColor = s.adjusted_residual > 2.0 ? tc.bad : s.adjusted_residual > 1.0 ? tc.warn : tc.good;
            h += '<tr><td>Adjusted |resid|/&sigma;</td><td style="font-weight:600;color:' + adjColor + ';">' +
                s.adjusted_residual.toFixed(4) + '</td></tr>';
        }}
        if (s.partition_size != null)
            h += '<tr><td>Partition size</td><td>' + s.partition_size + '</td></tr>';
        if (s.local_r2 != null) {{
            h += '<tr><td>Local R&sup2; (partition)</td><td>' +
                s.local_r2.toFixed(4);
            if (s.partition_size != null && s.partition_size < 20 && s.local_r2 > 0.99)
                h += ' <span style="font-size:10px;color:var(--text-muted);">(specialized partition)</span>';
            h += '</td></tr>';
        }}
        h += '</table>';

        // Features & contributions
        if (s.features && s.features.length > 0) {{
            const hasC = s.features.some(f => f.contrib !== 0);
            h += '<details open><summary style="font-weight:600;font-size:13px;cursor:pointer;margin:10px 0 6px;">Features &amp; Contributions</summary>';
            h += '<table class="dt"><tr><th>Feature</th><th>Value</th>';
            if (hasC) h += '<th>Main contrib</th>';
            h += '</tr>';
            s.features.forEach(fe => {{
                const color = fe.contrib >= 0 ? tc.divPos : tc.divNeg;
                h += '<tr><td>' + fe.name + '</td><td style="font-family:monospace;">' + fe.val.toFixed(4) + '</td>';
                if (hasC) h += '<td style="font-family:monospace;color:' + color + ';">' +
                    (fe.contrib >= 0 ? '+' : '') + fe.contrib.toFixed(4) + '</td>';
                h += '</tr>';
            }});
            h += '</table></details>';
        }}

        // Pairwise interactions from local model
        if (s.pairwise && Object.keys(s.pairwise).length > 0) {{
            h += '<details><summary style="font-weight:600;font-size:13px;cursor:pointer;margin:10px 0 6px;">Pairwise Interactions (local model)</summary>';
            h += '<table class="dt"><tr><th>Pair</th><th>Coefficient</th></tr>';
            const pw = Object.entries(s.pairwise).sort((a,b) => Math.abs(b[1])-Math.abs(a[1]));
            pw.slice(0,10).forEach(([p,v]) => {{
                const color = v >= 0 ? tc.divPos : tc.divNeg;
                h += '<tr><td>'+p+'</td><td style="font-family:monospace;color:'+color+';">'+
                    (v>=0?'+':'')+v.toFixed(4)+'</td></tr>';
            }});
            h += '</table></details>';
        }}

        // Interaction contributions (from contribution frame)
        if (s.interactions && s.interactions.length > 0) {{
            h += '<details><summary style="font-weight:600;font-size:13px;cursor:pointer;margin:10px 0 6px;">Interactions (top ' + s.interactions.length + ')</summary>';
            h += '<table class="dt"><tr><th>Interaction</th><th>Contribution</th><th style="width:40%;">Bar</th></tr>';
            const iMax = Math.max(...s.interactions.map(e => Math.abs(e.val))) || 1;
            s.interactions.forEach(e => {{
                const pct = Math.abs(e.val) / iMax * 100;
                const color = e.val >= 0 ? tc.divPos : tc.divNeg;
                h += '<tr><td>' + e.name + '</td>' +
                    '<td style="font-family:monospace;color:' + color + ';">' +
                    (e.val >= 0 ? '+' : '') + e.val.toFixed(4) + '</td>' +
                    '<td><div style="background:' + color +
                    ';height:10px;width:' + pct.toFixed(1) + '%;border-radius:3px;opacity:0.7;"></div></td></tr>';
            }});
            h += '</table></details>';
        }}

        content.innerHTML = h;
        showCoopDetail(s.idx);
    }}

    // Build normalized sample object from exemplar data
    function exemplarToSample(e) {{
        const feats = e.features ? Object.entries(e.features).map(([f,v]) => ({{
            name: f, val: v, contrib: e.contributions ? (e.contributions[f]||0) : 0
        }})) : [];
        if (e.contributions) feats.sort((a,b) => Math.abs(b.contrib) - Math.abs(a.contrib));
        return {{
            idx: e.sample_idx,
            y_true: e.y_true, y_pred: e.y_pred, residual: e.residual,
            local_noise_std: e.local_noise_std != null ? e.local_noise_std : null,
            adjusted_residual: e.adjusted_residual != null ? e.adjusted_residual : null,
            partition_size: e.partition_size != null ? e.partition_size : null,
            local_r2: e.local_r2 != null ? e.local_r2 : null,
            features: feats,
            pairwise: e.pairwise || {{}},
            interactions: []
        }};
    }}

    // Build normalized sample object from lookup data
    function lookupToSample(idx) {{
        const fnames = sl.feature_names;
        const feats_raw = sl.features[idx];
        const mc = sl.contributions_main;
        const entries = fnames.map((f, i) => ({{
            name: f, val: feats_raw[i], contrib: mc && mc[f] ? mc[f][idx] : 0
        }}));
        entries.sort((a,b) => Math.abs(b.contrib) - Math.abs(a.contrib));

        const interactions = [];
        if (sl.contributions_interaction) {{
            const ic = sl.contributions_interaction;
            Object.keys(ic).forEach(k => {{
                interactions.push({{ name: k, val: ic[k][idx] }});
            }});
            interactions.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        }}

        return {{
            idx: idx,
            y_true: sl.y_true[idx], y_pred: sl.y_pred[idx], residual: sl.residuals[idx],
            local_noise_std: sl.local_noise_std ? sl.local_noise_std[idx] : null,
            adjusted_residual: sl.adjusted_residual ? sl.adjusted_residual[idx] : null,
            partition_size: sl.partition_size ? sl.partition_size[idx] : null,
            local_r2: sl.local_r2 ? sl.local_r2[idx] : null,
            features: entries,
            pairwise: {{}},
            interactions: interactions
        }};
    }}

    // ---- Show exemplar ----
    function showExemplar(e, label) {{
        const tips = irrMode === "adjusted" ? {{
            "best": "Lowest |residual|/&sigma; — predicted well relative to local noise.",
            "worst": "Highest |residual|/&sigma; — surprisingly poor prediction for a low-noise region.",
            "median": "Median |residual|/&sigma; — typical noise-adjusted performance."
        }} : {{}};
        renderSample(exemplarToSample(e), tips[label] || null);
    }}

    // ---- Show lookup ----
    function showLookup(idx) {{
        if (!sl || isNaN(idx) || idx < 0 || idx >= sl.n) {{
            content.innerHTML = '<p style="color:var(--status-bad);">Invalid index. Enter 0\u2013' + ((sl||{{}}).n-1) + '.</p>';
            return;
        }}
        renderSample(lookupToSample(idx), null);
    }}

    // ---- Build tab buttons (shared logic) ----
    function _buildTabButtons() {{
        tabs.innerHTML = "";
        if (!ex) return;
        const data = irrMode === "adjusted" && ex.adjusted ? ex.adjusted : ex.raw;
        const keys = Object.keys(data);
        keys.forEach(k => {{
            const btn = document.createElement("button");
            const suffix = irrMode === "adjusted" ? " (adj)" : "";
            // None active if we're in lookup mode
            btn.className = "exemplar-tab" + (k === currentKey ? " active" : "");
            btn.textContent = k.charAt(0).toUpperCase() + k.slice(1) + suffix;
            btn.onclick = () => {{
                currentKey = k;
                tabs.querySelectorAll(".exemplar-tab").forEach(t => t.classList.remove("active"));
                btn.classList.add("active");
                showExemplar(data[k], k);
            }};
            tabs.appendChild(btn);
        }});
    }}

    // Rebuild tabs AND show the current exemplar
    function rebuildTabs() {{
        _buildTabButtons();
        if (ex) {{
            const data = irrMode === "adjusted" && ex.adjusted ? ex.adjusted : ex.raw;
            const keys = Object.keys(data);
            if (!keys.includes(currentKey) || currentKey === "lookup") currentKey = keys[0];
            showExemplar(data[currentKey], currentKey);
        }}
    }}

    // Rebuild tab labels only (don't change content — used when in lookup mode)
    function rebuildTabsOnly() {{
        _buildTabButtons();
    }}

    // Lookup wired globally
    window.lookupSample = function() {{
        const inp = document.getElementById("lookup-idx");
        const idx = parseInt(inp.value, 10);
        // Deselect exemplar tabs
        tabs.querySelectorAll(".exemplar-tab").forEach(t => t.classList.remove("active"));
        currentKey = "lookup";
        showLookup(idx);
    }};

    // Wire up irr toggle — only rebuild exemplar tabs, don't touch lookup view
    if (hasAdjusted) {{
        setTimeout(() => {{
            const toggle = document.getElementById("irr-toggle");
            if (toggle) toggle.onchange = function() {{
                irrMode = this.checked ? "adjusted" : "raw";
                // If currently viewing a looked-up sample, just update tab labels
                // but don't switch the content away from the lookup
                if (currentKey === "lookup") {{
                    rebuildTabsOnly();
                }} else {{
                    rebuildTabs();
                }}
            }};
        }}, 0);
    }}

    rebuildTabs();
}})();

// ========== RESIDUALS ==========
(function() {{
    const pred = D.predictions;
    if (!pred) return;
    const split = pred.test || pred.train;
    if (!split) return;

    function drawResiduals() {{
        const tc = themeColors();
        const yt = split.y_true, yp = split.y_pred, res = split.residuals;

        const [ctx1, W1, H1] = getCtx("cv-scatter");
        const p1 = {{l:60, r:15, t:15, b:40}};
        const cw1 = W1-p1.l-p1.r, ch1 = H1-p1.t-p1.b;
        const lo = Math.min(safeMin(yt), safeMin(yp));
        const hi = Math.max(safeMax(yt), safeMax(yp));
        const rng = (hi-lo)||1;
        ctx1.fillStyle = tc.primaryFill;
        yt.forEach((t,i) => {{
            const x = p1.l + (t-lo)/rng*cw1, y = p1.t + ch1 - (yp[i]-lo)/rng*ch1;
            ctx1.beginPath(); ctx1.arc(x,y,2.5,0,Math.PI*2); ctx1.fill();
        }});
        ctx1.strokeStyle = tc.grid; ctx1.lineWidth = 1; ctx1.setLineDash([4,4]);
        ctx1.beginPath(); ctx1.moveTo(p1.l, p1.t+ch1); ctx1.lineTo(p1.l+cw1, p1.t); ctx1.stroke();
        ctx1.setLineDash([]);
        ctx1.fillStyle = tc.text; ctx1.font = "12px -apple-system, sans-serif"; ctx1.textAlign = "center";
        ctx1.fillText("Actual", p1.l+cw1/2, H1-8);
        ctx1.save(); ctx1.translate(14, p1.t+ch1/2);
        ctx1.rotate(-Math.PI/2); ctx1.fillText("Predicted", 0, 0); ctx1.restore();

        if (!res) return;
        const [ctx2, W2, H2] = getCtx("cv-hist");
        const p2 = {{l:55, r:15, t:15, b:40}};
        const cw2 = W2-p2.l-p2.r, ch2 = H2-p2.t-p2.b;
        const nBins = 40;
        const rMin = safeMin(res), rMax = safeMax(res);
        const bw = (rMax-rMin)/nBins || 1;
        const counts = new Array(nBins).fill(0);
        res.forEach(r => {{ const b = Math.min(nBins-1, Math.floor((r-rMin)/bw)); counts[b]++; }});
        const maxC = Math.max(...counts) || 1;
        const barW = cw2/nBins;
        ctx2.fillStyle = tc.primary;
        counts.forEach((c,i) => {{
            const bh = c/maxC*ch2;
            ctx2.fillRect(p2.l + i*barW, p2.t + ch2 - bh, barW-1, bh);
        }});
        if (rMin < 0 && rMax > 0) {{
            const zx = p2.l + (-rMin)/(rMax-rMin)*cw2;
            ctx2.strokeStyle = tc.text; ctx2.lineWidth = 1; ctx2.setLineDash([4,4]);
            ctx2.beginPath(); ctx2.moveTo(zx, p2.t); ctx2.lineTo(zx, p2.t+ch2); ctx2.stroke();
            ctx2.setLineDash([]);
        }}
        ctx2.fillStyle = tc.text; ctx2.font = "12px -apple-system, sans-serif"; ctx2.textAlign = "center";
        ctx2.fillText("Residual", p2.l+cw2/2, H2-8);
    }}

    onPanelShow("residuals", drawResiduals);
    registerRedraw(() => {{ if (document.getElementById("panel-residuals").classList.contains("active")) drawResiduals(); }});
}})();

// ========== GEOMETRY (3D Manifold) ==========
(function() {{
    const ps = D.partition_summary;
    if (!ps || !ps.scatter_3d) return;

    const fnames = ps.feature_names;
    const parts = ps.partitions;
    const sc = ps.scatter_3d;
    const d = fnames.length;

    // Populate feature selectors
    const selX = document.getElementById("sel-geo-x");
    const selY = document.getElementById("sel-geo-y");
    if (!selX || !selY) return;
    fnames.forEach((f, i) => {{
        selX.appendChild(Object.assign(document.createElement("option"), {{value:i, textContent:f}}));
        selY.appendChild(Object.assign(document.createElement("option"), {{value:i, textContent:f}}));
    }});
    const imp = D.feature_importances || {{}};
    const ranked = fnames.map((f,i) => [i, imp[f]||0]).sort((a,b) => b[1]-a[1]);
    selX.value = ranked[0] ? ranked[0][0] : 0;
    selY.value = ranked[1] ? ranked[1][0] : Math.min(1, d-1);

    const partColors = [
        0x4e79a7,0xf28e2b,0xe15759,0x76b7b2,0x59a14f,
        0xedc948,0xb07aa1,0xff9da7,0x9c755f,0xbab0ac,
        0x86bcb6,0x8cd17d,0xb6992d,0x499894,0xd37295,
        0xa0cbe8,0xffbe7d,0xd4a6c8,0xfabfd2,0xd7b5a6
    ];

    let scene, camera, renderer, controls;
    let pointsMesh, planeGroup;
    let initialized = false;

    function showError(msg) {{
        const c = document.getElementById("three-container");
        if (c) c.innerHTML = '<p style="color:#ff6b6b;padding:20px;">' + msg + '</p>';
    }}

    function initScene() {{
        if (initialized) return true;
        if (typeof THREE === 'undefined') {{
            showError('Three.js failed to load. Check network/CDN availability.');
            return false;
        }}
        const container = document.getElementById("three-container");
        if (!container) return false;

        // Use actual dimensions or fallback
        const w = container.clientWidth || container.offsetWidth || 800;
        const h = container.clientHeight || container.offsetHeight || 550;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
        camera.position.set(1.8, 1.4, 1.8);
        camera.lookAt(0.5, 0.5, 0.5);

        renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(w, h);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0.5, 0.5, 0.5);
        controls.update();

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dl = new THREE.DirectionalLight(0xffffff, 0.4);
        dl.position.set(2, 3, 1);
        scene.add(dl);

        scene.add(new THREE.AxesHelper(1.1));

        initialized = true;
        animate();
        return true;
    }}

    function animate() {{
        if (!renderer) return;
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }}

    function buildScene() {{
        if (!initialized) return;
        const fi = parseInt(selX.value);
        const fj = parseInt(selY.value);
        const zMode = document.getElementById("sel-geo-z").value;
        const showPlanes = document.getElementById("geo-show-planes").checked;
        const showPoints = document.getElementById("geo-show-points").checked;

        // Remove old objects
        if (pointsMesh) {{ scene.remove(pointsMesh); pointsMesh.geometry.dispose(); pointsMesh = null; }}
        if (planeGroup) {{
            planeGroup.children.forEach(c => {{ if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); }});
            scene.remove(planeGroup); planeGroup = null;
        }}
        scene.children.filter(c => c.isSprite).forEach(s => scene.remove(s));

        const xs = sc.x.map(row => row[fi]);
        const ys = sc.x.map(row => row[fj]);
        const zs = zMode === "true" ? sc.y_true : sc.y_pred;
        const pids = sc.pid;
        const n = xs.length;

        const xMin = safeMin(xs), xMax = safeMax(xs);
        const yMin = safeMin(ys), yMax = safeMax(ys);
        const zMin = safeMin(zs), zMax = safeMax(zs);
        const xR = (xMax - xMin) || 1, yR = (yMax - yMin) || 1, zR = (zMax - zMin) || 1;
        const normX = x => (x - xMin) / xR;
        const normY = y => (y - yMin) / yR;
        const normZ = z => (z - zMin) / zR;

        if (showPoints) {{
            const geom = new THREE.BufferGeometry();
            const positions = new Float32Array(n * 3);
            const colors = new Float32Array(n * 3);
            for (let i = 0; i < n; i++) {{
                positions[i*3] = normX(xs[i]);
                positions[i*3+1] = normZ(zs[i]);
                positions[i*3+2] = normY(ys[i]);
                const c = new THREE.Color(partColors[pids[i] % partColors.length]);
                colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
            }}
            geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
            pointsMesh = new THREE.Points(geom, new THREE.PointsMaterial({{
                size: 3, vertexColors: true, sizeAttenuation: false, transparent: true, opacity: 0.6
            }}));
            scene.add(pointsMesh);
        }}

        if (showPlanes) {{
            planeGroup = new THREE.Group();
            parts.forEach(p => {{
                const fr = p.feature_ranges;
                if (!fr || !fr[fnames[fi]] || !fr[fnames[fj]]) return;
                const [xa, xb] = fr[fnames[fi]];
                const [ya, yb] = fr[fnames[fj]];
                const si = p.feature_slopes[fnames[fi]] || 0;
                const sj = p.feature_slopes[fnames[fj]] || 0;
                const mi = p.feature_means[fnames[fi]] || 0;
                const mj = p.feature_means[fnames[fj]] || 0;
                const z0 = p.mean_pred;
                const plane_z = (x, y) => z0 + si * (x - mi) + sj * (y - mj);

                const corners = [[xa,ya],[xb,ya],[xb,yb],[xa,yb]];
                const geom = new THREE.BufferGeometry();
                const verts = new Float32Array(6 * 3);
                const cNorm = corners.map(([cx,cy]) => [normX(cx), normZ(plane_z(cx,cy)), normY(cy)]);

                [0,1,2].forEach((vi,idx) => {{
                    verts[idx*3]=cNorm[vi][0]; verts[idx*3+1]=cNorm[vi][1]; verts[idx*3+2]=cNorm[vi][2];
                }});
                [0,2,3].forEach((vi,idx) => {{
                    const o = 9;
                    verts[o+idx*3]=cNorm[vi][0]; verts[o+idx*3+1]=cNorm[vi][1]; verts[o+idx*3+2]=cNorm[vi][2];
                }});
                geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
                geom.computeVertexNormals();

                const col = partColors[p.pid % partColors.length];
                planeGroup.add(new THREE.Mesh(geom, new THREE.MeshPhongMaterial({{
                    color: col, transparent: true, opacity: 0.4,
                    side: THREE.DoubleSide, shininess: 30
                }})));

                const edgeGeom = new THREE.BufferGeometry();
                const edgeVerts = new Float32Array(5 * 3);
                [...cNorm, cNorm[0]].forEach(([x,y,z], i) => {{
                    edgeVerts[i*3]=x; edgeVerts[i*3+1]=y; edgeVerts[i*3+2]=z;
                }});
                edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgeVerts, 3));
                planeGroup.add(new THREE.Line(edgeGeom, new THREE.LineBasicMaterial({{
                    color: col, transparent: true, opacity: 0.7
                }})));
            }});
            scene.add(planeGroup);
        }}

        const makeLabel = (text, pos) => {{
            const cv = document.createElement("canvas");
            cv.width = 256; cv.height = 64;
            const cx = cv.getContext("2d");
            cx.fillStyle = "#ffffff";
            cx.font = "bold 28px -apple-system, sans-serif";
            cx.textAlign = "center";
            cx.fillText(text, 128, 40);
            const tex = new THREE.CanvasTexture(cv);
            const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
            const spr = new THREE.Sprite(mat);
            spr.position.set(...pos);
            spr.scale.set(0.4, 0.1, 1);
            return spr;
        }};
        scene.add(makeLabel(fnames[fi] + " \u2192", [0.5, -0.08, -0.08]));
        scene.add(makeLabel(fnames[fj] + " \u2192", [-0.08, -0.08, 0.5]));
        scene.add(makeLabel(zMode === "true" ? "y (true)" : "y (pred)", [-0.12, 0.5, -0.08]));
    }}

    // Wire controls
    [selX, selY, document.getElementById("sel-geo-z"),
     document.getElementById("geo-show-planes"),
     document.getElementById("geo-show-points")].forEach(el => {{
        if (el) el.onchange = () => {{ if (initialized) buildScene(); }};
    }});

    function tryInit() {{
        try {{
            if (initScene()) buildScene();
        }} catch(e) {{
            showError('3D init error: ' + e.message);
        }}
    }}

    // Use persistent callback — retries init on every tab click until successful
    onPanelShowAlways("geometry", () => {{
        if (!initialized) requestAnimationFrame(tryInit);
    }});
}})();

// ========== GLOBAL MANIFOLD — Geometric Fingerprint ==========
(function() {{
    const ps = D.partition_summary;
    if (!ps || !ps.scatter_3d || !ps.scatter_3d.spec_x) return;

    const sc = ps.scatter_3d;
    const parts = ps.partitions;
    const nParts = ps.n_partitions || parts.length;
    const nSnaps = sc.n_snapshots || 0;
    const dFeat = sc.n_features || D.feature_names.length;

    // Subtitle and embedding info
    const sub = document.getElementById("gm-subtitle");
    if (sub) sub.textContent = nParts + " partitions \u00b7 "
        + sc.spec_x.length + " samples \u00b7 "
        + dFeat + "d z-space";
    const info = document.getElementById("global-pca-info");
    if (info) info.innerHTML = "Hyperboloid projection<br>"
        + "Q = \u00bd((" + (dFeat-1) + ")\u00b7u\u00b9\u00b2 \u2212 r\u00b2)";

    // Palette: softer, more luminous colors for the cinematic look
    const palette = [
        0x6ba3d6,0xf4a95b,0xe87c7e,0x8fd4cc,0x7ec47a,
        0xf0d76a,0xc99bbe,0xffb5c0,0xb8967a,0xcec5be,
        0x9ed1cb,0xa6e29a,0xc9ac48,0x6bb8af,0xe08fad,
        0xb5d8f0,0xffd1a0,0xe0c0db,0xfdd2e2,0xdfc8b8
    ];

    let scene2, camera2, renderer2, controls2;
    let pointsMesh2, glowMesh2;
    let init2 = false;
    let autoRotate = true;

    function showError2(msg) {{
        const c = document.getElementById("three-global-container");
        if (c) c.innerHTML = '<p style="color:#ff6b6b;padding:20px;">' + msg + '</p>';
    }}

    // Soft radial glow sprite texture
    function makeGlowTexture() {{
        const size = 64;
        const cv = document.createElement("canvas");
        cv.width = size; cv.height = size;
        const ctx = cv.getContext("2d");
        const half = size / 2;
        const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
        grad.addColorStop(0, "rgba(255,255,255,1.0)");
        grad.addColorStop(0.12, "rgba(255,255,255,0.6)");
        grad.addColorStop(0.4, "rgba(255,255,255,0.1)");
        grad.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, size, size);
        return new THREE.CanvasTexture(cv);
    }}

    function initGlobal() {{
        if (init2) return true;
        if (typeof THREE === 'undefined') {{
            showError2('Three.js failed to load.');
            return false;
        }}
        const container = document.getElementById("three-global-container");
        if (!container) return false;

        const w = container.clientWidth || container.offsetWidth || 800;
        const h = container.clientHeight || container.offsetHeight || 640;

        scene2 = new THREE.Scene();

        // Deep space gradient background
        const bgCv = document.createElement("canvas");
        bgCv.width = 2; bgCv.height = 512;
        const bgCtx = bgCv.getContext("2d");
        const bgGrad = bgCtx.createLinearGradient(0, 0, 0, 512);
        bgGrad.addColorStop(0, "#06060f");
        bgGrad.addColorStop(0.4, "#0a0d1e");
        bgGrad.addColorStop(1, "#080b18");
        bgCtx.fillStyle = bgGrad;
        bgCtx.fillRect(0, 0, 2, 512);
        scene2.background = new THREE.CanvasTexture(bgCv);

        // Subtle depth fog
        scene2.fog = new THREE.FogExp2(0x06060f, 0.3);

        camera2 = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
        camera2.position.set(2.0, 1.5, 2.0);
        camera2.lookAt(0.5, 0.5, 0.5);

        renderer2 = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
        renderer2.setSize(w, h);
        renderer2.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer2.toneMapping = THREE.ACESFilmicToneMapping;
        renderer2.toneMappingExposure = 1.1;
        renderer2.outputEncoding = THREE.sRGBEncoding;
        container.appendChild(renderer2.domElement);

        controls2 = new THREE.OrbitControls(camera2, renderer2.domElement);
        controls2.enableDamping = true;
        controls2.dampingFactor = 0.04;
        controls2.target.set(0.5, 0.5, 0.5);
        controls2.autoRotate = true;
        controls2.autoRotateSpeed = 0.6;
        controls2.enableZoom = true;
        controls2.minDistance = 0.8;
        controls2.maxDistance = 6.0;
        controls2.update();

        // Cinematic 3-point lighting
        const key = new THREE.DirectionalLight(0xfff0e0, 0.5);
        key.position.set(3, 4, 2);
        scene2.add(key);
        const fill = new THREE.DirectionalLight(0xd0e0ff, 0.2);
        fill.position.set(-3, 1, -1);
        scene2.add(fill);
        const rim = new THREE.DirectionalLight(0xa0c0ff, 0.35);
        rim.position.set(-1, 2, -3);
        scene2.add(rim);
        scene2.add(new THREE.AmbientLight(0x303050, 0.3));

        init2 = true;
        (function anim() {{
            if (!renderer2) return;
            requestAnimationFrame(anim);
            controls2.update();
            renderer2.render(scene2, camera2);
        }})();
        return true;
    }}

    let surfaceMesh2 = null;

    function clearScene() {{
        if (pointsMesh2) {{ scene2.remove(pointsMesh2); pointsMesh2.geometry.dispose(); pointsMesh2.material.dispose(); pointsMesh2 = null; }}
        if (glowMesh2) {{ scene2.remove(glowMesh2); glowMesh2.geometry.dispose(); glowMesh2.material.dispose(); glowMesh2 = null; }}
        if (surfaceMesh2) {{
            surfaceMesh2.traverse(c => {{ if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); }});
            scene2.remove(surfaceMesh2); surfaceMesh2 = null;
        }}
    }}

    function percentile(arr, p) {{
        const s = arr.slice().sort((a,b) => a - b);
        const k = (s.length - 1) * p;
        const f = Math.floor(k), c = Math.ceil(k);
        return f === c ? s[f] : s[f] + (k - f) * (s[c] - s[f]);
    }}

    function buildGlobal() {{
        if (!init2) return;
        clearScene();

        const showSurface = document.getElementById("gm-show-surface").checked;
        const hideOutliers = document.getElementById("gm-hide-outliers").checked;

        // Hyperboloid coordinates: X = transverse PC1, Y = u₁ (cooperative), Z = transverse PC2
        const xs = sc.spec_x;   // transverse PC1
        const ys = sc.spec_y;   // u₁ (cooperative axis)
        const zs = sc.spec_z;   // transverse PC2
        const pids = sc.pid;
        const n = xs.length;

        // Compute normalization bounds — optionally clamp to 2nd/98th percentile
        let xMin, xMax, yMin, yMax, zMin, zMax;
        if (hideOutliers) {{
            xMin = percentile(xs, 0.02); xMax = percentile(xs, 0.98);
            yMin = percentile(ys, 0.02); yMax = percentile(ys, 0.98);
            zMin = percentile(zs, 0.02); zMax = percentile(zs, 0.98);
        }} else {{
            xMin = safeMin(xs); xMax = safeMax(xs);
            yMin = safeMin(ys); yMax = safeMax(ys);
            zMin = safeMin(zs); zMax = safeMax(zs);
        }}
        const xR = (xMax - xMin) || 1, yR = (yMax - yMin) || 1, zR = (zMax - zMin) || 1;
        const normX = v => (v - xMin) / xR;
        const normY = v => (v - yMin) / yR;
        const normZ = v => (v - zMin) / zR;

        const glowTex = makeGlowTexture();

        // Build position + color arrays (hide outliers by sending them to NaN)
        const geom = new THREE.BufferGeometry();
        const pos = new Float32Array(n * 3);
        const col = new Float32Array(n * 3);
        let nVisible = 0;
        for (let i = 0; i < n; i++) {{
            const isOutlier = hideOutliers && (
                xs[i] < xMin || xs[i] > xMax ||
                ys[i] < yMin || ys[i] > yMax ||
                zs[i] < zMin || zs[i] > zMax);
            if (isOutlier) {{
                pos[i*3] = NaN; pos[i*3+1] = NaN; pos[i*3+2] = NaN;
            }} else {{
                pos[i*3]   = normX(xs[i]);
                pos[i*3+1] = normY(ys[i]);
                pos[i*3+2] = normZ(zs[i]);
                nVisible++;
            }}
            const c = new THREE.Color(palette[pids[i] % palette.length]);
            col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
        }}
        geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geom.setAttribute("color", new THREE.BufferAttribute(col, 3));

        // Core points
        pointsMesh2 = new THREE.Points(geom, new THREE.PointsMaterial({{
            size: 3, vertexColors: true, sizeAttenuation: false,
            transparent: true, opacity: 0.9, depthWrite: false
        }}));
        scene2.add(pointsMesh2);

        // Glow halo
        const glowGeom = geom.clone();
        glowMesh2 = new THREE.Points(glowGeom, new THREE.PointsMaterial({{
            size: 10, vertexColors: true, sizeAttenuation: false,
            transparent: true, opacity: 0.1, depthWrite: false,
            blending: THREE.AdditiveBlending, map: glowTex
        }}));
        scene2.add(glowMesh2);

        // --- Hyperboloid wireframe surface ---
        // Q = ½((d-1)·u₁² - r²) where r² = x² + z² in transverse space.
        // For a chosen Q level, the surface is: r = √((d-1)·u₁² - 2Q)
        // In normalized display coords, we parametrize u₁ and θ.
        if (showSurface) {{
            surfaceMesh2 = new THREE.Group();
            const dm1 = Math.max(dFeat - 1, 1);

            // Level sets: Q_min (bounding envelope), Q_median, Q_max (innermost core)
            // Q_min surface contains ALL points; Q_max is inside all points.
            const coops = sc.coop_score ? [...sc.coop_score].sort((a,b) => a-b) : [];
            const levels = [];
            if (coops.length > 0) {{
                const qMin = coops[0];
                const q50  = coops[Math.floor(coops.length * 0.50)];
                const qMax = coops[coops.length - 1];
                levels.push(qMin, q50, qMax);
            }}

            const nTheta = 32;   // circumferential resolution
            const nU = 24;       // u₁ resolution

            levels.forEach((Q, li) => {{
                // For this Q level, sweep u₁ and compute r = √((d-1)·u₁² - 2Q)
                // Only valid where (d-1)·u₁² - 2Q >= 0
                const u1_min_sq = 2 * Q / dm1;
                const u1_abs_min = u1_min_sq > 0 ? Math.sqrt(u1_min_sq) : 0;

                // Sweep u₁ from -max to +max (in raw z-space units)
                const u1_raw_min = safeMin(sc.spec_y);
                const u1_raw_max = safeMax(sc.spec_y);

                for (let th = 0; th < nTheta; th++) {{
                    const theta = (th / nTheta) * 2 * Math.PI;
                    const cosT = Math.cos(theta);
                    const sinT = Math.sin(theta);

                    const lineGeom = new THREE.BufferGeometry();
                    const pts = [];

                    for (let ui = 0; ui <= nU; ui++) {{
                        const u1_raw = u1_raw_min + (u1_raw_max - u1_raw_min) * ui / nU;
                        const r_sq = dm1 * u1_raw * u1_raw - 2 * Q;
                        if (r_sq < 0) continue;
                        const r_raw = Math.sqrt(r_sq);

                        // Convert raw transverse radius to display x/z coordinates
                        // r_raw is in z_perp units; we need to map to the spec_x/spec_z range
                        // The transverse PCA axes are unit-length, so r² = spec_x² + spec_z²
                        // for points on this level set.
                        const x_raw = r_raw * cosT;
                        const z_raw = r_raw * sinT;

                        pts.push(normX(x_raw), normY(u1_raw), normZ(z_raw));
                    }}

                    if (pts.length >= 6) {{
                        const arr = new Float32Array(pts);
                        lineGeom.setAttribute("position", new THREE.BufferAttribute(arr, 3));
                        // li: 0=Qmin(outer,faint) 1=Q50(mid,bright) 2=Qmax(inner,faint)
                        const opacity = li === 1 ? 0.18 : 0.07;
                        const col = li === 2 ? 0xcc8888 : 0x8888cc;
                        surfaceMesh2.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({{
                            color: col, transparent: true, opacity: opacity,
                            depthWrite: false
                        }})));
                    }}
                }}

                // Horizontal rings at several u₁ values
                for (let ui = 0; ui <= nU; ui += 4) {{
                    const u1_raw = u1_raw_min + (u1_raw_max - u1_raw_min) * ui / nU;
                    const r_sq = dm1 * u1_raw * u1_raw - 2 * Q;
                    if (r_sq < 0) continue;
                    const r_raw = Math.sqrt(r_sq);

                    const ringGeom = new THREE.BufferGeometry();
                    const rPts = [];
                    for (let th = 0; th <= nTheta; th++) {{
                        const theta = (th / nTheta) * 2 * Math.PI;
                        rPts.push(
                            normX(r_raw * Math.cos(theta)),
                            normY(u1_raw),
                            normZ(r_raw * Math.sin(theta))
                        );
                    }}
                    ringGeom.setAttribute("position",
                        new THREE.BufferAttribute(new Float32Array(rPts), 3));
                    const opacity = li === 1 ? 0.14 : 0.05;
                    const col = li === 2 ? 0xcc8888 : 0x8888cc;
                    surfaceMesh2.add(new THREE.Line(ringGeom, new THREE.LineBasicMaterial({{
                        color: col, transparent: true, opacity: opacity,
                        depthWrite: false
                    }})));
                }}
            }});
            scene2.add(surfaceMesh2);
        }}
    }}

    // Wire controls
    const selSurf = document.getElementById("gm-show-surface");
    if (selSurf) selSurf.onchange = () => {{ if (init2) buildGlobal(); }};
    const selOutliers = document.getElementById("gm-hide-outliers");
    if (selOutliers) selOutliers.onchange = () => {{ if (init2) buildGlobal(); }};

    // Pause / resume rotation
    const pauseBtn = document.getElementById("gm-pause");
    if (pauseBtn) pauseBtn.onclick = () => {{
        autoRotate = !autoRotate;
        if (controls2) controls2.autoRotate = autoRotate;
        pauseBtn.textContent = autoRotate ? "Pause" : "Rotate";
    }};

    function tryInitGlobal() {{
        try {{
            if (initGlobal()) buildGlobal();
        }} catch(e) {{
            showError2('3D error: ' + e.message);
        }}
    }}

    onPanelShowAlways("geometry", () => {{
        if (!init2) requestAnimationFrame(tryInitGlobal);
    }});
}})();

// ========== MAP ==========
function initMap() {{
    if (!D.geo_data) return;
    window._mapInit = true;
    const g = D.geo_data;
    const map = L.map("map").setView(
        [g.lats.reduce((a,b)=>a+b,0)/g.lats.length,
         g.lons.reduce((a,b)=>a+b,0)/g.lons.length], 6);
    window._map = map;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const tileUrl = isDark
        ? 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png'
        : 'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png';
    window._tileLayer = L.tileLayer(tileUrl, {{
        attribution: '&copy; OpenStreetMap &copy; CartoDB',
        maxZoom: 18,
    }});
    window._tileLayer.addTo(map);

    // ---------- Build feature selector dropdown ----------
    const sel = document.getElementById("sel-map-feature");
    const fi = g.feature_influence || {{}};
    const featureNames = Object.keys(fi);
    // Combined geo option first
    const geoOpt = document.createElement("option");
    geoOpt.value = "geo";
    geoOpt.textContent = g.lat_feature + " \u00d7 " + g.lon_feature + " (combined)";
    sel.appendChild(geoOpt);
    // Skip individual lat/lon — the combined option covers them
    const latF = g.lat_feature, lonF = g.lon_feature;
    featureNames.forEach(f => {{
        if (f === latF || f === lonF) return;
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        sel.appendChild(opt);
    }});

    // Current state
    let currentFeature = "geo";
    let currentMode = "raw";

    // Hide raw/net toggle when "geo" is selected (it has no modes)
    const modeWrap = document.getElementById("mode-toggle-wrap");

    // ---------- Sample points ----------
    let markers = [];
    function getPointValues() {{
        if (currentFeature === "geo" && g.geo_influence) return g.geo_influence;
        if (fi[currentFeature]) return fi[currentFeature][currentMode];
        return g.geo_influence || g.cooperation_scores;
    }}

    function renderPoints() {{
        markers.forEach(m => map.removeLayer(m));
        markers = [];
        const vals = getPointValues();
        const mn = safeMin(vals), mx = safeMax(vals);
        const rng = (mx - mn) || 1;

        // Pre-sort features by mean net influence for popups (top 5)
        const sortedFeats = featureNames.slice().sort((a, b) => {{
            const ma = fi[a] ? fi[a].net.reduce((s,v) => s+v, 0) : 0;
            const mb = fi[b] ? fi[b].net.reduce((s,v) => s+v, 0) : 0;
            return mb - ma;
        }}).slice(0, 5);

        const tc = themeColors();

        // Build popup HTML with mini bar chart for a given point index and mode
        function buildPopup(i, popMode) {{
            const mode = popMode || "net";
            const curVal = vals[i];
            const label = currentFeature === "geo"
                ? (g.lat_feature + " \u00d7 " + g.lon_feature)
                : (currentFeature + " (" + currentMode + ")");

            // Get top-5 features for this point in the selected mode
            const fEntries = [];
            featureNames.forEach(f => {{
                if (f === latF || f === lonF) return;
                if (fi[f]) fEntries.push({{name: f, val: fi[f][mode][i]}});
            }});
            // Add combined geo
            if (g.geo_influence) {{
                fEntries.push({{name: latF + " \u00d7 " + lonF, val: g.geo_influence[i]}});
            }}
            fEntries.sort((a,b) => b.val - a.val);
            const top5 = fEntries.slice(0, 5);
            const barMax = top5.length > 0 ? top5[0].val : 1;

            let h = '<div style="min-width:200px;font-family:-apple-system,sans-serif;font-size:12px;">';
            h += '<b>Sample ' + i + '</b> &mdash; pred: ' + (g.predictions[i]||0).toFixed(3) + '<br>';
            h += '<span style="font-size:11px;color:#888;">' + label + ': ' + (curVal*100).toFixed(1) + '%</span>';

            // Raw / Net toggle buttons
            h += '<div style="margin:6px 0 4px 0;display:flex;gap:2px;">';
            h += '<button onclick="window._popMode=&quot;raw&quot;; window._popRedraw(' + i + ')" '
                + 'style="padding:2px 8px;font-size:10px;border:1px solid #ccc;border-radius:3px 0 0 3px;'
                + 'cursor:pointer;background:' + (mode==="raw"?"var(--accent)":"var(--bg-card)")
                + ';color:' + (mode==="raw"?"var(--text-on-accent)":"var(--text)") + '">Raw</button>';
            h += '<button onclick="window._popMode=&quot;net&quot;; window._popRedraw(' + i + ')" '
                + 'style="padding:2px 8px;font-size:10px;border:1px solid #ccc;border-radius:0 3px 3px 0;'
                + 'cursor:pointer;background:' + (mode==="net"?"var(--accent)":"var(--bg-card)")
                + ';color:' + (mode==="net"?"var(--text-on-accent)":"var(--text)") + '">Net</button>';
            h += '</div>';

            // Mini horizontal bar chart
            h += '<div style="margin-top:4px;">';
            top5.forEach(e => {{
                const pct = barMax > 0 ? (e.val / barMax * 100) : 0;
                h += '<div style="display:flex;align-items:center;margin:2px 0;">';
                h += '<span style="width:80px;text-align:right;margin-right:6px;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + e.name + '</span>';
                h += '<div style="flex:1;background:var(--bg-body);border-radius:2px;height:12px;position:relative;">';
                h += '<div style="width:' + pct.toFixed(0) + '%;height:100%;background:' + mapSeqRgb(e.val) + ';border-radius:2px;"></div>';
                h += '</div>';
                h += '<span style="width:40px;text-align:right;font-size:10px;margin-left:4px;">' + (e.val*100).toFixed(1) + '%</span>';
                h += '</div>';
            }});
            h += '</div></div>';
            return h;
        }}

        // Global popup redraw: update an open popup's content in-place
        window._popMode = "net";
        window._popRedraw = function(idx) {{
            const m = markers[idx];
            if (m && m.getPopup() && m.getPopup().isOpen()) {{
                m.getPopup().setContent(buildPopup(idx, window._popMode));
            }}
        }};

        g.lats.forEach((lat, i) => {{
            const t = (vals[i] - mn) / rng;
            const color = mapSeqRgb(t);

            const m = L.circleMarker([lat, g.lons[i]], {{
                radius: 4, fillColor: color, color: '#333',
                weight: 0.3, fillOpacity: 0.8,
            }});

            // Hover → tooltip preview (lightweight)
            m.bindTooltip('<b>#' + i + '</b> ' + (vals[i]*100).toFixed(1) + '%', {{
                direction: 'top', offset: [0, -6], opacity: 0.9
            }});

            // Click → detailed popup with bar chart
            m.bindPopup(() => buildPopup(i, window._popMode), {{
                maxWidth: 280, closeOnClick: false, autoClose: false
            }});
            markers.push(m);
            m.addTo(map);
        }});
    }}

    // ---------- Update everything ----------
    function updateMap() {{
        modeWrap.style.display = currentFeature === "geo" ? "none" : "flex";
        renderPoints();
    }}

    // Wire controls
    sel.onchange = function() {{
        currentFeature = this.value;
        updateMap();
    }};
    window.setInfluenceMode = function(mode) {{
        currentMode = mode;
        document.getElementById("btn-raw").classList.toggle("mode-btn-active", mode === "raw");
        document.getElementById("btn-net").classList.toggle("mode-btn-active", mode === "net");
        updateMap();
    }};

    // Legend (uses mapSeqRgb for consistent theme-independent colormap)
    const legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function() {{
        const div = L.DomUtil.create('div', 'legend');
        let h = '<b>Feature Influence</b><br>';
        h += '<span style="font-size:10px">0%</span> ';
        [0, 0.25, 0.5, 0.75, 1.0].forEach(t => {{
            h += '<span style="color:'+mapSeqRgb(t)+'">&block;</span>';
        }});
        h += ' <span style="font-size:10px">100%</span>';
        div.innerHTML = h;
        return div;
    }};
    legend.addTo(map);

    // Initial render
    updateMap();
    registerRedraw(() => {{
        if (window._mapInit) {{
            renderPoints();
            map.removeControl(legend);
            legend.addTo(map);
        }}
    }});
}}

</script>
</body>
</html>"""

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_html, encoding="utf-8")

    return report_html
