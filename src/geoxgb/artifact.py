"""
ModelArtifact — serializable snapshot of a fitted GeoXGB model's behaviour.

A ModelArtifact captures everything needed to understand a model without
needing the model object itself: hyperparameters, feature importances,
cooperation structure, contribution curves, noise diagnostics, and
predictions.  It is the data layer that analytics tools consume.

Usage
-----
    from geoxgb import GeoXGBRegressor
    from geoxgb.artifact import build_artifact

    model = GeoXGBRegressor().fit(X_train, y_train)
    artifact = build_artifact(model, X_train, y_train,
                              X_test=X_test, y_test=y_test,
                              feature_names=feature_names)

    # Save / load
    artifact.save("model_artifact.json")
    loaded = ModelArtifact.load("model_artifact.json")

    # Consume with analytics tools
    from geoxgb.analytics import plot_importances, to_interactive_html_report
    plot_importances(artifact)
    to_interactive_html_report(artifact, "report.html")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helper: make everything JSON-serializable
# ---------------------------------------------------------------------------

def _jsonify(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# ModelArtifact
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    """
    Immutable snapshot of a fitted GeoXGB model.

    All fields are plain Python types (dicts, lists, floats) — no numpy
    arrays, no model objects.  Serializable to JSON or pickle.

    Attributes
    ----------
    model_type : str
        'GeoXGBRegressor', 'GeoXGBClassifier', etc.
    task : str
        'regression', 'binary', or 'multiclass'
    version : str
        geoxgb version at build time
    build_timestamp : str
        ISO 8601 timestamp

    hyperparameters : dict
        All model hyperparameters
    fit_stats : dict
        n_train, n_features, n_rounds_actual, convergence_round, fit_time_s

    feature_names : list[str]
        Feature names (auto-generated if not provided)
    feature_importances : dict[str, float]
        Normalised boosting importances (sum = 1)

    noise : dict
        Noise assessment: initial_modulation, final_modulation, assessment

    cooperation : dict or None
        Global cooperation matrix + top pairs.
        Keys: matrix (d x d list), top_pairs (list of {feat_a, feat_b, value})
    cooperation_tensor : dict or None
        Global 3-way tensor + top triples.

    contributions : dict or None
        Aggregated contribution statistics per feature/interaction.
        Keys: main (dict of {feat: {mean, std, min, max}}),
              interaction (dict of {pair: {mean, std, min, max}}),
              mean_local_r2

    predictions : dict or None
        train: {y_true, y_pred, residuals} and/or
        test:  {y_true, y_pred, residuals, metrics}

    custom : dict
        User-defined metadata (passed through unchanged)
    """

    # Identity
    model_type: str = ""
    task: str = ""
    version: str = ""
    build_timestamp: str = ""

    # Configuration
    hyperparameters: dict = field(default_factory=dict)
    fit_stats: dict = field(default_factory=dict)

    # Features
    feature_names: list = field(default_factory=list)
    feature_importances: dict = field(default_factory=dict)

    # Noise
    noise: dict = field(default_factory=dict)

    # Cooperation
    cooperation: dict | None = None
    cooperation_tensor: dict | None = None

    # Contributions (aggregated statistics, not raw per-sample)
    contributions: dict | None = None

    # EBM-style curves: pre-binned main effect & interaction surfaces
    # main_curves: {feat: {x: [...], y_mean: [...], y_lo: [...], y_hi: [...]}}
    # interaction_surfaces: {pair: {x: [...], y: [...], z: [[...]]}}
    main_curves: dict | None = None
    interaction_surfaces: dict | None = None

    # Sample exemplars: best, worst, median prediction local info
    # Contains both raw and noise-adjusted rankings
    exemplars: dict | None = None

    # Per-sample lookup table (compact parallel arrays for all contribution samples)
    # Keys: n, feature_names, features (n×d), y_true, y_pred, residuals,
    #        local_noise_std, contributions_main {feat: [...]},
    #        contributions_interaction {pair: [...]}
    sample_lookup: dict | None = None

    # Irreducible error estimate (k-NN local variance)
    # Keys: global_std, per_sample_std [...], method, k
    irreducible_error: dict | None = None

    # Geographic data: per-sample cooperation when lat/lon detected
    geo_data: dict | None = None

    # Partition summary: per-partition stats for the Partitions view
    partition_summary: dict | None = None

    # Predictions
    predictions: dict | None = None

    # User metadata
    custom: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain dict (JSON-ready)."""
        return _jsonify(asdict(self))

    def save(self, path: str | Path, format: str = "auto"):
        """
        Save artifact to disk.

        Parameters
        ----------
        path : str or Path
        format : 'json' | 'pickle' | 'auto'
            'auto' infers from extension (.json or .pkl/.pickle).
        """
        path = Path(path)
        if format == "auto":
            format = "pickle" if path.suffix in (".pkl", ".pickle") else "json"

        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        """Load artifact from JSON or pickle."""
        path = Path(path)
        if path.suffix in (".pkl", ".pickle"):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return cls(**d)

    def __repr__(self):
        n_feat = len(self.feature_names)
        has_contrib = self.contributions is not None
        has_coop = self.cooperation is not None
        return (
            f"ModelArtifact({self.model_type}, {self.task}, "
            f"{n_feat} features, "
            f"contributions={'yes' if has_contrib else 'no'}, "
            f"cooperation={'yes' if has_coop else 'no'})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _aggregate_contributions(cf) -> dict:
    """Summarise a ContributionFrame into JSON-friendly aggregate stats."""
    main = {}
    for feat, vals in cf.main.items():
        arr = np.asarray(vals)
        main[feat] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    interaction = {}
    for pair, vals in cf.interaction.items():
        arr = np.asarray(vals)
        key = f"{pair[0]} x {pair[1]}"
        interaction[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    return {
        "main": main,
        "interaction": interaction,
        "mean_local_r2": float(np.mean(cf.local_r2)),
        "median_local_r2": float(np.median(cf.local_r2)),
    }


def _top_pairs(matrix, feature_names, n=10):
    """Extract top-n cooperation pairs by absolute value."""
    d = len(feature_names)
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append({
                "feat_a": feature_names[i],
                "feat_b": feature_names[j],
                "value": float(matrix[i, j]),
            })
    pairs.sort(key=lambda p: abs(p["value"]), reverse=True)
    return pairs[:n]


def _top_triples(tensor, feature_names, n=10):
    """Extract top-n cooperation triples by absolute value."""
    d = len(feature_names)
    triples = []
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                triples.append({
                    "feat_a": feature_names[i],
                    "feat_b": feature_names[j],
                    "feat_c": feature_names[k],
                    "value": float(tensor[i, j, k]),
                })
    triples.sort(key=lambda t: abs(t["value"]), reverse=True)
    return triples[:n]


def _bin_curve(x_vals, y_vals, n_bins=30):
    """Bin y_vals by x_vals and return centers, means, CIs, and median/IQR."""
    edges = np.linspace(np.min(x_vals), np.max(x_vals), n_bins + 1)
    centers, means, lo, hi, lo90, hi90 = [], [], [], [], [], []
    medians, q25s, q75s, q10s, q90s = [], [], [], [], []
    for b in range(n_bins):
        mask = (x_vals >= edges[b]) & (x_vals < edges[b + 1])
        if b == n_bins - 1:
            mask |= (x_vals == edges[b + 1])
        if mask.sum() < 2:
            continue
        yb = y_vals[mask]
        m = float(np.mean(yb))
        se = float(np.std(yb) / np.sqrt(len(yb)))
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        means.append(m)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
        lo90.append(m - 1.645 * se)
        hi90.append(m + 1.645 * se)
        # Robust: median + quantiles (shows heterogeneity range, not uncertainty)
        medians.append(float(np.median(yb)))
        q25s.append(float(np.percentile(yb, 25)))
        q75s.append(float(np.percentile(yb, 75)))
        q10s.append(float(np.percentile(yb, 10)))
        q90s.append(float(np.percentile(yb, 90)))
    if not centers:
        return None
    return {
        "x": centers, "y_mean": means,
        "y_lo": lo, "y_hi": hi,
        "y_lo90": lo90, "y_hi90": hi90,
        "y_median": medians, "y_q25": q25s, "y_q75": q75s,
        "y_q10": q10s, "y_q90": q90s,
    }


def _build_main_curves(cf, X, feature_names, model=None,
                       n_bins=30, max_scatter=800):
    """Build per-partition scatter data and regression lines for each feature.

    Each feature gets:
      - scatter_x, scatter_main, scatter_net: raw per-sample points
      - scatter_pid: partition ID per scatter point (for color-coding)
      - partitions: list of {pid, n, slope, intercept, x_min, x_max} for main
      - partitions_net: same for net contributions
    """
    curves = {}
    n = X.shape[0]

    # Deterministic subsample for scatter points
    if n > max_scatter:
        step = n / max_scatter
        scatter_idx = np.array([int(i * step) for i in range(max_scatter)])
    else:
        scatter_idx = np.arange(n)

    # Get partition IDs if model available
    leaf_ids = None
    if model is not None:
        try:
            cpp, _ = model._get_geometry()
            leaf_ids = np.asarray(cpp.apply(X))
        except Exception:
            pass

    # Map leaf IDs to compact sequential IDs (0, 1, 2, ...)
    pid_map = {}
    compact_ids = None
    if leaf_ids is not None:
        unique_leaves = np.unique(leaf_ids)
        pid_map = {int(lf): i for i, lf in enumerate(unique_leaves)}
        compact_ids = np.array([pid_map[int(lf)] for lf in leaf_ids])

    for feat in feature_names:
        if feat not in cf.main:
            continue
        fi = feature_names.index(feat)
        x_vals = X[:, fi]

        y_main = np.asarray(cf.main[feat])

        # Net effect: main + Shapley share of interactions
        y_net = y_main.copy()
        for (fa, fb), ivals in cf.interaction.items():
            if fa == feat or fb == feat:
                y_net = y_net + 0.5 * np.asarray(ivals)

        # Scatter points (subsampled)
        entry = {}
        entry["scatter_x"] = np.round(x_vals[scatter_idx], 5).tolist()
        entry["scatter_main"] = np.round(y_main[scatter_idx], 5).tolist()
        entry["scatter_net"] = np.round(y_net[scatter_idx], 5).tolist()

        if compact_ids is not None:
            entry["scatter_pid"] = compact_ids[scatter_idx].tolist()

            # Per-partition regression lines
            for key, y_arr in [("partitions", y_main),
                               ("partitions_net", y_net)]:
                lines = []
                for lf, pid in pid_map.items():
                    mask = leaf_ids == lf
                    cnt = int(mask.sum())
                    if cnt < 2:
                        continue
                    xp = x_vals[mask]
                    yp = y_arr[mask]
                    # Simple OLS: y = slope * x + intercept
                    x_mean = np.mean(xp)
                    y_mean = np.mean(yp)
                    denom = np.sum((xp - x_mean) ** 2)
                    if denom < 1e-15:
                        slope = 0.0
                        intercept = y_mean
                    else:
                        slope = float(np.sum((xp - x_mean) * (yp - y_mean))
                                      / denom)
                        intercept = float(y_mean - slope * x_mean)
                    lines.append({
                        "pid": pid, "n": cnt,
                        "slope": round(slope, 6),
                        "intercept": round(intercept, 6),
                        "x_min": round(float(np.min(xp)), 5),
                        "x_max": round(float(np.max(xp)), 5),
                    })
                # Sort by partition size descending
                lines.sort(key=lambda l: l["n"], reverse=True)
                entry[key] = lines

        curves[feat] = entry
    return curves


def _build_partition_summary(model, X, y, feature_names, cf=None):
    """Build per-partition summary: sizes, local R², feature slopes, residuals.

    Returns dict with:
      - partitions: list of per-partition dicts (sorted by size desc)
      - feature_names: list of feature names
      - n_partitions: total count
    """
    try:
        cpp, _ = model._get_geometry()
        leaf_ids = np.asarray(cpp.apply(X))
    except Exception:
        return None

    y_pred = model.predict(X)
    residuals = y - y_pred
    n, d = X.shape

    unique_leaves = np.unique(leaf_ids)
    pid_map = {int(lf): i for i, lf in enumerate(unique_leaves)}
    compact_ids = np.array([pid_map[int(lf)] for lf in leaf_ids])

    partitions = []
    for lf in unique_leaves:
        pid = pid_map[int(lf)]
        mask = leaf_ids == lf
        cnt = int(mask.sum())
        xp = X[mask]
        yp = y[mask]
        ypp = y_pred[mask]
        res = residuals[mask]

        # Local R²
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((yp - np.mean(yp)) ** 2)
        local_r2 = float(1.0 - ss_res / max(ss_tot, 1e-12)) if cnt >= 2 else 1.0

        # Feature slopes (OLS per feature)
        slopes = {}
        for fi, feat in enumerate(feature_names):
            xf = xp[:, fi]
            x_mean = np.mean(xf)
            y_mean = np.mean(yp)
            denom = np.sum((xf - x_mean) ** 2)
            if denom < 1e-15 or cnt < 2:
                slopes[feat] = 0.0
            else:
                slopes[feat] = round(float(
                    np.sum((xf - x_mean) * (yp - y_mean)) / denom), 6)

        # Contribution slopes (if available)
        contrib_slopes = {}
        if cf is not None:
            for feat in feature_names:
                if feat not in cf.main:
                    continue
                yc = np.asarray(cf.main[feat])[mask]
                xf = xp[:, feature_names.index(feat)]
                x_mean = np.mean(xf)
                yc_mean = np.mean(yc)
                denom = np.sum((xf - x_mean) ** 2)
                if denom < 1e-15 or cnt < 2:
                    contrib_slopes[feat] = 0.0
                else:
                    contrib_slopes[feat] = round(float(
                        np.sum((xf - x_mean) * (yc - yc_mean)) / denom), 6)

        info = {
            "pid": pid,
            "n": cnt,
            "pct": round(cnt / n * 100, 1),
            "local_r2": round(local_r2, 4),
            "mean_y": round(float(np.mean(yp)), 4),
            "mean_pred": round(float(np.mean(ypp)), 4),
            "mean_abs_resid": round(float(np.mean(np.abs(res))), 4),
            "std_resid": round(float(np.std(res)), 4),
            "feature_slopes": slopes,
            "contrib_slopes": contrib_slopes,
            "feature_means": {feat: round(float(np.mean(xp[:, fi])), 4)
                              for fi, feat in enumerate(feature_names)},
            "feature_ranges": {feat: [round(float(np.min(xp[:, fi])), 4),
                                      round(float(np.max(xp[:, fi])), 4)]
                               for fi, feat in enumerate(feature_names)},
        }
        partitions.append(info)

    # Sort by size descending
    partitions.sort(key=lambda p: p["n"], reverse=True)

    # 3D scatter data — all points for the global fingerprint
    sc_idx = np.arange(n)

    scatter_3d = {
        "x": np.round(X[sc_idx], 4).tolist(),
        "y_pred": np.round(y_pred[sc_idx], 4).tolist(),
        "y_true": np.round(y[sc_idx], 4).tolist(),
        "pid": compact_ids[sc_idx].tolist(),
    }

    # --- Hyperboloid decomposition of cooperation geometry ---
    # The cooperation score Q = ½(S² - ||z||²) is a quadratic form with
    # eigenvalues ½(d-1) along [1,...,1] and -½ along the orthogonal complement.
    # Decomposing z = u₁·e₁ + z_perp gives: Q = ½((d-1)·u₁² - ||z_perp||²).
    # Level sets Q = k are exact hyperboloids of revolution around e₁.
    #
    # 3D projection:
    #   Y = u₁  (cooperative axis — the [1,...,1] direction)
    #   X, Z = PCA(z_perp)[:2]  (transverse plane — rotation-invariant)
    try:
        X_z = np.asarray(cpp.to_z(X))  # (n, d)
        d_feat = X_z.shape[1]

        # Cooperation score
        S_vec = X_z.sum(axis=1)
        l2sq = (X_z * X_z).sum(axis=1)
        coop_score = 0.5 * (S_vec * S_vec - l2sq)
        scatter_3d["coop_score"] = np.round(coop_score[sc_idx], 4).tolist()

        # Cooperative axis: projection onto e₁ = [1,...,1]/√d
        sqrt_d = np.sqrt(d_feat)
        u1 = S_vec / sqrt_d  # (n,)

        # Transverse component: z_perp = z - (S/d)·1
        z_perp = X_z - (S_vec / d_feat)[:, np.newaxis]  # (n, d)

        # Full transverse radius (all d-1 dimensions)
        r_full = np.sqrt((z_perp * z_perp).sum(axis=1))  # (n,)

        # Angular coordinate from PCA on z_perp (just for visual distribution;
        # the hyperboloid is rotationally symmetric so any angle works)
        zp_mean = z_perp.mean(axis=0)
        zp_c = z_perp - zp_mean
        cov_perp = zp_c.T @ zp_c / max(n - 1, 1)
        eigvals_p, eigvecs_p = np.linalg.eigh(cov_perp)
        order_p = np.argsort(eigvals_p)[::-1]
        pc_perp = eigvecs_p[:, order_p[:2]]  # (d, 2)
        xy_perp = zp_c @ pc_perp  # (n, 2)
        theta = np.arctan2(xy_perp[:, 1], xy_perp[:, 0])  # (n,)

        # Cylindrical → Cartesian: exact hyperboloid coordinates
        # X = r·cos(θ), Y = u₁, Z = r·sin(θ)
        # Q = ½((d-1)·u₁² - r²) is EXACTLY preserved for every point.
        hyp_x = r_full * np.cos(theta)
        hyp_z = r_full * np.sin(theta)

        scatter_3d["spec_x"] = np.round(hyp_x[sc_idx], 6).tolist()
        scatter_3d["spec_y"] = np.round(u1[sc_idx], 6).tolist()
        scatter_3d["spec_z"] = np.round(hyp_z[sc_idx], 6).tolist()
        scatter_3d["n_features"] = d_feat

        # Trajectory info (for overlay display)
        traj = cpp.partition_trajectory()
        scatter_3d["n_snapshots"] = len(traj)
    except Exception:
        pass

    return {
        "partitions": partitions,
        "feature_names": feature_names,
        "n_partitions": len(partitions),
        "scatter_3d": scatter_3d,
    }


def _build_interaction_surfaces(cf, X, feature_names, n_bins=20, top_n=10):
    """Build 2D binned interaction surfaces for top interactions."""
    surfaces = {}
    # Rank interactions by abs_mean
    inter_ranked = sorted(cf.interaction.items(),
                          key=lambda kv: np.mean(np.abs(kv[1])), reverse=True)
    for (feat_a, feat_b), vals in inter_ranked[:top_n]:
        ia = feature_names.index(feat_a)
        ib = feature_names.index(feat_b)
        xa, xb = X[:, ia], X[:, ib]
        yvals = np.asarray(vals)
        edges_a = np.linspace(np.min(xa), np.max(xa), n_bins + 1)
        edges_b = np.linspace(np.min(xb), np.max(xb), n_bins + 1)
        grid = np.full((n_bins, n_bins), np.nan)
        for i in range(n_bins):
            for j in range(n_bins):
                mask_a = (xa >= edges_a[i]) & (xa < edges_a[i + 1])
                mask_b = (xb >= edges_b[j]) & (xb < edges_b[j + 1])
                if i == n_bins - 1:
                    mask_a |= (xa == edges_a[i + 1])
                if j == n_bins - 1:
                    mask_b |= (xb == edges_b[j + 1])
                mask = mask_a & mask_b
                if mask.sum() >= 1:
                    grid[i, j] = float(np.mean(yvals[mask]))
        centers_a = [float((edges_a[k] + edges_a[k+1]) / 2) for k in range(n_bins)]
        centers_b = [float((edges_b[k] + edges_b[k+1]) / 2) for k in range(n_bins)]
        key = f"{feat_a} x {feat_b}"
        # Replace NaN with None for JSON
        grid_list = [[None if np.isnan(v) else float(v) for v in row] for row in grid]
        surfaces[key] = {"x": centers_a, "y": centers_b, "z": grid_list,
                         "feat_a": feat_a, "feat_b": feat_b}
    return surfaces


def _estimate_irreducible_error(X, y, k=10, model=None):
    """
    Estimate per-sample irreducible error by averaging two methods:

    1. **k-NN local variance** — for each sample, finds k nearest neighbors
       and computes std of their y values.
    2. **Partition variance** — uses the model's HVRT partitions; samples in
       the same leaf share a local y-variance estimate.

    When both are available the per-sample estimate is the geometric mean,
    giving a robust combined estimate.  Falls back to k-NN alone when the
    model is not available.

    Returns
    -------
    dict with keys:
        global_std : float — combined global noise floor
        per_sample_std : list[float] — per-sample local noise estimate
        k : int — number of neighbors used (k-NN method)
        method : str — method description
        methods_used : list[str]
        knn_global_std : float
        partition_global_std : float | None
    """
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    methods_used = []

    # ---- Method 1: k-NN ----
    k_actual = min(k, max(3, n // 30))
    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_y = y[indices[:, 1:]]
    knn_var = np.var(neighbor_y, axis=1)
    knn_std = np.sqrt(knn_var)
    knn_global = float(np.sqrt(np.mean(knn_var)))
    methods_used.append("knn_local_variance")

    # ---- Method 2: partition variance ----
    part_std = None
    part_global = None
    if model is not None:
        try:
            cpp, _ = model._get_geometry()
            leaf_ids = np.asarray(cpp.apply(X))
            part_var = np.zeros(n)
            for leaf in np.unique(leaf_ids):
                mask = leaf_ids == leaf
                if mask.sum() >= 2:
                    part_var[mask] = np.var(y[mask])
                else:
                    part_var[mask] = knn_var[mask]  # fallback
            part_std = np.sqrt(part_var)
            part_global = float(np.sqrt(np.mean(part_var)))
            methods_used.append("partition_variance")
        except Exception:
            pass

    # ---- Combine ----
    if part_std is not None:
        # Geometric mean: balances two estimates, robust to outliers in either
        combined_std = np.sqrt(knn_std * part_std)
        global_std = float(np.sqrt(knn_global * part_global))
        method_desc = "combined (geometric mean of k-NN and partition)"
    else:
        combined_std = knn_std
        global_std = knn_global
        method_desc = "knn_local_variance"

    return {
        "global_std": global_std,
        "per_sample_std": combined_std.tolist(),
        "k": int(k_actual),
        "method": method_desc,
        "methods_used": methods_used,
        "knn_global_std": knn_global,
        "partition_global_std": part_global,
    }


def _build_exemplars(model, X, y, feature_names, cf,
                     irr_error=None):
    """Find best/worst/median predictions and extract local model info.

    When irr_error is provided, also computes noise-adjusted rankings
    and includes both raw and adjusted exemplars.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    abs_resid = np.abs(residuals)

    # --- Raw rankings ---
    raw_indices = {
        "best": int(np.argmin(abs_resid)),
        "worst": int(np.argmax(abs_resid)),
        "median": int(np.argsort(abs_resid)[len(abs_resid) // 2]),
    }

    # --- Adjusted rankings (residual relative to local noise) ---
    adj_indices = None
    adjusted_resid = None
    if irr_error is not None:
        local_std = np.array(irr_error["per_sample_std"])
        # Adjusted residual: |residual| / max(local_std, floor)
        # Floor prevents division by zero in low-noise regions
        floor = max(irr_error["global_std"] * 0.1, 1e-8)
        adjusted_resid = abs_resid / np.maximum(local_std, floor)

        adj_indices = {
            "best": int(np.argmin(adjusted_resid)),
            "worst": int(np.argmax(adjusted_resid)),
            "median": int(np.argsort(adjusted_resid)[len(adjusted_resid) // 2]),
        }

    def _build_sample_info(idx):
        info = {
            "sample_idx": idx,
            "y_true": float(y[idx]),
            "y_pred": float(y_pred[idx]),
            "residual": float(residuals[idx]),
            "abs_residual": float(abs_resid[idx]),
            "features": {feature_names[j]: float(X[idx, j])
                         for j in range(X.shape[1])},
        }
        if irr_error is not None:
            info["local_noise_std"] = float(
                irr_error["per_sample_std"][idx])
            if adjusted_resid is not None:
                info["adjusted_residual"] = float(adjusted_resid[idx])

        # Local model
        try:
            lm = model.local_model(X[idx:idx+1],
                                   feature_names=feature_names)
            info["local_r2"] = float(lm["local_r2"])
            info["partition_size"] = int(lm["partition_size"])
            info["additive"] = {feature_names[j]: float(lm["additive"][j])
                                for j in range(len(lm["additive"]))}
            info["pairwise"] = {
                f"{feature_names[i]} x {feature_names[j]}": float(v)
                for (i, j), v in lm["pairwise"].items()}
        except Exception:
            pass

        # Per-feature contributions
        if cf is not None:
            contribs = {}
            for feat in feature_names:
                if feat in cf.main:
                    contribs[feat] = float(cf.main[feat][idx])
            info["contributions"] = contribs

        return info

    exemplars = {"raw": {}, "adjusted": None}
    for label, idx in raw_indices.items():
        exemplars["raw"][label] = _build_sample_info(idx)

    if adj_indices is not None:
        exemplars["adjusted"] = {}
        for label, idx in adj_indices.items():
            exemplars["adjusted"][label] = _build_sample_info(idx)

    return exemplars


def _build_sample_lookup(model, X, y, feature_names, cf, irr_error=None,
                         cooperation_result=None, global_coop_matrix=None):
    """
    Build a compact per-sample lookup table for all contribution samples.

    Stores parallel arrays rather than per-sample dicts for compactness.
    Enables the interactive report's sample ID lookup feature.
    """
    n, d = X.shape
    y_pred = model.predict(X)
    residuals = y - y_pred

    lookup = {
        "n": int(n),
        "feature_names": feature_names,
        "features": X.tolist(),
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
    }

    # Per-sample noise estimates
    if irr_error is not None:
        lookup["local_noise_std"] = irr_error["per_sample_std"]
        floor = max(irr_error["global_std"] * 0.1, 1e-8)
        local_std = np.array(irr_error["per_sample_std"])
        adjusted = np.abs(residuals) / np.maximum(local_std, floor)
        lookup["adjusted_residual"] = adjusted.tolist()

    # Per-sample main-effect contributions
    if cf is not None:
        main_c = {}
        for feat in feature_names:
            if feat in cf.main:
                main_c[feat] = np.asarray(cf.main[feat]).tolist()
        lookup["contributions_main"] = main_c

        # Per-sample interaction contributions (top interactions only)
        inter_c = {}
        inter_ranked = sorted(cf.interaction.items(),
                              key=lambda kv: np.mean(np.abs(kv[1])),
                              reverse=True)
        for (fa, fb), vals in inter_ranked[:15]:
            inter_c[f"{fa} x {fb}"] = np.asarray(vals).tolist()
        lookup["contributions_interaction"] = inter_c

    # Per-sample partition size and local R² (via HVRT leaf assignment)
    try:
        cpp, _ = model._get_geometry()
        leaf_ids = np.asarray(cpp.apply(X))
        y_pred_arr = np.asarray(lookup["y_pred"])
        part_sizes = np.zeros(n, dtype=int)
        local_r2s = np.zeros(n)
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            sz = int(mask.sum())
            part_sizes[mask] = sz
            if sz >= 2:
                ss_res = np.sum((y[mask] - y_pred_arr[mask]) ** 2)
                ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
                local_r2s[mask] = 1.0 - ss_res / max(ss_tot, 1e-12)
            else:
                local_r2s[mask] = 1.0
        lookup["partition_size"] = part_sizes.tolist()
        lookup["local_r2"] = local_r2s.tolist()
    except Exception:
        pass

    # Per-sample cooperation matrices (stored as unique partitions + index)
    # This is space-efficient: only store each unique partition matrix once.
    if cooperation_result is not None:
        matrices = np.asarray(cooperation_result["matrices"])  # (n, d, d)
        # Find unique matrices by partition assignment
        # Use apply() to get leaf IDs — samples in same partition share a matrix
        try:
            leaf_ids = np.asarray(model._get_geometry()[0].apply(X))
            unique_leaves, inverse = np.unique(leaf_ids, return_inverse=True)
            unique_mats = []
            for leaf in unique_leaves:
                idx = np.where(leaf_ids == leaf)[0][0]
                unique_mats.append(np.round(matrices[idx], 3).tolist())
            lookup["coop_partitions"] = unique_mats
            lookup["coop_partition_idx"] = inverse.tolist()
        except Exception:
            # Fallback: store all matrices (rounded)
            lookup["coop_partitions"] = [
                np.round(matrices[i], 3).tolist() for i in range(n)]
            lookup["coop_partition_idx"] = list(range(n))

    if global_coop_matrix is not None:
        lookup["global_coop_matrix"] = global_coop_matrix

    return lookup


def _build_geo_data(model, X, feature_names, cooperation_matrices=None,
                    contribution_frame=None):
    """If lat/lon columns detected, build geographic cooperation data."""
    # Auto-detect lat/lon columns
    lat_idx, lon_idx = None, None
    lat_names = {"lat", "latitude", "y_coord", "lat_rad"}
    lon_names = {"lon", "lng", "longitude", "x_coord", "lon_rad", "long"}
    for i, name in enumerate(feature_names):
        lower = name.lower().strip()
        if lower in lat_names:
            lat_idx = i
        elif lower in lon_names:
            lon_idx = i

    if lat_idx is None or lon_idx is None:
        return None

    lat_name = feature_names[lat_idx]
    lon_name = feature_names[lon_idx]
    n = X.shape[0]
    lats = X[:, lat_idx].tolist()
    lons = X[:, lon_idx].tolist()

    # Get cooperation scores
    try:
        scores = model.cooperation_score(X).tolist()
    except Exception:
        scores = [0.0] * n

    # Get per-sample predictions
    try:
        preds = model.predict(X).tolist()
    except Exception:
        preds = [0.0] * n

    # Per-sample feature influence indices (0–1)
    # raw_f = |main_f| / total_abs
    # net_f = (|main_f| + 0.5 * sum_g(|interaction_f×g|)) / total_abs
    # Net influences sum to 1.0 across all features.
    geo_influence = None
    latlon_interaction = None
    feature_influence = None
    if contribution_frame is not None:
        cf = contribution_frame

        # Lat×lon interaction
        key = (lat_name, lon_name)
        key_rev = (lon_name, lat_name)
        if key in cf.interaction:
            latlon_interaction = np.asarray(cf.interaction[key]).tolist()
        elif key_rev in cf.interaction:
            latlon_interaction = np.asarray(cf.interaction[key_rev]).tolist()

        # Total |contributions| denominator
        n_cf = len(next(iter(cf.main.values())))
        total_abs = np.zeros(n_cf)
        for feat, vals in cf.main.items():
            total_abs += np.abs(np.asarray(vals))
        for (_fa, _fb), vals in cf.interaction.items():
            total_abs += np.abs(np.asarray(vals))

        # Per-feature raw and net influence
        feature_influence = {}
        for fname in feature_names:
            raw_vals = np.zeros(n_cf)
            net_vals = np.zeros(n_cf)
            if fname in cf.main:
                main_abs = np.abs(np.asarray(cf.main[fname]))
                raw_vals = main_abs.copy()
                net_vals = main_abs.copy()
            for (fa, fb), ivals in cf.interaction.items():
                if fa == fname or fb == fname:
                    net_vals += 0.5 * np.abs(np.asarray(ivals))
            safe = total_abs > 1e-10
            feature_influence[fname] = {
                "raw": np.round(np.where(safe, raw_vals / total_abs, 0.0), 4).tolist(),
                "net": np.round(np.where(safe, net_vals / total_abs, 0.0), 4).tolist(),
            }

        # Combined geo influence = lat + lon + lat×lon interaction
        geo_abs = np.zeros(n_cf)
        if lat_name in cf.main:
            geo_abs += np.abs(np.asarray(cf.main[lat_name]))
        if lon_name in cf.main:
            geo_abs += np.abs(np.asarray(cf.main[lon_name]))
        if latlon_interaction is not None:
            geo_abs += np.abs(np.asarray(latlon_interaction))
        geo_influence = np.where(
            total_abs > 1e-10, geo_abs / total_abs, 0.0
        ).tolist()

    # Per-feature cooperation at each point (top 5 pairs for popup)
    per_point_top_pairs = []
    if cooperation_matrices is not None:
        matrices = np.asarray(cooperation_matrices)
        d = len(feature_names)
        for si in range(min(n, len(matrices))):
            mat = matrices[si]
            pairs = []
            for i in range(d):
                for j in range(i + 1, d):
                    pairs.append((feature_names[i], feature_names[j], float(mat[i, j])))
            pairs.sort(key=lambda p: abs(p[2]), reverse=True)
            per_point_top_pairs.append(
                [{"a": p[0], "b": p[1], "v": round(p[2], 3)} for p in pairs[:5]]
            )

    # Subsample if too many points for the map
    max_points = 2000
    if n > max_points:
        step = n // max_points
        indices = list(range(0, n, step))[:max_points]
        lats = [lats[i] for i in indices]
        lons = [lons[i] for i in indices]
        scores = [scores[i] for i in indices]
        preds = [preds[i] for i in indices]
        if latlon_interaction:
            latlon_interaction = [latlon_interaction[i] for i in indices]
        if geo_influence:
            geo_influence = [geo_influence[i] for i in indices]
        if feature_influence:
            for fname in feature_influence:
                for mode in ("raw", "net"):
                    feature_influence[fname][mode] = [
                        feature_influence[fname][mode][i] for i in indices
                    ]
        if per_point_top_pairs:
            per_point_top_pairs = [per_point_top_pairs[i] for i in indices]

    geo = {
        "lat_feature": lat_name,
        "lon_feature": lon_name,
        "lats": lats,
        "lons": lons,
        "cooperation_scores": scores,
        "predictions": preds,
    }
    if latlon_interaction:
        geo["latlon_interaction"] = latlon_interaction
    if geo_influence:
        geo["geo_influence"] = geo_influence
    if per_point_top_pairs:
        geo["top_pairs"] = per_point_top_pairs

    # Cap to top-k features by mean net influence for map data
    max_map_features = 10
    if feature_influence:
        ranked = sorted(feature_influence.keys(),
                        key=lambda f: np.mean(feature_influence[f]["net"]),
                        reverse=True)
        geo["feature_influence"] = {
            f: feature_influence[f] for f in ranked[:max_map_features]
        }

    return geo


def _build_cooperation_surface(lats, lons, scores, grid_res=80,
                               fixed_range=None):
    """
    Interpolate per-point cooperation scores onto a regular lat/lon grid.

    Uses scipy griddata (linear) with IDW fallback for NaN fill.
    Returns a dict with bounds, grid dimensions, and flattened row-major
    values suitable for rendering as a canvas ImageOverlay.
    """
    from scipy.interpolate import griddata

    lats_a = np.array(lats)
    lons_a = np.array(lons)
    scores_a = np.array(scores)

    # Grid bounds with small padding
    lat_min, lat_max = float(lats_a.min()), float(lats_a.max())
    lon_min, lon_max = float(lons_a.min()), float(lons_a.max())
    lat_pad = (lat_max - lat_min) * 0.02
    lon_pad = (lon_max - lon_min) * 0.02
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    # Build regular grid
    grid_lat = np.linspace(lat_min, lat_max, grid_res)
    grid_lon = np.linspace(lon_min, lon_max, grid_res)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    # Interpolate (linear, then fill NaN with nearest)
    points = np.column_stack([lats_a, lons_a])
    grid = griddata(points, scores_a,
                    (grid_lat_2d, grid_lon_2d), method='linear')
    mask = np.isnan(grid)
    if mask.any():
        fill = griddata(points, scores_a,
                        (grid_lat_2d[mask], grid_lon_2d[mask]),
                        method='nearest')
        grid[mask] = fill

    # Normalise to [0, 1] for colormap
    if fixed_range is not None:
        vmin, vmax = fixed_range
    else:
        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    rng = (vmax - vmin) if (vmax - vmin) > 1e-10 else 1.0

    # Store as flat list (row-major, top=lat_max, bottom=lat_min)
    # Flip rows so row 0 = northernmost latitude
    grid_flipped = np.flipud(grid)
    values = np.round((grid_flipped - vmin) / rng, 3).ravel().tolist()

    return {
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "rows": grid_res,
        "cols": grid_res,
        "values": values,
        "vmin": vmin,
        "vmax": vmax,
    }


def _compute_metrics(task, y_true, y_pred, y_proba=None):
    """Compute task-appropriate metrics."""
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error,
        roc_auc_score, accuracy_score, f1_score,
    )
    metrics = {}
    if task == "regression":
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        f1_avg = "binary" if task == "binary" else "macro"
        metrics["f1"] = float(f1_score(y_true, y_pred, average=f1_avg,
                                        zero_division=0))
        if y_proba is not None:
            try:
                if task == "binary":
                    metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["auc"] = float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                pass
    return metrics


def build_artifact(
    model,
    X_train,
    y_train,
    *,
    X_test=None,
    y_test=None,
    feature_names=None,
    include_cooperation: bool = True,
    include_contributions: bool = True,
    include_tensor: bool = False,
    include_predictions: bool = True,
    max_contribution_samples: int = 5000,
    min_pair_coop: float = 0.10,
    custom: dict | None = None,
) -> ModelArtifact:
    """
    Build a ModelArtifact from a fitted GeoXGB model.

    Parameters
    ----------
    model : fitted GeoXGBRegressor or GeoXGBClassifier
    X_train, y_train : training data
    X_test, y_test : optional test data (adds test predictions + metrics)
    feature_names : list of str, optional
    include_cooperation : bool, default True
        Compute global cooperation matrix
    include_contributions : bool, default True
        Compute contribution statistics (requires training data)
    include_tensor : bool, default False
        Compute 3-way cooperation tensor (expensive for high-d)
    include_predictions : bool, default True
        Include train/test predictions in the artifact
    max_contribution_samples : int, default 5000
        Subsample training data for contributions (speed)
    min_pair_coop : float, default 0.10
        Minimum cooperation threshold for interaction terms
    custom : dict, optional
        User metadata to attach

    Returns
    -------
    ModelArtifact
    """
    import geoxgb

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    n, d = X_train.shape

    # Feature names
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]
    feature_names = list(feature_names)

    # Model type and task
    model_type = type(model).__name__
    if "Classifier" in model_type:
        n_classes = len(np.unique(y_train))
        task = "binary" if n_classes == 2 else "multiclass"
    else:
        task = "regression"

    # Hyperparameters
    param_names = getattr(model, "_PARAM_NAMES", [])
    hyperparams = {}
    for p in param_names:
        if hasattr(model, p):
            v = getattr(model, p)
            hyperparams[p] = _jsonify(v)
    # Add loss for regressor
    if hasattr(model, "loss"):
        hyperparams["loss"] = model.loss

    # Fit stats
    convergence_round = getattr(model, "convergence_round_", None)
    n_rounds = hyperparams.get("n_rounds", None)
    n_rounds_actual = convergence_round if convergence_round else n_rounds
    fit_stats = {
        "n_train": int(n),
        "n_features": int(d),
        "n_rounds": n_rounds,
        "n_rounds_actual": n_rounds_actual,
        "convergence_round": convergence_round,
    }

    # Feature importances
    try:
        raw_imp = model._cpp_model.feature_importances()
        total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
        importances = {feature_names[i]: float(raw_imp[i] / total)
                       for i in range(min(len(raw_imp), d))}
    except Exception:
        importances = {}

    # Noise
    noise_info = {}
    try:
        from geoxgb.report import noise_report
        noise_info = _jsonify(noise_report(model))
    except Exception:
        pass

    # Cooperation
    coop_data = None
    if include_cooperation:
        try:
            coop = model.cooperation_matrix(X_train, feature_names=feature_names)
            global_mat = np.asarray(coop["global_matrix"])
            # Pearson correlation matrix for comparison
            corr_mat = np.corrcoef(X_train, rowvar=False)
            corr_mat = np.round(corr_mat, 4).tolist()
            coop_data = {
                "matrix": global_mat.tolist(),
                "correlation_matrix": corr_mat,
                "feature_names": feature_names,
                "top_pairs": _top_pairs(global_mat, feature_names),
            }
        except Exception:
            pass

    # Cooperation tensor
    tensor_data = None
    if include_tensor and d <= 30:  # skip for very high-d
        try:
            tens = model.cooperation_tensor(X_train, feature_names=feature_names)
            global_tens = np.asarray(tens["global_tensor"])
            tensor_data = {
                "top_triples": _top_triples(global_tens, feature_names),
            }
        except Exception:
            pass

    # Contributions + curves + exemplars + sample lookup
    contrib_data = None
    main_curves = None
    interaction_surfaces = None
    exemplars_data = None
    sample_lookup_data = None
    partition_summary_data = None
    irr_error = None
    cf = None
    X_c = X_train  # may be subsampled below
    if include_contributions:
        try:
            # Subsample for speed
            if n > max_contribution_samples:
                rng = np.random.RandomState(42)
                idx = rng.choice(n, max_contribution_samples, replace=False)
                X_c = X_train[idx]
            else:
                X_c = X_train
                idx = None

            y_c = y_train[idx] if idx is not None else y_train

            cf = model.contributions(X_c, feature_names=feature_names,
                                     min_pair_coop=min_pair_coop)
            contrib_data = _aggregate_contributions(cf)

            # EBM-style curves (main + net views)
            main_curves = _build_main_curves(cf, X_c, feature_names,
                                            model=model)

            # Partition summary
            try:
                partition_summary_data = _build_partition_summary(
                    model, X_c, y_c, feature_names, cf=cf)
            except Exception:
                partition_summary_data = None

            # Irreducible error estimation (k-NN local variance)
            try:
                irr_error = _estimate_irreducible_error(X_c, y_c, model=model)
            except Exception:
                pass

            # Sample exemplars (use subsampled data)
            exemplars_data = _build_exemplars(
                model, X_c, y_c, feature_names, cf,
                irr_error=irr_error)

            # Per-sample cooperation matrices (needed for lookup + geo)
            coop_full = None
            if include_cooperation:
                try:
                    coop_full = model.cooperation_matrix(
                        X_c, feature_names=feature_names)
                except Exception:
                    pass

            # Per-sample lookup table
            sample_lookup_data = _build_sample_lookup(
                model, X_c, y_c, feature_names, cf,
                irr_error=irr_error,
                cooperation_result=coop_full,
                global_coop_matrix=(coop_data or {}).get("matrix"))
        except Exception:
            pass

    # Geographic data (auto-detect lat/lon)
    # Use contribution subsample (X_c / cf) when available — cf indices
    # are aligned to X_c, and interpolation handles sparse coverage fine.
    geo = None
    if include_cooperation:
        try:
            if coop_full is None:
                X_geo = X_c if cf is not None else X_train
                coop_full = model.cooperation_matrix(
                    X_geo, feature_names=feature_names)
            geo = _build_geo_data(
                model, X_c if cf is not None else X_train, feature_names,
                cooperation_matrices=coop_full.get("matrices"),
                contribution_frame=cf)
        except Exception:
            pass

    # Predictions
    pred_data = None
    if include_predictions:
        pred_data = {}
        try:
            y_pred_train = model.predict(X_train)
            train_pred = {
                "y_true": y_train.tolist(),
                "y_pred": y_pred_train.tolist(),
                "residuals": (y_train - y_pred_train).tolist(),
            }
            train_pred["metrics"] = _compute_metrics(task, y_train, y_pred_train)
            pred_data["train"] = train_pred
        except Exception:
            pass

        if X_test is not None and y_test is not None:
            try:
                X_test = np.asarray(X_test, dtype=np.float64)
                y_test = np.asarray(y_test)
                y_pred_test = model.predict(X_test)
                test_pred = {
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred_test.tolist(),
                    "residuals": (y_test - y_pred_test).tolist(),
                }
                y_proba = None
                if task != "regression":
                    try:
                        y_proba = model.predict_proba(X_test)
                        test_pred["y_proba"] = y_proba.tolist()
                    except Exception:
                        pass
                test_pred["metrics"] = _compute_metrics(
                    task, y_test, y_pred_test, y_proba)
                pred_data["test"] = test_pred
            except Exception:
                pass

    return ModelArtifact(
        model_type=model_type,
        task=task,
        version=geoxgb.__version__,
        build_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hyperparameters=hyperparams,
        fit_stats=fit_stats,
        feature_names=feature_names,
        feature_importances=importances,
        noise=noise_info,
        cooperation=coop_data,
        cooperation_tensor=tensor_data,
        contributions=contrib_data,
        main_curves=main_curves,
        interaction_surfaces=None,
        exemplars=exemplars_data,
        sample_lookup=sample_lookup_data,
        irreducible_error=irr_error,
        geo_data=geo,
        partition_summary=partition_summary_data,
        predictions=pred_data,
        custom=custom or {},
    )
