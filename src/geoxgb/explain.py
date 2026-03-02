"""
geoxgb.explain
==============

Geometric structural explainer for GeoXGB models.

Uses the HVRT partition tree and z-space topology to explain individual
predictions without perturbation or post-hoc attribution.  Every piece of
information comes directly from the model's internal geometry:

  - *Partition routing* — the exact z-space decision path from root to leaf,
    with each split node mapped back to the most correlated original X feature.
  - *Cooperation geometry* — the T-value and cone fraction of the landing
    partition: how well the training samples there "agreed" on a direction.
  - *Neighbourhood provenance* — the k closest training samples in z-space,
    their original features, targets, and training residuals.
  - *Residual decomposition* — if a noiseless ground-truth function is
    supplied, the residual is split into model_error + irreducible noise.

Public API
----------
    GeoXGBExplainer
    Explanation       (dataclass)
    PathNode          (dataclass)
    Neighbour         (dataclass)
    PartitionGeometry (dataclass)
    format_explanation, print_explanation
    format_summary,    print_summary

Design notes
------------
- Returns Python-native types (no numpy arrays) so explanations are
  JSON-serialisable via to_dict().
- Backend-agnostic: works with CppGeoXGBRegressor or GeoXGBRegressor;
  only requires model.predict().  Fits its own Python HVRT internally.
- Z→X correlation matrix is precomputed once at construction time.
- Complexity: O(n_train · d²) at init for the correlation matrix; O(n_train)
  per explain_sample call for the k-NN search.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from hvrt import HVRT

__all__ = [
    "GeoXGBExplainer",
    "Explanation",
    "PathNode",
    "Neighbour",
    "PartitionGeometry",
    "format_explanation",
    "print_explanation",
    "format_summary",
    "print_summary",
]


# ===========================================================================
# Dataclasses — all fields are JSON-serialisable Python types
# ===========================================================================

@dataclass
class PathNode:
    """One split node on the HVRT tree path from root to leaf partition."""
    z_feat:      int        # z-space feature index used for the split
    threshold:   float      # split threshold value in z-space
    direction:   str        # "left" (≤ threshold) or "right" (> threshold)
    n_samples:   int        # training samples at this node before the split
    top_x_feat:  int        # original X feature index most correlated with z_feat
    top_x_name:  str        # feature name for top_x_feat
    top_x_corr:  float      # Spearman ρ between z_feat and top_x_feat (signed)
    all_x_corr:  list[float]  # Spearman ρ between z_feat and every X feature


@dataclass
class Neighbour:
    """One training-set sample from the z-space nearest-neighbour search."""
    rank:           int     # 1 = closest
    train_idx:      int
    z_distance:     float
    same_partition: bool    # shares the same HVRT leaf as the query sample
    y_train:        float
    pred_train:     float
    resid_train:    float   # y_train - pred_train
    x:              list[float]  # original X features


@dataclass
class PartitionGeometry:
    """Cooperation geometry of the HVRT leaf partition."""
    partition_id:  int
    n_train:       int
    t_value:       float    # mean pairwise T = 2·Σ zᵢzⱼ (cooperation); NaN if unavailable
    frac_in_cone:  float    # fraction of training samples with T > 0; NaN if unavailable
    resid_mean:    float    # mean training residual in this partition
    resid_std:     float    # std  training residual in this partition


@dataclass
class Explanation:
    """
    Full geometric explanation for one prediction.

    Use format_explanation() / print_explanation() for human-readable output,
    or to_dict() for a JSON-serialisable representation.
    """
    sample_idx:          int | None   # original row index in X_val (None if explain_sample)
    x:                   list[float]
    feature_names:       list[str]

    # ── Prediction ──────────────────────────────────────────────────────────
    pred:                float
    y_true:              float | None
    residual:            float | None  # y_true - pred
    model_error:         float | None  # noiseless_y - pred  (requires true_fn)
    noise:               float | None  # y_true - noiseless_y

    # ── Geometric structure ─────────────────────────────────────────────────
    tree_path:           list[PathNode]
    partition:           PartitionGeometry
    neighbours:          list[Neighbour]
    neighbour_y_std:     float   # spread of neighbour y values (signal ambiguity)
    neighbour_resid_std: float   # spread of neighbour training residuals

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (NaN values become None)."""
        def _clean(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj
        return _clean(asdict(self))


# ===========================================================================
# Internal helpers
# ===========================================================================

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Numpy-only Spearman rank correlation."""
    n = len(a)
    if n < 2:
        return 0.0
    def _ranks(x):
        order = np.argsort(x)
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        return r - r.mean()
    ra, rb = _ranks(a), _ranks(b)
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return 1.0 if np.allclose(ra, rb) else 0.0
    return float(np.dot(ra, rb) / denom)


def _fmt(v: float | None, fmt: str = ".4f", na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return format(v, fmt)


# ===========================================================================
# GeoXGBExplainer
# ===========================================================================

class GeoXGBExplainer:
    """
    Geometric structural explainer for any fitted GeoXGB model.

    Works with both ``CppGeoXGBRegressor`` and ``GeoXGBRegressor``; requires
    only ``model.predict()``.  Fits its own Python HVRT on the training data
    to provide partition routing, z-space topology, and cooperation geometry.

    Parameters
    ----------
    model : fitted GeoXGB model
        Must implement ``predict(X) -> np.ndarray``.
    X_train : array-like, shape (n, d)
        Training features (same data the model was trained on).
    y_train : array-like, shape (n,)
        Training targets.
    feature_names : list[str], optional
        Human-readable names for the d original features.
    k : int, default 5
        Number of z-space nearest neighbours to report per explanation.
    hvrt_params : dict, optional
        Override HVRT constructor kwargs.
        Defaults: ``{"y_weight": 0.5, "random_state": 0}``.
    true_fn : callable, optional
        ``true_fn(X) -> np.ndarray`` returning noiseless targets.
        If supplied, residuals are decomposed into model_error + noise.
        Useful when a ground-truth generative function is known (e.g. synthetic
        benchmarks).
    """

    _DEFAULT_HVRT = {"y_weight": 0.5, "random_state": 0}

    def __init__(
        self,
        model,
        X_train,
        y_train,
        *,
        feature_names: list[str] | None = None,
        k: int = 5,
        hvrt_params: dict | None = None,
        true_fn: Callable | None = None,
    ) -> None:
        self._model    = model
        self._X_tr     = np.asarray(X_train, dtype=np.float64)
        self._y_tr     = np.asarray(y_train,  dtype=np.float64).ravel()
        self._true_fn  = true_fn
        self._k        = k
        n, d = self._X_tr.shape
        self._n, self._d = n, d

        self._feature_names: list[str] = (
            list(feature_names) if feature_names is not None
            else [f"x{i}" for i in range(d)]
        )

        # ── Fit Python HVRT for geometry ────────────────────────────────────
        params = dict(self._DEFAULT_HVRT)
        if hvrt_params:
            params.update(hvrt_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._hvrt = HVRT(**params)
            self._hvrt.fit(self._X_tr, self._y_tr)

        # ── Precompute training geometry ─────────────────────────────────────
        self._X_tr_z  = self._hvrt._to_z(self._X_tr)
        self._tr_pids = self._hvrt.tree_.apply(self._X_tr_z)

        self._pred_tr  = self._model.predict(self._X_tr)
        self._resid_tr = self._y_tr - self._pred_tr

        # Geometry stats: partition_id -> raw dict from geometry_stats()
        geom_raw     = self._hvrt.geometry_stats()
        self._geom   = {
            int(pg["id"]): pg for pg in geom_raw.get("partitions", [])
        }

        # ── Z→X Spearman correlation matrix: shape (d_z, d_x) ───────────────
        # corr_zx[i, j] = Spearman(z_train[:, i], X_train[:, j])
        d_z = self._X_tr_z.shape[1]
        self._corr_zx = np.zeros((d_z, d), dtype=np.float64)
        for i in range(d_z):
            for j in range(d):
                self._corr_zx[i, j] = _spearman(self._X_tr_z[:, i], self._X_tr[:, j])

        # sklearn internal tree object (may be None for non-sklearn trees)
        self._sk_tree = getattr(self._hvrt.tree_, "tree_", None)

    # ── Core: explain one sample ─────────────────────────────────────────────

    def explain_sample(
        self,
        x,
        y_true: float | None = None,
        sample_idx: int | None = None,
    ) -> Explanation:
        """
        Explain a single prediction.

        Parameters
        ----------
        x : array-like, shape (d,)
        y_true : float, optional
        sample_idx : int, optional
            Row index label included in the explanation (cosmetic only).
        """
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        x_z   = self._hvrt._to_z(x_arr.reshape(1, -1))   # (1, d_z)

        # ── Prediction ───────────────────────────────────────────────────────
        pred = float(self._model.predict(x_arr.reshape(1, -1))[0])

        residual    = float(y_true) - pred if y_true is not None else None
        model_error = None
        noise       = None
        if self._true_fn is not None and y_true is not None:
            y_noiseless = float(
                np.asarray(self._true_fn(x_arr.reshape(1, -1))).ravel()[0]
            )
            model_error = y_noiseless - pred
            noise       = float(y_true) - y_noiseless

        # ── Partition geometry ───────────────────────────────────────────────
        pid  = int(self._hvrt.tree_.apply(x_z)[0])
        pg   = self._geom.get(pid, {})
        pmask = self._tr_pids == pid
        pr    = self._resid_tr[pmask]
        partition = PartitionGeometry(
            partition_id = pid,
            n_train      = int(pmask.sum()),
            t_value      = float(pg.get("E_T",          float("nan"))),
            frac_in_cone = float(pg.get("frac_in_cone", float("nan"))),
            resid_mean   = float(pr.mean()) if len(pr) > 0 else float("nan"),
            resid_std    = float(pr.std())  if len(pr) > 0 else float("nan"),
        )

        # ── Tree path ────────────────────────────────────────────────────────
        tree_path = self._decode_path(x_z)

        # ── Z-space k-NN ────────────────────────────────────────────────────
        dists   = np.linalg.norm(self._X_tr_z - x_z, axis=1)
        nn_idxs = np.argsort(dists)[: self._k]
        neighbours: list[Neighbour] = []
        for rank, idx in enumerate(nn_idxs, 1):
            neighbours.append(Neighbour(
                rank           = rank,
                train_idx      = int(idx),
                z_distance     = float(dists[idx]),
                same_partition = bool(self._tr_pids[idx] == pid),
                y_train        = float(self._y_tr[idx]),
                pred_train     = float(self._pred_tr[idx]),
                resid_train    = float(self._resid_tr[idx]),
                x              = [float(v) for v in self._X_tr[idx]],
            ))

        nn_ys = [nb.y_train    for nb in neighbours]
        nn_rs = [nb.resid_train for nb in neighbours]

        return Explanation(
            sample_idx          = sample_idx,
            x                   = [float(v) for v in x_arr],
            feature_names       = list(self._feature_names),
            pred                = pred,
            y_true              = float(y_true) if y_true is not None else None,
            residual            = residual,
            model_error         = model_error,
            noise               = noise,
            tree_path           = tree_path,
            partition           = partition,
            neighbours          = neighbours,
            neighbour_y_std     = float(np.std(nn_ys)),
            neighbour_resid_std = float(np.std(nn_rs)),
        )

    def _decode_path(self, x_z: np.ndarray) -> list[PathNode]:
        """Decode the HVRT decision path for a single z-space sample."""
        if not hasattr(self._hvrt.tree_, "decision_path"):
            return []
        if self._sk_tree is None:
            return []

        node_indicator = self._hvrt.tree_.decision_path(x_z)
        node_ids       = node_indicator.indices   # path nodes in root→leaf order

        feat_arr      = self._sk_tree.feature
        thresh_arr    = self._sk_tree.threshold
        n_samp_arr    = self._sk_tree.n_node_samples
        children_left = self._sk_tree.children_left

        path: list[PathNode] = []
        for i in range(len(node_ids) - 1):   # exclude the leaf itself
            node      = node_ids[i]
            next_node = node_ids[i + 1]
            z_feat    = int(feat_arr[node])
            if z_feat < 0:
                break   # leaf sentinel (-2)

            direction = "left" if next_node == children_left[node] else "right"
            corrs     = [float(self._corr_zx[z_feat, j]) for j in range(self._d)]
            top_j     = int(np.argmax(np.abs(self._corr_zx[z_feat])))

            path.append(PathNode(
                z_feat     = z_feat,
                threshold  = float(thresh_arr[node]),
                direction  = direction,
                n_samples  = int(n_samp_arr[node]),
                top_x_feat = top_j,
                top_x_name = self._feature_names[top_j],
                top_x_corr = corrs[top_j],
                all_x_corr = corrs,
            ))

        return path

    # ── Batch explain ────────────────────────────────────────────────────────

    def explain(
        self,
        X,
        y: np.ndarray | None = None,
        *,
        sort_by: str = "abs_residual",
        top_n:   int | None = None,
    ) -> list[Explanation]:
        """
        Explain predictions for a batch of samples.

        Parameters
        ----------
        X : array-like, shape (m, d)
        y : array-like, shape (m,), optional
            True targets; required for ``sort_by='abs_residual'`` or
            ``'abs_model_error'``.
        sort_by : str
            ``"abs_residual"`` — worst predictions first (default).
            ``"abs_model_error"`` — worst model error first (requires true_fn).
            ``"residual"`` — most negative residual first.
            ``"index"`` — original row order.
        top_n : int, optional
            Return only the first top_n after sorting.

        Returns
        -------
        list[Explanation]
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel() if y is not None else None

        explanations: list[Explanation] = [
            self.explain_sample(
                X_arr[i],
                float(y_arr[i]) if y_arr is not None else None,
                sample_idx=i,
            )
            for i in range(len(X_arr))
        ]

        def _key(e: Explanation) -> float:
            if sort_by == "abs_residual":
                return -abs(e.residual)  if e.residual    is not None else 0.0
            if sort_by == "abs_model_error":
                return -abs(e.model_error) if e.model_error is not None else 0.0
            if sort_by == "residual":
                return -(e.residual or 0.0)
            return float(e.sample_idx or 0)

        explanations.sort(key=_key)
        return explanations if top_n is None else explanations[:top_n]

    # ── Aggregate summary ────────────────────────────────────────────────────

    def summary(self, explanations: list[Explanation]) -> dict:
        """
        Aggregate statistics across a collection of Explanation objects.

        Returns a JSON-serialisable dict suitable for print_summary().
        """
        if not explanations:
            return {}

        # Partition distribution
        pid_counts: dict[int, int] = {}
        for e in explanations:
            pid = e.partition.partition_id
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
        top_partitions = sorted(pid_counts.items(), key=lambda kv: -kv[1])[:10]

        # T-value stats
        t_vals = [
            e.partition.t_value for e in explanations
            if not math.isnan(e.partition.t_value)
        ]

        # Neighbourhood quality
        nb_y_std   = [e.neighbour_y_std     for e in explanations]
        nb_r_std   = [e.neighbour_resid_std for e in explanations]

        # Most-used z-features across all tree paths
        z_feat_counts: dict[int, int] = {}
        for e in explanations:
            for node in e.tree_path:
                z_feat_counts[node.z_feat] = z_feat_counts.get(node.z_feat, 0) + 1
        top_z = sorted(z_feat_counts.items(), key=lambda kv: -kv[1])[:8]

        # Residual / model_error stats
        residuals    = [e.residual    for e in explanations if e.residual    is not None]
        model_errors = [e.model_error for e in explanations if e.model_error is not None]
        path_depths  = [len(e.tree_path) for e in explanations]

        # Fraction of neighbours sharing the same partition
        same_part_fracs = [
            sum(nb.same_partition for nb in e.neighbours) / max(len(e.neighbours), 1)
            for e in explanations
        ]

        return dict(
            n_explained              = len(explanations),
            residual_mean            = float(np.mean(residuals))    if residuals     else None,
            residual_std             = float(np.std(residuals))     if residuals     else None,
            model_error_mean         = float(np.mean(model_errors)) if model_errors  else None,
            model_error_std          = float(np.std(model_errors))  if model_errors  else None,
            t_value_mean             = float(np.mean(t_vals))       if t_vals        else None,
            t_value_std              = float(np.std(t_vals))        if t_vals        else None,
            neighbour_y_std_mean     = float(np.mean(nb_y_std)),
            neighbour_resid_std_mean = float(np.mean(nb_r_std)),
            same_partition_frac_mean = float(np.mean(same_part_fracs)),
            mean_path_depth          = float(np.mean(path_depths)),
            top_partitions           = [
                {"id": pid, "count": cnt} for pid, cnt in top_partitions
            ],
            top_z_feats_in_path      = [
                {
                    "z_feat": zf,
                    "count":  cnt,
                    "top_x_name": self._feature_names[
                        int(np.argmax(np.abs(self._corr_zx[zf])))
                    ],
                    "top_x_corr": float(
                        self._corr_zx[zf, int(np.argmax(np.abs(self._corr_zx[zf])))]
                    ),
                }
                for zf, cnt in top_z
            ],
        )


# ===========================================================================
# Formatting
# ===========================================================================

def format_explanation(
    e: Explanation,
    *,
    show_all_x: bool = False,
    max_neighbours: int | None = None,
    width: int = 62,
) -> str:
    """
    Return a human-readable string for one Explanation.

    Parameters
    ----------
    show_all_x : bool
        Show all feature values; default shows only the first 6.
    max_neighbours : int, optional
        Limit how many neighbours are printed.
    width : int
        Approximate line width for section dividers.
    """
    lines: list[str] = []
    div = "  " + "─" * (width - 2)

    # ── Header ────────────────────────────────────────────────────────────
    idx_str = f"[{e.sample_idx}]  " if e.sample_idx is not None else ""
    lines.append(f"Sample {idx_str}" + "─" * max(0, width - 7 - len(idx_str)))

    # ── X features ────────────────────────────────────────────────────────
    n_show = len(e.x) if show_all_x else min(6, len(e.x))
    x_parts = [f"{e.feature_names[i]}={e.x[i]:.3f}" for i in range(n_show)]
    if not show_all_x and len(e.x) > 6:
        x_parts.append(f"... (+{len(e.x)-6} more)")
    lines.append("  x:  " + "  ".join(x_parts))

    # ── Prediction summary ────────────────────────────────────────────────
    p = f"  pred={e.pred:.4f}"
    if e.y_true is not None:
        p += f"   y={e.y_true:.4f}   resid={e.residual:+.4f}"
    if e.model_error is not None:
        p += f"   [model_err={e.model_error:+.4f}  noise={e.noise:+.4f}]"
    lines.append(p)

    # ── Partition routing ─────────────────────────────────────────────────
    lines.append("")
    lines.append(div)
    lines.append("  Partition routing")
    lines.append(div)
    if e.tree_path:
        for node in e.tree_path:
            sym  = "≤" if node.direction == "left" else ">"
            lines.append(
                f"    z[{node.z_feat}] {sym} {node.threshold:+.4f}"
                f"  →  {node.direction:<5}"
                f"  (↔ {node.top_x_name}, ρ={node.top_x_corr:+.2f})"
                f"   n={node.n_samples}"
            )
    else:
        lines.append("    (tree path unavailable)")
    lines.append(f"    └─ partition {e.partition.partition_id}")

    # ── Partition geometry ────────────────────────────────────────────────
    lines.append("")
    lines.append(div)
    lines.append("  Partition geometry")
    lines.append(div)
    p = e.partition
    lines.append(
        f"    T={_fmt(p.t_value, '+.3f')}   "
        f"n_train={p.n_train}   "
        f"cone={_fmt(p.frac_in_cone, '.2f')}"
    )
    lines.append(
        f"    train residuals:  "
        f"mean={_fmt(p.resid_mean, '+.3f')}   "
        f"std={_fmt(p.resid_std, '.3f')}"
    )

    # ── Z-space neighbours ────────────────────────────────────────────────
    lines.append("")
    lines.append(div)
    nbs = e.neighbours if max_neighbours is None else e.neighbours[:max_neighbours]
    lines.append(f"  Z-space neighbours  (k={len(nbs)})")
    lines.append(div)
    # Header
    lines.append(
        f"    {'rank':>4}  {'z_dist':>6}  {'part?':>5}  "
        f"{'y_tr':>7}  {'resid':>7}  x (first 3 features)"
    )
    for nb in nbs:
        x3 = "  ".join(f"{nb.x[i]:.3f}" for i in range(min(3, len(nb.x))))
        lines.append(
            f"    {nb.rank:>4}  {nb.z_distance:>6.3f}  "
            f"{'yes' if nb.same_partition else 'no':>5}  "
            f"{nb.y_train:>7.3f}  {nb.resid_train:>+7.3f}  {x3}"
        )
    lines.append(
        f"    neighbour spread:  "
        f"y_std={e.neighbour_y_std:.3f}   "
        f"resid_std={e.neighbour_resid_std:.3f}"
    )

    return "\n".join(lines)


def print_explanation(e: Explanation, **kwargs) -> None:
    """Print a single Explanation.  kwargs forwarded to format_explanation."""
    print(format_explanation(e, **kwargs))


def format_summary(s: dict, *, width: int = 62) -> str:
    """Return a human-readable string for a summary dict from explain.summary()."""
    if not s:
        return "(empty summary)"
    lines: list[str] = []
    div = "─" * width

    lines.append(div)
    lines.append(f"GeoXGB Explanation Summary  (n={s['n_explained']})")
    lines.append(div)

    # Prediction quality
    lines.append("Prediction quality")
    if s.get("residual_mean") is not None:
        lines.append(
            f"  residual:     mean={s['residual_mean']:+.4f}   "
            f"std={s['residual_std']:.4f}"
        )
    if s.get("model_error_mean") is not None:
        lines.append(
            f"  model_error:  mean={s['model_error_mean']:+.4f}   "
            f"std={s['model_error_std']:.4f}"
        )

    # Geometry quality
    lines.append("Partition geometry")
    if s.get("t_value_mean") is not None:
        lines.append(
            f"  T:            mean={s['t_value_mean']:+.4f}   "
            f"std={s['t_value_std']:.4f}"
        )
    lines.append(f"  mean path depth:    {s['mean_path_depth']:.1f}")
    lines.append(
        f"  same-partition neighbour frac:  "
        f"{s['same_partition_frac_mean']:.2f}"
    )

    # Neighbourhood ambiguity
    lines.append("Neighbourhood ambiguity")
    lines.append(
        f"  mean neighbour y_std:     {s['neighbour_y_std_mean']:.4f}"
    )
    lines.append(
        f"  mean neighbour resid_std: {s['neighbour_resid_std_mean']:.4f}"
    )

    # Top partitions
    lines.append("Top partitions (by count in this set)")
    for entry in s["top_partitions"]:
        lines.append(f"  partition {entry['id']:>4}  →  {entry['count']} samples")

    # Top z-features used for routing
    lines.append("Top z-features driving partition routing")
    for entry in s["top_z_feats_in_path"]:
        lines.append(
            f"  z[{entry['z_feat']}]  {entry['count']:>4} appearances"
            f"   (↔ {entry['top_x_name']}, ρ={entry['top_x_corr']:+.2f})"
        )

    lines.append(div)
    return "\n".join(lines)


def print_summary(s: dict, **kwargs) -> None:
    """Print a summary dict from explain.summary().  kwargs → format_summary."""
    print(format_summary(s, **kwargs))
