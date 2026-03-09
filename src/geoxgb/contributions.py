"""
EBM-style per-prediction feature contributions for GeoXGB.

Public API
----------
ContributionFrame  -- dataclass holding per-sample main/interaction arrays
compute_contributions  -- batch Ridge loop (called by GeoXGBBase.contributions())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geoxgb._base import _GeoXGBBase


# ---------------------------------------------------------------------------
# Nadaraya-Watson smoothing helpers
# ---------------------------------------------------------------------------

def _bandwidth(x_feat: np.ndarray, bw) -> float:
    """Return bandwidth in data units."""
    r = float(x_feat.max() - x_feat.min())
    if r < 1e-12:
        return 1.0
    if bw == "auto":
        return r / 20.0
    return float(bw) * r


def _smooth_1d(
    x_feat: np.ndarray,
    contrib: np.ndarray,
    weights: np.ndarray,
    bw,
    n_grid: int = 100,
):
    """
    Nadaraya-Watson 1-D smoother.

    Returns
    -------
    grid   : (n_grid,) x values
    smooth : (n_grid,) weighted-mean contributions
    ci_std : (n_grid,) weighted std (use as ±1σ CI band)
    """
    lo, hi = float(x_feat.min()), float(x_feat.max())
    grid = np.linspace(lo, hi, n_grid)
    h = _bandwidth(x_feat, bw)

    smooth = np.empty(n_grid)
    ci_std = np.empty(n_grid)
    for k, g in enumerate(grid):
        k_w = weights * np.exp(-0.5 * ((x_feat - g) / h) ** 2)
        w_sum = k_w.sum()
        if w_sum < 1e-12:
            smooth[k] = 0.0
            ci_std[k] = 0.0
        else:
            mu = (k_w * contrib).sum() / w_sum
            smooth[k] = mu
            ci_std[k] = np.sqrt(
                (k_w * (contrib - mu) ** 2).sum() / w_sum
            )
    return grid, smooth, ci_std


def _smooth_2d(
    x_a: np.ndarray,
    x_b: np.ndarray,
    contrib: np.ndarray,
    weights: np.ndarray,
    bw,
    n_grid: int = 50,
):
    """
    Nadaraya-Watson 2-D smoother.

    Returns
    -------
    ga, gb : (n_grid,) grid coordinates for each axis
    Z      : (n_grid, n_grid) smoothed interaction values
    """
    ga = np.linspace(x_a.min(), x_a.max(), n_grid)
    gb = np.linspace(x_b.min(), x_b.max(), n_grid)
    ha = _bandwidth(x_a, bw)
    hb = _bandwidth(x_b, bw)

    Z = np.zeros((n_grid, n_grid))
    for ia, va in enumerate(ga):
        ka = np.exp(-0.5 * ((x_a - va) / ha) ** 2)
        for ib, vb in enumerate(gb):
            kb = np.exp(-0.5 * ((x_b - vb) / hb) ** 2)
            k_w = weights * ka * kb
            w_sum = k_w.sum()
            Z[ia, ib] = (k_w * contrib).sum() / w_sum if w_sum > 1e-12 else 0.0
    return ga, gb, Z


# ---------------------------------------------------------------------------
# Batch Ridge computation
# ---------------------------------------------------------------------------

def compute_contributions(
    model: "_GeoXGBBase",
    X: np.ndarray,
    feature_names=None,
    min_pair_coop: float = 0.10,
    alpha: float = 1e-3,
    target_class: int | None = None,
) -> "ContributionFrame":
    """
    Compute per-sample additive and interaction feature contributions.

    Parameters
    ----------
    model : fitted _GeoXGBBase
    X : array-like, shape (n_samples, n_features)
    feature_names : list of str, optional
    min_pair_coop : float, default=0.10
        Include pair (i,j) for a partition if |Pearson corr| >= threshold.
    alpha : float, default=1e-3
        Ridge regularisation.
    target_class : int, optional
        For multiclass classifiers, which class logit to explain.

    Returns
    -------
    ContributionFrame
    """
    cpp, _ = model._get_geometry()   # raises if not fitted / geometry missing

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    # ── batch z-transform and leaf assignment ───────────────────────────────
    test_leaves = np.asarray(cpp.apply(X))                       # (n,)
    X_z_query   = np.asarray(cpp.to_z(X))                        # (n, d)

    # Use full training data for local models (X_z/partition_ids are from the
    # HVRT reduced set and may be shorter than train_predictions which covers
    # the full training set).
    X_full = model._X_train                                     # (n_full, d_orig)
    X_z_train   = np.asarray(cpp.to_z(X_full))                  # (n_full, d)
    part_ids    = np.asarray(cpp.apply(X_full))                  # (n_full,)

    # Get training predictions (scalar or per-class slice)
    mc = getattr(model, '_mc_cpp_model', None)
    if mc is not None:
        if target_class is None:
            raise ValueError(
                "target_class is required for multiclass models. "
                f"Pass target_class in 0..{model._n_classes - 1}."
            )
        preds_train = np.asarray(mc.train_predictions_multi())[:, target_class]
    else:
        preds_train = np.asarray(cpp.train_predictions())  # (n_train,)

    # ── output arrays ───────────────────────────────────────────────────────
    main_arr   = np.zeros((n, d))    # additive contributions
    inter_arr  = {}                  # (i,j) -> (n,) — filled lazily
    intercepts = np.zeros(n)
    local_r2   = np.zeros(n)

    # ── per-partition Ridge loop ─────────────────────────────────────────────
    unique_query_leaves = np.unique(test_leaves)

    for leaf in unique_query_leaves:
        # query indices that land in this partition
        q_mask = test_leaves == leaf            # (n,) bool
        q_idx  = np.where(q_mask)[0]            # indices into X

        # training points in this partition
        t_mask = part_ids == leaf
        n_p    = int(t_mask.sum())

        Z_p      = X_z_train[t_mask]    # (n_p, d)
        y_hat_p  = preds_train[t_mask]  # (n_p,)

        # ── cooperation matrix → active pairs ───────────────────────────────
        if n_p >= 2:
            Z_c = Z_p - Z_p.mean(axis=0, keepdims=True)
            std = Z_c.std(axis=0)
            std[std < 1e-10] = 1.0
            Z_n = Z_c / std
            C_p = (Z_n.T @ Z_n) / n_p
        else:
            C_p = np.eye(d)

        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)
                 if abs(C_p[i, j]) >= min_pair_coop]

        # ── design matrix ────────────────────────────────────────────────────
        n_cols = 1 + d + len(pairs)
        M_train = np.empty((n_p, n_cols))
        M_train[:, 0] = 1.0
        M_train[:, 1:1 + d] = Z_p
        for col, (pi, pj) in enumerate(pairs, start=1 + d):
            M_train[:, col] = Z_p[:, pi] * Z_p[:, pj]

        # ── Ridge solve ──────────────────────────────────────────────────────
        A     = M_train.T @ M_train + alpha * np.eye(n_cols)
        b     = M_train.T @ y_hat_p
        theta = np.linalg.solve(A, b)   # (n_cols,)

        intercept_val = float(theta[0])
        alpha_vec     = theta[1:1 + d]      # additive coefficients in z-space
        beta_dict     = {(pi, pj): float(theta[1 + d + col])
                         for col, (pi, pj) in enumerate(pairs)}

        # ── local R² from training residuals ────────────────────────────────
        y_pred_p = M_train @ theta
        ss_res = float(np.sum((y_hat_p - y_pred_p) ** 2))
        ss_tot = float(np.sum((y_hat_p - y_hat_p.mean()) ** 2))
        r2_val = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        r2_val = float(np.clip(r2_val, 0.0, 1.0))

        # ── evaluate for all query samples in this partition ─────────────────
        Z_q = X_z_query[q_idx]   # (n_q, d)

        # additive contribution: α_i * z_i (in prediction units)
        main_arr[np.ix_(q_idx, np.arange(d))] = Z_q * alpha_vec[np.newaxis, :]

        intercepts[q_idx] = intercept_val
        local_r2[q_idx]   = r2_val

        # interaction contributions: β_ij * z_i * z_j
        for (pi, pj), beta_val in beta_dict.items():
            key = (pi, pj)
            if key not in inter_arr:
                inter_arr[key] = np.zeros(n)
            inter_arr[key][q_idx] = beta_val * Z_q[:, pi] * Z_q[:, pj]

    # ── wrap in named dicts ──────────────────────────────────────────────────
    main_named  = {feature_names[i]: main_arr[:, i] for i in range(d)}
    inter_named = {
        (feature_names[pi], feature_names[pj]): arr
        for (pi, pj), arr in inter_arr.items()
    }

    return ContributionFrame(
        X=X,
        main=main_named,
        interaction=inter_named,
        intercepts=intercepts,
        local_r2=local_r2,
        feature_names=list(feature_names),
    )


# ---------------------------------------------------------------------------
# ContributionFrame
# ---------------------------------------------------------------------------

@dataclass
class ContributionFrame:
    """
    Per-sample EBM-style feature contributions from a GeoXGB model.

    Attributes
    ----------
    X : ndarray, shape (n, d)
        Original feature values (not z-scores).
    main : dict[str, ndarray]
        Feature name → (n,) additive contributions in prediction units.
    interaction : dict[tuple[str,str], ndarray]
        (feat_i, feat_j) → (n,) interaction contributions (0 where inactive).
    intercepts : ndarray, shape (n,)
        Per-sample local polynomial intercept.
    local_r2 : ndarray, shape (n,)
        Ridge fit quality in each sample's partition.  Reliability weight.
    feature_names : list[str]
    """
    X:             np.ndarray
    main:          dict
    interaction:   dict
    intercepts:    np.ndarray
    local_r2:      np.ndarray
    feature_names: list

    # -- convenience ---------------------------------------------------------

    def _feat_index(self, name: str) -> int:
        return self.feature_names.index(name)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all arrays."""
        return {
            "feature_names": self.feature_names,
            "intercepts":    self.intercepts.tolist(),
            "local_r2":      self.local_r2.tolist(),
            "main": {k: v.tolist() for k, v in self.main.items()},
            "interaction": {
                f"{a}_x_{b}": v.tolist()
                for (a, b), v in self.interaction.items()
            },
        }

    def to_dataframe(self):
        """
        Return a pandas DataFrame with one column per contribution term.

        Requires pandas.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        cols = {}
        for feat in self.feature_names:
            cols[f"main_{feat}"] = self.main[feat]
        for (a, b), arr in self.interaction.items():
            cols[f"inter_{a}_x_{b}"] = arr
        cols["intercept"] = self.intercepts
        cols["local_r2"]  = self.local_r2
        return pd.DataFrame(cols)

    # -- plots ---------------------------------------------------------------

    def plot(
        self,
        feature,
        *,
        overlay=None,
        bandwidth="auto",
        ci=True,
        ci_alpha=0.20,
        interaction_slice="median",
        ax=None,
    ):
        """
        1-D smoothed main-effect curve for *feature*.

        Parameters
        ----------
        feature : str
            Primary feature to plot on the x-axis.
        overlay : list, optional
            List of feature names (str) or interaction tuples ``(a, b)`` to
            overlay on the same axes.
        bandwidth : 'auto' or float
            'auto' = range/20.  float = fraction of data range.
        ci : bool
            Show ±1σ shaded confidence band.
        ci_alpha : float
            Alpha for CI shading (0–1).
        interaction_slice : {'median', 'scatter'}
            When overlaying an interaction term: 'median' fixes the second
            feature at its median; 'scatter' plots each sample's value.
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        mpl = _require_matplotlib()
        fig, ax = _get_ax(ax, mpl)

        x_feat = self.X[:, self._feat_index(feature)]
        w = np.clip(self.local_r2, 1e-6, None)

        # primary curve
        grid, smooth, ci_std = _smooth_1d(
            x_feat, self.main[feature], w, bandwidth
        )
        ax.plot(grid, smooth, label=feature)
        if ci:
            ax.fill_between(grid, smooth - ci_std, smooth + ci_std,
                            alpha=ci_alpha)

        # overlays
        for term in (overlay or []):
            if isinstance(term, str):
                # another main effect
                grid2, sm2, cs2 = _smooth_1d(
                    x_feat, self.main[term], w, bandwidth
                )
                ax.plot(grid2, sm2, label=term)
                if ci:
                    ax.fill_between(grid2, sm2 - cs2, sm2 + cs2,
                                    alpha=ci_alpha)
            elif isinstance(term, (tuple, list)) and len(term) == 2:
                a, b = term
                key = (a, b) if (a, b) in self.interaction else (b, a)
                inter_vals = self.interaction.get(key, np.zeros(len(x_feat)))
                if interaction_slice == "scatter":
                    ax.scatter(x_feat, inter_vals, s=6, alpha=0.4,
                               label=f"{a}×{b}")
                else:
                    # 'median': treat as 1-D function of primary feature
                    grid2, sm2, cs2 = _smooth_1d(
                        x_feat, inter_vals, w, bandwidth
                    )
                    ax.plot(grid2, sm2, linestyle="--", label=f"{a}×{b}")
                    if ci:
                        ax.fill_between(grid2, sm2 - cs2, sm2 + cs2,
                                        alpha=ci_alpha)

        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.set_xlabel(feature)
        ax.set_ylabel("contribution (prediction units)")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_interaction(
        self,
        feat_a: str,
        feat_b: str,
        *,
        bandwidth="auto",
        show_points: bool = False,
        ax=None,
    ):
        """
        Continuous 2-D interaction heatmap (diverging colormap).

        Parameters
        ----------
        feat_a, feat_b : str
        bandwidth : 'auto' or float
        show_points : bool
            Scatter training samples on top of heatmap.
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        mpl = _require_matplotlib()
        fig, ax = _get_ax(ax, mpl)

        key = (feat_a, feat_b) if (feat_a, feat_b) in self.interaction \
              else (feat_b, feat_a)
        if key not in self.interaction:
            raise KeyError(
                f"No interaction found for ({feat_a!r}, {feat_b!r}). "
                "Try a lower min_pair_coop."
            )
        inter_vals = self.interaction[key]
        x_a = self.X[:, self._feat_index(feat_a)]
        x_b = self.X[:, self._feat_index(feat_b)]
        w   = np.clip(self.local_r2, 1e-6, None)

        ga, gb, Z = _smooth_2d(x_a, x_b, inter_vals, w, bandwidth)

        vmax = float(np.abs(Z).max()) or 1.0
        cm = ax.pcolormesh(
            gb, ga, Z,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )
        fig.colorbar(cm, ax=ax, label="interaction contribution")
        ax.contour(gb, ga, Z, levels=[0], colors="k", linewidths=0.8)
        if show_points:
            ax.scatter(x_b, x_a, s=4, c="k", alpha=0.3)
        ax.set_xlabel(feat_b)
        ax.set_ylabel(feat_a)
        ax.set_title(f"Interaction: {feat_a} × {feat_b}")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        return matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install geoxgb[plots]"
        ) from exc


def _get_ax(ax, mpl):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax
