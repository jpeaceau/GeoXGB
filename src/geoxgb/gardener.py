"""
geoxgb.gardener
===============

Post-hoc surgical editor for fitted GeoXGBClassifier / GeoXGBRegressor.

The Gardener wraps a fitted model and exposes two layers of tooling:

Manual tools (the pruning shears)
----------------------------------
  garden.adjust_leaf(tree_idx, leaf_id, delta)
      Shift a specific leaf's value by delta.

  garden.prune(tree_idx, leaf_id)
      Zero out a leaf's contribution entirely.

  garden.graft(X, residuals, n_rounds, learning_rate)
      Train correction trees on targeted samples and append to the ensemble.

  garden.rollback(n=1)
      Undo the last n operations.

  garden.reset()
      Restore the model to its original fitted state.

Automatic self-healing
-----------------------
  garden.diagnose(X, y)
      Scan all leaves for systematic bias. Returns a list of findings
      sorted by |mean_residual| descending.

  garden.heal(X_train, y_train, X_val, y_val, strategy="auto")
      Detect biased leaves, attempt corrections, validate on held-out data,
      commit only if beneficial. Iterates until convergence.

Reporting
---------
  garden.report()         — full audit trail of every edit
  garden.score(X, y)      — AUC (classifier) or R2 (regressor) on given data

Examples
--------
>>> from geoxgb import GeoXGBClassifier
>>> from geoxgb.gardener import Gardener
>>>
>>> model = GeoXGBClassifier(n_rounds=1000).fit(X_train, y_train)
>>> garden = Gardener(model)
>>>
>>> # Inspect biased leaves
>>> findings = garden.diagnose(X_train, y_train)
>>> print(f"Biased leaves: {len(findings)}")
>>>
>>> # Auto-heal with validation gate
>>> result = garden.heal(X_train, y_train, X_val, y_val)
>>> print(f"Improvement: {result['improvement']:+.4f}")
>>>
>>> # Predict with healed model
>>> proba = garden.predict_proba(X_test)
>>>
>>> # Inspect what changed
>>> garden.report()
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, r2_score
except ImportError:
    raise ImportError("scikit-learn is required for geoxgb.gardener")

from geoxgb._utils import _sigmoid


# ---------------------------------------------------------------------------
# Internal edit record
# ---------------------------------------------------------------------------

@dataclass
class _Edit:
    """Atomic record of a single model modification."""
    kind:    str
    desc:    str
    meta:    dict
    undo_fn: Callable

    def __repr__(self):
        return f"_Edit(kind={self.kind!r}, desc={self.desc!r})"


# ---------------------------------------------------------------------------
# Diagnostic finding
# ---------------------------------------------------------------------------

@dataclass
class LeafFinding:
    """A biased leaf identified by diagnose()."""
    tree_idx:         int
    leaf_id:          int
    n_samples:        int
    mean_residual:    float
    abs_mean_res:     float
    sign_consistency: float
    leaf_value:       float

    def __repr__(self):
        return (
            f"LeafFinding(tree={self.tree_idx}, leaf={self.leaf_id}, "
            f"n={self.n_samples}, mean_res={self.mean_residual:+.4f}, "
            f"sign_cons={self.sign_consistency:.2f})"
        )


# ---------------------------------------------------------------------------
# Gardener
# ---------------------------------------------------------------------------

class Gardener:
    """
    Post-hoc surgical editor for fitted GeoXGB models.

    Parameters
    ----------
    model : GeoXGBClassifier | GeoXGBRegressor
        A fitted GeoXGB model.  The original is never mutated — the Gardener
        works on a deep copy and maintains a full undo stack.
    random_state : int, default 42
        Seed for any correction models trained by graft() or heal().

    Notes
    -----
    The ``heal()`` method applies a validation gate: corrections are only
    committed if they improve the held-out score.  If a correction hurts,
    it is automatically rolled back.
    """

    # Default hyperparameter grid for heal()
    _DEFAULT_SURGERY_ALPHAS    = [0.1, 0.2, 0.3, 0.5]
    _DEFAULT_CORRECTION_ROUNDS = [50, 100, 200]
    _DEFAULT_CORRECTION_LR     = 0.05

    def __init__(self, model, random_state: int = 42):
        model._check_fitted()
        self._model    = copy.deepcopy(model)
        self._original = copy.deepcopy(model)
        self._edits: List[_Edit] = []
        self._rs       = random_state
        self._is_clf   = model._is_classifier

    # ------------------------------------------------------------------
    # Prediction interface (delegates to the modified copy)
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        """Probability estimates from the (possibly modified) model."""
        return self._model.predict_proba(X)

    def predict(self, X):
        """Regression predictions or class labels from the modified model."""
        return self._model.predict(X)

    def score(self, X, y) -> float:
        """AUC-ROC for classifiers, R² for regressors."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self._is_clf:
            return float(roc_auc_score(y, self.predict_proba(X)[:, 1]))
        return float(r2_score(y, self.predict(X)))

    # ------------------------------------------------------------------
    # Manual tools
    # ------------------------------------------------------------------

    def adjust_leaf(self, tree_idx: int, leaf_id: int, delta: float) -> None:
        """
        Shift a specific leaf value by ``delta``.

        Parameters
        ----------
        tree_idx : int
            Index into the boosting ensemble (0-indexed).
        leaf_id : int
            Node ID of the target leaf in that tree.
        delta : float
            Amount to add to the leaf value.  For classifiers this is in
            log-odds space; for regressors it is in the target's space.
        """
        tree    = self._model._trees[tree_idx]
        old_val = float(tree.tree_.value[leaf_id, 0, 0])
        tree.tree_.value[leaf_id, 0, 0] += delta

        def _undo(ti=tree_idx, li=leaf_id, v=old_val):
            self._model._trees[ti].tree_.value[li, 0, 0] = v

        self._edits.append(_Edit(
            kind    = "adjust_leaf",
            desc    = (f"tree[{tree_idx}] leaf[{leaf_id}] "
                       f"{old_val:+.4f} -> {old_val+delta:+.4f}"),
            meta    = {"tree_idx": tree_idx, "leaf_id": leaf_id,
                       "delta": delta, "old_val": old_val},
            undo_fn = _undo,
        ))

    def prune(self, tree_idx: int, leaf_id: int) -> None:
        """
        Zero out a leaf's contribution (set value to 0).

        Equivalent to removing this leaf's prediction from the ensemble
        for all samples that route to it.
        """
        tree    = self._model._trees[tree_idx]
        old_val = float(tree.tree_.value[leaf_id, 0, 0])
        self.adjust_leaf(tree_idx, leaf_id, -old_val)
        # Update last edit's kind for clarity
        self._edits[-1].kind = "prune"
        self._edits[-1].desc = f"tree[{tree_idx}] leaf[{leaf_id}] zeroed (was {old_val:+.4f})"

    def graft(
        self,
        X,
        residuals,
        n_rounds:      int   = 100,
        learning_rate: float = 0.05,
        max_depth:     int   = 3,
        refit_interval: int  = 10,
    ) -> int:
        """
        Train correction trees on targeted samples and append to ensemble.

        ``residuals`` should be the final residuals for samples in X:
          - classifier : ``y_true - sigmoid(raw_pred)``
          - regressor  : ``y_true - pred``

        Correction trees are trained as a regression on residuals and their
        predictions are added to the raw log-odds / regression output at the
        given learning rate.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        residuals : array-like (n_samples,)
        n_rounds : int
        learning_rate : float
        max_depth : int
        refit_interval : int

        Returns
        -------
        int — number of correction trees added
        """
        from geoxgb import GeoXGBRegressor

        X   = np.asarray(X, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64).ravel()

        n_before = len(self._model._trees)

        corr = GeoXGBRegressor(
            n_rounds       = n_rounds,
            learning_rate  = learning_rate,
            max_depth      = max_depth,
            refit_interval = refit_interval,
            random_state   = self._rs,
            auto_expand    = False,
        )
        corr.fit(X, res)

        self._model._trees.extend(corr._trees)
        self._model._lr_values.extend(corr._lr_values)
        n_added = len(corr._trees)

        def _undo(nb=n_before):
            del self._model._trees[nb:]
            del self._model._lr_values[nb:]

        self._edits.append(_Edit(
            kind    = "graft",
            desc    = (f"Grafted {n_added} correction trees "
                       f"(lr={learning_rate}, depth={max_depth}, "
                       f"n_samples={len(X)})"),
            meta    = {"n_added": n_added, "n_before": n_before,
                       "lr": learning_rate, "n_rounds": n_rounds},
            undo_fn = _undo,
        ))
        return n_added

    def rollback(self, n: int = 1) -> int:
        """
        Undo the last ``n`` operations.

        Returns the number of operations actually rolled back.
        """
        n = min(n, len(self._edits))
        for _ in range(n):
            edit = self._edits.pop()
            edit.undo_fn()
        return n

    def reset(self) -> None:
        """Restore the model to its original fitted state, clearing all edits."""
        self._model = copy.deepcopy(self._original)
        self._edits.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(
        self,
        X,
        y,
        min_samples:      int   = 20,
        bias_threshold:   float = 0.05,
        sign_consistency: float = 0.70,
    ) -> List[LeafFinding]:
        """
        Scan every leaf in the boosting ensemble for systematic bias.

        A leaf is flagged when:
          1. At least ``min_samples`` training samples land in it
          2. ``|mean(residual)| > bias_threshold``
          3. The fraction of same-sign residuals exceeds ``sign_consistency``

        Condition 3 distinguishes *systematic* over/under-prediction from
        high-variance noise: a leaf with mean_residual=0.05 but equal numbers
        of positive and negative residuals is noisy, not biased.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,)
        min_samples : int
        bias_threshold : float
        sign_consistency : float — fraction of same-sign residuals [0, 1]

        Returns
        -------
        list of LeafFinding, sorted by abs_mean_res descending
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self._is_clf:
            raw   = self._model._raw_predict(X)
            prob  = _sigmoid(raw)
            resid = y - prob
        else:
            pred  = self._model.predict(X)
            resid = y - pred

        findings: List[LeafFinding] = []

        for ti, tree in enumerate(self._model._trees):
            leaf_ids = tree.apply(X)
            for lid in np.unique(leaf_ids):
                mask = leaf_ids == lid
                n    = int(mask.sum())
                if n < min_samples:
                    continue
                r  = resid[mask]
                mr = float(r.mean())
                if abs(mr) < bias_threshold:
                    continue
                sc = float((np.sign(r) == np.sign(mr)).mean())
                if sc < sign_consistency:
                    continue
                findings.append(LeafFinding(
                    tree_idx         = ti,
                    leaf_id          = lid,
                    n_samples        = n,
                    mean_residual    = mr,
                    abs_mean_res     = abs(mr),
                    sign_consistency = sc,
                    leaf_value       = float(tree.tree_.value[lid, 0, 0]),
                ))

        findings.sort(key=lambda f: -f.abs_mean_res)
        return findings

    # ------------------------------------------------------------------
    # Auto self-healing
    # ------------------------------------------------------------------

    def heal(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        strategy:          Literal["surgery", "ensemble", "auto"] = "auto",
        min_samples:       int   = 20,
        bias_threshold:    float = 0.05,
        sign_consistency:  float = 0.70,
        surgery_alphas:    Optional[list] = None,
        correction_rounds: Optional[list] = None,
        correction_lr:     float = 0.05,
        max_iterations:    int   = 3,
        verbose:           bool  = True,
    ) -> dict:
        """
        Automatically detect biased leaves, attempt corrections, validate
        on held-out data, and commit only when beneficial.

        Parameters
        ----------
        X_train, y_train : training data used for residual analysis
        X_val,   y_val   : held-out data for validation gate
        strategy : "surgery" | "ensemble" | "auto"
            "surgery"  — leaf value adjustment only (fast, no retraining)
            "ensemble" — correction ensemble only (more powerful)
            "auto"     — surgery first; if insufficient, add correction ensemble
        min_samples : int
            Minimum leaf occupancy to flag as a candidate.
        bias_threshold : float
            |mean_residual| threshold to flag a leaf as biased.
        sign_consistency : float
            Fraction of same-sign residuals required to flag as systematic.
        surgery_alphas : list of float
            Fractions of mean_residual to try as correction magnitude.
        correction_rounds : list of int
            Correction ensemble sizes to try.
        correction_lr : float
            Learning rate for correction ensemble.
        max_iterations : int
            Number of diagnose → correct → validate cycles.
        verbose : bool

        Returns
        -------
        dict with keys: baseline_score, final_score, improvement, n_edits, history
        """
        if surgery_alphas   is None: surgery_alphas   = self._DEFAULT_SURGERY_ALPHAS
        if correction_rounds is None: correction_rounds = self._DEFAULT_CORRECTION_ROUNDS

        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()
        X_v  = np.asarray(X_val,   dtype=np.float64)
        y_v  = np.asarray(y_val,   dtype=np.float64).ravel()

        metric         = "AUC" if self._is_clf else "R2"
        baseline_score = self.score(X_v, y_v)
        history        = []

        if verbose:
            print(f"[Gardener] Baseline val {metric}: {baseline_score:.4f}")
            print(f"  Strategy={strategy}  max_iterations={max_iterations}")

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n  --- Iteration {iteration + 1} ---")

            findings = self.diagnose(
                X_tr, y_tr,
                min_samples      = min_samples,
                bias_threshold   = bias_threshold,
                sign_consistency = sign_consistency,
            )

            if not findings:
                if verbose:
                    print("  No biased leaves found. Model is clean.")
                break

            if verbose:
                bias_range = (findings[-1].abs_mean_res, findings[0].abs_mean_res)
                print(f"  Biased leaves: {len(findings)}  "
                      f"|bias| range [{bias_range[0]:.4f}, {bias_range[1]:.4f}]")

            current_score = self.score(X_v, y_v)
            improved      = False

            # ---- Surgery phase ----------------------------------------
            if strategy in ("surgery", "auto"):
                best_alpha = None
                best_score = current_score

                for alpha in surgery_alphas:
                    n_before = len(self._edits)
                    for f in findings:
                        self.adjust_leaf(f.tree_idx, f.leaf_id,
                                         alpha * f.mean_residual)
                    s = self.score(X_v, y_v)
                    # Always roll back for now; commit best at end
                    n_added = len(self._edits) - n_before
                    for _ in range(n_added):
                        e = self._edits.pop(); e.undo_fn()

                    if s > best_score:
                        best_score = s
                        best_alpha = alpha

                if best_alpha is not None:
                    for f in findings:
                        self.adjust_leaf(f.tree_idx, f.leaf_id,
                                         best_alpha * f.mean_residual)
                    new_score = self.score(X_v, y_v)
                    if verbose:
                        print(f"  Surgery (alpha={best_alpha}): "
                              f"{current_score:.4f} -> {new_score:.4f} "
                              f"({new_score - current_score:+.4f})")
                    history.append({"iteration": iteration + 1,
                                    "action": f"surgery(alpha={best_alpha})",
                                    "score": new_score})
                    current_score = new_score
                    improved      = True
                elif verbose:
                    print("  Surgery: no improvement found.")

            # ---- Ensemble phase ---------------------------------------
            if strategy in ("ensemble", "auto") and (
                strategy == "ensemble" or not improved
            ):
                # Residuals on current (possibly already-surgered) model
                if self._is_clf:
                    raw   = self._model._raw_predict(X_tr)
                    resid = y_tr - _sigmoid(raw)
                else:
                    resid = y_tr - self._model.predict(X_tr)

                # Re-diagnose so findings match current residuals
                findings_now = self.diagnose(
                    X_tr, y_tr,
                    min_samples      = min_samples,
                    bias_threshold   = bias_threshold,
                    sign_consistency = sign_consistency,
                )

                biased_mask = np.zeros(len(X_tr), dtype=bool)
                for f in findings_now:
                    leaf_ids = self._model._trees[f.tree_idx].apply(X_tr)
                    biased_mask |= (leaf_ids == f.leaf_id)

                X_c = X_tr[biased_mask]
                r_c = resid[biased_mask]

                if len(X_c) < 50:
                    if verbose:
                        print(f"  Correction ensemble: too few samples "
                              f"({len(X_c)}), skipping.")
                else:
                    best_ens_score  = current_score
                    best_ens_config = None

                    for n_corr in correction_rounds:
                        n_before = len(self._edits)
                        self.graft(X_c, r_c, n_rounds=n_corr,
                                   learning_rate=correction_lr)
                        s = self.score(X_v, y_v)
                        n_added = len(self._edits) - n_before
                        for _ in range(n_added):
                            e = self._edits.pop(); e.undo_fn()

                        if s > best_ens_score:
                            best_ens_score  = s
                            best_ens_config = n_corr

                    if best_ens_config is not None:
                        self.graft(X_c, r_c, n_rounds=best_ens_config,
                                   learning_rate=correction_lr)
                        new_score = self.score(X_v, y_v)
                        if verbose:
                            print(f"  Correction ensemble (rounds={best_ens_config}, "
                                  f"n={len(X_c)}): "
                                  f"{current_score:.4f} -> {new_score:.4f} "
                                  f"({new_score - current_score:+.4f})")
                        history.append({"iteration": iteration + 1,
                                        "action": (f"ensemble("
                                                   f"rounds={best_ens_config})"),
                                        "score": new_score})
                        current_score = new_score
                        improved      = True
                    elif verbose:
                        print("  Correction ensemble: no improvement found.")

            if not improved:
                if verbose:
                    print("  No improvement this iteration. Stopping.")
                break

        final_score = self.score(X_v, y_v)
        improvement = final_score - baseline_score

        if verbose:
            print(f"\n[Gardener] Healing complete.")
            print(f"  Baseline : {baseline_score:.4f}")
            print(f"  Final    : {final_score:.4f}")
            print(f"  Gain     : {improvement:+.4f}")
            print(f"  Edits    : {len(self._edits)}")

        return {
            "baseline_score": baseline_score,
            "final_score":    final_score,
            "improvement":    improvement,
            "n_edits":        len(self._edits),
            "history":        history,
        }

    # ------------------------------------------------------------------
    # Feature weight recommendation
    # ------------------------------------------------------------------

    def recommend_feature_weights(self, feature_names=None):
        """
        Recommend per-feature weights to correct geometry-gradient mismatch.

        Computes the ratio of boosting importance to partition importance for
        each feature.  Features where the gradient trees rely heavily but HVRT
        geometry underweights them get a weight > 1; features where geometry
        overweights them get a weight < 1.  The weights are normalized so their
        mean equals 1 (neutral baseline), making them a drop-in for
        ``GeoXGBClassifier(feature_weights=...)``.

        Parameters
        ----------
        feature_names : list of str, optional
            If None, uses ``["x0", "x1", ...]``.

        Returns
        -------
        dict {feature_name: weight}
            Weights ready to pass as ``feature_weights`` to a new GeoXGB model.
            Also accessible as a plain array via
            ``list(weights.values())``.

        Examples
        --------
        >>> garden = Gardener(model)
        >>> weights = garden.recommend_feature_weights(feature_names)
        >>> fw = list(weights.values())
        >>> new_model = GeoXGBClassifier(feature_weights=fw).fit(X, y)
        """
        n_features = self._model._n_features
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]

        boost_imp = self._model.feature_importances(feature_names)  # dict sorted desc

        part_hist = self._model.partition_feature_importances(feature_names)
        if not part_hist:
            return {f: 1.0 for f in feature_names}

        part_imp = part_hist[-1]["importances"]  # last refit round

        eps = 1e-6
        # Compute ratio for each feature in consistent order (feature_names order)
        raw = np.array([
            boost_imp.get(f, 0.0) / (part_imp.get(f, 0.0) + eps)
            for f in feature_names
        ])

        # Normalize: mean weight = 1.0 (no global scale change)
        weights = raw / (raw.mean() + eps)

        return {f: float(w) for f, w in zip(feature_names, weights)}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print the full audit trail of every edit made to the model."""
        metric = "AUC" if self._is_clf else "R2"
        print("=" * 60)
        print(f" Gardener Report — {self._model.__class__.__name__}")
        print("=" * 60)
        print(f"  Original trees : {len(self._original._trees)}")
        print(f"  Current trees  : {len(self._model._trees)}")
        print(f"  Total edits    : {len(self._edits)}")
        print()
        if not self._edits:
            print("  (no edits applied)")
        else:
            counts = {}
            for i, edit in enumerate(self._edits):
                counts[edit.kind] = counts.get(edit.kind, 0) + 1
                print(f"  [{i+1:4d}] {edit.kind:<18s}  {edit.desc}")
            print()
            print("  Edit summary:")
            for kind, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"    {kind:<20s}: {cnt}")
        print("=" * 60)

    def save(self, path) -> None:
        """
        Save the modified model (not the Gardener wrapper) to disk.

        Saves only the underlying GeoXGB model so it can be loaded with
        ``load_model()`` and used without the Gardener.
        """
        self._model.save(path)

    def __repr__(self) -> str:
        n_surgery = sum(1 for e in self._edits if e.kind in ("adjust_leaf", "prune"))
        n_grafts  = sum(1 for e in self._edits if e.kind == "graft")
        return (
            f"Gardener("
            f"model={self._model.__class__.__name__}, "
            f"n_trees={len(self._model._trees)}, "
            f"surgery_edits={n_surgery}, "
            f"grafts={n_grafts})"
        )
