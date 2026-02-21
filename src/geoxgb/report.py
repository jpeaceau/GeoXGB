"""
geoxgb.report
=============

Structured interpretability reports for fitted GeoXGB models.

All functions return dicts of JSON-serialisable Python types (str, int, float,
bool, list, dict — no numpy arrays, no model objects).  The caller decides how
to present the output; use print_report() for human-readable stdout.

Public API
----------
    model_report, noise_report, provenance_report, importance_report,
    partition_report, evolution_report, validation_report, compare_report,
    print_report

Design
------
- Structured data first: return dicts, never print directly.
- Layered depth: detail='summary'|'standard'|'full'.
- Uses only the model's public interpretability interface:
    model.feature_importances(), model.partition_trace(),
    model.sample_provenance(), model.noise_estimate(),
    model.partition_tree_rules(), model.partition_feature_importances().
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "model_report",
    "noise_report",
    "provenance_report",
    "importance_report",
    "partition_report",
    "evolution_report",
    "validation_report",
    "compare_report",
    "print_report",
]


# ===========================================================================
# Internal helpers
# ===========================================================================

def _f(v) -> float:
    return float(v)


def _i(v) -> int:
    return int(v)


def _rank_desc(a: np.ndarray) -> np.ndarray:
    """1-based descending ranks; ties resolved by average rank."""
    n = len(a)
    order = np.argsort(-a)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (numpy-only implementation)."""
    n = len(a)
    if n < 2:
        return 0.0
    ra = _rank_desc(a)
    rb = _rank_desc(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom < 1e-12:
        return 1.0 if np.allclose(ra, rb) else 0.0
    return _f(np.dot(ra, rb) / denom)


def _tree_depth_from_rules(text: str) -> int:
    """Infer tree depth by counting pipe indentation in export_text output."""
    max_depth = 0
    for line in text.split("\n"):
        depth = line.count("|   ")
        if depth > max_depth:
            max_depth = depth
    return max_depth


def _noise_assessment(mod: float) -> str:
    if mod > 0.7:
        return "clean"
    if mod >= 0.3:
        return "moderate"
    return "noisy"


def _trend(values: list) -> str:
    """Classify a sequence as 'stable', 'decreasing', or 'increasing'."""
    if len(values) < 2:
        return "stable"
    diff = values[-1] - values[0]
    span = max(abs(v) for v in values) + 1e-12
    if abs(diff / span) < 0.05:
        return "stable"
    return "decreasing" if diff < 0 else "increasing"


def _model_type(model) -> str:
    return type(model).__name__


def _is_classifier(model) -> bool:
    return "Classifier" in _model_type(model)


def _default_names(model, feature_names) -> list[str]:
    if feature_names is not None:
        return list(feature_names)
    return [f"x{i}" for i in range(model._n_features)]


def _performance_metrics(model, X_test, y_test) -> dict:
    X_test = np.asarray(X_test, dtype=np.float64)
    if _is_classifier(model):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        n_classes = _i(y_proba.shape[1])
        accuracy = _f(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
        # log_loss: map labels to class indices
        classes = list(model._classes)
        y_arr = np.asarray(y_test)
        y_idx = np.array([classes.index(lab) for lab in y_arr])
        probs = np.clip(y_proba[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
        log_loss_val = _f(-np.mean(np.log(probs)))
        return {"accuracy": accuracy, "log_loss": log_loss_val, "n_classes": n_classes}
    else:
        y_pred = model.predict(X_test)
        y_true = np.asarray(y_test, dtype=np.float64)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = _f(1.0 - ss_res / (ss_tot + 1e-12))
        rmse = _f(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"r2": r2, "rmse": rmse}


# ===========================================================================
# Public API
# ===========================================================================

def noise_report(model) -> dict:
    """
    How noisy is the data, and how did GeoXGB respond?

    Returns
    -------
    dict with keys:
        initial_modulation, assessment, final_modulation,
        modulation_trend, effective_reduce_ratio, interpretation
    """
    trace = model.partition_trace()
    mods = [e["noise_modulation"] for e in trace]
    init_mod = _f(mods[0])
    final_mod = _f(mods[-1])
    trend = _trend(mods)
    assessment = _noise_assessment(init_mod)

    prov = model.sample_provenance()
    eff_reduce = _f(prov["reduced_n"] / max(prov["original_n"], 1))

    if assessment == "clean":
        interp = "Clean data detected. Full geometric resampling applied."
    elif assessment == "moderate":
        interp = (
            "Moderate noise detected. "
            "Resampling partially dampened to preserve signal."
        )
    else:
        interp = (
            "High noise detected. Resampling mostly suppressed — "
            "model operated near-vanilla to avoid amplifying noise."
        )

    if trend == "decreasing":
        interp += (
            " Noise modulation decreased across refits, indicating the model"
            " progressively captured signal and residuals became noise-dominated."
            " This is expected healthy behaviour."
        )

    return {
        "initial_modulation": init_mod,
        "assessment": assessment,
        "final_modulation": final_mod,
        "modulation_trend": trend,
        "effective_reduce_ratio": eff_reduce,
        "interpretation": interp,
    }


def provenance_report(model, detail: str = "standard") -> dict:
    """
    Where did the training samples come from?

    Returns
    -------
    dict with keys:
        original_n, reduced_n, expanded_n, total_training, reduction_ratio,
        efficiency.  detail='standard' adds per_partition.
        detail='full' adds history.
    """
    prov = model.sample_provenance()
    orig_n    = _i(prov["original_n"])
    reduced_n = _i(prov["reduced_n"])
    expanded_n = _i(prov["expanded_n"])
    total     = _i(prov["total_training"])
    ratio     = _f(prov["reduction_ratio"])
    pct       = round(ratio * 100, 1)

    out: dict[str, Any] = {
        "original_n":     orig_n,
        "reduced_n":      reduced_n,
        "expanded_n":     expanded_n,
        "total_training": total,
        "reduction_ratio": ratio,
        "efficiency": f"{pct}% of original data used for training",
    }

    if detail in ("standard", "full"):
        trace = model.partition_trace()
        last_parts = trace[-1]["partitions"]
        out["per_partition"] = [
            {
                "id":          _i(p["id"]),
                "size":        _i(p["size"]),
                "mean_abs_z":  _f(p["mean_abs_z"]),
                "variance":    _f(p["variance"]),
            }
            for p in last_parts
        ]

    if detail == "full":
        trace = model.partition_trace()
        out["history"] = [
            {
                "round":            _i(e["round"]),
                "n_reduced":        _i(e["n_reduced"]),
                "n_expanded":       _i(e["n_expanded"]),
                "total":            _i(e["n_samples"]),
                "noise_modulation": _f(e["noise_modulation"]),
            }
            for e in trace
        ]

    return out


def importance_report(
    model,
    feature_names=None,
    ground_truth: dict | None = None,
    detail: str = "standard",
) -> dict:
    """
    Which features matter, and where?

    Parameters
    ----------
    ground_truth : dict, optional
        Keys: 'signal_features' (list[int]), 'noise_features' (list[int]).

    Returns
    -------
    dict with keys:
        boosting_importance, partition_importance, agreement, interpretation.
        detail='standard' adds top_boosting, top_partition, divergent_features.
        Adds 'validation' key if ground_truth provided.
    """
    names = _default_names(model, feature_names)

    boost_imp: dict[str, float] = model.feature_importances(names)
    pfi = model.partition_feature_importances(names)
    # Use the first resample event as representative geometry
    part_imp: dict[str, float] = pfi[0]["importances"] if pfi else {}

    # Ensure both dicts cover all feature names
    for n in names:
        boost_imp.setdefault(n, 0.0)
        part_imp.setdefault(n, 0.0)

    boost_arr = np.array([boost_imp[n] for n in names])
    part_arr  = np.array([part_imp[n]  for n in names])
    agreement = _spearman(boost_arr, part_arr)

    if abs(agreement) > 0.7:
        interp = (
            "Strong agreement between boosting and partition importance. "
            "Features that predict y also define the data geometry — "
            "the data has consistent structure."
        )
    elif abs(agreement) > 0.3:
        interp = (
            "Moderate agreement between boosting and partition importance. "
            "Some features drive geometry but not prediction, suggesting "
            "regime-based structure in the data."
        )
    else:
        interp = (
            "Low agreement between boosting and partition importance. "
            "The features that define geometry are largely distinct from those "
            "that predict y — strong regime-switching or context-dependent "
            "behaviour is likely."
        )

    out: dict[str, Any] = {
        "boosting_importance":  {k: _f(v) for k, v in boost_imp.items()},
        "partition_importance": {k: _f(v) for k, v in part_imp.items()},
        "agreement": _f(agreement),
        "interpretation": interp,
    }

    if detail in ("standard", "full"):
        top_n = 5
        boost_sorted = sorted(boost_imp.items(), key=lambda x: -x[1])
        part_sorted  = sorted(part_imp.items(),  key=lambda x: -x[1])
        out["top_boosting"]  = [[n, _f(v)] for n, v in boost_sorted[:top_n]]
        out["top_partition"] = [[n, _f(v)] for n, v in part_sorted[:top_n]]

        boost_rank = {n: i + 1 for i, (n, _) in enumerate(boost_sorted)}
        part_rank  = {n: i + 1 for i, (n, _) in enumerate(part_sorted)}
        divergent = []
        for n in names:
            br = boost_rank.get(n, len(names))
            pr = part_rank.get(n, len(names))
            if abs(br - pr) > 3:
                divergent.append({
                    "feature":       n,
                    "boosting_rank": br,
                    "partition_rank": pr,
                    "rank_diff":     abs(br - pr),
                })
        divergent.sort(key=lambda x: -x["rank_diff"])
        out["divergent_features"] = divergent

    if ground_truth is not None:
        sig_idx   = ground_truth.get("signal_features", [])
        noise_idx = ground_truth.get("noise_features",  [])
        sig_names   = [names[i] for i in sig_idx   if i < len(names)]
        noise_names = [names[i] for i in noise_idx if i < len(names)]

        def _pct(imp_dict: dict, feat_list: list[str]) -> float:
            total = sum(imp_dict.values()) + 1e-12
            return _f(100 * sum(imp_dict.get(n, 0.0) for n in feat_list) / total)

        out["validation"] = {
            "signal_pct_partition":    _pct(part_imp,  sig_names),
            "noise_pct_partition":     _pct(part_imp,  noise_names),
            "signal_pct_boosting":     _pct(boost_imp, sig_names),
            "noise_pct_boosting":      _pct(boost_imp, noise_names),
            "partition_ignores_noise": _pct(part_imp,  noise_names) < 10.0,
            "boosting_ignores_noise":  _pct(boost_imp, noise_names) < 30.0,
        }

    return out


def partition_report(
    model,
    round_idx: int = 0,
    feature_names=None,
    detail: str = "standard",
) -> dict:
    """
    What does the partition structure look like at a given resample event?

    Parameters
    ----------
    round_idx : int
        Index into resample history (0 = initial fit).

    Returns
    -------
    dict with keys:
        round, n_partitions, noise_modulation, total_samples,
        tree_rules, tree_depth, tree_feature_importances.
        detail='standard' adds partitions list and size_distribution.
        detail='full' adds tree_rules_full.
    """
    names = _default_names(model, feature_names)
    trace = model.partition_trace()
    event = trace[round_idx]
    partitions = event["partitions"]
    total = _i(event["n_samples"])

    rules = model.partition_tree_rules(round_idx=round_idx)
    tree_depth = _tree_depth_from_rules(rules)

    pfi = model.partition_feature_importances(names)
    if round_idx < len(pfi):
        tree_fi = {k: _f(v) for k, v in pfi[round_idx]["importances"].items()}
    else:
        tree_fi = {k: _f(v) for k, v in pfi[0]["importances"].items()}

    out: dict[str, Any] = {
        "round":                   _i(event["round"]),
        "n_partitions":            _i(len(partitions)),
        "noise_modulation":        _f(event["noise_modulation"]),
        "total_samples":           total,
        "tree_rules":              rules,
        "tree_depth":              _i(tree_depth),
        "tree_feature_importances": tree_fi,
    }

    if detail in ("standard", "full"):
        sizes = [_i(p["size"]) for p in partitions]
        min_sz = min(sizes) if sizes else 1
        max_sz = max(sizes) if sizes else 1
        imbalance = _f(max_sz / max(min_sz, 1))

        if imbalance < 3:
            imb_interp = "Fairly balanced partitions."
        elif imbalance <= 10:
            imb_interp = "Moderate imbalance — some regions denser than others."
        else:
            imb_interp = (
                "Highly imbalanced — data has strong density structure. "
                "FPS reduction is actively correcting this."
            )

        out["partitions"] = [
            {
                "id":          _i(p["id"]),
                "size":        _i(p["size"]),
                "mean_abs_z":  _f(p["mean_abs_z"]),
                "variance":    _f(p["variance"]),
                "pct_of_total": _f(round(100 * p["size"] / max(total, 1), 2)),
            }
            for p in partitions
        ]
        out["size_distribution"] = {
            "min":              min_sz,
            "max":              max_sz,
            "median":           _f(float(np.median(sizes))),
            "std":              _f(float(np.std(sizes))),
            "imbalance_ratio":  imbalance,
            "imbalance_interpretation": imb_interp,
        }

    if detail == "full":
        out["tree_rules_full"] = rules

    return out


def evolution_report(
    model,
    feature_names=None,
    detail: str = "standard",
) -> dict:
    """
    How did the partitioning evolve across refits?

    Returns
    -------
    dict with keys:
        n_resamples, refit_interval, rounds.
        detail='standard' adds noise_trend, sample_trend, interpretation.
        detail='full' adds importance_evolution.
    """
    names = _default_names(model, feature_names)
    trace = model.partition_trace()
    n_resamples = _i(len(trace))

    rounds = [
        {
            "round":            _i(e["round"]),
            "noise_modulation": _f(e["noise_modulation"]),
            "n_samples":        _i(e["n_samples"]),
            "n_reduced":        _i(e["n_reduced"]),
            "n_expanded":       _i(e["n_expanded"]),
            "n_partitions":     _i(len(e["partitions"])),
        }
        for e in trace
    ]

    partition_counts = [r["n_partitions"] for r in rounds]
    out: dict[str, Any] = {
        "n_resamples":    n_resamples,
        "refit_interval": model.refit_interval,
        "rounds":         rounds,
        "partition_stability": {
            "min_partitions": min(partition_counts) if partition_counts else 0,
            "max_partitions": max(partition_counts) if partition_counts else 0,
            "changed": len(set(partition_counts)) > 1,
            "interpretation": (
                "Partition count stable across all refits."
                if len(set(partition_counts)) <= 1
                else f"Partition count varied from {min(partition_counts)} to "
                     f"{max(partition_counts)} across refits, indicating the "
                     f"residual structure evolved during training."
            ),
        },
    }

    if detail in ("standard", "full") and n_resamples >= 1:
        mods    = [e["noise_modulation"] for e in trace]
        samples = [e["n_samples"] for e in trace]
        noise_dir  = _trend(mods)
        sample_dir = _trend(samples)

        out["noise_trend"] = {
            "direction": noise_dir,
            "start":     _f(mods[0]),
            "end":       _f(mods[-1]),
            "delta":     _f(mods[-1] - mods[0]),
        }
        out["sample_trend"] = {
            "direction": sample_dir,
            "start":     _i(samples[0]),
            "end":       _i(samples[-1]),
        }

        if noise_dir == "decreasing" and sample_dir == "increasing":
            interp = (
                "Model progressively learned signal. Later refits found less"
                " structure in residuals, indicating convergence."
            )
        elif all(abs(m - 1.0) < 0.05 for m in mods) and sample_dir == "stable":
            interp = (
                "Consistent structural signal across all refits. "
                "Data has persistent geometry that does not collapse with learning."
            )
        elif all(m < 0.1 for m in mods):
            interp = (
                "Noise dominated from the start. "
                "Model operated in near-vanilla mode throughout."
            )
        else:
            parts_interp = []
            if noise_dir != "stable":
                parts_interp.append(
                    f"Noise modulation {noise_dir} "
                    f"({mods[0]:.3f} -> {mods[-1]:.3f})."
                )
            if sample_dir != "stable":
                parts_interp.append(
                    f"Training set size {sample_dir} "
                    f"({samples[0]:,} -> {samples[-1]:,})."
                )
            interp = " ".join(parts_interp) if parts_interp else "Mixed evolution observed."

        out["interpretation"] = interp

        # UPDATE-010: importance drift across refits
        pfi = model.partition_feature_importances(names)
        if len(pfi) >= 2 and feature_names is not None:
            first_imp = pfi[0]["importances"]
            last_imp = pfi[-1]["importances"]

            if first_imp and last_imp:
                drifts = []
                for fn in names:
                    v0 = first_imp.get(fn, 0.0)
                    v1 = last_imp.get(fn, 0.0)
                    if abs(v1 - v0) > 0.05:
                        drifts.append({
                            "feature": fn,
                            "round_0": round(v0, 4),
                            "final_round": round(v1, 4),
                            "delta": round(v1 - v0, 4),
                        })

                out["importance_drift"] = {
                    "n_drifted": len(drifts),
                    "features": sorted(drifts, key=lambda x: -abs(x["delta"])),
                    "interpretation": (
                        "No meaningful partition importance drift detected."
                        if not drifts
                        else f"{len(drifts)} features showed meaningful drift in "
                             f"partition importance across refits."
                    ),
                }

    if detail == "full":
        pfi = model.partition_feature_importances(names)
        out["importance_evolution"] = [
            {
                "round":       e["round"],
                "importances": {k: _f(v) for k, v in e["importances"].items()},
            }
            for e in pfi
        ]

    return out


def validation_report(
    model,
    X_train,
    y_train,
    feature_names=None,
    ground_truth: dict | None = None,
) -> dict:
    """
    Validate that model decisions align with known data mechanisms.

    Parameters
    ----------
    ground_truth : dict, optional
        Any subset of: 'signal_features', 'noise_features',
        'cluster_sizes', 'mechanism'.

    Returns
    -------
    dict with keys: mechanism, checks, overall_pass, summary.
    """
    names = _default_names(model, feature_names)
    gt = ground_truth or {}
    mechanism = gt.get("mechanism", "not specified")
    checks: list[dict] = []

    # Gather imports once
    imp = importance_report(model, names, ground_truth=gt if gt else None)
    prov = model.sample_provenance()
    nr = noise_report(model)
    init_mod = nr["initial_modulation"]

    # ------------------------------------------------------------------
    # Check 1: partition_ignores_noise
    # ------------------------------------------------------------------
    if "signal_features" in gt and "noise_features" in gt:
        noise_names = [names[i] for i in gt["noise_features"] if i < len(names)]
        part_imp = imp["partition_importance"]
        noise_pct = _f(
            100 * sum(part_imp.get(n, 0) for n in noise_names)
            / max(sum(part_imp.values()), 1e-12)
        )
        checks.append({
            "name":   "partition_ignores_noise",
            "passed": noise_pct < 10.0,
            "detail": (
                f"Noise features hold {noise_pct:.1f}% of partition importance "
                f"(threshold: <10%)."
            ),
            "values": {"noise_pct_partition": _f(noise_pct)},
        })

    # ------------------------------------------------------------------
    # Check 2: boosting_deprioritises_noise
    # ------------------------------------------------------------------
    if "signal_features" in gt and "noise_features" in gt:
        noise_names = [names[i] for i in gt["noise_features"] if i < len(names)]
        boost_imp = imp["boosting_importance"]
        noise_pct_b = _f(
            100 * sum(boost_imp.get(n, 0) for n in noise_names)
            / max(sum(boost_imp.values()), 1e-12)
        )
        checks.append({
            "name":   "boosting_deprioritises_noise",
            "passed": noise_pct_b < 30.0,
            "detail": (
                f"Noise features hold {noise_pct_b:.1f}% of boosting importance "
                f"(threshold: <30%)."
            ),
            "values": {"noise_pct_boosting": _f(noise_pct_b)},
        })

    # ------------------------------------------------------------------
    # Check 3: fps_targets_dense
    # ------------------------------------------------------------------
    trace = model.partition_trace()
    if trace and trace[0]["partitions"]:
        parts = trace[0]["partitions"]
        sizes      = [p["size"]       for p in parts]
        mean_abs_z = [p["mean_abs_z"] for p in parts]
        largest_idx = int(np.argmax(sizes))
        median_z    = float(np.median(mean_abs_z))
        fps_correct = mean_abs_z[largest_idx] <= median_z
        checks.append({
            "name":   "fps_targets_dense",
            "passed": fps_correct,
            "detail": (
                f"Largest partition ({sizes[largest_idx]:,} samples) has "
                f"mean_abs_z={mean_abs_z[largest_idx]:.3f} "
                f"({'below' if fps_correct else 'above'} median {median_z:.3f}). "
                + (
                    "Dense, structurally similar region — correct FPS target."
                    if fps_correct else
                    "Structurally extreme — FPS allocation may be suboptimal."
                )
            ),
            "values": {
                "largest_partition_size":        _i(sizes[largest_idx]),
                "largest_partition_mean_abs_z":  _f(mean_abs_z[largest_idx]),
                "median_mean_abs_z":             _f(median_z),
            },
        })

    # ------------------------------------------------------------------
    # Check 4: kde_targets_sparse
    # ------------------------------------------------------------------
    if model.expand_ratio > 0:
        expanded_n = _i(prov["expanded_n"])
        checks.append({
            "name":   "kde_targets_sparse",
            "passed": expanded_n > 0,
            "detail": f"{expanded_n:,} samples generated by KDE expansion.",
            "values": {"expanded_n": expanded_n},
        })

    # ------------------------------------------------------------------
    # Check 5: noise_correctly_assessed
    # ------------------------------------------------------------------
    if "mechanism" in gt:
        mech_lower = mechanism.lower()
        is_noisy = any(w in mech_lower for w in ["noise", "noisy", "random"])
        if is_noisy:
            passed_noise = init_mod < 0.5
            noise_detail = (
                f"Mechanism suggests noisy data; modulation={init_mod:.3f} "
                + ("(correctly low)." if passed_noise else "(unexpectedly high).")
            )
        else:
            passed_noise = init_mod >= 0.5
            noise_detail = (
                f"Mechanism suggests structured data; modulation={init_mod:.3f} "
                + ("(correctly high)." if passed_noise else "(unexpectedly low).")
            )
        checks.append({
            "name":   "noise_correctly_assessed",
            "passed": passed_noise,
            "detail": noise_detail,
            "values": {"initial_modulation": _f(init_mod)},
        })

    # ------------------------------------------------------------------
    # Check 6: reduction_conservative_under_noise
    # ------------------------------------------------------------------
    if nr["assessment"] == "noisy":
        ratio = _f(prov["reduction_ratio"])
        checks.append({
            "name":   "reduction_conservative_under_noise",
            "passed": ratio > 0.85,
            "detail": (
                f"Data detected as noisy; reduction_ratio={ratio:.3f} "
                + (
                    "(conservatively kept most samples)."
                    if ratio > 0.85 else
                    "(may have removed too many samples)."
                )
            ),
            "values": {"reduction_ratio": ratio},
        })

    n_passed = sum(c["passed"] for c in checks)
    n_total  = len(checks)
    all_pass = all(c["passed"] for c in checks)

    return {
        "mechanism":    mechanism,
        "checks":       checks,
        "overall_pass": all_pass,
        "summary": (
            f"{n_passed}/{n_total} checks passed."
            + (" All good." if all_pass else " Review failing checks.")
        ),
    }


def compare_report(
    model,
    baseline_scores: dict,
    feature_names=None,
) -> dict:
    """
    Compare GeoXGB against a baseline model.

    Parameters
    ----------
    baseline_scores : dict
        Baseline metrics.  Recognised keys:
            'auc' | 'r2' | 'accuracy'  — baseline score
            'geoxgb_auc' | 'geoxgb_r2' | 'geoxgb_accuracy' — GeoXGB score
            'time'        — baseline wall time (s)
            'geoxgb_time' — GeoXGB wall time (s)
            'n_samples_used' — baseline training sample count

    Returns
    -------
    dict with keys:
        geoxgb, baseline, delta_score, sample_efficiency, interpretation.
        delta_time added if both times provided.
    """
    prov   = model.sample_provenance()
    geo_n  = _i(prov["total_training"])
    orig_n = _i(prov["original_n"])

    # Determine score metric
    score_key = None
    for k in ("auc", "r2", "accuracy"):
        if k in baseline_scores:
            score_key = k
            break
    if score_key is None:
        score_key = next(
            (k for k in baseline_scores if not k.startswith("geoxgb_")
             and k not in ("time", "n_samples_used")),
            None,
        )

    base_score = _f(baseline_scores.get(score_key, 0.0)) if score_key else 0.0
    geo_score  = _f(
        baseline_scores.get(f"geoxgb_{score_key}", base_score)
        if score_key else 0.0
    )

    geo_time  = baseline_scores.get("geoxgb_time")
    base_time = baseline_scores.get("time")
    base_n    = _i(baseline_scores.get("n_samples_used", orig_n))

    delta_score = _f(geo_score - base_score)
    # Sample efficiency: score-per-sample ratio of GeoXGB vs baseline
    sample_eff  = _f(
        (geo_score / max(base_score, 1e-12))
        / (geo_n / max(base_n, 1))
    )

    geo_pct      = round(100 * geo_n / max(base_n, 1), 1)
    diff_str     = f"{abs(delta_score):.4f} {'higher' if delta_score > 0 else 'lower'}"
    metric_label = (score_key or "score").upper()
    interp = (
        f"GeoXGB achieved {metric_label}={geo_score:.4f} vs "
        f"baseline {base_score:.4f} ({diff_str}) "
        f"using {geo_pct}% of training data with full sample provenance."
    )

    if geo_time is not None and base_time is not None:
        ratio = float(geo_time) / max(float(base_time), 1e-6)
        interp += (
            f" Wall time {ratio:.1f}x "
            + ("slower" if ratio > 1 else "faster") + "."
        )

    geo_entry: dict = {"score": geo_score, "n_samples": geo_n}
    if geo_time is not None:
        geo_entry["time"] = _f(geo_time)

    base_entry: dict = {"score": base_score, "n_samples": base_n}
    if base_time is not None:
        base_entry["time"] = _f(base_time)

    out: dict[str, Any] = {
        "geoxgb":           geo_entry,
        "baseline":         base_entry,
        "delta_score":      delta_score,
        "sample_efficiency": sample_eff,
        "interpretation":   interp,
    }
    if geo_time is not None and base_time is not None:
        out["delta_time"] = _f(float(geo_time) - float(base_time))

    return out


def model_report(
    model,
    X_test=None,
    y_test=None,
    feature_names=None,
    detail: str = "standard",
) -> dict:
    """
    Top-level report combining all sub-reports.

    Parameters
    ----------
    detail : 'summary' | 'standard' | 'full'
        Controls depth of sub-reports included.

    Returns
    -------
    dict with keys:
        model_type, n_rounds, n_trees, n_resamples,
        performance (if X_test/y_test provided),
        noise, provenance, importance,
        partitions (standard/full only),
        evolution (full only).
    """
    out: dict[str, Any] = {
        "model_type":  _model_type(model),
        "n_rounds":    _i(model.n_rounds),
        "n_trees":     _i(model.n_trees),
        "n_resamples": _i(model.n_resamples),
    }

    if X_test is not None and y_test is not None:
        out["performance"] = _performance_metrics(model, X_test, y_test)

    out["noise"]      = noise_report(model)
    out["provenance"] = provenance_report(model, detail=detail)
    out["importance"] = importance_report(model, feature_names=feature_names, detail=detail)

    if detail in ("standard", "full"):
        out["partitions"] = partition_report(
            model, feature_names=feature_names, detail=detail
        )

    if detail == "full":
        out["evolution"] = evolution_report(
            model, feature_names=feature_names, detail=detail
        )

    return out


# ===========================================================================
# print_report
# ===========================================================================

_BAR_WIDTH      = 28
_BAR_CHAR       = "#"
_IMPORTANCE_KEYS = frozenset({
    "boosting_importance", "partition_importance",
    "tree_feature_importances", "importances",
})
_PARTITION_LIST_KEYS = frozenset({"partitions", "per_partition"})
_CHECK_LIST_KEYS     = frozenset({"checks"})


def _bar(val: float, max_val: float) -> str:
    if max_val < 1e-12:
        return ""
    n = int(round(_BAR_WIDTH * val / max_val))
    return _BAR_CHAR * n


def _print_importance_dict(d: dict, prefix: str) -> None:
    max_val = max(d.values()) if d else 1.0
    for name, val in d.items():
        b = _bar(val, max_val)
        print(f"{prefix}  {name:<22s} {b:<{_BAR_WIDTH}s} {val:.4f}")


def _print_partition_table(parts: list[dict], prefix: str) -> None:
    if not parts:
        return
    keys = list(parts[0].keys())
    widths = {k: max(len(k), max(len(str(p.get(k, ""))) for p in parts))
              for k in keys}
    hdr = "  ".join(k.ljust(widths[k]) for k in keys)
    sep = "-" * len(hdr)
    print(f"{prefix}  {hdr}")
    print(f"{prefix}  {sep}")
    for p in parts:
        row = "  ".join(str(p.get(k, "")).ljust(widths[k]) for k in keys)
        print(f"{prefix}  {row}")


def _print_checks(checks: list[dict], prefix: str) -> None:
    for c in checks:
        symbol = "[PASS]" if c.get("passed") else "[FAIL]"
        print(f"{prefix}  {symbol}  {c.get('name', '')}")
        detail = c.get("detail", "")
        if detail:
            print(f"{prefix}        {detail}")


def _print_value(key: str, val: Any, indent: int) -> None:
    prefix = "  " * indent

    if isinstance(val, dict):
        print(f"{prefix}{key}:")
        if key in _IMPORTANCE_KEYS:
            _print_importance_dict(val, prefix)
        else:
            for k2, v2 in val.items():
                _print_value(k2, v2, indent + 1)

    elif isinstance(val, list):
        if not val:
            print(f"{prefix}{key}: []")
        elif key in _CHECK_LIST_KEYS:
            print(f"{prefix}{key}:")
            _print_checks(val, prefix)
        elif key in _PARTITION_LIST_KEYS:
            print(f"{prefix}{key} ({len(val)} partitions):")
            if val and isinstance(val[0], dict):
                _print_partition_table(val, prefix)
        elif val and isinstance(val[0], list) and len(val[0]) == 2:
            # top_boosting / top_partition: [[name, float], ...]
            print(f"{prefix}{key}:")
            max_val = max(x[1] for x in val) if val else 1.0
            for name, v in val:
                b = _bar(v, max_val)
                print(f"{prefix}  {name:<22s} {b:<{_BAR_WIDTH}s} {v:.4f}")
        elif val and isinstance(val[0], dict):
            print(f"{prefix}{key} ({len(val)} entries):")
            for i, item in enumerate(val):
                print(f"{prefix}  [{i}]")
                for k2, v2 in item.items():
                    _print_value(k2, v2, indent + 2)
        else:
            print(f"{prefix}{key}: {val}")

    elif isinstance(val, float):
        if math.isnan(val):
            print(f"{prefix}{key}: n/a")
        else:
            print(f"{prefix}{key}: {val:.4f}")

    elif isinstance(val, str) and "\n" in val:
        print(f"{prefix}{key}:")
        for line in val.split("\n"):
            print(f"{prefix}  {line}")

    else:
        print(f"{prefix}{key}: {val}")


def print_report(report: dict, title: str | None = None) -> None:
    """
    Pretty-print any report dict to stdout.

    Formatting rules
    ----------------
    - Importance dicts / top lists  -> horizontal bar chart (#-blocks)
    - Partition lists               -> aligned column table
    - Validation checks             -> [PASS] / [FAIL] with detail text
    - Multi-line strings            -> indented line-by-line
    - Nested dicts                  -> indented sections
    - Floats                        -> 4 decimal places
    - NaN floats                    -> 'n/a'
    """
    sep = "=" * 64
    if title:
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
    for k, v in report.items():
        _print_value(k, v, indent=0)
    print()
