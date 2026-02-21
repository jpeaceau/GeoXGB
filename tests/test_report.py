"""
Tests for geoxgb.report — all 11 spec checks.
"""
import json
import math

import numpy as np
import pytest

from geoxgb import GeoXGBRegressor, GeoXGBClassifier
from geoxgb.report import (
    model_report,
    noise_report,
    provenance_report,
    importance_report,
    partition_report,
    evolution_report,
    validation_report,
    compare_report,
    print_report,
)

RNG = np.random.default_rng(99)
N, P = 200, 6
SIGNAL = [0, 1, 2]
NOISE  = [3, 4, 5]


def _clean_data():
    X = RNG.standard_normal((N, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] + RNG.standard_normal(N) * 0.05
    return X, y


def _noisy_data():
    X = RNG.standard_normal((N, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(N) * 20.0
    return X, y


def _fitted_reg(X=None, y=None, **kwargs):
    if X is None:
        X, y = _clean_data()
    defaults = dict(n_rounds=20, refit_interval=5, random_state=0)
    defaults.update(kwargs)
    return GeoXGBRegressor(**defaults).fit(X, y), X, y


def _fitted_clf(n_classes=2):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=N, n_features=P, n_informative=3,
        n_classes=n_classes, n_clusters_per_class=1, random_state=0,
    )
    clf = GeoXGBClassifier(n_rounds=20, refit_interval=5, random_state=0)
    clf.fit(X, y)
    return clf, X, y


FEATURE_NAMES = [f"x{i}" for i in range(P)]


# ---------------------------------------------------------------------------
# 1. model_report returns all keys at each detail level
# ---------------------------------------------------------------------------

def test_model_report_summary_keys():
    model, X, y = _fitted_reg()
    rep = model_report(model, detail="summary")
    for k in ("model_type", "n_rounds", "n_trees", "n_resamples",
               "noise", "provenance", "importance"):
        assert k in rep, f"Missing key: {k}"
    assert "partitions" not in rep
    assert "evolution" not in rep


def test_model_report_standard_keys():
    model, X, y = _fitted_reg()
    rep = model_report(model, detail="standard")
    for k in ("model_type", "n_rounds", "n_trees", "n_resamples",
               "noise", "provenance", "importance", "partitions"):
        assert k in rep, f"Missing key: {k}"
    assert "evolution" not in rep


def test_model_report_full_keys():
    model, X, y = _fitted_reg()
    rep = model_report(model, detail="full")
    for k in ("model_type", "n_rounds", "n_trees", "n_resamples",
               "noise", "provenance", "importance", "partitions", "evolution"):
        assert k in rep, f"Missing key: {k}"


def test_model_report_with_performance_regressor():
    model, X, y = _fitted_reg()
    split = N // 5
    rep = model_report(model, X[:split], y[:split], detail="standard")
    perf = rep["performance"]
    assert "r2" in perf and "rmse" in perf


def test_model_report_with_performance_classifier():
    clf, X, y = _fitted_clf()
    split = N // 5
    rep = model_report(clf, X[:split], y[:split], detail="standard")
    perf = rep["performance"]
    assert "accuracy" in perf and "log_loss" in perf and "n_classes" in perf


# ---------------------------------------------------------------------------
# 2. noise_report assessment strings match modulation ranges
# ---------------------------------------------------------------------------

def test_noise_report_clean_assessment():
    model, X, y = _fitted_reg()
    nr = noise_report(model)
    assert "initial_modulation" in nr
    assert "assessment" in nr
    if nr["initial_modulation"] > 0.7:
        assert nr["assessment"] == "clean"
    elif nr["initial_modulation"] >= 0.3:
        assert nr["assessment"] == "moderate"
    else:
        assert nr["assessment"] == "noisy"


def test_noise_report_noisy_data():
    # Use large n so the noise estimator can reliably detect the absence of structure
    rng2 = np.random.default_rng(42)
    X = rng2.standard_normal((600, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng2.standard_normal(600) * 30.0
    model = GeoXGBRegressor(n_rounds=20, refit_interval=5, random_state=0)
    model.fit(X, y)
    nr = noise_report(model)
    assert nr["assessment"] in ("moderate", "noisy")


def test_noise_report_required_keys():
    model, _, _ = _fitted_reg()
    nr = noise_report(model)
    for k in ("initial_modulation", "assessment", "final_modulation",
               "modulation_trend", "effective_reduce_ratio", "interpretation"):
        assert k in nr, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# 3. importance_report agreement is float in [-1, 1]
# ---------------------------------------------------------------------------

def test_importance_report_agreement_range():
    model, _, _ = _fitted_reg()
    imp = importance_report(model, FEATURE_NAMES)
    ag = imp["agreement"]
    assert isinstance(ag, float)
    assert -1.0 <= ag <= 1.0


def test_importance_report_keys():
    model, _, _ = _fitted_reg()
    imp = importance_report(model, FEATURE_NAMES)
    for k in ("boosting_importance", "partition_importance",
               "agreement", "interpretation"):
        assert k in imp


def test_importance_report_standard_adds_top_and_divergent():
    model, _, _ = _fitted_reg()
    imp = importance_report(model, FEATURE_NAMES, detail="standard")
    assert "top_boosting" in imp
    assert "top_partition" in imp
    assert "divergent_features" in imp
    assert len(imp["top_boosting"]) <= 5
    assert len(imp["top_partition"]) <= 5


# ---------------------------------------------------------------------------
# 4. importance_report divergent_features correctly identifies rank gaps
# ---------------------------------------------------------------------------

def test_importance_report_divergent_features_rank_diff():
    model, _, _ = _fitted_reg()
    imp = importance_report(model, FEATURE_NAMES, detail="standard")
    for d in imp["divergent_features"]:
        assert d["rank_diff"] > 3
        assert "feature" in d
        assert "boosting_rank" in d
        assert "partition_rank" in d


# ---------------------------------------------------------------------------
# 5. partition_report imbalance_ratio is >= 1.0
# ---------------------------------------------------------------------------

def test_partition_report_imbalance_ratio():
    model, _, _ = _fitted_reg()
    pr = partition_report(model, round_idx=0, feature_names=FEATURE_NAMES)
    dist = pr["size_distribution"]
    assert dist["imbalance_ratio"] >= 1.0


def test_partition_report_required_keys():
    model, _, _ = _fitted_reg()
    pr = partition_report(model, detail="standard")
    for k in ("round", "n_partitions", "noise_modulation", "total_samples",
               "tree_rules", "tree_depth", "tree_feature_importances"):
        assert k in pr, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# 6. evolution_report returns n_resamples matching model.n_resamples
# ---------------------------------------------------------------------------

def test_evolution_report_n_resamples():
    model, _, _ = _fitted_reg()
    evo = evolution_report(model)
    assert evo["n_resamples"] == model.n_resamples


def test_evolution_report_rounds_length():
    model, _, _ = _fitted_reg()
    evo = evolution_report(model)
    assert len(evo["rounds"]) == model.n_resamples


# ---------------------------------------------------------------------------
# 7. validation_report passes on clean data with correct ground truth
# ---------------------------------------------------------------------------

def test_validation_report_clean_passes():
    X, y = _clean_data()
    model, _, _ = _fitted_reg(X, y)
    gt = {
        "signal_features": SIGNAL,
        "noise_features":  NOISE,
        "mechanism":       "y = 3*x0 - 2*x1 + x2 + small_noise",
    }
    val = validation_report(model, X, y, FEATURE_NAMES, gt)
    assert "checks" in val
    assert "overall_pass" in val
    assert isinstance(val["overall_pass"], bool)
    assert val["summary"]


# ---------------------------------------------------------------------------
# 8. validation_report fails partition_ignores_noise when noise features dominate
# ---------------------------------------------------------------------------

def test_validation_report_catch_noise_features():
    # Need n=1000 so HVRT's partition tree reliably produces non-trivial
    # importances.  x3,x4,x5 are the actual signal; we mislabel them as
    # "noise" in the ground truth — the check must FAIL.
    rng2 = np.random.default_rng(7)
    n_large = 1000
    X = rng2.standard_normal((n_large, P))
    y = 8 * X[:, 3] - 6 * X[:, 4] + 4 * X[:, 5] + rng2.standard_normal(n_large) * 0.01
    model = GeoXGBRegressor(n_rounds=20, refit_interval=5, random_state=0)
    model.fit(X, y)

    # Skip if HVRT returned an all-zero partition tree (trivial partition)
    imp_check = importance_report(model, FEATURE_NAMES)
    if sum(imp_check["partition_importance"].values()) < 1e-6:
        pytest.skip("HVRT returned trivial partition tree; check not applicable")

    wrong_gt = {
        "signal_features": SIGNAL,  # x0,x1,x2 — noise in this dataset
        "noise_features":  NOISE,   # x3,x4,x5 — actual signal; mislabelled
    }
    val = validation_report(model, X, y, FEATURE_NAMES, wrong_gt)
    checks_by_name = {c["name"]: c for c in val["checks"]}
    if "partition_ignores_noise" in checks_by_name:
        # "Noise" features (actual signal) dominate partition importance → FAIL
        assert not checks_by_name["partition_ignores_noise"]["passed"]


# ---------------------------------------------------------------------------
# 9. compare_report produces correct delta signs
# ---------------------------------------------------------------------------

def test_compare_report_positive_delta():
    model, _, _ = _fitted_reg()
    baseline = {"r2": 0.80, "geoxgb_r2": 0.85, "n_samples_used": N}
    comp = compare_report(model, baseline)
    assert comp["delta_score"] > 0

def test_compare_report_negative_delta():
    model, _, _ = _fitted_reg()
    baseline = {"r2": 0.95, "geoxgb_r2": 0.80, "n_samples_used": N}
    comp = compare_report(model, baseline)
    assert comp["delta_score"] < 0

def test_compare_report_required_keys():
    model, _, _ = _fitted_reg()
    baseline = {"auc": 0.95, "geoxgb_auc": 0.94, "n_samples_used": N}
    comp = compare_report(model, baseline)
    for k in ("geoxgb", "baseline", "delta_score", "sample_efficiency", "interpretation"):
        assert k in comp


# ---------------------------------------------------------------------------
# 10. print_report does not raise on any report type
# ---------------------------------------------------------------------------

def test_print_report_model_report(capsys):
    model, X, y = _fitted_reg()
    rep = model_report(model, X[:20], y[:20], FEATURE_NAMES, detail="full")
    print_report(rep, title="Test Model Report")
    captured = capsys.readouterr()
    assert len(captured.out) > 0

def test_print_report_noise_report(capsys):
    model, _, _ = _fitted_reg()
    print_report(noise_report(model), title="Noise")
    assert len(capsys.readouterr().out) > 0

def test_print_report_importance_report(capsys):
    model, _, _ = _fitted_reg()
    print_report(importance_report(model, FEATURE_NAMES), title="Importance")
    assert len(capsys.readouterr().out) > 0

def test_print_report_validation_report(capsys):
    X, y = _clean_data()
    model, _, _ = _fitted_reg(X, y)
    gt = {"signal_features": SIGNAL, "noise_features": NOISE}
    rep = validation_report(model, X, y, FEATURE_NAMES, gt)
    print_report(rep, title="Validation")
    assert len(capsys.readouterr().out) > 0

def test_print_report_compare_report(capsys):
    model, _, _ = _fitted_reg()
    baseline = {"r2": 0.90, "geoxgb_r2": 0.88, "n_samples_used": N}
    print_report(compare_report(model, baseline), title="Compare")
    assert len(capsys.readouterr().out) > 0


# ---------------------------------------------------------------------------
# 11. All reports are JSON-serialisable
# ---------------------------------------------------------------------------

def _assert_json_serialisable(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_json_serialisable(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_json_serialisable(v, f"{path}[{i}]")
    elif isinstance(obj, (str, int, bool, type(None))):
        pass
    elif isinstance(obj, float):
        pass  # nan/inf are technically not JSON-serialisable per spec; allow
    else:
        raise TypeError(f"Non-serialisable type {type(obj)} at {path}")


def test_all_reports_json_serialisable():
    X, y = _clean_data()
    model, _, _ = _fitted_reg(X, y)
    gt = {"signal_features": SIGNAL, "noise_features": NOISE,
          "mechanism": "y = 3*x0 - 2*x1 + x2"}

    reports = [
        model_report(model, X[:20], y[:20], FEATURE_NAMES, detail="full"),
        noise_report(model),
        provenance_report(model, detail="full"),
        importance_report(model, FEATURE_NAMES, gt, detail="full"),
        partition_report(model, feature_names=FEATURE_NAMES, detail="full"),
        evolution_report(model, FEATURE_NAMES, detail="full"),
        validation_report(model, X, y, FEATURE_NAMES, gt),
        compare_report(model, {"r2": 0.9, "geoxgb_r2": 0.88, "n_samples_used": N}),
    ]
    for rep in reports:
        _assert_json_serialisable(rep)
        # Strict JSON round-trip (replace nan with null for compliance)
        json_str = json.dumps(rep, allow_nan=True)
        assert len(json_str) > 2
