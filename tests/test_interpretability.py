"""
Interpretability API tests.
"""
import numpy as np
import pytest

from geoxgb import GeoXGBRegressor

RNG = np.random.default_rng(3)
N, P = 300, 6
SIGNAL_FEATURES = [0, 1, 2]   # x0, x1, x2 are informative
NOISE_FEATURES  = [3, 4, 5]   # x3, x4, x5 are pure noise

FEATURE_NAMES = [f"x{i}" for i in range(P)]


def _fitted_model(expand_ratio=0.0):
    X = RNG.standard_normal((N, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] + RNG.standard_normal(N) * 0.05
    reg = GeoXGBRegressor(
        n_rounds=30, refit_interval=10,
        expand_ratio=expand_ratio, random_state=0,
        auto_expand=False, hvrt_min_samples_leaf=20,
    )
    reg.fit(X, y)
    return reg


# ---------------------------------------------------------------------------
# 1. feature_importances
# ---------------------------------------------------------------------------

def test_feature_importances_format():
    reg = _fitted_model()
    fi = reg.feature_importances(FEATURE_NAMES)
    assert isinstance(fi, dict)
    assert set(fi.keys()) == set(FEATURE_NAMES)
    total = sum(fi.values())
    assert abs(total - 1.0) < 0.01, f"Importances sum={total}, expected ~1.0"


# ---------------------------------------------------------------------------
# 2. partition_feature_importances
# ---------------------------------------------------------------------------

def test_partition_feature_importances_format():
    reg = _fitted_model()
    pfi = reg.partition_feature_importances(FEATURE_NAMES)
    assert isinstance(pfi, list)
    assert len(pfi) >= 1
    for entry in pfi:
        assert "round" in entry
        assert "importances" in entry
        assert isinstance(entry["importances"], dict)

def test_partition_feature_importances_signal_over_noise():
    reg = _fitted_model()
    pfi = reg.partition_feature_importances(FEATURE_NAMES)
    # First resample: signal features should collectively dominate
    imps = pfi[0]["importances"]
    signal_imp = sum(imps.get(f"x{i}", 0) for i in SIGNAL_FEATURES)
    noise_imp  = sum(imps.get(f"x{i}", 0) for i in NOISE_FEATURES)
    assert signal_imp > noise_imp, (
        f"Signal importance={signal_imp:.3f} should exceed noise={noise_imp:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. partition_trace
# ---------------------------------------------------------------------------

def test_partition_trace_format():
    reg = _fitted_model()
    trace = reg.partition_trace()
    assert isinstance(trace, list)
    assert len(trace) >= 1
    required_keys = {"round", "noise_modulation", "n_samples",
                     "n_reduced", "n_expanded", "partitions"}
    for entry in trace:
        assert required_keys.issubset(entry.keys()), (
            f"Missing keys: {required_keys - entry.keys()}"
        )
        assert isinstance(entry["partitions"], list)


# ---------------------------------------------------------------------------
# 4. partition_tree_rules
# ---------------------------------------------------------------------------

def test_partition_tree_rules():
    reg = _fitted_model()
    rules = reg.partition_tree_rules(round_idx=0)
    assert isinstance(rules, str)
    assert len(rules) > 0

def test_partition_tree_rules_bad_index():
    reg = _fitted_model()
    with pytest.raises(IndexError):
        reg.partition_tree_rules(round_idx=9999)


# ---------------------------------------------------------------------------
# 5. sample_provenance
# ---------------------------------------------------------------------------

def test_sample_provenance_keys():
    reg = _fitted_model()
    prov = reg.sample_provenance()
    required = {"original_n", "reduced_n", "expanded_n",
                "total_training", "reduction_ratio"}
    assert required.issubset(prov.keys())

def test_sample_provenance_no_expansion():
    reg = _fitted_model(expand_ratio=0.0)
    prov = reg.sample_provenance()
    assert prov["total_training"] <= prov["original_n"], (
        "With expand_ratio=0, total training should not exceed original n"
    )

def test_sample_provenance_with_expansion():
    reg = _fitted_model(expand_ratio=0.2)
    prov = reg.sample_provenance()
    assert prov["expanded_n"] > 0


# ---------------------------------------------------------------------------
# 6. noise_estimate
# ---------------------------------------------------------------------------

def test_noise_estimate_clean():
    X = RNG.standard_normal((300, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(300) * 0.05
    reg = GeoXGBRegressor(n_rounds=10, random_state=0)
    reg.fit(X, y)
    ne = reg.noise_estimate()
    assert 0.0 <= ne <= 1.0
    assert ne > 0.5, f"Clean data noise_estimate={ne:.3f}, expected > 0.5"

def test_noise_estimate_noisy():
    X = RNG.standard_normal((300, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(300) * 20.0
    reg = GeoXGBRegressor(n_rounds=10, random_state=0)
    reg.fit(X, y)
    ne = reg.noise_estimate()
    assert 0.0 <= ne <= 1.0
    assert ne < 0.5, f"Noisy data noise_estimate={ne:.3f}, expected < 0.5"

def test_noise_estimate_not_fitted():
    reg = GeoXGBRegressor()
    with pytest.raises(RuntimeError):
        reg.noise_estimate()
