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
    # feature_types forces Python path (interpretability API requires Python backend)
    reg.fit(X, y, feature_types=["continuous"] * P)
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
    reg = GeoXGBRegressor(n_rounds=10, refit_interval=5, auto_noise=True, random_state=0)
    reg.fit(X, y, feature_types=["continuous"] * 5)
    ne = reg.noise_estimate()
    assert 0.0 <= ne <= 1.0
    assert ne > 0.5, f"Clean data noise_estimate={ne:.3f}, expected > 0.5"

def test_noise_estimate_noisy():
    X = RNG.standard_normal((300, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(300) * 20.0
    reg = GeoXGBRegressor(n_rounds=10, refit_interval=5, auto_noise=True, random_state=0)
    reg.fit(X, y, feature_types=["continuous"] * 5)
    ne = reg.noise_estimate()
    assert 0.0 <= ne <= 1.0
    assert ne < 0.5, f"Noisy data noise_estimate={ne:.3f}, expected < 0.5"

def test_noise_estimate_not_fitted():
    reg = GeoXGBRegressor()
    with pytest.raises(RuntimeError):
        reg.noise_estimate()


# ---------------------------------------------------------------------------
# 7. cooperation_matrix
# ---------------------------------------------------------------------------

def test_cooperation_matrix_shape():
    reg = _fitted_model()
    X_test = RNG.standard_normal((10, P))
    result = reg.cooperation_matrix(X_test, FEATURE_NAMES)
    assert result["matrices"].shape == (10, P, P)
    assert result["global_matrix"].shape == (P, P)
    assert result["feature_names"] == FEATURE_NAMES
    assert result["partitioner"] == reg.partitioner


def test_cooperation_matrix_diagonal_ones():
    reg = _fitted_model()
    X_test = RNG.standard_normal((20, P))
    result = reg.cooperation_matrix(X_test)
    mats = result["matrices"]
    for s in range(len(X_test)):
        diag = np.diag(mats[s])
        np.testing.assert_allclose(diag, np.ones(P), atol=0.01,
                                   err_msg=f"Diagonal not 1 at sample {s}")


def test_cooperation_matrix_symmetric():
    reg = _fitted_model()
    X_test = RNG.standard_normal((15, P))
    result = reg.cooperation_matrix(X_test)
    mats = result["matrices"]
    for s in range(len(X_test)):
        diff = np.abs(mats[s] - mats[s].T).max()
        assert diff < 1e-10, f"Matrix not symmetric at sample {s}, max diff={diff}"
    glob = result["global_matrix"]
    assert np.abs(glob - glob.T).max() < 1e-10, "Global matrix not symmetric"


def test_cooperation_matrix_bounded():
    reg = _fitted_model()
    X_test = RNG.standard_normal((30, P))
    result = reg.cooperation_matrix(X_test)
    mats = result["matrices"]
    assert mats.min() >= -1.0 - 1e-9
    assert mats.max() <=  1.0 + 1e-9


def test_cooperation_matrix_requires_python_path():
    # C++ path: no feature_types → no hvrt_model with X_z_
    from geoxgb._cpp_backend import _CPP_AVAILABLE
    if not _CPP_AVAILABLE:
        pytest.skip("C++ backend not available")
    X = RNG.standard_normal((N, P))
    y = RNG.standard_normal(N)
    reg = GeoXGBRegressor(n_rounds=10, random_state=0)
    reg.fit(X, y)   # no feature_types → C++ path
    X_test = RNG.standard_normal((5, P))
    with pytest.raises(RuntimeError):
        reg.cooperation_matrix(X_test)


# ---------------------------------------------------------------------------
# 8. cooperation_score
# ---------------------------------------------------------------------------

def test_cooperation_score_shape():
    reg = _fitted_model()
    X_test = RNG.standard_normal((20, P))
    scores = reg.cooperation_score(X_test)
    assert scores.shape == (20,)


def test_cooperation_score_pyramid_hart_nonpositive():
    # PyramidHART: A = |S| - ||z||_1 <= 0
    X = RNG.standard_normal((N, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(N) * 0.1
    reg = GeoXGBRegressor(
        n_rounds=20, partitioner='pyramid_hart', random_state=0
    )
    reg.fit(X, y, feature_types=["continuous"] * P)
    X_test = RNG.standard_normal((50, P))
    scores = reg.cooperation_score(X_test)
    assert scores.max() <= 1e-9, (
        f"PyramidHART cooperation scores should be <= 0, got max={scores.max():.4f}"
    )


def test_cooperation_score_hvrt_signed():
    # HVRT: T can be positive or negative
    X = RNG.standard_normal((N, P))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(N) * 0.1
    reg = GeoXGBRegressor(
        n_rounds=20, partitioner='hvrt', random_state=0
    )
    reg.fit(X, y, feature_types=["continuous"] * P)
    X_test = RNG.standard_normal((50, P))
    scores = reg.cooperation_score(X_test)
    # HVRT scores can be positive or negative — just check shape
    assert scores.shape == (50,)


# ---------------------------------------------------------------------------
# 9. cooperation_tensor
# ---------------------------------------------------------------------------

def test_cooperation_tensor_shape():
    reg = _fitted_model()
    X_test = RNG.standard_normal((8, P))
    result = reg.cooperation_tensor(X_test, FEATURE_NAMES)
    assert result["tensor"].shape == (8, P, P, P)
    assert result["global_tensor"].shape == (P, P, P)
    assert result["feature_names"] == FEATURE_NAMES


def test_cooperation_tensor_symmetry():
    # T[s, i, j, k] should equal T[s, j, i, k] = T[s, i, k, j] (all permutations)
    reg = _fitted_model()
    X_test = RNG.standard_normal((5, P))
    result = reg.cooperation_tensor(X_test)
    T = result["tensor"]
    for s in range(len(X_test)):
        np.testing.assert_allclose(T[s], T[s].transpose(1, 0, 2), atol=1e-10,
                                   err_msg=f"Tensor not sym in (i,j) at s={s}")
        np.testing.assert_allclose(T[s], T[s].transpose(0, 2, 1), atol=1e-10,
                                   err_msg=f"Tensor not sym in (j,k) at s={s}")
