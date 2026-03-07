"""
Interpretability API tests.
"""
import numpy as np
import pytest

from geoxgb import GeoXGBRegressor, GeoXGBClassifier

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
# 2. partition_feature_importances — now raises RuntimeError (C++ path only)
# ---------------------------------------------------------------------------

def test_partition_feature_importances_raises():
    reg = _fitted_model()
    with pytest.raises(RuntimeError):
        reg.partition_feature_importances(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 3. partition_trace — now raises RuntimeError (C++ path only)
# ---------------------------------------------------------------------------

def test_partition_trace_raises():
    reg = _fitted_model()
    with pytest.raises(RuntimeError):
        reg.partition_trace()


# ---------------------------------------------------------------------------
# 4. partition_tree_rules — now raises RuntimeError (C++ path only)
# ---------------------------------------------------------------------------

def test_partition_tree_rules_raises():
    reg = _fitted_model()
    with pytest.raises(RuntimeError):
        reg.partition_tree_rules(round_idx=0)


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
    # C++ path does not track expansion; expanded_n is always 0
    reg = _fitted_model(expand_ratio=0.2)
    prov = reg.sample_provenance()
    assert prov["expanded_n"] == 0


# ---------------------------------------------------------------------------
# 6. noise_estimate
# ---------------------------------------------------------------------------

def test_noise_estimate_clean():
    X = RNG.standard_normal((300, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(300) * 0.05
    reg = GeoXGBRegressor(n_rounds=10, refit_interval=5, auto_noise=True, random_state=0)
    reg.fit(X, y)
    ne = reg.noise_estimate()
    assert 0.0 <= ne <= 1.0
    assert ne > 0.5, f"Clean data noise_estimate={ne:.3f}, expected > 0.5"

def test_noise_estimate_noisy():
    X = RNG.standard_normal((300, 5))
    y = 3 * X[:, 0] - 2 * X[:, 1] + RNG.standard_normal(300) * 20.0
    reg = GeoXGBRegressor(n_rounds=10, refit_interval=5, auto_noise=True, random_state=0)
    reg.fit(X, y)
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
    reg.fit(X, y)
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
    reg.fit(X, y)
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


# ---------------------------------------------------------------------------
# 10. local_model
# ---------------------------------------------------------------------------

# Deterministic synthetic dataset: y = 2*x1 + 3*x2 - x3 + 0.5*x1*x2 - 1.5*x2*x3
_VALS = np.linspace(0.0, 2.0, 8)
_X_SYNTH = np.array([(x1, x2, x3) for x1 in _VALS for x2 in _VALS for x3 in _VALS],
                    dtype=np.float64)
_Y_SYNTH  = (2.0 * _X_SYNTH[:, 0] + 3.0 * _X_SYNTH[:, 1] - 1.0 * _X_SYNTH[:, 2]
             + 0.5 * _X_SYNTH[:, 0] * _X_SYNTH[:, 1]
             - 1.5 * _X_SYNTH[:, 1] * _X_SYNTH[:, 2])


def _local_model_fixture():
    reg = GeoXGBRegressor(
        n_rounds=500, learning_rate=0.02, max_depth=3,
        y_weight=0.25, refit_interval=50, random_state=0,
    )
    reg.fit(_X_SYNTH, _Y_SYNTH)
    return reg


def test_local_model_return_keys():
    reg = _local_model_fixture()
    lm = reg.local_model(np.array([1.0, 1.0, 1.0]),
                         feature_names=["x1", "x2", "x3"])
    for key in ("intercept", "additive", "pairwise", "prediction",
                "partition_size", "local_r2", "feature_names"):
        assert key in lm, f"missing key: {key}"
    assert lm["additive"].shape == (3,)
    assert isinstance(lm["pairwise"], dict)
    assert lm["partition_size"] >= 1
    assert 0.0 <= lm["local_r2"] <= 1.0 + 1e-9


def test_local_model_additive_signs():
    # At x=[1,1,1]: df/dx1>0, df/dx2>0, df/dx3<0 — must hold in z-space too
    reg = _local_model_fixture()
    lm = reg.local_model(np.array([1.0, 1.0, 1.0]))
    assert lm["additive"][0] > 0, "alpha_x1 should be positive"
    assert lm["additive"][1] > 0, "alpha_x2 should be positive"
    assert lm["additive"][2] < 0, "alpha_x3 should be negative"


def test_local_model_pairwise_signs():
    # Drop min_pair_coop to 0 to force all pairs into model
    reg = _local_model_fixture()
    lm = reg.local_model(np.array([1.0, 1.0, 1.0]), min_pair_coop=0.0)
    # True: beta_12>0, beta_23<0
    b12 = lm["pairwise"].get((0, 1), 0.0)
    b23 = lm["pairwise"].get((1, 2), 0.0)
    assert b12 > 0, f"beta_(x1,x2) should be positive, got {b12:.4f}"
    assert b23 < 0, f"beta_(x2,x3) should be negative, got {b23:.4f}"


def test_local_model_prediction_close_to_ensemble():
    reg = _local_model_fixture()
    x = np.array([1.0, 1.0, 1.0])
    lm = reg.local_model(x)
    ensemble_pred = float(reg.predict(x.reshape(1, -1))[0])
    # local polynomial should approximate ensemble within reasonable tolerance
    assert abs(lm["prediction"] - ensemble_pred) < 1.0, (
        f"local_model prediction {lm['prediction']:.4f} too far from "
        f"ensemble {ensemble_pred:.4f}"
    )


def test_local_model_works_without_feature_types():
    # C++ path always has geometry — local_model should work
    reg = GeoXGBRegressor(n_rounds=10, random_state=0)
    reg.fit(_X_SYNTH, _Y_SYNTH)
    lm = reg.local_model(np.array([1.0, 1.0, 1.0]))
    assert "intercept" in lm
    assert "additive" in lm


# ---------------------------------------------------------------------------
# 11. Multiclass classifier interpretability
# ---------------------------------------------------------------------------

def _fitted_multiclass():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4))
    y = np.array([0, 1, 2] * 66 + [0, 1])
    X[y == 0, 0] += 2
    X[y == 1, 1] += 2
    X[y == 2, 2] += 2
    clf = GeoXGBClassifier(
        n_rounds=50, learning_rate=0.1, max_depth=3,
        refit_interval=25, random_state=0,
    )
    clf.fit(X, y)
    return clf, X


def test_multiclass_cooperation_matrix():
    clf, X = _fitted_multiclass()
    result = clf.cooperation_matrix(X[:5])
    assert result["matrices"].shape == (5, 4, 4)
    assert result["global_matrix"].shape == (4, 4)


def test_multiclass_cooperation_score():
    clf, X = _fitted_multiclass()
    scores = clf.cooperation_score(X[:10])
    assert scores.shape == (10,)


def test_multiclass_cooperation_tensor():
    clf, X = _fitted_multiclass()
    result = clf.cooperation_tensor(X[:5])
    assert result["tensor"].shape == (5, 4, 4, 4)


def test_multiclass_feature_importances():
    clf, _ = _fitted_multiclass()
    fi = clf.feature_importances()
    assert isinstance(fi, dict)
    assert abs(sum(fi.values()) - 1.0) < 0.01


def test_multiclass_sample_provenance():
    clf, _ = _fitted_multiclass()
    prov = clf.sample_provenance()
    assert prov["original_n"] == 200


def test_multiclass_noise_estimate():
    clf, _ = _fitted_multiclass()
    ne = clf.noise_estimate()
    assert 0.0 <= ne <= 1.0


def test_multiclass_local_model_per_class():
    clf, X = _fitted_multiclass()
    for k in range(3):
        lm = clf.local_model(X[0], target_class=k)
        assert "intercept" in lm
        assert lm["additive"].shape == (4,)
        assert 0.0 <= lm["local_r2"] <= 1.0 + 1e-9


def test_multiclass_local_model_requires_target_class():
    clf, X = _fitted_multiclass()
    with pytest.raises(ValueError, match="target_class"):
        clf.local_model(X[0])


def test_multiclass_contributions_per_class():
    clf, X = _fitted_multiclass()
    for k in range(3):
        cf = clf.contributions(X[:10], target_class=k)
        assert len(cf.main) == 4
        assert cf.intercepts.shape == (10,)
        assert cf.local_r2.shape == (10,)


def test_multiclass_contributions_requires_target_class():
    clf, X = _fitted_multiclass()
    with pytest.raises(ValueError, match="target_class"):
        clf.contributions(X[:10])
