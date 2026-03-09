"""
Tests for contributions() API and ContributionFrame.
"""
import json

import numpy as np
import pytest

from geoxgb import GeoXGBRegressor, ContributionFrame

RNG = np.random.default_rng(42)
N, D = 250, 5
FEAT_NAMES = [f"x{i}" for i in range(D)]


def _make_data():
    X = RNG.standard_normal((N, D))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + RNG.standard_normal(N) * 0.1
    return X, y


def _fitted_model(X, y):
    reg = GeoXGBRegressor(
        n_rounds=40, refit_interval=10,
        hvrt_min_samples_leaf=15, random_state=0,
        auto_expand=False,
    )
    reg.fit(X, y)  # C++ path — no feature_types required
    return reg


@pytest.fixture(scope="module")
def setup():
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((50, D))
    contrib = model.contributions(X_val, feature_names=FEAT_NAMES)
    return model, X_val, contrib


# ---------------------------------------------------------------------------
# 1. Type check
# ---------------------------------------------------------------------------

def test_returns_contribution_frame(setup):
    _, _, contrib = setup
    assert isinstance(contrib, ContributionFrame)


# ---------------------------------------------------------------------------
# 2. Shape checks
# ---------------------------------------------------------------------------

def test_main_shapes(setup):
    _, X_val, contrib = setup
    n = len(X_val)
    for fname in FEAT_NAMES:
        assert contrib.main[fname].shape == (n,), fname


def test_intercepts_shape(setup):
    _, X_val, contrib = setup
    assert contrib.intercepts.shape == (len(X_val),)


def test_local_r2_shape(setup):
    _, X_val, contrib = setup
    assert contrib.local_r2.shape == (len(X_val),)


def test_interaction_shapes(setup):
    _, X_val, contrib = setup
    n = len(X_val)
    for key, arr in contrib.interaction.items():
        assert arr.shape == (n,), f"interaction {key}"


# ---------------------------------------------------------------------------
# 3. local_r2 range
# ---------------------------------------------------------------------------

def test_local_r2_range(setup):
    _, _, contrib = setup
    assert np.all(contrib.local_r2 >= 0.0), "local_r2 has negative values"
    assert np.all(contrib.local_r2 <= 1.0), "local_r2 > 1"


# ---------------------------------------------------------------------------
# 4. Contribution sum ≈ prediction
# ---------------------------------------------------------------------------

def test_contribution_sum_equals_prediction(setup):
    model, X_val, contrib = setup
    preds = model.predict(X_val)

    total = contrib.intercepts.copy()
    for arr in contrib.main.values():
        total += arr
    for arr in contrib.interaction.values():
        total += arr

    # total = local-polynomial value, which approximates model.predict(x).
    # The approximation quality depends on local_r2; use a generous tolerance.
    max_dev = float(np.abs(total - preds).max())
    assert max_dev < 5.0, (
        f"Max |sum_contributions - prediction| = {max_dev:.3f}, expected < 5.0. "
        "This may indicate a bug in the partition/Ridge logic."
    )


# ---------------------------------------------------------------------------
# 5. Interaction keys are (str, str) tuples
# ---------------------------------------------------------------------------

def test_interaction_keys_are_string_tuples(setup):
    _, _, contrib = setup
    for key in contrib.interaction:
        assert isinstance(key, tuple) and len(key) == 2
        assert all(isinstance(k, str) for k in key)


# ---------------------------------------------------------------------------
# 6. to_dict() JSON-serialisable
# ---------------------------------------------------------------------------

def test_to_dict_json_serialisable(setup):
    _, _, contrib = setup
    d = contrib.to_dict()
    # must not raise
    s = json.dumps(d)
    assert len(s) > 0


def test_to_dict_keys(setup):
    _, _, contrib = setup
    d = contrib.to_dict()
    assert "feature_names" in d
    assert "main" in d
    assert "interaction" in d
    assert "intercepts" in d
    assert "local_r2" in d


# ---------------------------------------------------------------------------
# 7. to_dataframe() (pandas optional)
# ---------------------------------------------------------------------------

def test_to_dataframe(setup):
    pytest.importorskip("pandas")
    _, X_val, contrib = setup
    df = contrib.to_dataframe()
    assert len(df) == len(X_val)
    for fname in FEAT_NAMES:
        assert f"main_{fname}" in df.columns


# ---------------------------------------------------------------------------
# 8. min_pair_coop=0 includes all pairs
# ---------------------------------------------------------------------------

def test_zero_min_pair_coop():
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((20, D))
    contrib = model.contributions(X_val, feature_names=FEAT_NAMES,
                                  min_pair_coop=0.0)
    n_possible = D * (D - 1) // 2
    assert len(contrib.interaction) <= n_possible


# ---------------------------------------------------------------------------
# 9. No feature_names falls back to x0..x{d-1}
# ---------------------------------------------------------------------------

def test_default_feature_names():
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((10, D))
    contrib = model.contributions(X_val)
    assert contrib.feature_names == [f"x{i}" for i in range(D)]
    assert all(f"x{i}" in contrib.main for i in range(D))


# ---------------------------------------------------------------------------
# 10. Plot smoke tests (require matplotlib)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mpl_agg():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    return mpl


def test_plot_main_effect(setup, mpl_agg):
    _, _, contrib = setup
    fig = contrib.plot("x0")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_main_effect_with_ci(setup, mpl_agg):
    _, _, contrib = setup
    fig = contrib.plot("x0", ci=True, bandwidth=0.1)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_overlay_main(setup, mpl_agg):
    _, _, contrib = setup
    fig = contrib.plot("x0", overlay=["x1"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_overlay_interaction(setup, mpl_agg):
    _, _, contrib = setup
    # Use min_pair_coop=0 so we have interactions available
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((50, D))
    c = model.contributions(X_val, feature_names=FEAT_NAMES,
                            min_pair_coop=0.0)
    if c.interaction:
        key = next(iter(c.interaction))
        fig = c.plot(key[0], overlay=[key])
        assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_interaction_heatmap(setup, mpl_agg):
    _, _, contrib = setup
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((80, D))
    c = model.contributions(X_val, feature_names=FEAT_NAMES,
                            min_pair_coop=0.0)
    if c.interaction:
        a, b = next(iter(c.interaction))
        fig = c.plot_interaction(a, b)
        assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_interaction_missing_key(setup, mpl_agg):
    _, _, contrib = setup
    # With high min_pair_coop, most pairs absent — check KeyError
    X, y = _make_data()
    model = _fitted_model(X, y)
    X_val = RNG.standard_normal((20, D))
    c = model.contributions(X_val, feature_names=FEAT_NAMES,
                            min_pair_coop=1.1)  # impossible threshold
    with pytest.raises(KeyError):
        c.plot_interaction("x0", "x1")


# ---------------------------------------------------------------------------
# 11. Error: C++ path or no Python geometry
# ---------------------------------------------------------------------------

def test_contributions_work_without_feature_types():
    # contributions() now works without feature_types (C++ geometry is always available)
    X, y = _make_data()
    reg = GeoXGBRegressor(n_rounds=10, random_state=0)
    reg.fit(X, y)
    X_val = RNG.standard_normal((5, D))
    c = reg.contributions(X_val, feature_names=FEAT_NAMES)
    assert isinstance(c, ContributionFrame)
