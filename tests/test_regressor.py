"""
Regression tests for GeoXGBRegressor.
"""
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from geoxgb import GeoXGBRegressor

RNG = np.random.default_rng(0)
N, P = 200, 5


def _vanilla_r2(X, y):
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
    split = int(0.8 * len(X))
    gb.fit(X[:split], y[:split])
    return r2_score(y[split:], gb.predict(X[split:]))


def _geo_r2(X, y, **kwargs):
    reg = GeoXGBRegressor(
        n_rounds=100, learning_rate=0.1, max_depth=3,
        refit_interval=10, auto_noise=True, random_state=42, **kwargs
    )
    split = int(0.8 * len(X))
    reg.fit(X[:split], y[:split])
    return r2_score(y[split:], reg.predict(X[split:]))


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------

def test_smoke():
    X = RNG.standard_normal((N, P))
    y = RNG.standard_normal(N)
    reg = GeoXGBRegressor(n_rounds=20, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (N,)
    assert r2_score(y, preds) > -1.0


# ---------------------------------------------------------------------------
# 2. Clean linear
# ---------------------------------------------------------------------------

def test_clean_linear():
    X = RNG.standard_normal((500, 5))
    y = 2 * X[:, 0] + X[:, 1] - X[:, 2] + RNG.normal(0, 0.1, 500)
    geo = _geo_r2(X, y)
    van = _vanilla_r2(X, y)
    assert geo >= van - 0.01, f"GeoXGB R²={geo:.4f} vs VanillaGB R²={van:.4f}"


# ---------------------------------------------------------------------------
# 3. Clean nonlinear
# ---------------------------------------------------------------------------

def test_clean_nonlinear():
    X = RNG.standard_normal((500, 5))
    y = 3 * np.sin(X[:, 0]) + X[:, 1] ** 2 - 2 * X[:, 2] * X[:, 3]
    geo = _geo_r2(X, y)
    van = _vanilla_r2(X, y)
    assert geo >= van - 0.02, f"GeoXGB R²={geo:.4f} vs VanillaGB R²={van:.4f}"


# ---------------------------------------------------------------------------
# 4. Noisy linear
# ---------------------------------------------------------------------------

def test_noisy_linear():
    X = RNG.standard_normal((500, 5))
    y = 2 * X[:, 0] + X[:, 1] - X[:, 2] + RNG.normal(0, 10.0, 500)
    geo = _geo_r2(X, y)
    van = _vanilla_r2(X, y)
    assert geo >= van - 0.05, (
        f"GeoXGB R²={geo:.4f} vs VanillaGB R²={van:.4f} — "
        "noise modulation should prevent catastrophic degradation"
    )


# ---------------------------------------------------------------------------
# 5. Density-dependent (imbalanced clusters)
# ---------------------------------------------------------------------------

def test_density_dependent():
    # Dense cluster (80%) + sparse cluster (20%)
    n_dense, n_sparse = 400, 100
    X_dense = RNG.normal([0, 0], 0.3, (n_dense, 2))
    X_sparse = RNG.normal([5, 5], 1.0, (n_sparse, 2))
    X = np.vstack([X_dense, X_sparse])
    y = X[:, 0] * 2 + X[:, 1]
    geo = _geo_r2(X, y)
    van = _vanilla_r2(X, y)
    assert geo >= van - 0.01, f"GeoXGB R²={geo:.4f} vs VanillaGB R²={van:.4f}"


# ---------------------------------------------------------------------------
# 6. Refit disabled
# ---------------------------------------------------------------------------

def test_refit_disabled():
    X = RNG.standard_normal((N, P))
    y = RNG.standard_normal(N)
    reg = GeoXGBRegressor(n_rounds=20, refit_interval=None, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (N,)
    assert reg.n_resamples == 1


# ---------------------------------------------------------------------------
# 7. Expand enabled
# ---------------------------------------------------------------------------

def test_expand_enabled():
    X = RNG.standard_normal((N, P))
    y = RNG.standard_normal(N)
    reg = GeoXGBRegressor(n_rounds=20, expand_ratio=0.2, random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (N,)
    assert reg.sample_provenance()["expanded_n"] > 0
