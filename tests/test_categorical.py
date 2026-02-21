"""
Categorical feature tests for GeoXGBRegressor.
"""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from geoxgb import GeoXGBRegressor

RNG = np.random.default_rng(2)
N = 200


def _encode_categorical(col):
    le = LabelEncoder()
    return le.fit_transform(col).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Mixed features: 3 continuous + 1 categorical
# ---------------------------------------------------------------------------

def test_mixed_features():
    X_cont = RNG.standard_normal((N, 3))
    X_cat = _encode_categorical(RNG.choice(["a", "b", "c"], size=N))
    X = np.column_stack([X_cont, X_cat])
    y = 2 * X[:, 0] - X[:, 1] + X[:, 3] + RNG.standard_normal(N) * 0.1
    feature_types = ["continuous", "continuous", "continuous", "categorical"]
    reg = GeoXGBRegressor(n_rounds=20, random_state=0)
    reg.fit(X, y, feature_types=feature_types)
    preds = reg.predict(X)
    assert preds.shape == (N,)


# ---------------------------------------------------------------------------
# 2. All categorical
# ---------------------------------------------------------------------------

def test_all_categorical():
    cats = [
        _encode_categorical(RNG.choice(["x", "y", "z"], size=N)),
        _encode_categorical(RNG.choice(["p", "q"], size=N)),
        _encode_categorical(RNG.choice(["r", "s", "t", "u"], size=N)),
    ]
    X = np.column_stack(cats)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + RNG.standard_normal(N) * 0.1
    feature_types = ["categorical", "categorical", "categorical"]
    reg = GeoXGBRegressor(n_rounds=20, random_state=0)
    reg.fit(X, y, feature_types=feature_types)
    preds = reg.predict(X)
    assert preds.shape == (N,)


# ---------------------------------------------------------------------------
# 3. feature_types=None — defaults to all continuous, no crash
# ---------------------------------------------------------------------------

def test_feature_types_none():
    X = RNG.standard_normal((N, 4))
    y = RNG.standard_normal(N)
    reg = GeoXGBRegressor(n_rounds=20, random_state=0)
    reg.fit(X, y, feature_types=None)
    preds = reg.predict(X)
    assert preds.shape == (N,)
