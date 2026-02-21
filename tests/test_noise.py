"""
Noise modulation behaviour tests.
"""
import numpy as np
import pytest

from geoxgb import GeoXGBRegressor

RNG = np.random.default_rng(4)
N = 300


def _fit_and_noise(X, y):
    reg = GeoXGBRegressor(n_rounds=10, auto_noise=True, random_state=0)
    reg.fit(X, y)
    return reg.noise_estimate(), reg.sample_provenance()["reduction_ratio"]


# ---------------------------------------------------------------------------
# 1. Clean data → high modulation
# ---------------------------------------------------------------------------

def test_clean_data_modulation():
    X = RNG.standard_normal((N, 5))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + RNG.standard_normal(N) * 0.05
    ne, _ = _fit_and_noise(X, y)
    assert ne > 0.7, f"Clean data noise_estimate={ne:.3f}, expected > 0.7"


# ---------------------------------------------------------------------------
# 2. Noisy data → low modulation
# ---------------------------------------------------------------------------

def test_noisy_data_modulation():
    X = RNG.standard_normal((N, 5))
    y = 2 * X[:, 0] - X[:, 1] + RNG.standard_normal(N) * 15.0
    ne, _ = _fit_and_noise(X, y)
    assert ne < 0.4, f"Noisy data noise_estimate={ne:.3f}, expected < 0.4"


# ---------------------------------------------------------------------------
# 3. Pure noise → very low modulation
# ---------------------------------------------------------------------------

def test_pure_noise_modulation():
    X = RNG.standard_normal((N, 5))
    y = RNG.standard_normal(N)  # no signal at all
    ne, _ = _fit_and_noise(X, y)
    assert ne < 0.2, f"Pure noise noise_estimate={ne:.3f}, expected < 0.2"


# ---------------------------------------------------------------------------
# 4. Noise dampens resampling (keeps more samples)
# ---------------------------------------------------------------------------

def test_noise_dampens_resampling():
    X = RNG.standard_normal((N, 5))
    y_clean = 2 * X[:, 0] - X[:, 1] + RNG.standard_normal(N) * 0.05
    y_noisy = 2 * X[:, 0] - X[:, 1] + RNG.standard_normal(N) * 15.0

    _, ratio_clean = _fit_and_noise(X, y_clean)
    _, ratio_noisy = _fit_and_noise(X, y_noisy)

    # Noisy data should retain a higher fraction (ratio closer to 1.0)
    assert ratio_noisy > ratio_clean, (
        f"Noisy ratio={ratio_noisy:.3f} should exceed clean ratio={ratio_clean:.3f}"
    )
