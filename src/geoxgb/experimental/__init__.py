"""
geoxgb.experimental
====================
Experimental strategies not yet part of the stable API.

Contents
--------
detect_spikes_pyramid(X_z, sigma_threshold=2.5)
    Detect ell-infty spike samples via suppressed |A|/||z||_1 ratio.
    Spike samples have A determined by remaining non-spike features
    (Proposition 1 item 3), so ||z||_1 is dominated by the spike while
    |A| stays at the non-spike level -- causing the ratio to fall well
    below the ambient expectation ~(1 - 1/sqrt(d)).

_pyramid_hvrt_expand(pyramid_model, y, n_expand, ...)
    Hierarchical augmentation: HVRT expand() within each PyramidHART cell.
    Synthetic samples respect both the polyhedral outer structure (spike
    robustness) and the inner hyperboloidal T-structure (Theorem 3 holds
    per-partition as an algebraic identity).

Use generation_strategy='pyramid_hvrt' with partitioner='pyramid_hart'
in GeoXGBRegressor to activate Strategy 2 during training.
"""

from geoxgb.experimental._synergy import (
    detect_spikes_pyramid,
    _pyramid_hvrt_expand,
)

__all__ = [
    "detect_spikes_pyramid",
    "_pyramid_hvrt_expand",
]
