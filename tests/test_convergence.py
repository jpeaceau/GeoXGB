"""
Tests for convergence_tol early stopping.

convergence_tol is a compute-efficiency feature: it stops boosting when the
mean-absolute-gradient improvement over the last 2 refit cycles falls below the
threshold.  It is NOT anti-overfitting regularisation (HVRT already prevents that).
"""
import numpy as np
import pytest
from sklearn.datasets import make_friedman1

from geoxgb import GeoXGBRegressor, GeoXGBClassifier

RNG = np.random.default_rng(42)


def _make_reg():
    X, y = make_friedman1(n_samples=500, n_features=5, noise=1.0, random_state=42)
    return X, y


def _make_clf():
    X = RNG.standard_normal((300, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 1. tol=None — no early stopping, convergence_round_ stays None
# ---------------------------------------------------------------------------

def test_no_convergence_tol_runs_all_rounds():
    X, y = _make_reg()
    m = GeoXGBRegressor(n_rounds=100, refit_interval=20,
                        convergence_tol=None, random_state=42)
    m.fit(X, y)
    assert m.convergence_round_ is None
    assert m.n_trees == 100
    assert m._convergence_losses == []


# ---------------------------------------------------------------------------
# 2. Loose threshold fires early, fewer trees
# ---------------------------------------------------------------------------

def test_loose_tol_stops_early():
    X, y = _make_reg()
    # tol=2.0 is absurdly large — should fire after the first eligible check
    m = GeoXGBRegressor(n_rounds=500, refit_interval=20,
                        convergence_tol=2.0, random_state=42)
    m.fit(X, y)
    assert m.convergence_round_ is not None
    assert m.n_trees < 500


# ---------------------------------------------------------------------------
# 3. Very tight threshold does not fire on a rapidly-improving dataset
# ---------------------------------------------------------------------------

def test_tight_tol_does_not_stop_prematurely():
    X, y = _make_reg()
    # 0.0001 — gradient on Friedman #1 drops >40% per cycle early on
    m = GeoXGBRegressor(n_rounds=200, refit_interval=20,
                        convergence_tol=0.0001, random_state=42)
    m.fit(X, y)
    # Should run to completion (or at least most of the way)
    assert m.n_trees > 100


# ---------------------------------------------------------------------------
# 4. convergence_losses are populated when tol is set
# ---------------------------------------------------------------------------

def test_convergence_losses_populated():
    X, y = _make_reg()
    m = GeoXGBRegressor(n_rounds=100, refit_interval=20,
                        convergence_tol=0.001, random_state=42)
    m.fit(X, y)
    # At least initial + round-20 entry
    assert len(m._convergence_losses) >= 2
    # Losses should be positive
    assert all(v > 0 for v in m._convergence_losses)


# ---------------------------------------------------------------------------
# 5. repr shows convergence info when fired
# ---------------------------------------------------------------------------

def test_repr_converged():
    X, y = _make_reg()
    m = GeoXGBRegressor(n_rounds=500, refit_interval=20,
                        convergence_tol=2.0, random_state=42)
    m.fit(X, y)
    r = repr(m)
    assert "converged at round" in r


def test_repr_not_converged():
    X, y = _make_reg()
    m = GeoXGBRegressor(n_rounds=60, refit_interval=20,
                        convergence_tol=None, random_state=42)
    m.fit(X, y)
    assert "fitted" in repr(m)
    assert "converged" not in repr(m)


# ---------------------------------------------------------------------------
# 6. Classifier also supports convergence_tol (binary)
# ---------------------------------------------------------------------------

def test_convergence_tol_classifier():
    X, y = _make_clf()
    m = GeoXGBClassifier(n_rounds=500, refit_interval=20,
                         convergence_tol=2.0, random_state=42)
    m.fit(X, y)
    # Should fire early on a clean linear-separable problem
    assert m.convergence_round_ is not None
    assert m.n_trees < 500


# ---------------------------------------------------------------------------
# 7. Model still predicts after early stopping
# ---------------------------------------------------------------------------

def test_predict_after_convergence():
    X, y = _make_reg()
    m = GeoXGBRegressor(n_rounds=500, refit_interval=20,
                        convergence_tol=2.0, random_state=42)
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
