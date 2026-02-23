"""
Classification tests for GeoXGBClassifier.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from geoxgb import GeoXGBClassifier

RNG = np.random.default_rng(1)


def _make_binary(n=300, noise=False):
    X, y = make_classification(
        n_samples=n, n_features=5, n_informative=3,
        n_redundant=1, random_state=0,
    )
    return X, y


# ---------------------------------------------------------------------------
# 1. Binary smoke
# ---------------------------------------------------------------------------

def test_binary_smoke():
    X, y = _make_binary()
    clf = GeoXGBClassifier(n_rounds=30, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert accuracy_score(y, preds) > 0.8


# ---------------------------------------------------------------------------
# 2. Binary predict_proba shape and row sums
# ---------------------------------------------------------------------------

def test_binary_proba_shape():
    X, y = _make_binary()
    clf = GeoXGBClassifier(n_rounds=30, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Multiclass (3 classes)
# ---------------------------------------------------------------------------

def test_multiclass_3():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    clf = GeoXGBClassifier(n_rounds=30, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert accuracy_score(y, preds) > 0.7


# ---------------------------------------------------------------------------
# 4. Multiclass (5 classes) — predict_proba shape
# ---------------------------------------------------------------------------

def test_multiclass_5_proba_shape():
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5,
        n_classes=5, n_clusters_per_class=1, random_state=0,
    )
    clf = GeoXGBClassifier(n_rounds=20, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. String labels
# ---------------------------------------------------------------------------

def test_string_labels():
    X, y_int = make_classification(
        n_samples=200, n_features=6, n_informative=4,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    label_map = {0: "cat", 1: "dog", 2: "bird"}
    y_str = np.array([label_map[v] for v in y_int])
    clf = GeoXGBClassifier(n_rounds=20, random_state=0)
    clf.fit(X, y_str)
    preds = clf.predict(X)
    assert set(preds).issubset({"cat", "dog", "bird"})


# ---------------------------------------------------------------------------
# 6. Binary imbalanced (90/10 split)
# ---------------------------------------------------------------------------

def test_binary_imbalanced():
    n = 300
    X_pos = RNG.normal(2.0, 1.0, (n // 10, 5))
    X_neg = RNG.normal(0.0, 1.0, (n - n // 10, 5))
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([np.zeros(n - n // 10), np.ones(n // 10)])
    clf = GeoXGBClassifier(n_rounds=30, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert accuracy_score(y, preds) > 0.7
