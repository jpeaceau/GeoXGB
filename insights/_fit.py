"""
Model fitting helpers — GeoXGB and XGBoost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from geoxgb import GeoXGBClassifier


@dataclass
class FitResult:
    """Container returned by fit_geoxgb / fit_xgboost."""
    model: object
    proba: np.ndarray   # predict_proba on test set, shape (n_test, 2)
    auc: float
    elapsed: float      # wall-clock seconds


def fit_geoxgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_types=None,
    **params,
) -> FitResult:
    """
    Fit a GeoXGBClassifier and evaluate on the test set.

    Parameters
    ----------
    feature_types : list of str or None
        Per-column type hints ('continuous' or 'categorical').
        Passed directly to GeoXGBClassifier.fit().
    **params : forwarded to GeoXGBClassifier constructor.
    """
    t0 = time.perf_counter()
    clf = GeoXGBClassifier(**params)
    clf.fit(X_train, y_train, feature_types=feature_types)
    elapsed = time.perf_counter() - t0

    proba = clf.predict_proba(X_test)
    auc = float(roc_auc_score(y_test, proba[:, 1]))
    return FitResult(model=clf, proba=proba, auc=auc, elapsed=elapsed)


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **params,
) -> FitResult:
    """
    Fit an XGBClassifier and evaluate on the test set.

    Parameters
    ----------
    **params : forwarded to XGBClassifier (overrides XGB_PARAMS defaults).
    """
    t0 = time.perf_counter()
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    proba = clf.predict_proba(X_test)
    auc = float(roc_auc_score(y_test, proba[:, 1]))
    return FitResult(model=clf, proba=proba, auc=auc, elapsed=elapsed)
