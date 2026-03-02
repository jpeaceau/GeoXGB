from geoxgb.regressor import GeoXGBRegressor
from geoxgb.classifier import GeoXGBClassifier
from geoxgb.gardener import Gardener
from geoxgb import report
from geoxgb.explain import (
    GeoXGBExplainer,
    Explanation,
    PathNode,
    Neighbour,
    PartitionGeometry,
    format_explanation,
    print_explanation,
    format_summary,
    print_summary,
)

try:
    from geoxgb.optimizer import GeoXGBOptimizer
    __all__ = ["GeoXGBRegressor", "GeoXGBClassifier", "GeoXGBOptimizer",
               "Gardener", "load_model", "report",
               "GeoXGBExplainer", "Explanation", "PathNode", "Neighbour",
               "PartitionGeometry", "format_explanation", "print_explanation",
               "format_summary", "print_summary"]
except ImportError:
    __all__ = ["GeoXGBRegressor", "GeoXGBClassifier", "Gardener",
               "load_model", "report",
               "GeoXGBExplainer", "Explanation", "PathNode", "Neighbour",
               "PartitionGeometry", "format_explanation", "print_explanation",
               "format_summary", "print_summary"]


def load_model(path):
    """
    Load a GeoXGB model saved with ``model.save(path)``.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    GeoXGBRegressor | GeoXGBClassifier

    Examples
    --------
    >>> from geoxgb import load_model
    >>> model = load_model("heart_disease.pkl")
    >>> model.predict_proba(X_test)
    """
    import joblib
    return joblib.load(path)


__version__ = "0.1.4"
