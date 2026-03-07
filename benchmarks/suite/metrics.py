"""Metric computation for regression and classification tasks."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
)


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def compute_binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def compute_multiclass_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_metrics(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    if task == "regression":
        return compute_regression_metrics(y_true, y_pred)
    elif task == "binary":
        return compute_binary_metrics(y_true, y_pred, y_proba)
    else:
        return compute_multiclass_metrics(y_true, y_pred, y_proba)
