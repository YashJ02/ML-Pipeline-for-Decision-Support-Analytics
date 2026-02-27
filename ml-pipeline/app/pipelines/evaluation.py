from __future__ import annotations

import math

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_predictions(task_type: str, y_true, y_pred) -> dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def primary_metric_name(task_type: str) -> str:
    return "f1" if task_type == "classification" else "rmse"


def is_higher_better(metric_name: str) -> bool:
    return metric_name != "rmse"
