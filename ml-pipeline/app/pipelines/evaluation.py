from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

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


def _safe_feature_names(model_pipeline: Any) -> list[str]:
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return []

    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return []


def _safe_transformed_matrix(model_pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    feature_engineer = model_pipeline.named_steps.get("feature_engineer")
    preprocessor = model_pipeline.named_steps.get("preprocessor")
    if feature_engineer is None or preprocessor is None:
        return np.zeros((len(X), 0), dtype=float)

    engineered = feature_engineer.transform(X)
    transformed = preprocessor.transform(engineered)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed, dtype=float)


def _global_importance_vector(model: Any) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.abs(np.asarray(model.feature_importances_, dtype=float))

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.mean(np.abs(coef), axis=0)

    return np.array([], dtype=float)


def summarize_feature_importance(
    model_pipeline: Any,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    model = model_pipeline.named_steps.get("model")
    if model is None:
        return {
            "method": "unsupported",
            "global_feature_importance": [],
            "note": "Model step is missing in pipeline.",
        }

    feature_names = _safe_feature_names(model_pipeline)
    importance = _global_importance_vector(model)

    if importance.size == 0:
        return {
            "method": "unsupported",
            "global_feature_importance": [],
            "note": "Model does not expose feature importances or coefficients.",
        }

    if not feature_names or len(feature_names) != len(importance):
        feature_names = [f"feature_{index}" for index in range(len(importance))]

    total = float(np.sum(importance))
    if total <= 0:
        normalized = np.zeros_like(importance)
    else:
        normalized = importance / total

    ranked_indices = np.argsort(normalized)[::-1][:top_k]
    ranked = [
        {
            "feature": feature_names[index],
            "importance": float(normalized[index]),
        }
        for index in ranked_indices
    ]

    method = "feature_importances" if hasattr(model, "feature_importances_") else "coefficients"
    return {
        "method": method,
        "global_feature_importance": ranked,
        "note": "SHAP-style approximation from model internals (not exact SHAP values).",
    }


def summarize_prediction_explanations(
    model_pipeline: Any,
    records: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not records:
        return []

    model = model_pipeline.named_steps.get("model")
    if model is None:
        return []

    X = pd.DataFrame(records)
    transformed = _safe_transformed_matrix(model_pipeline, X)
    if transformed.size == 0:
        return []

    feature_names = _safe_feature_names(model_pipeline)
    if not feature_names or len(feature_names) != transformed.shape[1]:
        feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            weight = coef
        else:
            weight = np.mean(coef, axis=0)
        contribution_matrix = transformed * weight
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
        contribution_matrix = np.abs(transformed) * np.abs(importance)
    else:
        return []

    explanations: list[dict[str, Any]] = []
    for row_index in range(contribution_matrix.shape[0]):
        row = contribution_matrix[row_index]
        ranked_indices = np.argsort(np.abs(row))[::-1][:top_k]
        top_features = [
            {
                "feature": feature_names[index],
                "contribution": float(row[index]),
                "direction": "positive" if row[index] >= 0 else "negative",
            }
            for index in ranked_indices
        ]
        explanations.append(
            {
                "record_index": row_index,
                "top_features": top_features,
            }
        )

    return explanations
