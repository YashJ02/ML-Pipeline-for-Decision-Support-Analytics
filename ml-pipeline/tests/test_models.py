from __future__ import annotations

import pandas as pd

from app.models.model_factory import build_models
from app.models.trainer import train_candidates


def test_model_factory_has_baseline_models_for_classification():
    models = build_models(task_type="classification", random_seed=42)
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models


def test_train_candidates_returns_model_and_metrics():
    X = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.9, 1.0, 0.15, 0.95, 0.3, 0.8],
            "f2": [1, 1, 2, 2, 1, 2, 1, 2],
            "cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])

    X_train = X.iloc[:6]
    y_train = y.iloc[:6]
    X_val = X.iloc[6:]
    y_val = y.iloc[6:]

    result, candidates = train_candidates(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="classification",
        random_seed=42,
        hyperparams=None,
        allowed_models=["logistic_regression"],
    )

    assert result.model_name == "logistic_regression"
    assert "f1" in result.validation_metrics
    assert "logistic_regression" in candidates
