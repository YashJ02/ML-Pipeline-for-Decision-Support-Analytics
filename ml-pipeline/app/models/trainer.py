from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.models.model_factory import build_models
from app.pipelines.evaluation import evaluate_predictions, is_higher_better, primary_metric_name
from app.pipelines.feature_engineering import FeatureEngineer
from app.pipelines.preprocessing import IQRClipper

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model_name: str
    model: Pipeline
    validation_metrics: dict[str, float]
    cross_validation_score: float


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = list(X.select_dtypes(include=["number"]).columns)
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _score_for_selection(metric_name: str, metric_value: float) -> float:
    if is_higher_better(metric_name):
        return metric_value
    return -metric_value


def train_candidates(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    random_seed: int,
    hyperparams: dict[str, dict[str, Any]] | None,
    allowed_models: list[str] | None,
) -> tuple[TrainingResult, dict[str, dict[str, Any]]]:
    feature_engineer = FeatureEngineer()
    engineered_train = feature_engineer.fit_transform(X_train)
    preprocessor = _build_preprocessor(engineered_train)
    models = build_models(task_type=task_type, random_seed=random_seed, hyperparams=hyperparams)

    if allowed_models:
        models = {name: model for name, model in models.items() if name in set(allowed_models)}

    if not models:
        raise ValueError("No eligible models found for training.")

    metric_name = primary_metric_name(task_type)
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] = {}
    best_cv_score = float("-inf")
    best_selection_score = float("-inf")
    candidate_results: dict[str, dict[str, Any]] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", feature_engineer),
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        val_metrics = evaluate_predictions(task_type, y_val, y_val_pred)

        scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=3,
            scoring=scoring,
            n_jobs=None,
        )
        cv_mean = float(np.mean(cv_scores))

        selection_score = _score_for_selection(metric_name, val_metrics[metric_name])
        candidate_results[model_name] = {
            "validation_metrics": val_metrics,
            "cross_validation_score": cv_mean,
        }

        logger.info(
            "Candidate %s: validation=%s cv=%.4f",
            model_name,
            val_metrics,
            cv_mean,
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = val_metrics
            best_cv_score = cv_mean

    if best_name is None or best_pipeline is None:
        raise RuntimeError("Model selection failed unexpectedly.")

    return (
        TrainingResult(
            model_name=best_name,
            model=best_pipeline,
            validation_metrics=best_metrics,
            cross_validation_score=best_cv_score,
        ),
        candidate_results,
    )
