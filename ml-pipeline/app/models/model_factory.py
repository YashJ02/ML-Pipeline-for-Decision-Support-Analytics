from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

logger = logging.getLogger(__name__)


def _with_overrides(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return defaults
    merged = defaults.copy()
    merged.update(overrides)
    return merged


def build_models(
    task_type: str,
    random_seed: int,
    hyperparams: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hyperparams = hyperparams or {}

    if task_type == "classification":
        models: dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                **_with_overrides({"max_iter": 1000, "random_state": random_seed}, hyperparams.get("logistic_regression"))
            ),
            "random_forest": RandomForestClassifier(
                **_with_overrides({"n_estimators": 200, "random_state": random_seed}, hyperparams.get("random_forest"))
            ),
            "gradient_boosting": GradientBoostingClassifier(
                **_with_overrides({"random_state": random_seed}, hyperparams.get("gradient_boosting"))
            ),
        }
        try:
            from xgboost import XGBClassifier  # type: ignore

            models["xgboost"] = XGBClassifier(
                **_with_overrides(
                    {
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "max_depth": 6,
                        "subsample": 0.9,
                        "colsample_bytree": 0.9,
                        "eval_metric": "logloss",
                        "random_state": random_seed,
                    },
                    hyperparams.get("xgboost"),
                )
            )
        except Exception:
            logger.info("xgboost not available; skipping xgboost model.")

        return models

    models = {
        "linear_regression": LinearRegression(
            **_with_overrides({}, hyperparams.get("linear_regression"))
        ),
        "random_forest": RandomForestRegressor(
            **_with_overrides({"n_estimators": 200, "random_state": random_seed}, hyperparams.get("random_forest"))
        ),
        "gradient_boosting": GradientBoostingRegressor(
            **_with_overrides({"random_state": random_seed}, hyperparams.get("gradient_boosting"))
        ),
    }

    try:
        from xgboost import XGBRegressor  # type: ignore

        models["xgboost"] = XGBRegressor(
            **_with_overrides(
                {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "random_state": random_seed,
                },
                hyperparams.get("xgboost"),
            )
        )
    except Exception:
        logger.info("xgboost not available; skipping xgboost model.")

    return models
