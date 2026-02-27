from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import get_settings
from app.core.database import (
    get_experiment,
    get_latest_prediction_log,
    init_db,
    log_experiment,
    log_prediction,
)
from app.core.state import pipeline_status
from app.models.model_registry import ModelRegistry
from app.models.trainer import train_candidates
from app.pipelines.eda import generate_eda_artifacts
from app.pipelines.evaluation import (
    evaluate_predictions,
    primary_metric_name,
    summarize_feature_importance,
    summarize_prediction_explanations,
)
from app.pipelines.ingestion import load_dataset, validate_schema
from app.utils.experiment import new_run_id, save_model, write_json
from app.utils.monitoring import detect_distribution_drift, summarize_prediction_distribution

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(self) -> None:
        self.settings = get_settings()
        init_db()
        self.registry = ModelRegistry(self.settings.model_store_dir)
        self.latest_data_pointer = self.settings.processed_data_dir / "latest_uploaded_path.txt"

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _resolve_dataset_path(self, dataset_path: str | None) -> Path:
        if dataset_path:
            candidate = Path(dataset_path)
            if candidate.is_absolute():
                return candidate
            return self.settings.base_dir / candidate

        if self.latest_data_pointer.exists():
            latest_path = self.latest_data_pointer.read_text(encoding="utf-8").strip()
            if latest_path:
                return Path(latest_path)

        raise FileNotFoundError("No dataset path provided and no uploaded dataset found.")

    def set_latest_dataset_path(self, path: Path) -> None:
        self.latest_data_pointer.write_text(str(path.resolve()), encoding="utf-8")

    def inspect_dataset(self, dataset_path: Path, target_column: str | None = None) -> dict[str, Any]:
        df = load_dataset(dataset_path)
        schema = {column: str(dtype) for column, dtype in df.dtypes.items()}
        target_info: dict[str, Any] = {}

        if target_column and target_column in df.columns:
            target_info = {
                "target_column": target_column,
                "target_unique_values": int(df[target_column].nunique(dropna=False)),
                "target_dtype": str(df[target_column].dtype),
            }

        return {
            "path": str(dataset_path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "schema": schema,
            "preview": df.head(5).to_dict(orient="records"),
            **target_info,
        }

    def _infer_task_type(self, y: pd.Series, requested: str) -> str:
        if requested in {"classification", "regression"}:
            return requested

        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return "classification"

        unique_count = y.nunique(dropna=True)
        if unique_count <= max(20, int(len(y) * 0.05)):
            return "classification"
        return "regression"

    def _train_val_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        random_seed: int,
        test_size: float,
        val_size: float,
    ):
        stratify = y if task_type == "classification" else None

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )

        val_relative = val_size / (1 - test_size)
        stratify_train = y_train_val if task_type == "classification" else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_relative,
                random_state=random_seed,
                stratify=stratify_train,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_relative,
                random_state=random_seed,
            )

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

    def run_training(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id: str | None = None
        try:
            pipeline_status.update(state="running", message="Pipeline execution started.")
            dataset_path = self._resolve_dataset_path(payload.get("dataset_path"))
            target_column = payload["target_column"]
            task_requested = payload.get("task_type", "auto")
            hyperparams = payload.get("hyperparameters")
            allowed_models = payload.get("models")
            test_size = float(payload.get("test_size", 0.2))
            val_size = float(payload.get("val_size", 0.1))

            if not (0.05 <= test_size <= 0.4):
                raise ValueError("test_size must be between 0.05 and 0.4")
            if not (0.05 <= val_size <= 0.3):
                raise ValueError("val_size must be between 0.05 and 0.3")

            df = load_dataset(dataset_path)
            schema = validate_schema(df, target_column)
            run_id = new_run_id()
            run_dir = self.settings.experiments_dir / run_id
            eda_artifacts = generate_eda_artifacts(df, target_column, run_dir)

            y = df[target_column]
            X = df.drop(columns=[target_column])
            task_type = self._infer_task_type(y, task_requested)

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                X_train_val,
                y_train_val,
            ) = self._train_val_test_split(
                X,
                y,
                task_type,
                self.settings.random_seed,
                test_size,
                val_size,
            )

            best_result, candidates = train_candidates(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task_type=task_type,
                random_seed=self.settings.random_seed,
                hyperparams=hyperparams,
                allowed_models=allowed_models,
            )

            # Refit the selected pipeline on train+validation for stronger final model.
            best_result.model.fit(X_train_val, y_train_val)
            y_test_pred = best_result.model.predict(X_test)
            test_metrics = evaluate_predictions(task_type, y_test, y_test_pred)
            explainability_payload = summarize_feature_importance(best_result.model)
            sample_records = X_test.head(3).to_dict(orient="records")
            explainability_payload["test_sample_explanations"] = summarize_prediction_explanations(
                best_result.model,
                sample_records,
                top_k=5,
            )

            created_at = self._timestamp()
            artifact_path = run_dir / "model.joblib"

            primary_metric = primary_metric_name(task_type)
            metrics_payload = {
                "task_type": task_type,
                "primary_metric": primary_metric,
                "validation_metrics": best_result.validation_metrics,
                "test_metrics": test_metrics,
                "cross_validation_score": best_result.cross_validation_score,
                "candidate_results": candidates,
            }

            config_payload = {
                "dataset_path": str(dataset_path),
                "target_column": target_column,
                "task_type": task_type,
                "models": allowed_models,
                "hyperparameters": hyperparams,
                "test_size": test_size,
                "val_size": val_size,
                "random_seed": self.settings.random_seed,
            }

            numeric_predictions = pd.Series(y_test_pred).astype("category").cat.codes.to_list()
            prediction_stats = summarize_prediction_distribution(numeric_predictions)

            metadata_payload = {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": best_result.model_name,
                "task_type": task_type,
                "artifact_path": str(artifact_path),
                "metrics": metrics_payload,
                "explainability": explainability_payload,
                "schema": schema,
                "prediction_baseline": prediction_stats,
                "eda_artifacts": eda_artifacts,
            }

            save_model(artifact_path, best_result.model)
            write_json(run_dir / "metrics.json", metrics_payload)
            write_json(run_dir / "config.json", config_payload)
            write_json(run_dir / "schema.json", schema)
            write_json(run_dir / "metadata.json", metadata_payload)

            self.registry.save_latest(best_result.model, metadata_payload)
            log_experiment(
                run_id=run_id,
                created_at=created_at,
                task_type=task_type,
                model_name=best_result.model_name,
                primary_metric=primary_metric,
                metrics=metrics_payload,
                config=config_payload,
                artifact_path=artifact_path,
            )

            pipeline_status.update(
                state="completed",
                message="Pipeline completed successfully.",
                last_run_id=run_id,
                details={
                    "model_name": best_result.model_name,
                    "primary_metric": primary_metric,
                    "test_metrics": test_metrics,
                },
            )

            return {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": best_result.model_name,
                "task_type": task_type,
                "metrics": metrics_payload,
                "explainability": explainability_payload,
                "eda_artifacts": eda_artifacts,
            }

        except Exception as exc:
            logger.exception("Pipeline run failed")
            pipeline_status.update(
                state="failed",
                message=str(exc),
                last_run_id=run_id,
            )
            raise

    def evaluate_latest(self) -> dict[str, Any]:
        metadata = self.registry.load_latest_metadata()
        return {
            "run_id": metadata["run_id"],
            "model_name": metadata["model_name"],
            "task_type": metadata["task_type"],
            "metrics": metadata["metrics"],
            "explainability": metadata.get("explainability", {}),
        }

    def get_run_details(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.settings.experiments_dir / run_id / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        experiment = get_experiment(run_id)
        if experiment is None:
            raise FileNotFoundError(f"Run '{run_id}' was not found.")

        return {
            "run_id": experiment["run_id"],
            "created_at": experiment["created_at"],
            "model_name": experiment["model_name"],
            "task_type": experiment["task_type"],
            "metrics": experiment["metrics"],
            "config": experiment["config"],
            "artifact_path": experiment["artifact_path"],
            "explainability": {},
        }

    def latest_prediction_payload(self) -> dict[str, Any]:
        latest = get_latest_prediction_log()
        if latest is None:
            raise FileNotFoundError("No prediction logs found yet.")

        return {
            "id": latest["id"],
            "created_at": latest["created_at"],
            "run_id": latest["model_version"],
            "records": latest["payload"],
            "predictions": latest["predictions"],
            "drift": latest["drift"],
        }

    def predict(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("Prediction payload is empty.")

        model = self.registry.load_latest_model()
        metadata = self.registry.load_latest_metadata()

        payload_df = pd.DataFrame(records)
        predictions = model.predict(payload_df)
        prediction_list = [
            value.item() if isinstance(value, np.generic) else value for value in predictions.tolist()
        ]
        local_explanations = summarize_prediction_explanations(model, records, top_k=5)

        numeric_predictions = pd.Series(prediction_list).astype("category").cat.codes.to_list()
        current_distribution = summarize_prediction_distribution(numeric_predictions)
        drift = detect_distribution_drift(
            baseline=metadata.get("prediction_baseline"),
            current=current_distribution,
        )

        created_at = self._timestamp()
        log_prediction(
            created_at=created_at,
            model_version=metadata["run_id"],
            payload=records,
            predictions=prediction_list,
            drift=drift,
        )

        return {
            "run_id": metadata["run_id"],
            "model_name": metadata["model_name"],
            "predictions": prediction_list,
            "explainability": {
                "global": metadata.get("explainability", {}),
                "local": local_explanations,
            },
            "drift": drift,
            "created_at": created_at,
        }
