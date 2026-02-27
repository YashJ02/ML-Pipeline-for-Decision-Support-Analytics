from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    dataset_path: str | None = Field(default=None)
    target_column: str
    task_type: Literal["auto", "classification", "regression"] = "auto"
    models: list[str] | None = None
    hyperparameters: dict[str, dict[str, Any]] | None = None
    test_size: float = 0.2
    val_size: float = 0.1


class ModelTrainRequest(PipelineRunRequest):
    pass


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


class DatasetInspectResponse(BaseModel):
    path: str
    rows: int
    columns: int
    schema: dict[str, str]
    preview: list[dict[str, Any]]
    target_column: str | None = None
    target_unique_values: int | None = None
    target_dtype: str | None = None


class PipelineStatusResponse(BaseModel):
    state: str
    message: str
    updated_at: str
    last_run_id: str | None
    details: dict[str, Any]
