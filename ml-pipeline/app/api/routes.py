from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    DatasetInspectResponse,
    ModelTrainRequest,
    PipelineRunRequest,
    PipelineStatusResponse,
    PredictRequest,
)
from app.core.config import get_settings
from app.core.database import list_experiments
from app.core.security import verify_token
from app.core.state import pipeline_status
from app.pipelines.pipeline_service import PipelineService

router = APIRouter()
service = PipelineService()
settings = get_settings()


@router.post("/data/upload", response_model=DatasetInspectResponse)
def upload_data(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
    _: str = Depends(verify_token),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    destination = settings.raw_data_dir / file.filename
    destination.write_bytes(file.file.read())
    service.set_latest_dataset_path(destination)

    try:
        details = service.inspect_dataset(destination, target_column=target_column)
        return DatasetInspectResponse(**details)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pipeline/run")
def run_pipeline(request: PipelineRunRequest, _: str = Depends(verify_token)):
    try:
        return service.run_training(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status_endpoint(_: str = Depends(verify_token)):
    return PipelineStatusResponse(
        state=pipeline_status.state,
        message=pipeline_status.message,
        updated_at=pipeline_status.updated_at,
        last_run_id=pipeline_status.last_run_id,
        details=pipeline_status.details,
    )


@router.post("/model/train")
def train_model(request: ModelTrainRequest, _: str = Depends(verify_token)):
    try:
        return service.run_training(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model/evaluate")
def evaluate_model(_: str = Depends(verify_token)):
    try:
        return service.evaluate_latest()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/model/predict")
def predict(request: PredictRequest, _: str = Depends(verify_token)):
    try:
        return service.predict(request.records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments")
def experiments(limit: int = 100, _: str = Depends(verify_token)):
    return {"items": list_experiments(limit=limit)}


@router.get("/experiments/{run_id}")
def experiment_details(run_id: str, _: str = Depends(verify_token)):
    try:
        return service.get_run_details(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/predictions/latest")
def latest_prediction_payload(_: str = Depends(verify_token)):
    try:
        return service.latest_prediction_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
