from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                task_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL,
                drift_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_experiment(
    *,
    run_id: str,
    created_at: str,
    task_type: str,
    model_name: str,
    primary_metric: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    artifact_path: Path,
) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                run_id, created_at, task_type, model_name, primary_metric,
                metrics_json, config_json, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                task_type,
                model_name,
                primary_metric,
                json.dumps(metrics),
                json.dumps(config),
                str(artifact_path),
            ),
        )
        conn.commit()


def list_experiments(limit: int = 100) -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT run_id, created_at, task_type, model_name, primary_metric,
                   metrics_json, config_json, artifact_path
            FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "task_type": row["task_type"],
            "model_name": row["model_name"],
            "primary_metric": row["primary_metric"],
            "metrics": json.loads(row["metrics_json"]),
            "config": json.loads(row["config_json"]),
            "artifact_path": row["artifact_path"],
        }
        for row in rows
    ]


def get_experiment(run_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT run_id, created_at, task_type, model_name, primary_metric,
                   metrics_json, config_json, artifact_path
            FROM experiments
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "task_type": row["task_type"],
        "model_name": row["model_name"],
        "primary_metric": row["primary_metric"],
        "metrics": json.loads(row["metrics_json"]),
        "config": json.loads(row["config_json"]),
        "artifact_path": row["artifact_path"],
    }


def log_prediction(
    *,
    created_at: str,
    model_version: str,
    payload: list[dict[str, Any]],
    predictions: list[Any],
    drift: dict[str, Any],
) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (
                created_at, model_version, payload_json, predictions_json, drift_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                created_at,
                model_version,
                json.dumps(payload),
                json.dumps(predictions),
                json.dumps(drift),
            ),
        )
        conn.commit()


def get_latest_prediction_log() -> dict[str, Any] | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT id, created_at, model_version, payload_json, predictions_json, drift_json
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "payload": json.loads(row["payload_json"]),
        "predictions": json.loads(row["predictions_json"]),
        "drift": json.loads(row["drift_json"]),
    }
