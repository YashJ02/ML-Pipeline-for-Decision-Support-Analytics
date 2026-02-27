from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _make_dataset(path: Path, rows: int = 100) -> Path:
    records = []
    for idx in range(rows):
        label = 1 if idx % 2 == 0 else 0
        records.append(
            {
                "feature1": 4.0 + (idx % 10) * 0.2,
                "feature2": 2.0 + (idx % 5) * 0.3,
                "feature3": 1.0 + label * 1.5 + (idx % 3) * 0.1,
                "feature4": 0.1 + label * 0.2,
                "category": "A" if label else "B",
                "target": label,
            }
        )
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)
    return path


def test_pipeline_outputs_expected_artifacts(app_client, tmp_path: Path):
    client, app_base_dir = app_client
    dataset_path = _make_dataset(tmp_path / "regression_fixture.csv")

    with dataset_path.open("rb") as handle:
        upload_response = client.post(
            "/data/upload",
            headers=_auth_headers(),
            files={"file": ("fixture.csv", handle, "text/csv")},
            data={"target_column": "target"},
        )
    assert upload_response.status_code == 200

    run_response = client.post(
        "/model/train",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content=json.dumps(
            {
                "target_column": "target",
                "task_type": "classification",
                "models": ["logistic_regression"],
                "test_size": 0.2,
                "val_size": 0.1,
            }
        ),
    )
    assert run_response.status_code == 200
    run_id = run_response.json()["run_id"]

    run_dir = app_base_dir / "experiments" / run_id
    expected_files = [
        "model.joblib",
        "metrics.json",
        "config.json",
        "schema.json",
        "metadata.json",
        "eda_summary.json",
        "target_distribution.png",
    ]
    for name in expected_files:
        assert (run_dir / name).exists(), f"Expected artifact missing: {name}"
