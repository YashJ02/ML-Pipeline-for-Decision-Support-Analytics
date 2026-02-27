from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _make_dataset(path: Path, rows: int = 80) -> Path:
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
    df.loc[0, "feature2"] = None
    df.to_csv(path, index=False)
    return path


def test_rejects_missing_auth(app_client):
    client, _ = app_client
    response = client.get("/pipeline/status")
    assert response.status_code == 401


def test_api_end_to_end_flow(app_client, tmp_path: Path):
    client, _ = app_client
    dataset_path = _make_dataset(tmp_path / "upload.csv")

    with dataset_path.open("rb") as handle:
        upload_response = client.post(
            "/data/upload",
            headers=_auth_headers(),
            files={"file": ("upload.csv", handle, "text/csv")},
            data={"target_column": "target"},
        )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["rows"] > 10
    assert payload["target_column"] == "target"

    run_response = client.post(
        "/pipeline/run",
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
    run_payload = run_response.json()
    assert "run_id" in run_payload
    assert run_payload["model_name"] == "logistic_regression"

    eval_response = client.get("/model/evaluate", headers=_auth_headers())
    assert eval_response.status_code == 200
    eval_payload = eval_response.json()
    assert eval_payload["run_id"] == run_payload["run_id"]

    predict_response = client.post(
        "/model/predict",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content=json.dumps(
            {
                "records": [
                    {
                        "feature1": 4.8,
                        "feature2": 2.3,
                        "feature3": 1.6,
                        "feature4": 0.3,
                        "category": "A",
                    }
                ]
            }
        ),
    )
    assert predict_response.status_code == 200
    pred_payload = predict_response.json()
    assert len(pred_payload["predictions"]) == 1
    assert "drift" in pred_payload

    experiments_response = client.get("/experiments", headers=_auth_headers())
    assert experiments_response.status_code == 200
    items = experiments_response.json()["items"]
    assert any(item["run_id"] == run_payload["run_id"] for item in items)
