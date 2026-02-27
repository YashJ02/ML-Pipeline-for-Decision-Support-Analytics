from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


def make_classification_csv(path: Path, rows: int = 80) -> Path:
    records: list[dict[str, float | int | str]] = []
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


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app_base_dir = tmp_path / "ml-pipeline-runtime"
    monkeypatch.setenv("APP_BASE_DIR", str(app_base_dir))
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("APP_ENV", "test")

    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.api.routes as routes_module
    import app.main as main_module

    importlib.reload(routes_module)
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, app_base_dir
