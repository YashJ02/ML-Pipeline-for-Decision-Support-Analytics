from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, model_store_dir: Path) -> None:
        self.model_store_dir = model_store_dir
        self.model_store_dir.mkdir(parents=True, exist_ok=True)
        self.latest_model_path = self.model_store_dir / "latest_model.joblib"
        self.latest_metadata_path = self.model_store_dir / "latest_metadata.json"

    def save_latest(self, model: Any, metadata: dict[str, Any]) -> None:
        joblib.dump(model, self.latest_model_path)
        self.latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_latest_model(self) -> Any:
        if not self.latest_model_path.exists():
            raise FileNotFoundError("No approved model found in registry.")
        return joblib.load(self.latest_model_path)

    def load_latest_metadata(self) -> dict[str, Any]:
        if not self.latest_metadata_path.exists():
            raise FileNotFoundError("No metadata found for latest model.")
        return json.loads(self.latest_metadata_path.read_text(encoding="utf-8"))
