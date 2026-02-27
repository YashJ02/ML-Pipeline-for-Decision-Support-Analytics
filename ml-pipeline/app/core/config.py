from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    env: str
    api_token: str
    base_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    experiments_dir: Path
    model_store_dir: Path
    db_path: Path
    log_level: str
    random_seed: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_dir = Path(os.getenv("APP_BASE_DIR", str(Path(__file__).resolve().parents[2])))
    data_dir = base_dir / "data"
    experiments_dir = base_dir / "experiments"
    model_store_dir = experiments_dir / "model_store"

    settings = Settings(
        app_name=os.getenv("APP_NAME", "ML Pipeline for Decision Support & Analytics"),
        env=os.getenv("APP_ENV", "dev"),
        api_token=os.getenv("API_TOKEN", "dev-token"),
        base_dir=base_dir,
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        experiments_dir=experiments_dir,
        model_store_dir=model_store_dir,
        db_path=experiments_dir / "ml_pipeline.db",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )

    for directory in (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.experiments_dir,
        settings.model_store_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return settings
