from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists() or path.suffix.lower() != ".csv":
        raise DataValidationError(f"Dataset path is invalid or not a CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise DataValidationError("Uploaded dataset is empty.")
    logger.info("Loaded dataset with shape %s from %s", df.shape, path)
    return df


def validate_schema(df: pd.DataFrame, target_column: str) -> dict[str, str]:
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' not found in dataset.")

    schema = {column: str(dtype) for column, dtype in df.dtypes.items()}
    logger.info("Schema validation complete for %s columns", len(schema))
    return schema
