from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import seaborn as sns

from app.utils.experiment import write_json

# Use a non-interactive backend for API/server execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_eda_artifacts(df: pd.DataFrame, target_column: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
        "describe_numeric": df.describe(include=["number"]).to_dict(),
    }
    write_json(output_dir / "eda_summary.json", summary_payload)

    artifacts: dict[str, str] = {
        "summary": str(output_dir / "eda_summary.json"),
    }

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        correlation = numeric_df.corr(numeric_only=True)
        write_json(output_dir / "correlation_matrix.json", correlation.fillna(0.0).to_dict())
        artifacts["correlation_matrix"] = str(output_dir / "correlation_matrix.json")

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        heatmap_path = output_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=120)
        plt.close()
        artifacts["correlation_heatmap"] = str(heatmap_path)

    plt.figure(figsize=(8, 4))
    if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        sns.histplot(df[target_column], kde=False)
    elif target_column in df.columns:
        value_counts = df[target_column].astype(str).value_counts().head(30)
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.xticks(rotation=45, ha="right")
    plt.title(f"Target Distribution: {target_column}")
    plt.tight_layout()
    target_plot_path = output_dir / "target_distribution.png"
    plt.savefig(target_plot_path, dpi=120)
    plt.close()
    artifacts["target_distribution"] = str(target_plot_path)

    return artifacts
