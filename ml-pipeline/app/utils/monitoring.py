from __future__ import annotations

from typing import Any

import numpy as np


def summarize_prediction_distribution(predictions: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(predictions, dtype=float)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def detect_distribution_drift(
    *, baseline: dict[str, Any] | None, current: dict[str, float]
) -> dict[str, Any]:
    if not baseline:
        return {"drift_detected": False, "reason": "no_baseline"}

    baseline_mean = float(baseline.get("mean", 0.0))
    baseline_std = float(baseline.get("std", 0.0))
    current_mean = float(current.get("mean", 0.0))

    threshold = baseline_std * 2 if baseline_std > 0 else max(0.1 * abs(baseline_mean), 0.1)
    mean_shift = abs(current_mean - baseline_mean)

    return {
        "drift_detected": bool(mean_shift > threshold),
        "baseline": baseline,
        "current": current,
        "mean_shift": mean_shift,
        "threshold": threshold,
    }
