from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PipelineStatus:
    state: str = "idle"
    message: str = "No runs started yet."
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    last_run_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        state: str,
        message: str,
        last_run_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.state = state
        self.message = message
        self.last_run_id = last_run_id
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.details = details or {}


pipeline_status = PipelineStatus()
