from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CameraHealthMonitor:
    camera_id: str
    status_path: Path
    reconnect_attempts: int = 0
    read_failures: int = 0
    detector_failures: int = 0
    last_frame_timestamp: str | None = None
    last_successful_read_timestamp: str | None = None
    last_detector_failure_timestamp: str | None = None
    last_reconnect_timestamp: str | None = None
    status: str = "initializing"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

    def mark_frame_read(self) -> None:
        now = _utc_now()
        iso = now.isoformat().replace("+00:00", "Z")
        self.last_frame_timestamp = iso
        self.last_successful_read_timestamp = iso
        self.status = "online"
        self.persist()

    def mark_read_failure(self) -> None:
        self.read_failures += 1
        self.status = "degraded"
        self.persist()

    def mark_detector_failure(self) -> None:
        self.detector_failures += 1
        self.last_detector_failure_timestamp = (
            _utc_now().isoformat().replace("+00:00", "Z")
        )
        self.status = "degraded"
        self.persist()

    def mark_reconnect(self, successful: bool) -> None:
        self.reconnect_attempts += 1
        self.last_reconnect_timestamp = _utc_now().isoformat().replace("+00:00", "Z")
        self.status = "online" if successful else "reconnecting"
        self.persist()

    def mark_offline(self) -> None:
        self.status = "offline"
        self.persist()

    def snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "read_failures": self.read_failures,
            "detector_failures": self.detector_failures,
            "reconnect_attempts": self.reconnect_attempts,
            "last_frame_timestamp": self.last_frame_timestamp,
            "last_successful_read_timestamp": self.last_successful_read_timestamp,
            "last_detector_failure_timestamp": self.last_detector_failure_timestamp,
            "last_reconnect_timestamp": self.last_reconnect_timestamp,
            **self.extra,
        }

    def persist(self) -> None:
        self.status_path.write_text(
            json.dumps(self.snapshot(), indent=2), encoding="utf-8"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
