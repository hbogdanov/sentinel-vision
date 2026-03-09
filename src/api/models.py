from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AlertEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    event_id: str
    timestamp: datetime
    camera_id: str
    event_type: str
    track_id: int | None = None
    object_class: str = Field(alias="class")
    confidence: float | None = None
    zone: str
    frame_index: int
    snapshot_path: str | None = None
    clip_path: str | None = None
    metadata_path: str | None = None
    fps: float | None = None


class HealthResponse(BaseModel):
    status: str
    db_path: str
    alerts: int


class IngestResponse(BaseModel):
    stored: bool
    event_id: str
    total_alerts: int


class StatsResponse(BaseModel):
    total_alerts: int
    by_event_type: dict[str, int]
    by_camera_id: dict[str, int]
    latest_timestamp: str | None
    db_path: str


class CameraSummary(BaseModel):
    camera_id: str
    alerts: int
    latest_timestamp: str | None


class CameraListResponse(BaseModel):
    cameras: list[CameraSummary]


class AlertListResponse(BaseModel):
    alerts: list[AlertEvent]
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]
