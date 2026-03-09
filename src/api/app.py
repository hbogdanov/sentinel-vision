from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from src.api.models import AlertEvent, AlertListResponse, CameraListResponse, CameraSummary, HealthResponse, IngestResponse, StatsResponse
from src.api.storage import SQLiteAlertStore


def _default_db_path() -> str:
    return os.getenv("SENTINEL_ALERTS_DB_PATH", "data/outputs/alerts.db")


app = FastAPI(title="Sentinel Vision Alerts API")
store = SQLiteAlertStore(_default_db_path())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", db_path=str(store.db_path).replace("\\", "/"), alerts=store.count())


@app.get("/alerts", response_model=AlertListResponse)
def list_alerts(
    camera_id: str | None = None,
    event_type: str | None = None,
    zone: str | None = None,
    start_time: str | None = Query(default=None, description="Inclusive ISO-8601 UTC timestamp"),
    end_time: str | None = Query(default=None, description="Inclusive ISO-8601 UTC timestamp"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AlertListResponse:
    alerts = store.list_alerts(
        camera_id=camera_id,
        event_type=event_type,
        zone=zone,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return AlertListResponse(alerts=[AlertEvent.model_validate(alert) for alert in alerts], limit=limit, offset=offset)


@app.get("/alerts/{event_id}", response_model=AlertEvent)
def get_alert(event_id: str) -> AlertEvent:
    alert = store.get_alert(event_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert '{event_id}' was not found.")
    return AlertEvent.model_validate(alert)


@app.get("/stats", response_model=StatsResponse)
def stats(
    camera_id: str | None = None,
    event_type: str | None = None,
    zone: str | None = None,
    start_time: str | None = Query(default=None, description="Inclusive ISO-8601 UTC timestamp"),
    end_time: str | None = Query(default=None, description="Inclusive ISO-8601 UTC timestamp"),
) -> StatsResponse:
    return StatsResponse.model_validate(
        store.stats(
        camera_id=camera_id,
        event_type=event_type,
        zone=zone,
        start_time=start_time,
        end_time=end_time,
    ))


@app.get("/cameras", response_model=CameraListResponse)
def list_cameras() -> CameraListResponse:
    return CameraListResponse(cameras=[CameraSummary.model_validate(camera) for camera in store.camera_summaries()])


@app.post("/ingest", response_model=IngestResponse)
def ingest_alert(event: AlertEvent) -> IngestResponse:
    store.ingest(event)
    return IngestResponse(stored=True, event_id=event.event_id, total_alerts=store.count())
