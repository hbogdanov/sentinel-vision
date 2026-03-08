from __future__ import annotations

from typing import Any

from fastapi import FastAPI

app = FastAPI(title="Sentinel Vision Alerts API")
_events: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/alerts")
def list_alerts(limit: int = 50) -> list[dict[str, Any]]:
    return _events[-limit:]


@app.post("/ingest")
def ingest_alert(event: dict[str, Any]) -> dict[str, int]:
    _events.append(event)
    return {"stored": len(_events)}
