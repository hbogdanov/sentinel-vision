from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.storage import SQLiteAlertStore


def test_alert_api_endpoints_support_persistence_and_filters(tmp_path: Path) -> None:
    app_module.store = SQLiteAlertStore(tmp_path / "alerts.db")
    client = TestClient(app_module.app)

    alert_a = {
        "event_id": "evt_1",
        "timestamp": "2026-03-09T10:00:00Z",
        "camera_id": "cam_a",
        "event_type": "intrusion",
        "zone": "restricted_lab",
        "track_id": 1,
        "class": "person",
        "frame_index": 10,
    }
    alert_b = {
        "event_id": "evt_2",
        "timestamp": "2026-03-09T11:00:00Z",
        "camera_id": "cam_b",
        "event_type": "loitering",
        "zone": "front_door",
        "track_id": 2,
        "class": "person",
        "frame_index": 20,
    }

    assert client.post("/ingest", json=alert_a).status_code == 200
    assert client.post("/ingest", json=alert_b).status_code == 200

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["alerts"] == 2

    alerts = client.get("/alerts", params={"camera_id": "cam_a"}).json()
    assert [item["event_id"] for item in alerts["alerts"]] == ["evt_1"]
    assert alerts["limit"] == 50

    alert = client.get("/alerts/evt_2").json()
    assert alert["zone"] == "front_door"

    stats = client.get("/stats", params={"start_time": "2026-03-09T10:30:00Z"}).json()
    assert stats["total_alerts"] == 1
    assert stats["by_event_type"]["loitering"] == 1

    cameras = client.get("/cameras").json()
    assert cameras["cameras"] == [
        {"camera_id": "cam_a", "alerts": 1, "latest_timestamp": "2026-03-09T10:00:00Z"},
        {"camera_id": "cam_b", "alerts": 1, "latest_timestamp": "2026-03-09T11:00:00Z"},
    ]


def test_alert_api_rejects_invalid_payload(tmp_path: Path) -> None:
    app_module.store = SQLiteAlertStore(tmp_path / "alerts.db")
    client = TestClient(app_module.app)

    response = client.post(
        "/ingest",
        json={
            "event_id": "evt_bad",
            "timestamp": "not-a-timestamp",
            "camera_id": "cam_a",
            "event_type": "intrusion",
            "zone": "restricted_lab",
            "frame_index": 10,
        },
    )

    assert response.status_code == 422


def test_alert_api_rejects_payload_missing_required_class(tmp_path: Path) -> None:
    app_module.store = SQLiteAlertStore(tmp_path / "alerts.db")
    client = TestClient(app_module.app)

    response = client.post(
        "/ingest",
        json={
            "event_id": "evt_missing_class",
            "timestamp": "2026-03-09T10:00:00Z",
            "camera_id": "cam_a",
            "event_type": "intrusion",
            "zone": "restricted_lab",
            "frame_index": 10,
        },
    )

    assert response.status_code == 422
