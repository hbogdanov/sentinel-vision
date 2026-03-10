from __future__ import annotations

from pathlib import Path

from src.api.storage import SQLiteAlertStore


def test_sqlite_alert_store_persists_and_filters_alerts(tmp_path: Path) -> None:
    store = SQLiteAlertStore(tmp_path / "alerts.db")
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

    store.ingest(alert_a)
    store.ingest(alert_b)

    assert store.count() == 2
    assert store.get_alert("evt_1")["camera_id"] == "cam_a"
    assert [item["event_id"] for item in store.list_alerts(camera_id="cam_b")] == [
        "evt_2"
    ]
    assert [item["event_id"] for item in store.list_alerts(event_type="intrusion")] == [
        "evt_1"
    ]
    assert [item["event_id"] for item in store.list_alerts(zone="restricted_lab")] == [
        "evt_1"
    ]
    assert [
        item["event_id"]
        for item in store.list_alerts(start_time="2026-03-09T10:30:00Z")
    ] == ["evt_2"]

    stats = store.stats()
    assert stats["total_alerts"] == 2
    assert stats["by_event_type"]["intrusion"] == 1
    assert stats["by_camera_id"]["cam_b"] == 1
    assert store.camera_summaries() == [
        {"camera_id": "cam_a", "alerts": 1, "latest_timestamp": "2026-03-09T10:00:00Z"},
        {"camera_id": "cam_b", "alerts": 1, "latest_timestamp": "2026-03-09T11:00:00Z"},
    ]
