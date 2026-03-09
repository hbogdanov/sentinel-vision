from __future__ import annotations

import json
from pathlib import Path

from src.io.health import CameraHealthMonitor


def test_camera_health_monitor_persists_status(tmp_path: Path) -> None:
    status_path = tmp_path / "camera_health.json"
    monitor = CameraHealthMonitor(camera_id="cam_a", status_path=status_path)

    monitor.mark_frame_read()
    monitor.mark_detector_failure()
    monitor.mark_reconnect(successful=False)
    monitor.mark_offline()

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["camera_id"] == "cam_a"
    assert payload["status"] == "offline"
    assert payload["detector_failures"] == 1
    assert payload["reconnect_attempts"] == 1
