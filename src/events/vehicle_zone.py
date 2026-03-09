from __future__ import annotations

from datetime import datetime
from typing import Any

from src.events.zones import PolygonZone
from src.inference.tracker import Track


class VehicleInPedestrianZoneDetector:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 5.0,
        target_classes: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.target_classes = set(target_classes or ["car", "truck", "bus", "motorcycle", "bicycle"])
        self._inside_state: dict[tuple[str, int], bool] = {}
        self._last_event_ts: dict[tuple[str, int], datetime] = {}
        self._counter = 0

    def evaluate(
        self,
        tracks: list[Track],
        zones: list[PolygonZone],
        frame_index: int,
        timestamp: datetime,
        fps: float,
        camera_id: str,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        events: list[dict[str, Any]] = []
        for zone in zones:
            for track in tracks:
                key = (zone.name, track.track_id)
                inside_now = track.label in self.target_classes and zone.contains_track(track)
                inside_before = self._inside_state.get(key, False)
                self._inside_state[key] = inside_now
                if not inside_now or inside_before:
                    continue
                last_event = self._last_event_ts.get(key)
                if last_event and (timestamp - last_event).total_seconds() < self.cooldown_seconds:
                    continue
                self._counter += 1
                self._last_event_ts[key] = timestamp
                events.append(
                    {
                        "event_id": f"evt_{self._counter:06d}",
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "camera_id": camera_id,
                        "event_type": "vehicle_in_pedestrian_zone",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "fps": round(fps, 2),
                    }
                )
        return events
