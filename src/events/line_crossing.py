from __future__ import annotations

from datetime import datetime
from typing import Any

from src.events.zones import LineZone
from src.inference.tracker import Track


class LineCrossingDetector:
    def __init__(self, enabled: bool = True, cooldown_seconds: float = 3.0, direction: str = "any") -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.direction = direction
        self._last_side: dict[tuple[str, int], float] = {}
        self._last_event_ts: dict[tuple[str, int], datetime] = {}
        self._counter = 0

    def evaluate(
        self,
        tracks: list[Track],
        zones: list[LineZone],
        frame_index: int,
        timestamp: datetime,
        fps: float,
        camera_id: str,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        events: list[dict[str, Any]] = []
        for zone in zones:
            zone_direction = str(zone.metadata.get("direction", self.direction)).lower()
            for track in tracks:
                key = (zone.name, track.track_id)
                current_side = zone.side_of_point(track.center)
                previous_side = self._last_side.get(key)
                self._last_side[key] = current_side
                if previous_side is None or previous_side == 0 or current_side == 0:
                    continue
                if previous_side * current_side > 0:
                    continue
                movement = "a_to_b" if previous_side > 0 and current_side < 0 else "b_to_a"
                if zone_direction != "any" and movement != zone_direction:
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
                        "event_type": "line_crossing",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "fps": round(fps, 2),
                        "direction": movement,
                    }
                )
        return events
