from __future__ import annotations

from datetime import datetime
from typing import Any

from src.events.zones import PolygonZone
from src.inference.tracker import Track


class LoiteringDetector:
    def __init__(self, enabled: bool = True, threshold_seconds: float = 10.0, cooldown_seconds: float = 20.0) -> None:
        self.enabled = enabled
        self.threshold_seconds = threshold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._entry_ts: dict[tuple[str, int], datetime] = {}
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
        active_keys: set[tuple[str, int]] = set()
        for zone in zones:
            for track in tracks:
                key = (zone.name, track.track_id)
                inside_now = zone.contains_track(track)
                if not inside_now:
                    continue

                active_keys.add(key)
                entry_ts = self._entry_ts.setdefault(key, timestamp)
                dwell_seconds = (timestamp - entry_ts).total_seconds()
                if dwell_seconds < self.threshold_seconds:
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
                        "event_type": "loitering",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "dwell_seconds": round(dwell_seconds, 2),
                        "fps": round(fps, 2),
                    }
                )

        inactive_keys = set(self._entry_ts) - active_keys
        for key in inactive_keys:
            del self._entry_ts[key]

        return events
