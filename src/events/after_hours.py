from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.events.zones import PolygonZone
from src.inference.tracker import Track


class AfterHoursOccupancyDetector:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 60.0,
        start_time: str = "08:00",
        end_time: str = "18:00",
        timezone_name: str = "UTC",
        target_classes: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.start_time = start_time
        self.end_time = end_time
        self.timezone_name = timezone_name
        self.target_classes = set(target_classes or ["person"])
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
        if not self.enabled or self._is_within_allowed_window(timestamp):
            return []

        events: list[dict[str, Any]] = []
        local_ts = timestamp.astimezone(ZoneInfo(self.timezone_name))
        for zone in zones:
            for track in tracks:
                if track.label not in self.target_classes or not zone.contains_track(
                    track
                ):
                    continue
                key = (zone.name, track.track_id)
                last_event = self._last_event_ts.get(key)
                if (
                    last_event
                    and (timestamp - last_event).total_seconds() < self.cooldown_seconds
                ):
                    continue
                self._counter += 1
                self._last_event_ts[key] = timestamp
                events.append(
                    {
                        "event_id": f"evt_{self._counter:06d}",
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "camera_id": camera_id,
                        "event_type": "after_hours_occupancy",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "fps": round(fps, 2),
                        "local_time": local_ts.isoformat(),
                    }
                )
        return events

    def _is_within_allowed_window(self, timestamp: datetime) -> bool:
        local_ts = timestamp.astimezone(ZoneInfo(self.timezone_name))
        current_minutes = local_ts.hour * 60 + local_ts.minute
        start_minutes = _parse_minutes(self.start_time)
        end_minutes = _parse_minutes(self.end_time)
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes < end_minutes
        return current_minutes >= start_minutes or current_minutes < end_minutes


def _parse_minutes(value: str) -> int:
    hours, minutes = value.split(":", maxsplit=1)
    return int(hours) * 60 + int(minutes)
