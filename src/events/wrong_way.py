from __future__ import annotations

from datetime import datetime
from typing import Any

from src.events.zones import PolygonZone
from src.inference.tracker import Track


class WrongWayDetector:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 10.0,
        min_displacement_pixels: float = 25.0,
        target_classes: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.min_displacement_pixels = min_displacement_pixels
        self.target_classes = set(target_classes or ["person", "car", "truck", "bus"])
        self._entry_centers: dict[tuple[str, int], tuple[float, float]] = {}
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
            expected_direction = zone.metadata.get("expected_direction")
            if not expected_direction:
                continue
            direction_vector = _direction_vector(expected_direction)
            for track in tracks:
                if track.label not in self.target_classes or not zone.contains_track(
                    track
                ):
                    continue
                key = (zone.name, track.track_id)
                active_keys.add(key)
                current_point = track.reasoning_point
                origin = self._entry_centers.setdefault(key, current_point)
                displacement = (
                    current_point[0] - origin[0],
                    current_point[1] - origin[1],
                )
                if _vector_norm(displacement) < self.min_displacement_pixels:
                    continue
                if _dot_product(displacement, direction_vector) >= 0:
                    continue
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
                        "event_type": "wrong_way",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "fps": round(fps, 2),
                        "expected_direction": str(expected_direction),
                    }
                )

        inactive_keys = set(self._entry_centers) - active_keys
        for key in inactive_keys:
            del self._entry_centers[key]
        return events


def _direction_vector(
    value: str | list[float] | tuple[float, float],
) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    lookup = {
        "right": (1.0, 0.0),
        "left": (-1.0, 0.0),
        "down": (0.0, 1.0),
        "up": (0.0, -1.0),
    }
    return lookup.get(str(value).lower(), (1.0, 0.0))


def _vector_norm(value: tuple[float, float]) -> float:
    return (value[0] ** 2 + value[1] ** 2) ** 0.5


def _dot_product(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]
