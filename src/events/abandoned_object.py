from __future__ import annotations

from datetime import datetime
from math import hypot
from typing import Any

from src.events.zones import PolygonZone
from src.inference.tracker import Track


class AbandonedObjectDetector:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 30.0,
        unattended_seconds: float = 20.0,
        min_stationary_seconds: float = 8.0,
        stationary_radius_pixels: float = 20.0,
        owner_max_distance_pixels: float = 80.0,
        target_classes: list[str] | None = None,
        owner_classes: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.unattended_seconds = unattended_seconds
        self.min_stationary_seconds = min_stationary_seconds
        self.stationary_radius_pixels = stationary_radius_pixels
        self.owner_max_distance_pixels = owner_max_distance_pixels
        self.target_classes = set(target_classes or ["backpack", "suitcase", "handbag", "bicycle"])
        self.owner_classes = set(owner_classes or ["person"])
        self._anchor_state: dict[tuple[str, int], tuple[tuple[float, float], datetime]] = {}
        self._unattended_since: dict[tuple[str, int], datetime] = {}
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
        owner_tracks = [track for track in tracks if track.label in self.owner_classes]

        for zone in zones:
            for track in tracks:
                if track.label not in self.target_classes or not zone.contains_track(track):
                    continue

                key = (zone.name, track.track_id)
                active_keys.add(key)
                stationary_seconds = self._stationary_seconds(key, track, timestamp)
                if stationary_seconds < self.min_stationary_seconds:
                    self._unattended_since.pop(key, None)
                    continue

                nearest_owner_distance = _nearest_owner_distance(track, owner_tracks)
                if nearest_owner_distance is None or nearest_owner_distance > self.owner_max_distance_pixels:
                    unattended_since = self._unattended_since.setdefault(key, timestamp)
                    unattended_seconds = (timestamp - unattended_since).total_seconds()
                else:
                    self._unattended_since.pop(key, None)
                    unattended_seconds = 0.0

                if unattended_seconds < self.unattended_seconds:
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
                        "event_type": "abandoned_object",
                        "track_id": track.track_id,
                        "class": track.label,
                        "confidence": round(track.score, 4),
                        "zone": zone.name,
                        "frame_index": frame_index,
                        "stationary_seconds": round(stationary_seconds, 2),
                        "unattended_seconds": round(unattended_seconds, 2),
                        "owner_max_distance_pixels": round(self.owner_max_distance_pixels, 2),
                        "nearest_owner_distance_pixels": round(nearest_owner_distance, 2) if nearest_owner_distance is not None else None,
                        "fps": round(fps, 2),
                    }
                )

        stale_keys = set(self._anchor_state) - active_keys
        for key in stale_keys:
            self._anchor_state.pop(key, None)
            self._unattended_since.pop(key, None)

        return events

    def _stationary_seconds(self, key: tuple[str, int], track: Track, timestamp: datetime) -> float:
        center = track.center
        anchor = self._anchor_state.get(key)
        if anchor is None:
            self._anchor_state[key] = (center, timestamp)
            return 0.0

        anchor_center, anchor_ts = anchor
        if _distance(center, anchor_center) > self.stationary_radius_pixels:
            self._anchor_state[key] = (center, timestamp)
            return 0.0
        return (timestamp - anchor_ts).total_seconds()


def _nearest_owner_distance(track: Track, owner_tracks: list[Track]) -> float | None:
    distances = [_distance(track.center, owner.center) for owner in owner_tracks]
    if not distances:
        return None
    return min(distances)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
