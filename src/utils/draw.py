from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.events.zones import LineZone, PolygonZone, Zone
from src.inference.tracker import Track

SEVERITY_COLORS = {
    "after_hours_occupancy": (0, 64, 255),
    "abandoned_object": (60, 60, 255),
    "intrusion": (0, 96, 255),
    "wrong_way": (0, 140, 255),
    "vehicle_in_pedestrian_zone": (0, 180, 255),
    "line_crossing": (0, 215, 255),
    "loitering": (0, 255, 255),
}


def draw_frame(
    frame,
    zones: list[Zone],
    tracks: list[Track],
    events: list[dict],
    fps: float,
    track_history: dict[int, deque[tuple[int, int]]] | None = None,
    dwell_timers: dict[int, float] | None = None,
    zone_heatmap_overlay=None,
    zone_heatmap_opacity: float = 0.35,
):
    if zone_heatmap_overlay is not None:
        heatmap_mask = zone_heatmap_overlay.any(axis=2)
        blended = cv2.addWeighted(
            zone_heatmap_overlay,
            zone_heatmap_opacity,
            frame,
            1.0 - zone_heatmap_opacity,
            0.0,
        )
        frame[heatmap_mask] = blended[heatmap_mask]

    overlay = frame.copy()
    for zone in zones:
        if isinstance(zone, PolygonZone):
            points = np.array(
                [(int(x), int(y)) for x, y in zone.points], dtype=np.int32
            )
            cv2.fillPoly(overlay, [points], (0, 120, 255))
            cv2.polylines(
                frame, [points], isClosed=True, color=(0, 165, 255), thickness=2
            )
            cv2.putText(
                frame,
                zone.name,
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        elif isinstance(zone, LineZone):
            start = (int(zone.start[0]), int(zone.start[1]))
            end = (int(zone.end[0]), int(zone.end[1]))
            cv2.line(frame, start, end, (255, 215, 0), 2)
            cv2.putText(
                frame, zone.name, start, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2
            )
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0.0)

    history_map = track_history or {}
    dwell_map = dwell_timers or {}
    for track in tracks:
        color = _track_color(track.label)
        history = history_map.get(track.track_id)
        if history and len(history) > 1:
            _draw_trail(frame, list(history), color)
        if track.predicted_path:
            _draw_prediction(frame, track.center, list(track.predicted_path), color)

        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track.label} #{track.track_id} {track.score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if track.track_id in dwell_map:
            dwell_text = f"dwell {dwell_map[track.track_id]:.1f}s"
            cv2.putText(
                frame,
                dwell_text,
                (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if events:
        _draw_alert_stack(frame, events)

    return frame


def _draw_trail(
    frame, history: list[tuple[int, int]], color: tuple[int, int, int]
) -> None:
    for idx in range(1, len(history)):
        alpha = idx / len(history)
        thickness = max(1, int(1 + 3 * alpha))
        cv2.line(frame, history[idx - 1], history[idx], color, thickness)


def _draw_alert_stack(frame, events: list[dict]) -> None:
    visible_events = events[-3:]
    banner_height = 34 + (len(visible_events) - 1) * 30
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (22, 24, 30), -1)
    for idx, event in enumerate(reversed(visible_events)):
        color = SEVERITY_COLORS.get(str(event.get("event_type", "")), (0, 0, 180))
        y = 24 + idx * 28
        text = f"{str(event.get('event_type', 'alert')).upper()}  {event.get('zone', 'unknown')}"
        cv2.circle(frame, (18, y - 5), 6, color, -1)
        cv2.putText(
            frame, text, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )


def _draw_prediction(
    frame,
    start: tuple[float, float],
    predicted_path: list[tuple[float, float]],
    color: tuple[int, int, int],
) -> None:
    previous = (int(start[0]), int(start[1]))
    prediction_color = tuple(min(255, channel + 25) for channel in color)
    for idx, point in enumerate(predicted_path):
        current = (int(point[0]), int(point[1]))
        if idx % 2 == 0:
            cv2.line(frame, previous, current, prediction_color, 1)
        cv2.circle(frame, current, 2, prediction_color, -1)
        previous = current


def _track_color(label: str) -> tuple[int, int, int]:
    if label == "person":
        return (80, 230, 120)
    if label in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return (255, 120, 60)
    return (255, 200, 80)
