from __future__ import annotations

import cv2

from src.events.zones import PolygonZone
from src.inference.tracker import Track


def draw_frame(frame, zones: list[PolygonZone], tracks: list[Track], events: list[dict], fps: float):
    for zone in zones:
        points = [(int(x), int(y)) for x, y in zone.points]
        for idx in range(len(points)):
            cv2.line(frame, points[idx], points[(idx + 1) % len(points)], (0, 165, 255), 2)
        cv2.putText(frame, zone.name, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = (0, 255, 0) if track.label == "person" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track.label} #{track.track_id} {track.score:.2f}"
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if events:
        banner = f"ALERT: {events[-1]['event_type'].upper()} in {events[-1]['zone']}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (0, 0, 180), -1)
        cv2.putText(frame, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame
