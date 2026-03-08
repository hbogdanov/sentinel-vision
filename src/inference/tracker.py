from __future__ import annotations

from dataclasses import dataclass

from src.inference.detector import Detection


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    last_seen_frame: int
    hits: int = 1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 30, min_hits: int = 1) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        matches, _, unmatched_detection_ids = self._match(detections)

        for track_id, det_idx in matches:
            detection = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.label = detection.label
            track.score = detection.score
            track.last_seen_frame = frame_index
            track.hits += 1

        for det_idx in unmatched_detection_ids:
            detection = detections[det_idx]
            self._tracks[self._next_id] = Track(
                track_id=self._next_id,
                bbox=detection.bbox,
                label=detection.label,
                score=detection.score,
                last_seen_frame=frame_index,
            )
            self._next_id += 1

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen_frame > self.max_age_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        visible_tracks = [
            track
            for track in self._tracks.values()
            if frame_index - track.last_seen_frame <= self.max_age_frames and track.hits >= self.min_hits
        ]
        return sorted(visible_tracks, key=lambda track: track.track_id)

    def _match(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_ids = set(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, detection in enumerate(detections):
                if detection.label != track.label:
                    continue
                iou = _iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, det_idx))

        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_detections:
                continue
            matches.append((track_id, det_idx))
            used_tracks.add(track_id)
            used_detections.add(det_idx)
            unmatched_track_ids.discard(track_id)
            unmatched_detection_ids.discard(det_idx)

        return matches, unmatched_track_ids, unmatched_detection_ids


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
