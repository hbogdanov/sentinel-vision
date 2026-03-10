from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from src.inference.detector import Detection
from src.inference.motion import compensate_bbox


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    last_seen_frame: int
    hits: int = 1
    world_center: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def footpoint(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def reasoning_point(self) -> tuple[float, float]:
        return self.world_center if self.world_center is not None else self.center


class Tracker(Protocol):
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame=None,
        motion_transform=None,
    ) -> list[Track]: ...


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    last_seen_frame: int
    hits: int = 1
    misses: int = 0
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confirmed: bool = False
    embedding: np.ndarray | None = field(default=None, repr=False)

    def predicted_bbox(self) -> tuple[float, float, float, float]:
        return tuple(coord + delta for coord, delta in zip(self.bbox, self.velocity))

    def update(
        self,
        detection: Detection,
        frame_index: int,
        embedding: np.ndarray | None = None,
    ) -> None:
        self.velocity = tuple(new - old for new, old in zip(detection.bbox, self.bbox))
        self.bbox = detection.bbox
        self.label = detection.label
        self.score = detection.score
        self.last_seen_frame = frame_index
        self.hits += 1
        self.misses = 0
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                self.embedding = 0.8 * self.embedding + 0.2 * embedding
                norm = np.linalg.norm(self.embedding)
                if norm > 0:
                    self.embedding = self.embedding / norm

    def mark_missed(self) -> None:
        self.misses += 1

    def to_public(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            label=self.label,
            score=self.score,
            last_seen_frame=self.last_seen_frame,
            hits=self.hits,
            world_center=None,
        )


class SimpleTracker:
    def __init__(
        self, iou_threshold: float = 0.3, max_age_frames: int = 30, min_hits: int = 1
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame=None,
        motion_transform=None,
    ) -> list[Track]:
        compensated_boxes = [
            compensate_bbox(detection.bbox, motion_transform)
            for detection in detections
        ]
        matches, _, unmatched_detection_ids = self._match(detections, compensated_boxes)

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
            if frame_index - track.last_seen_frame <= self.max_age_frames
            and track.hits >= self.min_hits
        ]
        return sorted(visible_tracks, key=lambda track: track.track_id)

    def _match(
        self,
        detections: list[Detection],
        compensated_boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_ids = set(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, detection in enumerate(detections):
                if detection.label != track.label:
                    continue
                iou = _iou(track.bbox, compensated_boxes[det_idx])
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


class ByteTracker:
    def __init__(
        self,
        high_score_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
        new_track_score_threshold: float = 0.6,
        match_iou_threshold: float = 0.3,
        secondary_match_iou_threshold: float = 0.15,
        max_age_frames: int = 30,
        min_hits: int = 2,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.new_track_score_threshold = new_track_score_threshold
        self.match_iou_threshold = match_iou_threshold
        self.secondary_match_iou_threshold = secondary_match_iou_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame=None,
        motion_transform=None,
    ) -> list[Track]:
        compensated_boxes = [
            compensate_bbox(detection.bbox, motion_transform)
            for detection in detections
        ]
        high_conf_ids = [
            idx
            for idx, detection in enumerate(detections)
            if detection.score >= self.high_score_threshold
        ]
        low_conf_ids = [
            idx
            for idx, detection in enumerate(detections)
            if self.low_score_threshold <= detection.score < self.high_score_threshold
        ]

        confirmed_ids = [
            track_id for track_id, track in self._tracks.items() if track.confirmed
        ]
        tentative_ids = [
            track_id for track_id, track in self._tracks.items() if not track.confirmed
        ]

        matches, unmatched_confirmed, unmatched_high = self._match_tracks(
            track_ids=confirmed_ids,
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=high_conf_ids,
            threshold=self.match_iou_threshold,
        )
        self._apply_matches(matches, detections, frame_index)

        secondary_matches, still_unmatched_confirmed, _ = self._match_tracks(
            track_ids=list(unmatched_confirmed),
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=low_conf_ids,
            threshold=self.secondary_match_iou_threshold,
        )
        self._apply_matches(secondary_matches, detections, frame_index)

        tentative_matches, unmatched_tentative, unmatched_high = self._match_tracks(
            track_ids=tentative_ids,
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=list(unmatched_high),
            threshold=self.match_iou_threshold,
        )
        self._apply_matches(tentative_matches, detections, frame_index)

        for track_id in list(still_unmatched_confirmed | unmatched_tentative):
            track = self._tracks.get(track_id)
            if track is not None:
                track.mark_missed()

        for det_idx in unmatched_high:
            detection = detections[det_idx]
            if detection.score < self.new_track_score_threshold:
                continue
            self._tracks[self._next_id] = _TrackState(
                track_id=self._next_id,
                bbox=detection.bbox,
                label=detection.label,
                score=detection.score,
                last_seen_frame=frame_index,
                confirmed=self.min_hits <= 1,
            )
            self._next_id += 1

        self._expire_stale_tracks()
        self._promote_confirmed_tracks()

        visible_tracks = [
            track.to_public()
            for track in self._tracks.values()
            if track.last_seen_frame == frame_index and track.confirmed
        ]
        return sorted(visible_tracks, key=lambda track: track.track_id)

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        detections: list[Detection],
        frame_index: int,
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> None:
        for track_id, det_idx in matches:
            embedding = embeddings.get(det_idx) if embeddings else None
            self._tracks[track_id].update(
                detections[det_idx], frame_index=frame_index, embedding=embedding
            )

    def _match_tracks(
        self,
        track_ids: list[int],
        detections: list[Detection],
        compensated_boxes: list[tuple[float, float, float, float]],
        detection_ids: list[int],
        threshold: float,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        unmatched_track_ids = set(track_ids)
        unmatched_detection_ids = set(detection_ids)
        candidates: list[tuple[float, int, int]] = []

        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            predicted_bbox = track.predicted_bbox()
            for det_idx in detection_ids:
                detection = detections[det_idx]
                if detection.label != track.label:
                    continue
                iou = _iou(predicted_bbox, compensated_boxes[det_idx])
                if iou >= threshold:
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

    def _expire_stale_tracks(self) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.misses > self.max_age_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _promote_confirmed_tracks(self) -> None:
        for track in self._tracks.values():
            if track.hits >= self.min_hits:
                track.confirmed = True


class BoTSORTTracker(ByteTracker):
    def __init__(
        self,
        high_score_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
        new_track_score_threshold: float = 0.6,
        match_iou_threshold: float = 0.3,
        secondary_match_iou_threshold: float = 0.15,
        max_age_frames: int = 30,
        min_hits: int = 2,
        appearance_weight: float = 0.35,
        appearance_threshold: float = 0.2,
    ) -> None:
        super().__init__(
            high_score_threshold=high_score_threshold,
            low_score_threshold=low_score_threshold,
            new_track_score_threshold=new_track_score_threshold,
            match_iou_threshold=match_iou_threshold,
            secondary_match_iou_threshold=secondary_match_iou_threshold,
            max_age_frames=max_age_frames,
            min_hits=min_hits,
        )
        self.appearance_weight = appearance_weight
        self.appearance_threshold = appearance_threshold

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame=None,
        motion_transform=None,
    ) -> list[Track]:
        embeddings = self._compute_embeddings(frame, detections)
        compensated_boxes = [
            compensate_bbox(detection.bbox, motion_transform)
            for detection in detections
        ]
        high_conf_ids = [
            idx
            for idx, detection in enumerate(detections)
            if detection.score >= self.high_score_threshold
        ]
        low_conf_ids = [
            idx
            for idx, detection in enumerate(detections)
            if self.low_score_threshold <= detection.score < self.high_score_threshold
        ]

        confirmed_ids = [
            track_id for track_id, track in self._tracks.items() if track.confirmed
        ]
        tentative_ids = [
            track_id for track_id, track in self._tracks.items() if not track.confirmed
        ]

        matches, unmatched_confirmed, unmatched_high = self._match_tracks(
            track_ids=confirmed_ids,
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=high_conf_ids,
            threshold=self.match_iou_threshold,
            embeddings=embeddings,
        )
        self._apply_matches(matches, detections, frame_index, embeddings=embeddings)

        secondary_matches, still_unmatched_confirmed, _ = self._match_tracks(
            track_ids=list(unmatched_confirmed),
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=low_conf_ids,
            threshold=self.secondary_match_iou_threshold,
            embeddings=embeddings,
        )
        self._apply_matches(
            secondary_matches, detections, frame_index, embeddings=embeddings
        )

        tentative_matches, unmatched_tentative, unmatched_high = self._match_tracks(
            track_ids=tentative_ids,
            detections=detections,
            compensated_boxes=compensated_boxes,
            detection_ids=list(unmatched_high),
            threshold=self.match_iou_threshold,
            embeddings=embeddings,
        )
        self._apply_matches(
            tentative_matches, detections, frame_index, embeddings=embeddings
        )

        for track_id in list(still_unmatched_confirmed | unmatched_tentative):
            track = self._tracks.get(track_id)
            if track is not None:
                track.mark_missed()

        for det_idx in unmatched_high:
            detection = detections[det_idx]
            if detection.score < self.new_track_score_threshold:
                continue
            self._tracks[self._next_id] = _TrackState(
                track_id=self._next_id,
                bbox=detection.bbox,
                label=detection.label,
                score=detection.score,
                last_seen_frame=frame_index,
                confirmed=self.min_hits <= 1,
                embedding=embeddings.get(det_idx),
            )
            self._next_id += 1

        self._expire_stale_tracks()
        self._promote_confirmed_tracks()

        visible_tracks = [
            track.to_public()
            for track in self._tracks.values()
            if track.last_seen_frame == frame_index and track.confirmed
        ]
        return sorted(visible_tracks, key=lambda track: track.track_id)

    def _match_tracks(
        self,
        track_ids: list[int],
        detections: list[Detection],
        compensated_boxes: list[tuple[float, float, float, float]],
        detection_ids: list[int],
        threshold: float,
        embeddings: dict[int, np.ndarray] | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        unmatched_track_ids = set(track_ids)
        unmatched_detection_ids = set(detection_ids)
        candidates: list[tuple[float, int, int]] = []

        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            predicted_bbox = track.predicted_bbox()
            for det_idx in detection_ids:
                detection = detections[det_idx]
                if detection.label != track.label:
                    continue
                iou = _iou(predicted_bbox, compensated_boxes[det_idx])
                appearance_score = _cosine_similarity(
                    track.embedding, embeddings.get(det_idx) if embeddings else None
                )
                if iou < threshold and appearance_score < self.appearance_threshold:
                    continue
                score = (
                    1.0 - self.appearance_weight
                ) * iou + self.appearance_weight * appearance_score
                candidates.append((score, track_id, det_idx))

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

    def _compute_embeddings(
        self, frame, detections: list[Detection]
    ) -> dict[int, np.ndarray]:
        if frame is None:
            return {}

        embeddings: dict[int, np.ndarray] = {}
        height, width = frame.shape[:2]
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            ix1 = max(0, min(width - 1, int(x1)))
            iy1 = max(0, min(height - 1, int(y1)))
            ix2 = max(ix1 + 1, min(width, int(x2)))
            iy2 = max(iy1 + 1, min(height, int(y2)))
            crop = frame[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = (
                cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                .flatten()
                .astype(np.float32)
            )
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            embeddings[det_idx] = hist
        return embeddings


def create_tracker(config: dict) -> Tracker:
    tracker_type = str(config.get("type", "bytetrack")).lower()
    common = {
        "max_age_frames": int(config.get("max_age_frames", 30)),
        "min_hits": int(config.get("min_hits", 2)),
    }

    if tracker_type == "simple":
        return SimpleTracker(
            iou_threshold=float(
                config.get("iou_threshold", config.get("match_iou_threshold", 0.3))
            ),
            max_age_frames=common["max_age_frames"],
            min_hits=common["min_hits"],
        )

    if tracker_type == "botsort":
        return BoTSORTTracker(
            high_score_threshold=float(config.get("high_score_threshold", 0.5)),
            low_score_threshold=float(config.get("low_score_threshold", 0.1)),
            new_track_score_threshold=float(
                config.get("new_track_score_threshold", 0.6)
            ),
            match_iou_threshold=float(config.get("match_iou_threshold", 0.3)),
            secondary_match_iou_threshold=float(
                config.get("secondary_match_iou_threshold", 0.15)
            ),
            max_age_frames=common["max_age_frames"],
            min_hits=common["min_hits"],
            appearance_weight=float(config.get("appearance_weight", 0.35)),
            appearance_threshold=float(config.get("appearance_threshold", 0.2)),
        )

    return ByteTracker(
        high_score_threshold=float(config.get("high_score_threshold", 0.5)),
        low_score_threshold=float(config.get("low_score_threshold", 0.1)),
        new_track_score_threshold=float(config.get("new_track_score_threshold", 0.6)),
        match_iou_threshold=float(config.get("match_iou_threshold", 0.3)),
        secondary_match_iou_threshold=float(
            config.get("secondary_match_iou_threshold", 0.15)
        ),
        max_age_frames=common["max_age_frames"],
        min_hits=common["min_hits"],
    )


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return max(0.0, float(np.dot(a, b) / denominator))


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
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
