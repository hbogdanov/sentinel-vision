from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - fallback path when scipy is unavailable
    linear_sum_assignment = None

from src.inference.appearance import AppearanceEmbedder, build_appearance_embedder
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
    velocity: tuple[float, float] = (0.0, 0.0)
    predicted_path: tuple[tuple[float, float], ...] = ()

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
    recent_boxes: deque[tuple[float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=6), repr=False
    )

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
        self.recent_boxes.append(detection.bbox)
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

    def to_public(
        self,
        *,
        velocity: tuple[float, float] = (0.0, 0.0),
        predicted_path: tuple[tuple[float, float], ...] = (),
    ) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            label=self.label,
            score=self.score,
            last_seen_frame=self.last_seen_frame,
            hits=self.hits,
            world_center=None,
            velocity=velocity,
            predicted_path=predicted_path,
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
        trajectory_prediction_enabled: bool = True,
        trajectory_history_size: int = 6,
        trajectory_prediction_horizon: int = 3,
        trajectory_smoothing: float = 0.65,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.new_track_score_threshold = new_track_score_threshold
        self.match_iou_threshold = match_iou_threshold
        self.secondary_match_iou_threshold = secondary_match_iou_threshold
        self.max_age_frames = max_age_frames
        self.min_hits = min_hits
        self.trajectory_prediction_enabled = trajectory_prediction_enabled
        self.trajectory_history_size = max(1, trajectory_history_size)
        self.trajectory_prediction_horizon = max(1, trajectory_prediction_horizon)
        self.trajectory_smoothing = trajectory_smoothing
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
                recent_boxes=deque(
                    [detection.bbox], maxlen=self.trajectory_history_size
                ),
            )
            self._next_id += 1

        self._expire_stale_tracks()
        self._promote_confirmed_tracks()

        visible_tracks = [
            self._build_public_track(track)
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
        unmatched_track_ids = set(track_ids)
        unmatched_detection_ids = set(detection_ids)
        candidates: dict[tuple[int, int], float] = {}

        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            predicted_bbox = self._predict_bbox(track)
            for det_idx in detection_ids:
                detection = detections[det_idx]
                if detection.label != track.label:
                    continue
                iou = _iou(predicted_bbox, compensated_boxes[det_idx])
                if iou >= threshold:
                    candidates[(track_id, det_idx)] = iou

        matches = _solve_assignment(track_ids, detection_ids, candidates)
        for track_id, det_idx in matches:
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

    def _build_public_track(self, track: _TrackState) -> Track:
        velocity = self._center_velocity(track)
        predicted_path = self._predict_path(track)
        return track.to_public(velocity=velocity, predicted_path=predicted_path)

    def _predict_bbox(self, track: _TrackState) -> tuple[float, float, float, float]:
        if not self.trajectory_prediction_enabled:
            return track.bbox
        delta = self._smoothed_bbox_delta(track)
        horizon = min(track.misses + 1, self.trajectory_prediction_horizon)
        return tuple(
            coord + horizon * change for coord, change in zip(track.bbox, delta)
        )

    def _predict_path(self, track: _TrackState) -> tuple[tuple[float, float], ...]:
        if not self.trajectory_prediction_enabled:
            return ()
        dx1, dy1, dx2, dy2 = self._smoothed_bbox_delta(track)
        step_x = (dx1 + dx2) / 2.0
        step_y = (dy1 + dy2) / 2.0
        if abs(step_x) < 1e-3 and abs(step_y) < 1e-3:
            return ()
        center_x, center_y = track.to_public().center
        return tuple(
            (
                center_x + step_x * step,
                center_y + step_y * step,
            )
            for step in range(1, self.trajectory_prediction_horizon + 1)
        )

    def _center_velocity(self, track: _TrackState) -> tuple[float, float]:
        dx1, dy1, dx2, dy2 = self._smoothed_bbox_delta(track)
        return ((dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0)

    def _smoothed_bbox_delta(
        self, track: _TrackState
    ) -> tuple[float, float, float, float]:
        history = list(track.recent_boxes)
        if len(history) < 2:
            return track.velocity

        deltas = [
            tuple(curr - prev for curr, prev in zip(current, previous))
            for previous, current in zip(history[:-1], history[1:])
        ]
        smoothed = deltas[0]
        for delta in deltas[1:]:
            smoothed = tuple(
                ((1.0 - self.trajectory_smoothing) * prior)
                + (self.trajectory_smoothing * current)
                for prior, current in zip(smoothed, delta)
            )
        track.velocity = smoothed
        return smoothed


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
        appearance_ambiguous_iou_margin: float = 0.1,
        appearance_embedder: AppearanceEmbedder | None = None,
        trajectory_prediction_enabled: bool = True,
        trajectory_history_size: int = 6,
        trajectory_prediction_horizon: int = 3,
        trajectory_smoothing: float = 0.65,
    ) -> None:
        super().__init__(
            high_score_threshold=high_score_threshold,
            low_score_threshold=low_score_threshold,
            new_track_score_threshold=new_track_score_threshold,
            match_iou_threshold=match_iou_threshold,
            secondary_match_iou_threshold=secondary_match_iou_threshold,
            max_age_frames=max_age_frames,
            min_hits=min_hits,
            trajectory_prediction_enabled=trajectory_prediction_enabled,
            trajectory_history_size=trajectory_history_size,
            trajectory_prediction_horizon=trajectory_prediction_horizon,
            trajectory_smoothing=trajectory_smoothing,
        )
        self.appearance_weight = appearance_weight
        self.appearance_threshold = appearance_threshold
        self.appearance_ambiguous_iou_margin = appearance_ambiguous_iou_margin
        self.appearance_embedder = appearance_embedder

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame=None,
        motion_transform=None,
    ) -> list[Track]:
        embeddings = (
            self.appearance_embedder.embed(frame, detections)
            if self.appearance_embedder is not None
            else {}
        )
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
                recent_boxes=deque(
                    [detection.bbox], maxlen=self.trajectory_history_size
                ),
            )
            self._next_id += 1

        self._expire_stale_tracks()
        self._promote_confirmed_tracks()

        visible_tracks = [
            self._build_public_track(track)
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
        unmatched_track_ids = set(track_ids)
        unmatched_detection_ids = set(detection_ids)
        candidates: dict[tuple[int, int], float] = {}

        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            predicted_bbox = self._predict_bbox(track)
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
                score = iou
                if iou < threshold + self.appearance_ambiguous_iou_margin:
                    score = (
                        1.0 - self.appearance_weight
                    ) * iou + self.appearance_weight * appearance_score
                candidates[(track_id, det_idx)] = score

        matches = _solve_assignment(track_ids, detection_ids, candidates)
        for track_id, det_idx in matches:
            unmatched_track_ids.discard(track_id)
            unmatched_detection_ids.discard(det_idx)

        return matches, unmatched_track_ids, unmatched_detection_ids


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
            trajectory_prediction_enabled=bool(
                config.get("trajectory_prediction_enabled", True)
            ),
            trajectory_history_size=int(config.get("trajectory_history_size", 6)),
            trajectory_prediction_horizon=int(
                config.get("trajectory_prediction_horizon", 3)
            ),
            trajectory_smoothing=float(config.get("trajectory_smoothing", 0.65)),
            appearance_weight=float(config.get("appearance_weight", 0.35)),
            appearance_threshold=float(config.get("appearance_threshold", 0.2)),
            appearance_ambiguous_iou_margin=float(
                config.get("appearance_ambiguous_iou_margin", 0.1)
            ),
            appearance_embedder=build_appearance_embedder(config),
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
        trajectory_prediction_enabled=bool(
            config.get("trajectory_prediction_enabled", True)
        ),
        trajectory_history_size=int(config.get("trajectory_history_size", 6)),
        trajectory_prediction_horizon=int(
            config.get("trajectory_prediction_horizon", 3)
        ),
        trajectory_smoothing=float(config.get("trajectory_smoothing", 0.65)),
    )


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return max(0.0, float(np.dot(a, b) / denominator))


def _solve_assignment(
    track_ids: list[int],
    detection_ids: list[int],
    scores: dict[tuple[int, int], float],
) -> list[tuple[int, int]]:
    if not track_ids or not detection_ids or not scores:
        return []

    if linear_sum_assignment is None:
        ranked = sorted(
            ((score, track_id, det_idx) for (track_id, det_idx), score in scores.items()),
            reverse=True,
        )
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, det_idx in ranked:
            if track_id in used_tracks or det_idx in used_detections:
                continue
            matches.append((track_id, det_idx))
            used_tracks.add(track_id)
            used_detections.add(det_idx)
        return matches

    track_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
    detection_index = {det_idx: idx for idx, det_idx in enumerate(detection_ids)}
    cost = np.full((len(track_ids), len(detection_ids)), 1e6, dtype=np.float32)
    for (track_id, det_idx), score in scores.items():
        cost[track_index[track_id], detection_index[det_idx]] = 1.0 - float(score)

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    for row_idx, col_idx in zip(rows.tolist(), cols.tolist()):
        if cost[row_idx, col_idx] >= 1e5:
            continue
        matches.append((track_ids[row_idx], detection_ids[col_idx]))
    return matches


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
