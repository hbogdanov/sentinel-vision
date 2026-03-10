import numpy as np

from src.inference.appearance import HistogramAppearanceEmbedder, build_appearance_embedder
from src.inference.detector import Detection
from src.inference.tracker import BoTSORTTracker, ByteTracker, _solve_assignment


class _StaticEmbedder:
    def __init__(self, embeddings_by_call):
        self.embeddings_by_call = embeddings_by_call
        self.calls = 0

    def embed(self, frame, detections):
        if not detections:
            return {}
        payload = self.embeddings_by_call[self.calls]
        self.calls += 1
        return {
            idx: np.asarray(vector, dtype=np.float32) for idx, vector in payload.items()
        }


def test_bytetrack_recovers_same_id_after_short_missed_detection() -> None:
    tracker = ByteTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.1,
        secondary_match_iou_threshold=0.05,
        max_age_frames=2,
        min_hits=1,
    )

    frame0 = [Detection(bbox=(0, 0, 10, 10), score=0.95, class_id=0, label="person")]
    frame1 = [Detection(bbox=(8, 0, 18, 10), score=0.94, class_id=0, label="person")]
    frame3 = [Detection(bbox=(24, 0, 34, 10), score=0.96, class_id=0, label="person")]

    initial = tracker.update(frame0, frame_index=0)
    moved = tracker.update(frame1, frame_index=1)
    missed = tracker.update([], frame_index=2)
    recovered = tracker.update(frame3, frame_index=3)

    assert [track.track_id for track in initial] == [1]
    assert [track.track_id for track in moved] == [1]
    assert missed == []
    assert [track.track_id for track in recovered] == [1]


def test_botsort_uses_appearance_to_avoid_identity_swap() -> None:
    tracker = BoTSORTTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.3,
        secondary_match_iou_threshold=0.1,
        max_age_frames=2,
        min_hits=1,
        appearance_weight=0.8,
        appearance_threshold=0.1,
        appearance_ambiguous_iou_margin=0.8,
        appearance_embedder=_StaticEmbedder(
            [
                {
                    0: np.array([1.0, 0.0], dtype=np.float32),
                    1: np.array([0.0, 1.0], dtype=np.float32),
                },
                {
                    0: np.array([0.0, 1.0], dtype=np.float32),
                    1: np.array([1.0, 0.0], dtype=np.float32),
                },
            ]
        ),
    )

    frame0 = np.zeros((12, 32, 3), dtype=np.uint8)
    frame0[:, 0:10] = (0, 0, 255)
    frame0[:, 20:30] = (255, 0, 0)
    detections0 = [
        Detection(bbox=(0, 0, 10, 10), score=0.95, class_id=0, label="person"),
        Detection(bbox=(20, 0, 30, 10), score=0.96, class_id=0, label="person"),
    ]

    frame1 = np.zeros((12, 32, 3), dtype=np.uint8)
    frame1[:, 0:10] = (255, 0, 0)
    frame1[:, 20:30] = (0, 0, 255)
    detections1 = [
        Detection(bbox=(8, 0, 18, 10), score=0.93, class_id=0, label="person"),
        Detection(bbox=(12, 0, 22, 10), score=0.94, class_id=0, label="person"),
    ]

    first = tracker.update(detections0, frame_index=0, frame=frame0)
    second = tracker.update(detections1, frame_index=1, frame=frame1)

    assert [track.track_id for track in first] == [1, 2]
    assert second[0].track_id == 1
    assert second[0].bbox == (12, 0, 22, 10)
    assert second[1].track_id == 2
    assert second[1].bbox == (8, 0, 18, 10)


def test_botsort_relinks_track_with_low_iou_when_embedding_matches() -> None:
    tracker = BoTSORTTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.3,
        secondary_match_iou_threshold=0.1,
        max_age_frames=3,
        min_hits=1,
        appearance_weight=0.85,
        appearance_threshold=0.7,
        appearance_ambiguous_iou_margin=0.15,
        appearance_embedder=_StaticEmbedder(
            [
                {0: np.array([1.0, 0.0], dtype=np.float32)},
                {0: np.array([1.0, 0.0], dtype=np.float32)},
            ]
        ),
    )

    first = tracker.update(
        [Detection(bbox=(0, 0, 10, 10), score=0.95, class_id=0, label="person")],
        frame_index=0,
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    tracker.update([], frame_index=1, frame=np.zeros((20, 20, 3), dtype=np.uint8))
    relinked = tracker.update(
        [Detection(bbox=(8, 0, 18, 10), score=0.94, class_id=0, label="person")],
        frame_index=2,
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert [track.track_id for track in first] == [1]
    assert [track.track_id for track in relinked] == [1]


def test_bytetrack_keeps_consistent_ids_for_two_tracks_across_multiple_frames() -> None:
    tracker = ByteTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.2,
        secondary_match_iou_threshold=0.1,
        max_age_frames=3,
        min_hits=1,
    )

    frames = [
        [
            Detection(bbox=(0, 0, 10, 10), score=0.95, class_id=0, label="person"),
            Detection(bbox=(30, 0, 40, 10), score=0.95, class_id=0, label="person"),
        ],
        [
            Detection(bbox=(4, 0, 14, 10), score=0.94, class_id=0, label="person"),
            Detection(bbox=(34, 0, 44, 10), score=0.94, class_id=0, label="person"),
        ],
        [
            Detection(bbox=(8, 0, 18, 10), score=0.93, class_id=0, label="person"),
            Detection(bbox=(38, 0, 48, 10), score=0.93, class_id=0, label="person"),
        ],
    ]

    ids_by_frame = []
    for frame_index, detections in enumerate(frames):
        tracks = tracker.update(detections, frame_index=frame_index)
        ids_by_frame.append([track.track_id for track in tracks])

    assert ids_by_frame == [[1, 2], [1, 2], [1, 2]]


def test_bytetrack_uses_global_motion_compensation_for_camera_shift() -> None:
    tracker = ByteTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.2,
        secondary_match_iou_threshold=0.1,
        max_age_frames=3,
        min_hits=1,
    )

    first = tracker.update(
        [Detection(bbox=(50, 20, 70, 60), score=0.95, class_id=0, label="person")],
        frame_index=0,
    )
    second = tracker.update(
        [Detection(bbox=(30, 20, 50, 60), score=0.94, class_id=0, label="person")],
        frame_index=1,
        motion_transform=np.array(
            [[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )

    assert [track.track_id for track in first] == [1]
    assert [track.track_id for track in second] == [1]
    assert second[0].bbox == (30, 20, 50, 60)


def test_bytetrack_exposes_predicted_trajectory_from_smoothed_motion() -> None:
    tracker = ByteTracker(
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        new_track_score_threshold=0.5,
        match_iou_threshold=0.2,
        secondary_match_iou_threshold=0.1,
        max_age_frames=3,
        min_hits=1,
        trajectory_prediction_enabled=True,
        trajectory_history_size=4,
        trajectory_prediction_horizon=3,
        trajectory_smoothing=0.7,
    )

    tracker.update(
        [Detection(bbox=(0, 0, 10, 10), score=0.95, class_id=0, label="person")],
        frame_index=0,
    )
    tracks = tracker.update(
        [Detection(bbox=(4, 0, 14, 10), score=0.95, class_id=0, label="person")],
        frame_index=1,
    )

    assert len(tracks) == 1
    assert len(tracks[0].predicted_path) == 3
    assert tracks[0].predicted_path[0][0] > tracks[0].center[0]
    assert tracks[0].velocity[0] > 0


def test_assignment_solver_prefers_global_optimum_over_greedy_pick() -> None:
    matches = _solve_assignment(
        track_ids=[1, 2],
        detection_ids=[10, 11],
        scores={
            (1, 10): 0.95,
            (1, 11): 0.94,
            (2, 10): 0.93,
        },
    )

    assert sorted(matches) == [(1, 11), (2, 10)]


def test_build_appearance_embedder_falls_back_to_histogram(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("no pretrained weights")

    monkeypatch.setattr("src.inference.appearance.TorchvisionReIDEmbedder", _raise)

    embedder = build_appearance_embedder({"appearance_model": "mobilenet_v3_small"})

    assert isinstance(embedder, HistogramAppearanceEmbedder)
