from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeMotionResult:
    estimated = False
    matrix = None


class _FakeMotionCompensator:
    def update(self, frame):
        return _FakeMotionResult()


class _FakeTrack:
    def __init__(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        label: str = "person",
        score: float = 0.95,
    ) -> None:
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.score = score
        self.world_center = None


class _FakeTracker:
    def update(self, detections, frame_index, frame=None, motion_transform=None):
        if not detections:
            return []
        return [
            _FakeTrack(
                track_id=7,
                bbox=detections[0].bbox,
                label=detections[0].label,
                score=detections[0].score,
            )
        ]


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        detection_cls = sys.modules["src.inference.detector"].Detection
        sequence = [
            [
                detection_cls(
                    bbox=(10, 10, 20, 30), score=0.95, class_id=0, label="person"
                )
            ],
            [
                detection_cls(
                    bbox=(12, 10, 22, 30), score=0.94, class_id=0, label="person"
                )
            ],
            [],
        ]
        result = sequence[self.calls] if self.calls < len(sequence) else []
        self.calls += 1
        return result


class _FakeVideoSource:
    def __init__(self) -> None:
        self.frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        self.index = 0
        self.fps = 5.0
        self.width = 64
        self.height = 48

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        return None

    def reopen(self) -> bool:
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _FakePipeline:
    def __init__(self, config):
        self.config = config
        self.detector = _FakeDetector()
        self.tracker = _FakeTracker()
        self.motion_compensator = _FakeMotionCompensator()

    def _resize_for_inference(self, frame):
        return frame

    def _scale_detections_to_frame(self, detections, inference_frame, frame):
        return detections

    def _project_tracks_to_ground_plane(self, tracks):
        return None

    def _evaluate_events(self, tracks, frame_index, timestamp, fps):
        if frame_index != 1:
            return []
        return [
            {
                "event_type": "intrusion",
                "zone": "restricted_lab",
                "class": "person",
                "track_id": 7,
                "entry_frame_index": 1,
                "frame_index": 1,
                "confidence": 0.88,
            }
        ]


def test_run_video_benchmark_exports_predictions_and_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "SentinelPipeline", _FakePipeline)
    monkeypatch.setattr(module, "open_video_source", lambda source: _FakeVideoSource())

    video_spec = {
        "video_id": "synthetic_clip",
        "video_path": "videos/synthetic_clip.mp4",
        "ground_truth": "annotations/synthetic_clip.json",
        "predictions": "predictions/synthetic_clip.json",
        "fps": 5.0,
        "duration_seconds": 0.6,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
camera_id: benchmark_cam
model:
  path: fake.pt
  confidence: 0.25
  device: cpu
  classes: [person]
tracking:
  type: bytetrack
events:
  intrusion:
    enabled: true
zones: []
""".strip(),
        encoding="utf-8",
    )

    payload = module.run_video_benchmark(
        video_spec=video_spec,
        base_dir=tmp_path,
        default_config_path=config_path,
        default_profile=None,
        device="cpu",
        model_path=None,
    )

    assert payload["video_id"] == "synthetic_clip"
    assert len(payload["detections"]) == 2
    assert payload["events"][0]["event_type"] == "intrusion"
    assert payload["runtime"]["device"] == "cpu"
    assert payload["runtime"]["frames_processed"] == 3


def test_run_video_benchmark_supports_frame_skip(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "SentinelPipeline", _FakePipeline)
    monkeypatch.setattr(module, "open_video_source", lambda source: _FakeVideoSource())

    video_spec = {
        "video_id": "synthetic_clip",
        "video_path": "videos/synthetic_clip.mp4",
        "ground_truth": "annotations/synthetic_clip.json",
        "predictions": "predictions/synthetic_clip.json",
        "fps": 5.0,
        "duration_seconds": 0.6,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
camera_id: benchmark_cam
model:
  path: fake.pt
  confidence: 0.25
  device: cpu
  classes: [person]
tracking:
  type: bytetrack
events:
  intrusion:
    enabled: true
zones: []
""".strip(),
        encoding="utf-8",
    )

    payload = module.run_video_benchmark(
        video_spec=video_spec,
        base_dir=tmp_path,
        default_config_path=config_path,
        default_profile=None,
        device="cpu",
        model_path=None,
        frame_skip=1,
    )

    assert payload["runtime"]["frames_processed"] == 2
    assert [item["frame_index"] for item in payload["detections"]] == [0, 2]


def test_run_video_benchmark_supports_max_frames(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "SentinelPipeline", _FakePipeline)
    monkeypatch.setattr(module, "open_video_source", lambda source: _FakeVideoSource())

    video_spec = {
        "video_id": "synthetic_clip",
        "video_path": "videos/synthetic_clip.mp4",
        "ground_truth": "annotations/synthetic_clip.json",
        "predictions": "predictions/synthetic_clip.json",
        "fps": 5.0,
        "duration_seconds": 0.6,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
camera_id: benchmark_cam
model:
  path: fake.pt
  confidence: 0.25
  device: cpu
  classes: [person]
tracking:
  type: bytetrack
events:
  intrusion:
    enabled: true
zones: []
""".strip(),
        encoding="utf-8",
    )

    payload = module.run_video_benchmark(
        video_spec=video_spec,
        base_dir=tmp_path,
        default_config_path=config_path,
        default_profile=None,
        device="cpu",
        model_path=None,
        max_frames=1,
    )

    assert payload["runtime"]["frames_processed"] == 1
    assert len(payload["detections"]) == 1
    assert payload["events"] == []
