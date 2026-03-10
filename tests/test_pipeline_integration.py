from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import src.inference.pipeline as pipeline_module
from src.inference.detector import Detection
from src.inference.pipeline import SentinelPipeline


class _FakeVideoSource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.source = "fake"
        self.frames = frames
        self.index = 0
        self.width = frames[0].shape[1]
        self.height = frames[0].shape[0]
        self.fps = 5.0
        self.is_rtsp = False

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def reopen(self) -> bool:
        return False

    def release(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _FakeDetector:
    def __init__(self, *args, **kwargs) -> None:
        self._calls = 0

    def detect(self, frame) -> list[Detection]:
        sequence = [
            [Detection(bbox=(30, 30, 50, 60), score=0.95, class_id=0, label="person")],
            [Detection(bbox=(45, 30, 65, 60), score=0.95, class_id=0, label="person")],
            [Detection(bbox=(60, 30, 80, 60), score=0.95, class_id=0, label="person")],
        ]
        result = sequence[self._calls] if self._calls < len(sequence) else []
        self._calls += 1
        return result


def test_pipeline_runs_synthetic_sequence_and_logs_intrusion(
    tmp_path: Path, monkeypatch
) -> None:
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)]
    fake_source = _FakeVideoSource(frames)

    monkeypatch.setattr(
        pipeline_module, "open_video_source", lambda source: fake_source
    )
    monkeypatch.setattr(pipeline_module, "YoloDetector", _FakeDetector)

    config = {
        "camera_id": "cam_test",
        "model": {
            "path": "fake.pt",
            "confidence": 0.3,
            "device": "cpu",
            "classes": ["person"],
        },
        "tracking": {
            "type": "bytetrack",
            "high_score_threshold": 0.5,
            "low_score_threshold": 0.1,
            "new_track_score_threshold": 0.5,
            "match_iou_threshold": 0.1,
            "secondary_match_iou_threshold": 0.05,
            "max_age_frames": 5,
            "min_hits": 1,
            "appearance_weight": 0.35,
            "appearance_threshold": 0.2,
        },
        "events": {
            "intrusion": {"enabled": True, "cooldown_seconds": 5},
            "loitering": {
                "enabled": False,
                "threshold_seconds": 10,
                "cooldown_seconds": 20,
            },
            "line_crossing": {
                "enabled": False,
                "cooldown_seconds": 3,
                "direction": "any",
            },
            "wrong_way": {
                "enabled": False,
                "cooldown_seconds": 10,
                "min_displacement_pixels": 25,
                "target_classes": ["person"],
            },
            "after_hours_occupancy": {
                "enabled": False,
                "cooldown_seconds": 60,
                "start_time": "08:00",
                "end_time": "18:00",
                "timezone": "America/New_York",
                "target_classes": ["person"],
            },
            "vehicle_in_pedestrian_zone": {
                "enabled": False,
                "cooldown_seconds": 5,
                "target_classes": ["car"],
            },
            "abandoned_object": {
                "enabled": False,
                "cooldown_seconds": 30,
                "unattended_seconds": 20,
                "min_stationary_seconds": 8,
                "stationary_radius_pixels": 20,
                "owner_max_distance_pixels": 80,
                "target_classes": ["backpack"],
                "owner_classes": ["person"],
            },
        },
        "input": {
            "source": "fake",
            "read_failure_threshold": 1,
            "reconnect_attempts": 0,
            "reconnect_backoff_seconds": 0.0,
        },
        "runtime": {
            "frame_skip": 0,
            "adaptive_frame_skip": False,
            "target_processing_fps": 10.0,
            "resize_width": 0,
            "resize_height": 0,
            "timing_log_interval_frames": 100,
        },
        "output": {
            "show_window": False,
            "save_annotated_video": False,
            "annotated_video_path": str(tmp_path / "annotated.mp4"),
            "alerts_dir": str(tmp_path / "alerts"),
            "log_path": str(tmp_path / "events.jsonl"),
            "buffer_seconds": 0.0,
            "post_event_seconds": 0.0,
            "duplicate_suppression_seconds": 10.0,
            "clip_writer_queue_size": 4,
        },
        "dashboard": {"enabled": False, "endpoint": "", "timeout_seconds": 1.0},
        "zones": [
            {
                "name": "restricted_lab",
                "type": "polygon",
                "tags": ["restricted"],
                "metadata": {},
                "points": [(50, 0), (140, 0), (140, 100), (50, 100)],
            }
        ],
    }

    pipeline = SentinelPipeline(config)
    pipeline.run()

    log_path = Path(config["output"]["log_path"])
    assert log_path.exists()
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(events) == 1
    assert events[0]["event_type"] == "intrusion"
    assert events[0]["camera_id"] == "cam_test"


def test_pipeline_dispatch_logs_failure_when_endpoint_errors(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(pipeline_module, "YoloDetector", _FakeDetector)
    config = {
        "camera_id": "cam_test",
        "model": {
            "path": "fake.pt",
            "confidence": 0.3,
            "device": "cpu",
            "classes": ["person"],
        },
        "tracking": {
            "type": "bytetrack",
            "high_score_threshold": 0.5,
            "low_score_threshold": 0.1,
            "new_track_score_threshold": 0.5,
            "match_iou_threshold": 0.1,
            "secondary_match_iou_threshold": 0.05,
            "max_age_frames": 5,
            "min_hits": 1,
            "appearance_weight": 0.35,
            "appearance_threshold": 0.2,
        },
        "events": {
            "intrusion": {"enabled": True, "cooldown_seconds": 5},
            "loitering": {
                "enabled": False,
                "threshold_seconds": 10,
                "cooldown_seconds": 20,
            },
            "line_crossing": {
                "enabled": False,
                "cooldown_seconds": 3,
                "direction": "any",
            },
            "wrong_way": {
                "enabled": False,
                "cooldown_seconds": 10,
                "min_displacement_pixels": 25,
                "target_classes": ["person"],
            },
            "after_hours_occupancy": {
                "enabled": False,
                "cooldown_seconds": 60,
                "start_time": "08:00",
                "end_time": "18:00",
                "timezone": "America/New_York",
                "target_classes": ["person"],
            },
            "vehicle_in_pedestrian_zone": {
                "enabled": False,
                "cooldown_seconds": 5,
                "target_classes": ["car"],
            },
            "abandoned_object": {
                "enabled": False,
                "cooldown_seconds": 30,
                "unattended_seconds": 20,
                "min_stationary_seconds": 8,
                "stationary_radius_pixels": 20,
                "owner_max_distance_pixels": 80,
                "target_classes": ["backpack"],
                "owner_classes": ["person"],
            },
        },
        "input": {
            "source": "fake",
            "read_failure_threshold": 1,
            "reconnect_attempts": 0,
            "reconnect_backoff_seconds": 0.0,
        },
        "runtime": {
            "frame_skip": 0,
            "adaptive_frame_skip": False,
            "target_processing_fps": 10.0,
            "resize_width": 0,
            "resize_height": 0,
            "timing_log_interval_frames": 100,
        },
        "output": {
            "show_window": False,
            "save_annotated_video": False,
            "annotated_video_path": str(tmp_path / "annotated.mp4"),
            "alerts_dir": str(tmp_path / "alerts"),
            "log_path": str(tmp_path / "events.jsonl"),
            "buffer_seconds": 0.0,
            "post_event_seconds": 0.0,
            "duplicate_suppression_seconds": 10.0,
            "clip_writer_queue_size": 4,
        },
        "dashboard": {
            "enabled": True,
            "endpoint": "http://127.0.0.1:9999/ingest",
            "timeout_seconds": 0.1,
        },
        "zones": [],
    }
    pipeline = SentinelPipeline(config)

    def _failing_urlopen(req, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module.request, "urlopen", _failing_urlopen)
    caplog.set_level("ERROR")
    pipeline._dispatch_alert(
        {
            "event_id": "evt_1",
            "timestamp": "2026-03-09T10:00:00Z",
            "camera_id": "cam_test",
            "event_type": "intrusion",
            "zone": "restricted_lab",
            "track_id": 1,
            "class": "person",
            "frame_index": 10,
        }
    )

    assert "Failed to dispatch alert evt_1" in caplog.text
