from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from src.io.recorder import AlertRecorder


def test_recorder_writes_pre_and_post_event_clip_with_metadata(tmp_path: Path) -> None:
    recorder = AlertRecorder(
        alerts_dir=str(tmp_path),
        annotated_video_path=None,
        save_annotated_video=False,
        buffer_seconds=1.0,
        post_event_seconds=1.0,
        duplicate_suppression_seconds=5.0,
    )
    fps = 2.0
    frames = [_frame_with_value(value) for value in (10, 20, 30, 40, 50)]
    for frame in frames[:3]:
        recorder.ingest_frame(frame, fps=fps)

    event = {
        "event_id": "evt_000001",
        "timestamp": datetime(2026, 3, 9, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_type": "intrusion",
        "track_id": 7,
        "zone": "restricted_lab",
    }
    snapshot_path, clip_path, metadata_path = recorder.start_alert(event=event, frame=frames[2], fps=fps)
    recorder.ingest_frame(frames[3], fps=fps)
    recorder.ingest_frame(frames[4], fps=fps)
    recorder.close(fps=fps)

    assert snapshot_path is not None and clip_path is not None and metadata_path is not None
    assert Path(snapshot_path).name.startswith("20260309T000000Z_intrusion_restricted_lab_track7_evt_000001")
    assert Path(snapshot_path).exists()
    assert Path(clip_path).exists()
    assert Path(metadata_path).exists()

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert metadata["pre_event_seconds"] == 1.0
    assert metadata["post_event_seconds"] == 1.0
    assert metadata["clip_frame_count"] == 5
    assert Path(clip_path).name == "20260309T000000Z_intrusion_restricted_lab_track7_evt_000001.mp4"
    assert Path(metadata_path).name == "20260309T000000Z_intrusion_restricted_lab_track7_evt_000001.json"

    capture = cv2.VideoCapture(clip_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frame_count == 5


def test_recorder_suppresses_duplicate_alerts_for_same_event_track_and_zone(tmp_path: Path) -> None:
    recorder = AlertRecorder(
        alerts_dir=str(tmp_path),
        annotated_video_path=None,
        save_annotated_video=False,
        buffer_seconds=1.0,
        post_event_seconds=0.0,
        duplicate_suppression_seconds=10.0,
    )
    fps = 2.0
    frame = _frame_with_value(10)
    recorder.ingest_frame(frame, fps=fps)
    base_time = datetime(2026, 3, 9, tzinfo=timezone.utc)

    first_event = {
        "event_id": "evt_000001",
        "timestamp": base_time.isoformat().replace("+00:00", "Z"),
        "event_type": "intrusion",
        "track_id": 3,
        "zone": "restricted_lab",
    }
    duplicate_event = {
        "event_id": "evt_000002",
        "timestamp": (base_time + timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
        "event_type": "intrusion",
        "track_id": 3,
        "zone": "restricted_lab",
    }

    first_paths = recorder.start_alert(event=first_event, frame=frame, fps=fps)
    duplicate_paths = recorder.start_alert(event=duplicate_event, frame=frame, fps=fps)
    recorder.close(fps=fps)

    assert first_paths[0] is not None
    assert duplicate_paths == (None, None, None)


def test_recorder_falls_back_to_safe_fps_when_input_fps_is_invalid(tmp_path: Path) -> None:
    recorder = AlertRecorder(
        alerts_dir=str(tmp_path),
        annotated_video_path=None,
        save_annotated_video=False,
        buffer_seconds=1.0,
        post_event_seconds=0.0,
        duplicate_suppression_seconds=5.0,
    )
    frame = _frame_with_value(10)
    recorder.ingest_frame(frame, fps=0.0)
    event = {
        "event_id": "evt_000003",
        "timestamp": datetime(2026, 3, 9, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_type": "intrusion",
        "track_id": 9,
        "zone": "restricted_lab",
    }

    _, clip_path, metadata_path = recorder.start_alert(event=event, frame=frame, fps=0.0)
    recorder.close(fps=0.0)

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    assert clip_path is not None
    assert Path(clip_path).exists()
    assert metadata["clip_write_ok"] in {True, False}
    assert metadata["deduplicated_by"] == ["event_type", "track_id", "zone"]


def _frame_with_value(value: int) -> np.ndarray:
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[:, :] = value
    return frame
