import pytest
from pydantic import ValidationError

from pathlib import Path

from src.utils.config import expand_camera_configs, load_config, merge_config_overlay


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera_id: cam_2\nzones: []\nmodel:\n  confidence: 0.5\n", encoding="utf-8")

    config = load_config(str(config_path))

    assert config["camera_id"] == "cam_2"
    assert config["model"]["confidence"] == 0.5
    assert config["tracking"]["type"] == "bytetrack"
    assert config["tracking"]["max_age_frames"] == 30
    assert config["events"]["line_crossing"]["enabled"] is True
    assert config["events"]["after_hours_occupancy"]["timezone"] == "America/New_York"


def test_load_config_rejects_invalid_zone_points(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "camera_id: cam_2\nzones:\n  - name: bad_zone\n    points:\n      - [0, 0]\n      - [10, 10]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_load_config_rejects_unsupported_classes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera_id: cam_2\nzones: []\nmodel:\n  classes:\n    - person\n    - toaster\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_load_config_rejects_invalid_dashboard_url(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "camera_id: cam_2\nzones: []\ndashboard:\n  enabled: true\n  endpoint: not-a-url\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_load_config_rejects_negative_runtime_frame_skip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera_id: cam_2\nzones: []\nruntime:\n  frame_skip: -1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_load_config_rejects_invalid_motion_compensation_method(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "camera_id: cam_2\nzones: []\nruntime:\n  motion_compensation:\n    method: projective\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_load_config_rejects_incomplete_perspective_correspondences(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "camera_id: cam_2\nzones: []\nperspective:\n  enabled: true\n  image_points: [[0, 0], [1, 0], [1, 1]]\n  world_points: [[0, 0], [1, 0], [1, 1]]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(str(config_path))


def test_expand_camera_configs_namespaces_outputs() -> None:
    config = {
        "camera_id": "base_cam",
        "model": {"path": "fake.pt", "confidence": 0.3, "device": "cpu", "classes": ["person"]},
        "tracking": {"type": "bytetrack", "max_age_frames": 30, "min_hits": 2},
        "events": {
            "intrusion": {"enabled": True, "cooldown_seconds": 5},
            "loitering": {"enabled": True, "threshold_seconds": 10, "cooldown_seconds": 20},
            "line_crossing": {"enabled": True, "cooldown_seconds": 3, "direction": "any"},
            "wrong_way": {"enabled": True, "cooldown_seconds": 10, "min_displacement_pixels": 25, "target_classes": ["person"]},
            "after_hours_occupancy": {
                "enabled": True,
                "cooldown_seconds": 60,
                "start_time": "08:00",
                "end_time": "18:00",
                "timezone": "America/New_York",
                "target_classes": ["person"],
            },
            "vehicle_in_pedestrian_zone": {"enabled": True, "cooldown_seconds": 5, "target_classes": ["car"]},
            "abandoned_object": {
                "enabled": True,
                "cooldown_seconds": 30,
                "unattended_seconds": 20,
                "min_stationary_seconds": 8,
                "stationary_radius_pixels": 20,
                "owner_max_distance_pixels": 80,
                "target_classes": ["backpack"],
                "owner_classes": ["person"],
            },
        },
        "input": {"source": 0, "read_failure_threshold": 30, "reconnect_attempts": 3, "reconnect_backoff_seconds": 1.0},
        "runtime": {"frame_skip": 0, "adaptive_frame_skip": False, "target_processing_fps": 10.0, "resize_width": 0, "resize_height": 0, "timing_log_interval_frames": 120, "motion_compensation": {"enabled": False, "static_camera_assumption": True, "method": "affine", "max_corners": 300, "quality_level": 0.01, "min_distance": 8.0, "min_matches": 12, "ransac_threshold": 3.0, "smoothing_factor": 0.8}},
        "perspective": {"enabled": False, "image_points": [], "world_points": []},
        "output": {
            "show_window": False,
            "save_annotated_video": False,
            "annotated_video_path": "data/outputs/annotated.mp4",
            "alerts_dir": "data/outputs/alerts",
            "log_path": "data/outputs/events.jsonl",
            "buffer_seconds": 0.0,
            "post_event_seconds": 0.0,
            "duplicate_suppression_seconds": 10.0,
            "clip_writer_queue_size": 4,
            "health_status_path": "data/outputs/camera_health.json",
        },
        "dashboard": {"enabled": False, "endpoint": "", "timeout_seconds": 1.0},
        "zones": [],
        "cameras": [
            {"camera_id": "north_gate", "input": {"source": "rtsp://north"}, "zones": []},
            {"camera_id": "south_gate", "input": {"source": "rtsp://south"}, "zones": []},
        ],
    }

    expanded = expand_camera_configs(config)

    assert len(expanded) == 2
    assert expanded[0]["camera_id"] == "north_gate"
    assert expanded[0]["output"]["alerts_dir"].endswith("alerts/north_gate")
    assert expanded[0]["output"]["log_path"].endswith("events_north_gate.jsonl")
    assert expanded[1]["output"]["annotated_video_path"].endswith("annotated_south_gate.mp4")


def test_merge_config_overlay_applies_runtime_profile() -> None:
    base = load_config("configs/default.yaml")

    merged = merge_config_overlay(
        base,
        {
            "model": {"device": "cpu"},
            "runtime": {"frame_skip": 2, "target_processing_fps": 7.5},
            "output": {"save_annotated_video": False},
        },
    )

    assert merged["model"]["device"] == "cpu"
    assert merged["runtime"]["frame_skip"] == 2
    assert merged["runtime"]["target_processing_fps"] == 7.5
    assert merged["output"]["save_annotated_video"] is False
