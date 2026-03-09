from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_schema import AppConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "camera_id": "default_camera",
    "model": {"path": "yolo11n.pt", "confidence": 0.35, "device": "cpu", "classes": ["person"]},
    "tracking": {
        "type": "bytetrack",
        "high_score_threshold": 0.5,
        "low_score_threshold": 0.1,
        "new_track_score_threshold": 0.6,
        "match_iou_threshold": 0.3,
        "secondary_match_iou_threshold": 0.15,
        "max_age_frames": 30,
        "min_hits": 2,
        "appearance_weight": 0.35,
        "appearance_threshold": 0.2,
    },
    "events": {
        "intrusion": {"enabled": True, "cooldown_seconds": 5},
        "loitering": {"enabled": True, "threshold_seconds": 10, "cooldown_seconds": 20},
        "line_crossing": {"enabled": True, "cooldown_seconds": 3, "direction": "any"},
        "wrong_way": {
            "enabled": True,
            "cooldown_seconds": 10,
            "min_displacement_pixels": 25,
            "target_classes": ["person", "car", "truck", "bus"],
        },
        "after_hours_occupancy": {
            "enabled": True,
            "cooldown_seconds": 60,
            "start_time": "08:00",
            "end_time": "18:00",
            "timezone": "America/New_York",
            "target_classes": ["person"],
        },
        "vehicle_in_pedestrian_zone": {
            "enabled": True,
            "cooldown_seconds": 5,
            "target_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
        },
    },
    "input": {"source": 0, "read_failure_threshold": 30, "reconnect_attempts": 3, "reconnect_backoff_seconds": 1.0},
    "runtime": {
        "frame_skip": 0,
        "adaptive_frame_skip": False,
        "target_processing_fps": 10.0,
        "resize_width": 0,
        "resize_height": 0,
        "timing_log_interval_frames": 120,
        "motion_compensation": {
            "enabled": False,
            "static_camera_assumption": True,
            "method": "affine",
            "max_corners": 300,
            "quality_level": 0.01,
            "min_distance": 8.0,
            "min_matches": 12,
            "ransac_threshold": 3.0,
            "smoothing_factor": 0.8,
        },
    },
    "perspective": {"enabled": False, "image_points": [], "world_points": []},
    "output": {
        "show_window": False,
        "save_annotated_video": True,
        "annotated_video_path": "data/outputs/annotated.mp4",
        "alerts_dir": "data/outputs/alerts",
        "log_path": "data/outputs/events.jsonl",
        "buffer_seconds": 3,
        "post_event_seconds": 3,
        "duplicate_suppression_seconds": 10,
        "clip_writer_queue_size": 16,
        "health_status_path": "data/outputs/camera_health.json",
    },
    "dashboard": {"enabled": False, "endpoint": "", "timeout_seconds": 1.0},
    "zones": [],
    "cameras": [],
}


def load_config(path: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    loaded = load_yaml_config(path)
    merged = _deep_merge(config, loaded)
    return validate_config(merged)


def load_yaml_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    return AppConfig.model_validate(config).model_dump()


def merge_config_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return validate_config(_deep_merge(deepcopy(config), deepcopy(overlay)))


def expand_camera_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    cameras = list(config.get("cameras", []))
    if not cameras:
        return [config]

    base = deepcopy(config)
    base.pop("cameras", None)
    camera_configs: list[dict[str, Any]] = []
    for camera_override in cameras:
        merged = _deep_merge(deepcopy(base), deepcopy(camera_override))
        merged["output"] = _namespace_output_paths(merged["output"], merged["camera_id"])
        camera_configs.append(AppConfig.model_validate({**merged, "cameras": []}).model_dump())
    return camera_configs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _namespace_output_paths(output: dict[str, Any], camera_id: str) -> dict[str, Any]:
    namespaced = deepcopy(output)
    safe_camera_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in camera_id) or "camera"
    alerts_dir = Path(str(namespaced.get("alerts_dir", "data/outputs/alerts")))
    namespaced["alerts_dir"] = str(alerts_dir / safe_camera_id).replace("\\", "/")
    for key in ("annotated_video_path", "log_path", "health_status_path"):
        raw = namespaced.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        suffix = path.suffix
        stem = path.stem
        namespaced[key] = str(path.with_name(f"{stem}_{safe_camera_id}{suffix}")).replace("\\", "/")
    return namespaced
