from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "camera_id": "default_camera",
    "model": {"path": "yolo11n.pt", "confidence": 0.35, "device": "cpu", "classes": ["person"]},
    "tracking": {"iou_threshold": 0.3, "max_age_frames": 30, "min_hits": 1},
    "events": {
        "intrusion": {"enabled": True, "cooldown_seconds": 5},
        "loitering": {"enabled": True, "threshold_seconds": 10, "cooldown_seconds": 20},
    },
    "input": {"source": 0},
    "output": {
        "show_window": False,
        "save_annotated_video": True,
        "annotated_video_path": "data/outputs/annotated.mp4",
        "alerts_dir": "data/outputs/alerts",
        "log_path": "data/outputs/events.jsonl",
        "buffer_seconds": 3,
    },
    "dashboard": {"enabled": False, "endpoint": "", "timeout_seconds": 1.0},
    "zones": [],
}


def load_config(path: str) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(config, loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
