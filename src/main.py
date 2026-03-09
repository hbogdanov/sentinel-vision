from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.inference.multi_camera import MultiCameraRunner
from src.inference.pipeline import SentinelPipeline
from src.utils.config import expand_camera_configs, load_config, load_yaml_config, merge_config_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel Vision pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default=None, help="Config profile name under configs/profiles/ or explicit YAML path")
    parser.add_argument("--source", default=None, help="Override input source: webcam index, file path, or RTSP URL")
    parser.add_argument("--device", default=None, help="Override model device, for example cpu, cuda, or cuda:0")
    parser.add_argument("--model", default=None, help="Override model path, for example yolo11n.pt or models/best.pt")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.profile is not None:
        config = merge_config_overlay(config, load_yaml_config(str(_resolve_profile_path(args.profile))))
    if args.source is not None and config.get("cameras"):
        raise ValueError("--source override is only supported for single-camera configs.")
    if args.source is not None:
        config["input"]["source"] = _normalize_source(args.source)
    config = _apply_global_overrides(config, model_path=args.model, device=args.device)
    camera_configs = expand_camera_configs(config)
    camera_configs = [_apply_global_overrides(camera_config, model_path=args.model, device=args.device) for camera_config in camera_configs]
    if len(camera_configs) == 1:
        SentinelPipeline(camera_configs[0]).run()
        return
    MultiCameraRunner(camera_configs).run()


def _normalize_source(raw: str) -> int | str:
    if raw.isdigit():
        return int(raw)
    return raw


def _resolve_profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    profile_path = Path("configs") / "profiles" / f"{profile}.yaml"
    if profile_path.exists():
        return profile_path
    raise FileNotFoundError(f"Could not resolve config profile '{profile}'.")


def _apply_global_overrides(
    config: dict[str, Any],
    *,
    model_path: str | None,
    device: str | None,
) -> dict[str, Any]:
    updated = dict(config)
    if model_path is not None:
        updated["model"] = {**updated.get("model", {}), "path": model_path}
    if device is not None:
        updated["model"] = {**updated.get("model", {}), "device": device}
    return updated


if __name__ == "__main__":
    main()
