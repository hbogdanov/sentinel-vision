from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts import evaluate_events as evaluator
from src.inference.pipeline import SentinelPipeline
from src.io.video import open_video_source
from src.utils.config import load_config, load_yaml_config, merge_config_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Sentinel Vision inference over a benchmark manifest and score it."
    )
    parser.add_argument(
        "--manifest",
        default="data/eval/benchmark_manifest.json",
        help="Benchmark manifest path.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Default config path used when a video entry does not override it.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional default config profile name or YAML path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override inference device, for example cpu or cuda:0.",
    )
    parser.add_argument("--model", default=None, help="Override model path.")
    parser.add_argument(
        "--predictions-dir",
        default="data/eval/predictions",
        help="Directory for generated prediction JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="data/eval/results/latest.json",
        help="Evaluation output JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        default="docs/results.md",
        help="Optional markdown summary path.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used during evaluation.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every N+1th frame for faster ad hoc benchmark runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after processing this many frames per video. 0 means no limit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    base_dir = manifest_path.parent
    predictions_dir = Path(args.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for video_spec in manifest["videos"]:
        prediction_path = predictions_dir / f"{video_spec['video_id']}.json"
        generated = run_video_benchmark(
            video_spec=video_spec,
            base_dir=base_dir,
            default_config_path=Path(args.config),
            default_profile=args.profile,
            device=args.device,
            model_path=args.model,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
        )
        prediction_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
        video_spec["predictions"] = str(prediction_path.relative_to(base_dir)).replace(
            "\\", "/"
        )

    bundles = [
        evaluator._load_video_bundle(base_dir, video_spec)
        for video_spec in manifest["videos"]
    ]
    results = evaluator.evaluate_manifest(bundles, iou_threshold=args.iou_threshold)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.output_markdown:
        output_markdown_path = Path(args.output_markdown)
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(
            evaluator.render_markdown(results), encoding="utf-8"
        )

    print(json.dumps(results["summary"], indent=2))


def run_video_benchmark(
    *,
    video_spec: dict[str, Any],
    base_dir: Path,
    default_config_path: Path,
    default_profile: str | None,
    device: str | None,
    model_path: str | None,
    frame_skip: int = 0,
    max_frames: int = 0,
) -> dict[str, Any]:
    if frame_skip < 0:
        raise ValueError("frame_skip must be >= 0.")
    if max_frames < 0:
        raise ValueError("max_frames must be >= 0.")

    pipeline = _build_pipeline(
        video_spec=video_spec,
        base_dir=base_dir,
        default_config_path=default_config_path,
        default_profile=default_profile,
        device=device,
        model_path=model_path,
    )

    source_path = base_dir / video_spec["video_path"]
    detections: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    stage_total_seconds: dict[str, float] = defaultdict(float)
    frames_processed = 0
    fps = float(video_spec.get("fps", 0.0))
    benchmark_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    wall_clock_start = time.perf_counter()

    with open_video_source(str(source_path)) as capture:
        fps = fps or capture.fps or 30.0
        frame_index = 0
        while True:
            read_started = time.perf_counter()
            ok, frame = capture.read()
            stage_total_seconds["read"] += time.perf_counter() - read_started
            if not ok:
                break

            if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                frame_index += 1
                continue

            if max_frames > 0 and frames_processed >= max_frames:
                break

            frames_processed += 1
            inference_frame = pipeline._resize_for_inference(frame)

            motion_started = time.perf_counter()
            motion_result = pipeline.motion_compensator.update(frame)
            stage_total_seconds["motion"] += time.perf_counter() - motion_started

            detect_started = time.perf_counter()
            raw_detections = pipeline.detector.detect(inference_frame)
            raw_detections = pipeline._scale_detections_to_frame(
                raw_detections, inference_frame, frame
            )
            stage_total_seconds["detect"] += time.perf_counter() - detect_started

            track_started = time.perf_counter()
            tracks = pipeline.tracker.update(
                raw_detections,
                frame_index=frame_index,
                frame=frame,
                motion_transform=(
                    motion_result.matrix if motion_result.estimated else None
                ),
            )
            pipeline._project_tracks_to_ground_plane(tracks)
            stage_total_seconds["track"] += time.perf_counter() - track_started

            timestamp = benchmark_start + timedelta(
                seconds=frame_index / max(fps, 1e-6)
            )
            events_started = time.perf_counter()
            current_events = pipeline._evaluate_events(
                tracks, frame_index, timestamp, fps
            )
            stage_total_seconds["events"] += time.perf_counter() - events_started

            for track in tracks:
                detections.append(
                    {
                        "frame_index": frame_index,
                        "track_id": track.track_id,
                        "class": track.label,
                        "bbox": [round(value, 3) for value in track.bbox],
                        "score": round(track.score, 4),
                    }
                )

            for event in current_events:
                events.append(
                    {
                        "event_type": str(event["event_type"]),
                        "zone": str(event["zone"]),
                        "class": str(event["class"]),
                        "track_id": (
                            int(event["track_id"])
                            if event.get("track_id") is not None
                            else None
                        ),
                        "entry_frame_index": int(
                            event.get("entry_frame_index", event["frame_index"])
                        ),
                        "frame_index": int(event["frame_index"]),
                        "confidence": (
                            round(float(event.get("confidence", 0.0)), 4)
                            if event.get("confidence") is not None
                            else None
                        ),
                    }
                )

            frame_index += 1

    wall_clock_seconds = time.perf_counter() - wall_clock_start
    duration_seconds = float(video_spec.get("duration_seconds", 0.0)) or (
        frames_processed / max(fps, 1e-6)
    )
    return {
        "video_id": str(video_spec["video_id"]),
        "fps": round(fps, 4),
        "duration_seconds": round(duration_seconds, 4),
        "detections": detections,
        "events": events,
        "runtime": {
            "device": str(device or pipeline.config["model"].get("device", "cpu")),
            "frames_processed": frames_processed,
            "wall_clock_seconds": round(wall_clock_seconds, 6),
            "effective_fps": (
                round(frames_processed / wall_clock_seconds, 4)
                if wall_clock_seconds
                else 0.0
            ),
            "stage_total_seconds": {
                stage: round(value, 6)
                for stage, value in sorted(stage_total_seconds.items())
            },
        },
    }


def _build_pipeline(
    *,
    video_spec: dict[str, Any],
    base_dir: Path,
    default_config_path: Path,
    default_profile: str | None,
    device: str | None,
    model_path: str | None,
) -> SentinelPipeline:
    config_path = (
        base_dir / video_spec["config"]
        if "config" in video_spec
        else default_config_path
    )
    config = load_config(str(config_path))

    profile = str(video_spec.get("profile", default_profile or "")).strip()
    if profile:
        profile_path = _resolve_profile_path(profile)
        config = merge_config_overlay(config, load_yaml_config(str(profile_path)))

    if "config_override" in video_spec:
        config = merge_config_overlay(config, dict(video_spec["config_override"]))

    config["input"]["source"] = str(base_dir / video_spec["video_path"])
    config["output"]["show_window"] = False
    config["output"]["save_annotated_video"] = False
    config["output"]["buffer_seconds"] = 0.0
    config["output"]["post_event_seconds"] = 0.0

    if device is not None:
        config["model"]["device"] = device
    if model_path is not None:
        config["model"]["path"] = model_path
    return SentinelPipeline(config)


def _resolve_profile_path(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate
    profile_path = Path("configs") / "profiles" / f"{profile}.yaml"
    if profile_path.exists():
        return profile_path
    raise FileNotFoundError(f"Could not resolve config profile '{profile}'.")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
