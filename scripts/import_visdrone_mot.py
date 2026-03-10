from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VISDRONE_CLASS_MAP = {
    1: "person",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "car",
    6: "truck",
    9: "bus",
    10: "motorcycle",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone MOT annotations into Sentinel Vision annotation JSON."
    )
    parser.add_argument("--gt", required=True, help="Path to VisDrone annotation txt")
    parser.add_argument("--out", required=True, help="Output annotation JSON path")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate to store in the output payload",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="1-based start frame for the clip window",
    )
    parser.add_argument(
        "--end-frame", type=int, default=None, help="1-based end frame, inclusive"
    )
    parser.add_argument(
        "--scale-x",
        type=float,
        default=1.0,
        help="Optional x scaling factor for bbox export",
    )
    parser.add_argument(
        "--scale-y",
        type=float,
        default=1.0,
        help="Optional y scaling factor for bbox export",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = convert_visdrone_annotations(
        gt_path=Path(args.gt),
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(str(output_path))


def convert_visdrone_annotations(
    *,
    gt_path: Path,
    fps: float,
    start_frame: int,
    end_frame: int | None,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    video_id = gt_path.stem.lower()
    detections: list[dict[str, Any]] = []
    clip_end_frame = end_frame if end_frame is not None else 10**9

    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            frame_id, track_id, left, top, width, height, score, category_id = (
                _parse_visdrone_line(line)
            )
            if frame_id < start_frame or frame_id > clip_end_frame:
                continue
            label = VISDRONE_CLASS_MAP.get(category_id)
            if label is None:
                continue
            detections.append(
                {
                    "frame_index": frame_id - start_frame,
                    "track_id": track_id + 1,
                    "class": label,
                    "bbox": [
                        round(left * scale_x, 3),
                        round(top * scale_y, 3),
                        round((left + width) * scale_x, 3),
                        round((top + height) * scale_y, 3),
                    ],
                    "score": round(score, 4),
                }
            )

    duration_seconds = (
        ((max((item["frame_index"] for item in detections), default=-1) + 1) / fps)
        if detections
        else 0.0
    )
    return {
        "video_id": video_id,
        "fps": round(fps, 4),
        "duration_seconds": round(duration_seconds, 4),
        "detections": detections,
        "events": [],
    }


def _parse_visdrone_line(
    line: str,
) -> tuple[int, int, float, float, float, float, float, int]:
    parts = [item.strip() for item in line.split(",")]
    return (
        int(parts[0]),
        int(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
        float(parts[5]),
        float(parts[6]),
        int(parts[7]),
    )


if __name__ == "__main__":
    main()
