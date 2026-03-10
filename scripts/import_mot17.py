from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MOT17 gt.txt into Sentinel Vision annotation JSON."
    )
    parser.add_argument("--gt", required=True, help="Path to MOT17 gt.txt")
    parser.add_argument("--out", required=True, help="Output annotation JSON path")
    parser.add_argument(
        "--seqinfo", default=None, help="Optional MOT17 seqinfo.ini path"
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
        "--min-visibility", type=float, default=0.2, help="Minimum visibility to keep"
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
    payload = convert_mot17_annotations(
        gt_path=Path(args.gt),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        min_visibility=args.min_visibility,
        seqinfo_path=Path(args.seqinfo) if args.seqinfo else None,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(str(output_path))


def convert_mot17_annotations(
    *,
    gt_path: Path,
    start_frame: int,
    end_frame: int | None,
    min_visibility: float,
    seqinfo_path: Path | None,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    fps = 30.0
    video_id = gt_path.parents[1].name.lower().replace("-frcnn", "").replace("-", "_")
    if seqinfo_path is not None and seqinfo_path.exists():
        parser = configparser.ConfigParser()
        parser.read(seqinfo_path, encoding="utf-8")
        fps = parser.getfloat("Sequence", "frameRate", fallback=fps)

    detections: list[dict[str, Any]] = []
    clip_end_frame = end_frame if end_frame is not None else 10**9
    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            frame_id, track_id, left, top, width, height, conf, class_id, visibility = (
                _parse_mot_line(line)
            )
            if frame_id < start_frame or frame_id > clip_end_frame:
                continue
            if conf <= 0 or class_id != 1 or visibility < min_visibility:
                continue
            detections.append(
                {
                    "frame_index": frame_id - start_frame,
                    "track_id": track_id,
                    "class": "person",
                    "bbox": [
                        round(left * scale_x, 3),
                        round(top * scale_y, 3),
                        round((left + width) * scale_x, 3),
                        round((top + height) * scale_y, 3),
                    ],
                    "score": round(visibility, 4),
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


def _parse_mot_line(
    line: str,
) -> tuple[int, int, float, float, float, float, float, int, float]:
    parts = [item.strip() for item in line.split(",")]
    return (
        int(parts[0]),
        int(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
        float(parts[5]),
        float(parts[6]),
        int(float(parts[7])),
        float(parts[8]),
    )


if __name__ == "__main__":
    main()
