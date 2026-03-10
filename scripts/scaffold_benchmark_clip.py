from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create annotation/prediction templates and a manifest snippet for a new benchmark clip."
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="Stable clip identifier, for example virat_loading_dock_01.",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the clip relative to data/eval/, for example videos/virat_loading_dock_01.mp4.",
    )
    parser.add_argument(
        "--fps", type=float, required=True, help="Clip frames per second."
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        required=True,
        help="Clip duration in seconds.",
    )
    parser.add_argument(
        "--scene-type",
        action="append",
        default=[],
        help="Scene type tag. Repeat for multiple values.",
    )
    parser.add_argument(
        "--challenge-tag",
        action="append",
        default=[],
        help="Challenge tag. Repeat for multiple values.",
    )
    parser.add_argument(
        "--subject-class",
        action="append",
        default=[],
        help="Subject class tag. Repeat for multiple values.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Short description of why the clip is in the benchmark.",
    )
    parser.add_argument(
        "--manifest-snippet-out",
        default=None,
        help="Optional path to write the manifest snippet JSON.",
    )
    parser.add_argument(
        "--eval-root", default="data/eval", help="Evaluation root directory."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eval_root = Path(args.eval_root)
    annotations_dir = eval_root / "annotations"
    predictions_dir = eval_root / "predictions"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = annotations_dir / f"{args.video_id}.json"
    prediction_path = predictions_dir / f"{args.video_id}.json"

    template_payload = {
        "video_id": args.video_id,
        "fps": round(args.fps, 4),
        "duration_seconds": round(args.duration_seconds, 4),
        "detections": [],
        "events": [],
    }
    annotation_path.write_text(
        json.dumps(template_payload, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path.write_text(
        json.dumps(template_payload, indent=2) + "\n", encoding="utf-8"
    )

    snippet = {
        "video_id": args.video_id,
        "video_path": _normalize_rel_path(args.video_path),
        "fps": round(args.fps, 4),
        "duration_seconds": round(args.duration_seconds, 4),
        "ground_truth": f"annotations/{args.video_id}.json",
        "predictions": f"predictions/{args.video_id}.json",
        "scene_types": args.scene_type,
        "challenge_tags": args.challenge_tag,
        "subject_classes": args.subject_class,
        "notes": args.notes,
    }

    if args.manifest_snippet_out:
        snippet_path = Path(args.manifest_snippet_out)
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet_path.write_text(json.dumps(snippet, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "annotation": str(annotation_path),
                "prediction": str(prediction_path),
                "manifest_snippet": snippet,
            },
            indent=2,
        )
    )


def _normalize_rel_path(value: str) -> str:
    return value.replace("\\", "/")


if __name__ == "__main__":
    main()
