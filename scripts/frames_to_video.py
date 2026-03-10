from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a dataset frame directory into an mp4 clip without ffmpeg."
    )
    parser.add_argument(
        "--frames-dir", required=True, help="Directory containing ordered image frames."
    )
    parser.add_argument("--out", required=True, help="Output mp4 path.")
    parser.add_argument(
        "--fps", type=float, required=True, help="Output frames per second."
    )
    parser.add_argument(
        "--start-frame", type=int, default=1, help="1-based start frame."
    )
    parser.add_argument(
        "--end-frame", type=int, default=None, help="1-based end frame, inclusive."
    )
    parser.add_argument(
        "--resize-width", type=int, default=0, help="Optional output width."
    )
    parser.add_argument(
        "--resize-height", type=int, default=0, help="Optional output height."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}.")

    start_index = max(args.start_frame - 1, 0)
    end_index = args.end_frame if args.end_frame is not None else len(frame_paths)
    selected_paths = frame_paths[start_index:end_index]
    if not selected_paths:
        raise ValueError("No frames selected for the requested clip window.")

    first_frame = cv2.imread(str(selected_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame {selected_paths[0]}.")
    height, width = first_frame.shape[:2]
    width, height = _resolve_output_size(
        width, height, args.resize_width, args.resize_height
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}.")

    try:
        for frame_path in selected_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame {frame_path}.")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    print(str(output_path))


def _resolve_output_size(
    src_width: int, src_height: int, requested_width: int, requested_height: int
) -> tuple[int, int]:
    if requested_width <= 0 and requested_height <= 0:
        return src_width, src_height
    if requested_width > 0 and requested_height > 0:
        return requested_width, requested_height
    if requested_width > 0:
        scale = requested_width / max(src_width, 1)
        return requested_width, max(1, int(round(src_height * scale)))
    scale = requested_height / max(src_height, 1)
    return max(1, int(round(src_width * scale))), requested_height


if __name__ == "__main__":
    main()
