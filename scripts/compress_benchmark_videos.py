from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress benchmark videos using ffmpeg."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input .mp4 benchmark videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write compressed .mp4 files.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="CRF value for H.264 compression. Higher means smaller files.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="slow",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="ffmpeg x264 preset.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Optional max output width. 0 keeps the original width.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original files after successful compression.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=Path,
        default=None,
        help="Optional explicit ffmpeg executable path.",
    )
    return parser


def resolve_ffmpeg_binary(explicit_path: Path | None) -> str:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"ffmpeg binary not found: {explicit_path}")
        return str(explicit_path)

    discovered = shutil.which("ffmpeg")
    if discovered is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install ffmpeg or pass --ffmpeg-bin."
        )
    return discovered


def build_ffmpeg_command(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    max_width: int,
) -> list[str]:
    command = [ffmpeg_bin, "-y", "-i", str(input_path)]
    if max_width > 0:
        command.extend(["-vf", f"scale='min(iw,{max_width})':-2"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    )
    return command


def compress_video(
    *,
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    max_width: int,
) -> None:
    command = build_ffmpeg_command(
        ffmpeg_bin=ffmpeg_bin,
        input_path=input_path,
        output_path=output_path,
        crf=crf,
        preset=preset,
        max_width=max_width,
    )
    subprocess.run(command, check=True)


def human_megabytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def main() -> None:
    args = build_parser().parse_args()
    ffmpeg_bin = resolve_ffmpeg_binary(args.ffmpeg_bin)

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    total_before = 0
    total_after = 0

    for input_path in mp4_files:
        output_path = output_dir / input_path.name
        print(f"\nCompressing: {input_path.name}")
        compress_video(
            ffmpeg_bin=ffmpeg_bin,
            input_path=input_path,
            output_path=output_path,
            crf=args.crf,
            preset=args.preset,
            max_width=args.max_width,
        )

        before_size = input_path.stat().st_size
        after_size = output_path.stat().st_size
        total_before += before_size
        total_after += after_size
        reduction = (
            100.0 * (before_size - after_size) / before_size if before_size else 0.0
        )
        print(
            f"  Before: {human_megabytes(before_size)} | "
            f"After: {human_megabytes(after_size)} | "
            f"Reduction: {reduction:.1f}%"
        )

        if args.replace:
            backup_path = input_path.with_suffix(".mp4.bak")
            if backup_path.exists():
                backup_path.unlink()
            input_path.rename(backup_path)
            output_path.replace(input_path)
            print(f"  Replaced original. Backup saved as: {backup_path.name}")

    total_reduction = (
        100.0 * (total_before - total_after) / total_before if total_before else 0.0
    )
    print("\nDone.")
    print(
        f"Total before: {human_megabytes(total_before)} | "
        f"Total after: {human_megabytes(total_after)} | "
        f"Reduction: {total_reduction:.1f}%"
    )
    if args.replace:
        print("Originals were replaced in-place, with .bak backups.")
    else:
        print(f"Compressed files are in: {output_dir}")


if __name__ == "__main__":
    main()
