from __future__ import annotations

import argparse

from src.inference.pipeline import SentinelPipeline
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentinel Vision pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--source", default=None, help="Override input source: webcam index, file path, or RTSP URL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.source is not None:
        config["input"]["source"] = _normalize_source(args.source)
    pipeline = SentinelPipeline(config)
    pipeline.run()


def _normalize_source(raw: str) -> int | str:
    if raw.isdigit():
        return int(raw)
    return raw


if __name__ == "__main__":
    main()
