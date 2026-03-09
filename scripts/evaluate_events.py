from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DetectionRecord:
    frame_index: int
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    score: float | None = None


@dataclass(slots=True)
class EventRecord:
    event_type: str
    zone: str
    label: str
    frame_index: int
    entry_frame_index: int
    track_id: int | None = None
    confidence: float | None = None
    match_tolerance_frames: int = 15


@dataclass(slots=True)
class VideoBundle:
    video_id: str
    fps: float
    duration_seconds: float
    gt_detections: list[DetectionRecord]
    pred_detections: list[DetectionRecord]
    gt_events: list[EventRecord]
    pred_events: list[EventRecord]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detection, tracking, and event quality on labeled clips.")
    parser.add_argument(
        "--manifest",
        default="data/eval/benchmark_manifest.json",
        help="Path to the evaluation manifest.",
    )
    parser.add_argument(
        "--output-json",
        default="data/eval/results/latest.json",
        help="Where to write the aggregated metrics JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional path to write a markdown summary.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for detection/tracking matching.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    base_dir = manifest_path.parent
    bundles = [_load_video_bundle(base_dir, video_spec) for video_spec in manifest["videos"]]
    results = evaluate_manifest(bundles, iou_threshold=args.iou_threshold)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.output_markdown:
        markdown_path = Path(args.output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(results), encoding="utf-8")

    print(json.dumps(results["summary"], indent=2))


def evaluate_manifest(bundles: list[VideoBundle], iou_threshold: float = 0.5) -> dict[str, Any]:
    per_video: list[dict[str, Any]] = []
    aggregate_detection: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    aggregate_tracking = {"gt_count": 0, "pred_count": 0, "matched_count": 0, "fp": 0, "fn": 0, "id_switches": 0}
    aggregate_event: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "latencies": [], "latency_errors": []}
    )
    total_duration_seconds = 0.0

    for bundle in bundles:
        detection_metrics = evaluate_detections(bundle.gt_detections, bundle.pred_detections, iou_threshold=iou_threshold)
        tracking_metrics = evaluate_tracking(bundle.gt_detections, bundle.pred_detections, iou_threshold=iou_threshold)
        event_metrics = evaluate_events(bundle.gt_events, bundle.pred_events, fps=bundle.fps, duration_seconds=bundle.duration_seconds)

        total_duration_seconds += bundle.duration_seconds
        per_video.append(
            {
                "video_id": bundle.video_id,
                "fps": bundle.fps,
                "duration_seconds": bundle.duration_seconds,
                "detection": detection_metrics,
                "tracking": tracking_metrics,
                "events": event_metrics,
            }
        )

        for label, counts in detection_metrics["counts_by_class"].items():
            aggregate_detection[label]["tp"] += counts["tp"]
            aggregate_detection[label]["fp"] += counts["fp"]
            aggregate_detection[label]["fn"] += counts["fn"]

        for key in aggregate_tracking:
            aggregate_tracking[key] += tracking_metrics["counts"][key]

        for event_type, counts in event_metrics["counts_by_type"].items():
            aggregate_event[event_type]["tp"] += counts["tp"]
            aggregate_event[event_type]["fp"] += counts["fp"]
            aggregate_event[event_type]["fn"] += counts["fn"]
            aggregate_event[event_type]["latencies"].extend(counts["latencies"])
            aggregate_event[event_type]["latency_errors"].extend(counts["latency_errors"])

    detection_summary = {
        label: _counts_to_metrics(counts["tp"], counts["fp"], counts["fn"]) | counts
        for label, counts in sorted(aggregate_detection.items())
    }
    tracking_summary = _tracking_counts_to_metrics(aggregate_tracking)
    event_summary = {
        event_type: _event_counts_to_metrics(counts, total_duration_seconds)
        for event_type, counts in sorted(aggregate_event.items())
    }

    overall_event_counts = {
        "tp": sum(item["tp"] for item in aggregate_event.values()),
        "fp": sum(item["fp"] for item in aggregate_event.values()),
        "fn": sum(item["fn"] for item in aggregate_event.values()),
        "latencies": [latency for item in aggregate_event.values() for latency in item["latencies"]],
        "latency_errors": [error for item in aggregate_event.values() for error in item["latency_errors"]],
    }

    summary = {
        "num_videos": len(bundles),
        "total_duration_seconds": round(total_duration_seconds, 3),
        "detection_by_class": detection_summary,
        "tracking": tracking_summary,
        "events_by_type": event_summary,
        "events_overall": _event_counts_to_metrics(overall_event_counts, total_duration_seconds),
    }
    return {"summary": summary, "videos": per_video}


def evaluate_detections(
    gt_detections: list[DetectionRecord],
    pred_detections: list[DetectionRecord],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = _group_by_frame(gt_detections)
    pred_by_frame = _group_by_frame(pred_detections)
    counts_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        matches, unmatched_gt, unmatched_pred = match_detections(
            gt_by_frame.get(frame_index, []),
            pred_by_frame.get(frame_index, []),
            iou_threshold=iou_threshold,
        )
        for gt_item, _ in matches:
            counts_by_class[gt_item.label]["tp"] += 1
        for gt_item in unmatched_gt:
            counts_by_class[gt_item.label]["fn"] += 1
        for pred_item in unmatched_pred:
            counts_by_class[pred_item.label]["fp"] += 1

    metrics_by_class = {
        label: _counts_to_metrics(counts["tp"], counts["fp"], counts["fn"]) | counts
        for label, counts in sorted(counts_by_class.items())
    }
    return {"counts_by_class": metrics_by_class}


def evaluate_tracking(
    gt_detections: list[DetectionRecord],
    pred_detections: list[DetectionRecord],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame = _group_by_frame(gt_detections)
    pred_by_frame = _group_by_frame(pred_detections)
    last_pred_for_gt: dict[int, int] = {}
    counts = {"gt_count": len(gt_detections), "pred_count": len(pred_detections), "matched_count": 0, "fp": 0, "fn": 0, "id_switches": 0}

    for frame_index in sorted(set(gt_by_frame) | set(pred_by_frame)):
        matches, unmatched_gt, unmatched_pred = match_detections(
            gt_by_frame.get(frame_index, []),
            pred_by_frame.get(frame_index, []),
            iou_threshold=iou_threshold,
        )
        counts["matched_count"] += len(matches)
        counts["fn"] += len(unmatched_gt)
        counts["fp"] += len(unmatched_pred)

        for gt_item, pred_item in matches:
            previous_pred_id = last_pred_for_gt.get(gt_item.track_id)
            if previous_pred_id is not None and previous_pred_id != pred_item.track_id:
                counts["id_switches"] += 1
            last_pred_for_gt[gt_item.track_id] = pred_item.track_id

    return {"counts": counts, "metrics": _tracking_counts_to_metrics(counts)}


def evaluate_events(
    gt_events: list[EventRecord],
    pred_events: list[EventRecord],
    fps: float,
    duration_seconds: float,
) -> dict[str, Any]:
    grouped_gt: dict[str, list[EventRecord]] = defaultdict(list)
    grouped_pred: dict[str, list[EventRecord]] = defaultdict(list)

    for event in gt_events:
        grouped_gt[event.event_type].append(event)
    for event in pred_events:
        grouped_pred[event.event_type].append(event)

    counts_by_type: dict[str, dict[str, Any]] = {}
    for event_type in sorted(set(grouped_gt) | set(grouped_pred)):
        matches, unmatched_gt, unmatched_pred = match_events(grouped_gt.get(event_type, []), grouped_pred.get(event_type, []))
        latencies = [round((pred.frame_index - gt.entry_frame_index) / fps, 4) for gt, pred in matches]
        latency_errors = [round((pred.frame_index - gt.frame_index) / fps, 4) for gt, pred in matches]
        counts = {
            "tp": len(matches),
            "fp": len(unmatched_pred),
            "fn": len(unmatched_gt),
            "latencies": latencies,
            "latency_errors": latency_errors,
        }
        counts_by_type[event_type] = _event_counts_to_metrics(counts, duration_seconds) | counts

    return {"counts_by_type": counts_by_type}


def match_detections(
    gt_items: list[DetectionRecord],
    pred_items: list[DetectionRecord],
    iou_threshold: float,
) -> tuple[list[tuple[DetectionRecord, DetectionRecord]], list[DetectionRecord], list[DetectionRecord]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_item in enumerate(gt_items):
        for pred_idx, pred_item in enumerate(pred_items):
            if gt_item.label != pred_item.label:
                continue
            iou_value = iou(gt_item.bbox, pred_item.bbox)
            if iou_value >= iou_threshold:
                candidates.append((iou_value, gt_idx, pred_idx))

    matches: list[tuple[DetectionRecord, DetectionRecord]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for _, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        matches.append((gt_items[gt_idx], pred_items[pred_idx]))
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    unmatched_gt = [item for idx, item in enumerate(gt_items) if idx not in used_gt]
    unmatched_pred = [item for idx, item in enumerate(pred_items) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def match_events(
    gt_events: list[EventRecord],
    pred_events: list[EventRecord],
) -> tuple[list[tuple[EventRecord, EventRecord]], list[EventRecord], list[EventRecord]]:
    candidates: list[tuple[int, int, int]] = []
    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(pred_events):
            if gt_event.zone != pred_event.zone or gt_event.label != pred_event.label:
                continue
            frame_gap = abs(pred_event.frame_index - gt_event.frame_index)
            tolerance = gt_event.match_tolerance_frames
            if frame_gap <= tolerance:
                candidates.append((frame_gap, gt_idx, pred_idx))

    matches: list[tuple[EventRecord, EventRecord]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for frame_gap, gt_idx, pred_idx in sorted(candidates, key=lambda item: item[0]):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        matches.append((gt_events[gt_idx], pred_events[pred_idx]))
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    unmatched_gt = [item for idx, item in enumerate(gt_events) if idx not in used_gt]
    unmatched_pred = [item for idx, item in enumerate(pred_events) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def render_markdown(results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# Evaluation Results",
        "",
        "This file is generated by `python scripts/evaluate_events.py --output-markdown docs/results.md`.",
        "",
        f"- Videos: {summary['num_videos']}",
        f"- Total benchmark duration: {summary['total_duration_seconds']:.2f}s",
        "",
        "## Detection",
        "",
        "| Class | Precision | Recall | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in summary["detection_by_class"].items():
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
        )

    tracking = summary["tracking"]
    lines.extend(
        [
            "",
            "## Tracking",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| MOTA | {tracking['mota']:.3f} |",
            f"| ID switches | {tracking['id_switches']} |",
            f"| Matched detections | {tracking['matched_count']} |",
            f"| FP | {tracking['fp']} |",
            f"| FN | {tracking['fn']} |",
            "",
            "## Events",
            "",
            "| Event | Precision | Recall | False Alerts/Min | Mean Alert Latency (s) | Mean Alert Timing Error (s) | TP | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for event_type, metrics in summary["events_by_type"].items():
        lines.append(
            f"| {event_type} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['false_alerts_per_minute']:.3f} | {metrics['mean_alert_latency_seconds']:.3f} | {metrics['mean_alert_timing_error_seconds']:.3f} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
        )

    overall = summary["events_overall"]
    lines.extend(
        [
            "",
            "## Overall Event Summary",
            "",
            f"- Precision: {overall['precision']:.3f}",
            f"- Recall: {overall['recall']:.3f}",
            f"- False alerts per minute: {overall['false_alerts_per_minute']:.3f}",
            f"- Mean alert latency from entry: {overall['mean_alert_latency_seconds']:.3f}s",
            f"- Mean alert timing error vs expected event frame: {overall['mean_alert_timing_error_seconds']:.3f}s",
        ]
    )
    return "\n".join(lines) + "\n"


def _group_by_frame(items: list[DetectionRecord]) -> dict[int, list[DetectionRecord]]:
    grouped: dict[int, list[DetectionRecord]] = defaultdict(list)
    for item in items:
        grouped[item.frame_index].append(item)
    return grouped


def _counts_to_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4)}


def _tracking_counts_to_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    gt_count = counts["gt_count"]
    mota = 1.0 - ((counts["fn"] + counts["fp"] + counts["id_switches"]) / gt_count) if gt_count else 0.0
    return {
        "mota": round(mota, 4),
        "id_switches": counts["id_switches"],
        "matched_count": counts["matched_count"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "gt_count": gt_count,
        "pred_count": counts["pred_count"],
    }


def _event_counts_to_metrics(counts: dict[str, Any], duration_seconds: float) -> dict[str, float | int]:
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    false_alerts_per_minute = counts["fp"] / (duration_seconds / 60.0) if duration_seconds else 0.0
    mean_latency = sum(counts["latencies"]) / len(counts["latencies"]) if counts["latencies"] else 0.0
    mean_latency_error = (
        sum(abs(value) for value in counts["latency_errors"]) / len(counts["latency_errors"])
        if counts["latency_errors"]
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_alerts_per_minute": round(false_alerts_per_minute, 4),
        "mean_alert_latency_seconds": round(mean_latency, 4),
        "mean_alert_timing_error_seconds": round(mean_latency_error, 4),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
    }


def _load_video_bundle(base_dir: Path, video_spec: dict[str, Any]) -> VideoBundle:
    gt_payload = _load_json(base_dir / video_spec["ground_truth"])
    pred_payload = _load_json(base_dir / video_spec["predictions"])
    fps = float(video_spec.get("fps", gt_payload.get("fps", pred_payload.get("fps", 30.0))))
    duration_seconds = float(
        video_spec.get("duration_seconds", gt_payload.get("duration_seconds", pred_payload.get("duration_seconds", 0.0)))
    )
    return VideoBundle(
        video_id=str(video_spec["video_id"]),
        fps=fps,
        duration_seconds=duration_seconds,
        gt_detections=_load_detections(gt_payload.get("detections", [])),
        pred_detections=_load_detections(pred_payload.get("detections", [])),
        gt_events=_load_events(gt_payload.get("events", [])),
        pred_events=_load_events(pred_payload.get("events", [])),
    )


def _load_detections(items: list[dict[str, Any]]) -> list[DetectionRecord]:
    return [
        DetectionRecord(
            frame_index=int(item["frame_index"]),
            track_id=int(item["track_id"]),
            label=str(item["class"]),
            bbox=tuple(float(value) for value in item["bbox"]),
            score=float(item["score"]) if "score" in item else None,
        )
        for item in items
    ]


def _load_events(items: list[dict[str, Any]]) -> list[EventRecord]:
    return [
        EventRecord(
            event_type=str(item["event_type"]),
            zone=str(item["zone"]),
            label=str(item["class"]),
            frame_index=int(item["frame_index"]),
            entry_frame_index=int(item.get("entry_frame_index", item["frame_index"])),
            track_id=int(item["track_id"]) if "track_id" in item and item["track_id"] is not None else None,
            confidence=float(item["confidence"]) if "confidence" in item else None,
            match_tolerance_frames=int(item.get("match_tolerance_frames", 15)),
        )
        for item in items
    ]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    main()
