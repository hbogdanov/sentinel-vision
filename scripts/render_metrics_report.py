from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sentinel Vision Metrics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f4efe6; color:#1f1b18; margin:0; padding:32px; }}
    .wrap {{ max-width:1100px; margin:0 auto; }}
    h1,h2 {{ margin:0 0 12px; }}
    .hero {{ background:#fffaf2; border:2px solid #2a241f; border-radius:18px; padding:24px; margin-bottom:24px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin:20px 0; }}
    .card {{ background:#fff; border-radius:16px; padding:16px; border:1px solid #d7c8b5; }}
    .label {{ font-size:12px; text-transform:uppercase; color:#7b6c5a; }}
    .value {{ font-size:28px; font-weight:700; margin-top:6px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:14px; overflow:hidden; }}
    th,td {{ padding:12px 14px; border-bottom:1px solid #ece3d8; text-align:left; }}
    th {{ background:#f7efe3; }}
    .section {{ margin-top:28px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Sentinel Vision Metrics Report</h1>
      <p>Generated from evaluation JSON for demo, review, and portfolio walkthroughs.</p>
      <div class="grid">
        <div class="card"><div class="label">Videos</div><div class="value">{num_videos}</div></div>
        <div class="card"><div class="label">Duration (s)</div><div class="value">{duration_seconds:.1f}</div></div>
        <div class="card"><div class="label">MOTA</div><div class="value">{mota:.3f}</div></div>
        <div class="card"><div class="label">IDF1</div><div class="value">{idf1:.3f}</div></div>
        <div class="card"><div class="label">Event Precision</div><div class="value">{event_precision:.3f}</div></div>
        <div class="card"><div class="label">Event Recall</div><div class="value">{event_recall:.3f}</div></div>
      </div>
    </div>

    <div class="section">
      <h2>Detection</h2>
      {detection_table}
    </div>

    <div class="section">
      <h2>Tracking</h2>
      {tracking_table}
    </div>

    <div class="section">
      <h2>Events</h2>
      {events_table}
    </div>

    <div class="section">
      <h2>Runtime</h2>
      {runtime_table}
    </div>

    <div class="section">
      <h2>Per Video</h2>
      {videos_table}
    </div>
  </div>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render evaluation JSON into an HTML dashboard report.")
    parser.add_argument("--input-json", default="data/eval/results/latest.json")
    parser.add_argument("--output-html", default="docs/results_dashboard.html")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    summary = payload["summary"]
    html = HTML_TEMPLATE.format(
        num_videos=summary["num_videos"],
        duration_seconds=summary["total_duration_seconds"],
        mota=summary["tracking"]["mota"],
        idf1=summary["tracking"]["idf1"],
        event_precision=summary["events_overall"]["precision"],
        event_recall=summary["events_overall"]["recall"],
        detection_table=_detection_table(summary["detection_by_class"]),
        tracking_table=_tracking_table(summary["tracking"]),
        events_table=_events_table(summary["events_by_type"]),
        runtime_table=_runtime_table(summary.get("runtime_by_device", {})),
        videos_table=_videos_table(payload["videos"]),
    )
    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(str(output_path))


def _detection_table(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td>{label}</td><td>{item['precision']:.3f}</td><td>{item['recall']:.3f}</td><td>{item['tp']}</td><td>{item['fp']}</td><td>{item['fn']}</td></tr>"
        for label, item in metrics.items()
    )
    return f"<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th><th>FN</th></tr>{rows}</table>"


def _tracking_table(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td>{label}</td><td>{value:.3f}</td></tr>"
        for label, value in [
            ("MOTA", metrics["mota"]),
            ("MOTP", metrics["motp"]),
            ("IDF1", metrics["idf1"]),
            ("Precision", metrics["precision"]),
            ("Recall", metrics["recall"]),
        ]
    )
    return f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def _events_table(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td>{label}</td><td>{item['precision']:.3f}</td><td>{item['recall']:.3f}</td><td>{item['false_alerts_per_minute']:.3f}</td><td>{item['mean_alert_latency_seconds']:.3f}</td></tr>"
        for label, item in metrics.items()
    )
    return f"<table><tr><th>Event</th><th>Precision</th><th>Recall</th><th>False Alerts/Min</th><th>Mean Latency (s)</th></tr>{rows}</table>"


def _runtime_table(metrics: dict) -> str:
    if not metrics:
        return "<p>No runtime samples were present in the evaluation payload.</p>"
    rows = "".join(
        f"<tr><td>{device}</td><td>{item['videos']}</td><td>{item['frames_processed']}</td><td>{item['wall_clock_seconds']:.3f}</td><td>{item['effective_fps']:.3f}</td></tr>"
        for device, item in metrics.items()
    )
    return f"<table><tr><th>Device</th><th>Videos</th><th>Frames</th><th>Wall Clock (s)</th><th>Effective FPS</th></tr>{rows}</table>"


def _videos_table(videos: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{video['video_id']}</td><td>{video['duration_seconds']:.1f}</td><td>{video['tracking']['metrics']['mota']:.3f}</td><td>{video['tracking']['metrics']['idf1']:.3f}</td><td>{', '.join(video['events']['counts_by_type'].keys()) or 'none'}</td></tr>"
        for video in videos
    )
    return f"<table><tr><th>Video</th><th>Duration (s)</th><th>MOTA</th><th>IDF1</th><th>Event Types</th></tr>{rows}</table>"


if __name__ == "__main__":
    main()
