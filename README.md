# Sentinel Vision

Sentinel Vision is a real-time safety, anomaly, and intrusion detection pipeline for webcam feeds, video files, or RTSP streams. It detects people and vehicles, overlays tracks, evaluates event logic such as restricted-zone entry and loitering, and saves structured alerts for downstream dashboards.

## Overview

Sentinel Vision processes live or recorded video streams and produces annotated outputs plus structured safety events:

- Input: webcam, local video file, or RTSP stream
- Detection: people, vehicles, restricted-zone entry, unusual motion hooks
- Tracking: stable IDs across frames
- Event logic: intrusion, loitering, cooldown suppression
- Output: annotated video, snapshots, alert clips, JSON event logs
- Deployment story: optional FastAPI dashboard ingestion

## Features

- YOLO-backed object detection via Ultralytics
- Lightweight multi-object tracking with stable IDs
- Polygon zone definitions from YAML config
- Intrusion alerts when a tracked subject enters a restricted region
- Loitering alerts when a subject remains inside a region for a threshold duration
- Structured JSONL event logging
- Snapshot and alert clip generation
- Optional event forwarding to a lightweight dashboard endpoint

## Architecture

```text
sentinel-vision/
├── configs/
├── data/
├── demo/
├── docs/
├── models/
├── src/
│   ├── main.py
│   ├── inference/
│   ├── events/
│   ├── io/
│   ├── utils/
│   └── api/
└── tests/
```

More detail lives in [docs/architecture.md](/c:/Users/Ivan/sentinel-vision/docs/architecture.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run against a local video file:

```bash
python -m src.main --config configs/default.yaml --source data/samples/demo.mp4
```

Use a webcam:

```bash
python -m src.main --config configs/default.yaml --source 0
```

Use RTSP:

```bash
python -m src.main --config configs/default.yaml --source rtsp://user:pass@camera/stream
```

## Example output event

```json
{
  "event_id": "evt_000001",
  "timestamp": "2026-03-08T18:42:31Z",
  "camera_id": "office_cam_1",
  "event_type": "intrusion",
  "track_id": 7,
  "class": "person",
  "confidence": 0.91,
  "zone": "restricted_lab",
  "frame_index": 3812,
  "snapshot_path": "data/outputs/alerts/evt_000001.jpg",
  "clip_path": "data/outputs/alerts/evt_000001.mp4"
}
```

## Config

The default YAML config shows the expected fields for sources, classes, thresholds, zones, logging, and optional dashboard dispatch. See [configs/default.yaml](/c:/Users/Ivan/sentinel-vision/configs/default.yaml).

## Testing

```bash
python -m pytest -q
```

## Demo checklist

- Person starts outside the restricted zone
- Box and track ID are visible
- Person crosses the polygon boundary
- Intrusion banner appears
- Snapshot and clip are saved
- JSON event is appended to disk

## Roadmap

- Replace the lightweight tracker with ByteTrack or DeepSORT
- Add abandoned-object logic
- Add RTSP reconnection policy and camera health checks
- Add a Streamlit or React operator dashboard
- Package the service with Docker for edge deployment
