# Sentinel Vision

Sentinel Vision is a real-time safety analytics pipeline for webcam feeds, video files, or RTSP streams. It detects people and vehicles, tracks them across frames, evaluates safety events, saves operator-friendly alert evidence, and exposes a persistent alert backend with filtering and dashboard views.

## Overview

- Input: webcam, local video file, or RTSP stream
- Detection: YOLO-based people and vehicle detection
- Tracking: ByteTrack by default, with an appearance-aware BoT-SORT-style option
- Event logic: intrusion, loitering, line crossing, wrong-way, after-hours occupancy, vehicle zone violations, abandoned-object detection
- Output: annotated video, snapshots, pre/post alert clips, JSON events, SQLite alert history
- Backend: FastAPI alert API plus Streamlit dashboard

## Architecture

![Architecture Diagram](docs/architecture.svg)

More detail lives in [docs/architecture.md](docs/architecture.md).

## Features

- YOLO-backed object detection via Ultralytics
- ByteTrack-based multi-object tracking for stronger ID continuity
- Optional appearance-aware BoT-SORT-style tracking for harder crossings and re-identification
- Optional feature-based camera motion compensation with affine or homography estimation
- Optional image-to-ground-plane homography for normalized zone reasoning and occupancy estimates
- Polygon and line zone definitions with tags and metadata
- Multi-event safety analytics for restricted areas, tripwires, motion direction, occupancy policy, and unattended asset detection
- Snapshot plus pre/post event alert clip generation with metadata sidecars
- SQLite-backed alert API with filterable history and stats
- Streamlit dashboard for operator triage
- Multi-camera runtime with per-camera outputs and a central alert service
- Benchmark/evaluation pipeline for detection, tracking, and event quality
- HTML metrics dashboard report for evaluation results
- Docker packaging for API, dashboard, and pipeline services
- Camera health status snapshots with reconnect/degradation tracking

## Demo

- Checked-in demo clip: [office_intrusion_short.mp4](data/eval/videos/office_intrusion_short.mp4)
- Additional checked-in benchmark clips live under [data/eval/videos](data/eval/videos)
- Evaluation bundle and reference clips: [data/eval/README.md](data/eval/README.md)

Demo assets that are committed to the repo live in `data/eval/videos/`. There is no separate `data/samples/` bundle checked in.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For linting, tests, and local CI-style checks:

```bash
pip install -r requirements-dev.txt
```

## Run

Run against a local video file:

```bash
python -m src.main --config configs/default.yaml --source data/eval/videos/office_intrusion_short.mp4
```

Use a webcam:

```bash
python -m src.main --config configs/default.yaml --source 0
```

Use RTSP:

```bash
python -m src.main --config configs/default.yaml --source rtsp://user:pass@camera/stream
```

Run multiple cameras from one config:

```bash
python -m src.main --config configs/multi_camera.yaml
```

Run the API:

```bash
uvicorn src.api.app:app --reload
```

Run the dashboard:

```bash
streamlit run src/api/dashboard.py
```

Run the stack with Docker Compose:

```bash
docker compose up --build
```

Use a deployment profile and explicit device/model overrides:

```bash
python -m src.main --config configs/default.yaml --profile edge_cpu --device cpu --model yolo11n.pt
```

## Example Alert

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
  "snapshot_path": "data/outputs/alerts/20260308T184231Z_intrusion_restricted_lab_track7_evt_000001.jpg",
  "clip_path": "data/outputs/alerts/20260308T184231Z_intrusion_restricted_lab_track7_evt_000001.mp4",
  "metadata_path": "data/outputs/alerts/20260308T184231Z_intrusion_restricted_lab_track7_evt_000001.json"
}
```

## Config

The default YAML config is at [configs/default.yaml](configs/default.yaml).

- Tracking backends: `bytetrack`, `botsort`, `simple`
- Zones can be `polygon` or `line`
- Zone `tags` and `metadata` route zones to specific event logic
- `abandoned_object` zones detect stationary assets whose nearest owner has moved beyond a configurable distance for a configurable dwell time
- `perspective.image_points` and `perspective.world_points` enable ground-plane projection for normalized-space intrusion/loitering reasoning
- Event payloads include `world_position` and zone occupancy stats when perspective calibration is configured
- Alert output settings include `buffer_seconds`, `post_event_seconds`, and `duplicate_suppression_seconds`
- Recorder writes clips on a background queue, guards invalid FPS, and records codec fallback status in metadata
- Runtime controls include RTSP reconnect attempts, frame skip policy, inference resize, and per-stage timing logs
- Motion compensation can be toggled per camera with `runtime.motion_compensation`, including a `static_camera_assumption` switch and `affine`/`homography` modes
- Deployment profiles under `configs/profiles/` can tune runtime/model/output settings for edge CPU, edge GPU, or low-latency use cases
- `--device` and `--model` CLI flags override inference backend and model selection without editing YAML
- Multi-camera configs can define a top-level `cameras` list; each camera gets its own event log, health file, annotated output, and alert clip directory automatically
- The API and dashboard use a persistent SQLite DB at `data/outputs/alerts.db` by default
- Supported detector classes now include `backpack`, `handbag`, and `suitcase` for unattended-object workflows

Config loading now uses schema validation with explicit checks for malformed zones, invalid thresholds, unsupported classes, and invalid dashboard URLs.

## API

Minimum operator API:

- `GET /health`
- `GET /cameras`
- `GET /alerts?camera_id=...&event_type=...&zone=...&start_time=...&end_time=...`
- `GET /alerts/{event_id}`
- `GET /stats`
- `POST /ingest`

## Benchmark Results

Reproducibility details live in [docs/reproducibility.md](docs/reproducibility.md).

Generate predictions and score the benchmark in one pass:

```bash
python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest.json --config configs/default.yaml --device cpu --output-json data/eval/results/latest.json --output-markdown docs/results.md
```

Or score an existing set of prediction JSON files:

```bash
python -m scripts.evaluate_events --manifest data/eval/benchmark_manifest.json --output-json data/eval/results/latest.json --output-markdown docs/results.md
```

Current benchmark summary is in [docs/results.md](docs/results.md).
Visual metrics dashboard report: [docs/results_dashboard.html](docs/results_dashboard.html)

For the checked-in `docs/results.md` artifact:

- Reproduction path: score-only from checked-in prediction JSON files
- Manifest: `data/eval/benchmark_manifest.json`
- Config used for live benchmark runs: `configs/default.yaml`
- Default model for live benchmark runs: `yolo11n.pt`
- Hardware: not applicable for the checked-in score-only artifact because no live inference was run

- Detection `person`: precision `1.000`, recall `0.944`
- Tracking: MOTA `0.833`, MOTP `1.000`, IDF1 `0.778`, ID switches `2`
- Events overall: precision `0.667`, recall `1.000`, false alerts `2.308/min`

The benchmark manifest now supports per-clip `scene_types`, `challenge_tags`, `subject_classes`, clip-specific `config_override`, and optional runtime payloads so CPU and GPU passes can be compared directly.

Exact reproduction commands:

```bash
python -m scripts.evaluate_events --manifest data/eval/benchmark_manifest.json --output-json data/eval/results/latest.json --output-markdown docs/results.md
python -m scripts.render_metrics_report --input-json data/eval/results/latest.json --output-html docs/results_dashboard.html
```

## Public Dataset Subset

The repo now includes a curated public-dataset tracking subset built from:

- `MOT17-04-FRCNN`
- `MOT17-09-FRCNN`
- `VisDrone2019-MOT-train/uav0000099_02109_v`
- `VisDrone2019-MOT-val/uav0000268_05773_v`

Artifacts:

- Manifest: [benchmark_manifest_public_datasets.json](data/eval/benchmark_manifest_public_datasets.json)
- CPU results: [public_dataset_cpu.json](data/eval/results/public_dataset_cpu.json)
- CPU report: [results_public_dataset_cpu.md](docs/results_public_dataset_cpu.md)
- CPU dashboard: [results_public_dataset_cpu.html](docs/results_public_dataset_cpu.html)

Important distinction:

- Tracking metrics on this subset are real dataset-derived results.
- Event metrics are currently false-alert-only stress numbers because MOT17 and VisDrone do not ship event-level ground truth for this repo's rule-based events.

Original datasets are not included in Git. Raw extractions live under `data/external/`, which is ignored by the repo.

## Camera Health

The pipeline writes camera health status to the configured `output.health_status_path`. In the default single-camera config that is `data/outputs/camera_health.json`; in multi-camera mode each camera gets its own namespaced health file automatically.

Health status includes:

- online/degraded/offline state
- read failure count
- detector failure count
- reconnect attempts
- last successful frame timestamp

## Testing

```bash
python -m pytest -q
```

Local lint and format checks:

```bash
ruff check src tests scripts
black --check src tests scripts
```

## CI

GitHub Actions runs:

- `ruff check src tests scripts`
- `black --check src tests scripts`
- `pytest -q`
- an offline smoke path that evaluates the bundled benchmark, renders the HTML report, and checks `python -m scripts.run_benchmark --help`

## Deployment

Deployment notes for Docker, Compose profiles, CPU/GPU selection, config profiles, and `systemd` startup are in [docs/deployment.md](docs/deployment.md).

## Known Limitations

- The bundled asset set is still a starter benchmark; for a credible external evaluation pass you should expand it to roughly 8 to 15 labeled clips with occlusion, false-positive traps, and both person and vehicle footage.
- The BoT-SORT-style tracker uses lightweight appearance features, not a full learned re-identification model.
- The Streamlit dashboard is intentionally minimal and optimized for speed of implementation over UI polish.
- Event logic is rule-based and depends on well-configured zones and camera placement.
- Motion compensation is intended for mild camera jitter, not large PTZ moves or scene cuts.
- Ground-plane occupancy is approximate and depends on usable camera calibration and reasonably planar scenes.
- Multi-camera mode centralizes alerts and per-camera outputs, but it does not yet do cross-camera re-identification.

## Roadmap

- Expand the benchmark asset pack toward MOT17, VIRAT, or VisDrone-backed evaluation clips
- Add stronger learned re-identification for BoT-SORT-style tracking
