# Sentinel Vision

Sentinel Vision is a real-time safety analytics pipeline for webcam feeds, video files, or RTSP streams. It detects people and vehicles, tracks them across frames, evaluates safety events, saves operator-friendly alert evidence, and exposes a persistent alert backend with filtering and dashboard views.

## Overview

- Input: webcam, local video file, or RTSP stream
- Detection: YOLO-based people and vehicle detection
- Tracking: ByteTrack by default, with an optional BoT-SORT-style learned re-ID path for harder occlusion cases
- Event logic: intrusion, loitering, line crossing, wrong-way, after-hours occupancy, vehicle zone violations, abandoned-object detection
- Output: annotated video, snapshots, pre/post alert clips, JSON events, SQLite alert history
- Backend: FastAPI alert API plus Streamlit dashboard

## Results At A Glance

| Evaluation slice | MOTA | IDF1 | ID switches | Effective FPS | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Bundled micro-benchmark | 0.833 | 0.778 | 2 | n/a | Checked-in score-only artifact from bundled reference predictions |
| Public-dataset CPU baseline | 0.228 | 0.347 | 89 | 19.869 | `edge_cpu` stress-test baseline on MOT17 + VisDrone subset |
| Public-dataset tuned tracking pass | 0.247 | 0.369 | 89 | 19.159 | `benchmark_association_tuned` profile with stricter association |

Current checked-in public-dataset results are a CPU baseline stress test used to guide tracker and event-logic tuning, not a final tuned deployment profile.

## Demo

![Sentinel Vision Demo Montage](demo/screenshots/demo_montage.gif)

| Annotated frame | Benchmark snapshot |
| --- | --- |
| ![Annotated Sentinel Vision frame](demo/screenshots/annotated_demo_frame.jpg) | ![Public benchmark snapshot](demo/screenshots/public_benchmark_snapshot.jpg) |

## Architecture

![Architecture Diagram](docs/architecture.svg)

More detail lives in [docs/architecture.md](docs/architecture.md).

## Design Notes

- Detector, tracker, event logic, rendering, and persistence are split into separate modules so the runtime can be tuned or swapped without rewriting the whole pipeline.
- Config-driven profiles under `configs/profiles/` make it easy to compare low-latency, edge CPU, crowded-scene, and benchmark-oriented settings on the same code path.
- Benchmarking is wired into the repo with MOT17/VisDrone subset manifests, JSON result artifacts, markdown reports, and reproducible commands instead of one-off notebook screenshots.
- The backend is intentionally simple: FastAPI handles persistence and filtering, while Streamlit provides operator-facing triage on top of the same alert store.

## Features

- YOLO-backed object detection via Ultralytics
- ByteTrack-based multi-object tracking for stronger ID continuity
- Optional BoT-SORT-style tracking with lightweight learned re-ID embeddings for harder crossings, occlusion recovery, and lower ID-switch rates
- Configurable trajectory prediction that smooths recent motion for short-occlusion association and renders forecast paths on annotated video
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

- Checked-in demo clip: [office_intrusion_short.mp4](data/eval/videos/office_intrusion_short.mp4)
- Additional checked-in benchmark clips live under [data/eval/videos](data/eval/videos)
- Evaluation bundle and reference clips: [data/eval/README.md](data/eval/README.md)
- Generated README visuals live under [demo/screenshots](demo/screenshots)

Demo assets that are committed to the repo live in `data/eval/videos/` and `demo/screenshots/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For linting, tests, and local CI-style checks:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run repo commands with the project virtual environment to avoid Windows PATH conflicts between system Python, Conda, and `.venv`:

```bash
.\.venv\Scripts\python.exe -m pytest -q
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

Use the learned-embedding tracking profile for crowded or occlusion-heavy scenes:

```bash
python -m src.main --config configs/default.yaml --profile crowded_tracking --source data/eval/videos/mot17_04_clip.mp4
```

## Alert Payload

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
- `configs/profiles/crowded_tracking.yaml` switches tracking to BoT-SORT with a lightweight MobileNetV3 appearance encoder for harder multi-object tracking cases
- `configs/profiles/benchmark_association_tuned.yaml` applies stricter ByteTrack association and threshold tuning for the checked-in MOT17/VisDrone public benchmark subset
- `--device` and `--model` CLI flags override inference backend and model selection without editing YAML
- Multi-camera configs can define a top-level `cameras` list; each camera gets its own event log, health file, annotated output, and alert clip directory automatically
- `tracking.trajectory_*` tunes the tracker-side motion predictor used for short-gap association and predicted path overlays
- `output.zone_heatmap` can render and persist a zone occupancy heatmap PNG plus JSON summary for demos and traffic-pattern analysis
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

For quicker local smoke runs on larger clips, you can subsample the benchmark:

```bash
python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest_public_datasets.json --config configs/default.yaml --profile edge_cpu --device cpu --frame-skip 2 --max-frames 200 --output-json data/eval/results/quickcheck.json --output-markdown docs/results_quickcheck.md
```

Use `--frame-skip` and `--max-frames` for iteration speed only. Full benchmark numbers in the README and reports should come from unskipped runs.

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

Profile comparison snapshot:

| Profile | MOTA | IDF1 | ID switches | Effective FPS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `edge_cpu` / `public_dataset_cpu` | 0.228 | 0.347 | 89 | 19.869 | Checked-in CPU baseline |
| `benchmark_association_tuned` | 0.247 | 0.369 | 89 | 19.159 | Global assignment + stricter association thresholds |

Tracking baseline context:

- These public-dataset numbers come from the `edge_cpu` profile on CPU with `yolo11n.pt` and the repo's default ByteTrack-style tracker.
- The subset mixes crowded MOT17 street scenes with moving-camera VisDrone aerial scenes and does not use dataset-specific detector or tracker tuning.
- Read the public-dataset results as a baseline stress test, not as a tuned SOTA tracking claim.
- For harder occlusion cases, switch to `--profile crowded_tracking` to enable the learned-embedding BoT-SORT path. That mode is intended to improve IDF1 and reduce ID switches rather than maximize raw throughput.

Current public-dataset CPU baseline:

- MOTA `0.228`
- IDF1 `0.347`
- ID switches `89`
- Effective CPU runtime `19.869 FPS`

Checked-in tuned public-dataset profile:

- Profile: `benchmark_association_tuned`
- Result artifact: [public_dataset_benchmark_association_tuned.json](data/eval/results/public_dataset_benchmark_association_tuned.json)
- MOTA `0.247`
- IDF1 `0.369`
- ID switches `89`
- Effective CPU runtime `19.159 FPS`

The checked-in VisDrone clip `visdrone_uav0000268_05773_clip.mp4` was re-encoded with H.264 to keep the repo lighter. If you re-encode benchmark videos, rerun `python -m scripts.run_benchmark` and regenerate the reports so predictions and published metrics stay aligned.
The repeatable compression workflow is documented in `data/eval/README.md` and implemented by `python -m scripts.compress_benchmark_videos`.

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
.\.venv\Scripts\python.exe -m pytest -q
```

Local lint and format checks:

```bash
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m black --check src tests scripts
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

- Add stronger learned re-identification for BoT-SORT-style tracking
- Expand the checked-in benchmark beyond the current MOT17 and VisDrone subset with more labeled clips, harder occlusions, and richer event-ground-truth coverage
