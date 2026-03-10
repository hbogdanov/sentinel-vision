# Sentinel Vision Architecture

## Data flow

1. `src.main` loads YAML configuration and expands it into one or more camera pipelines.
2. `src.inference.detector` runs object detection on each frame.
3. `src.inference.motion` estimates optional global camera motion for jitter compensation.
4. `src.inference.ground_plane` projects image coordinates into a calibrated ground plane when homography data is configured.
5. `src.inference.tracker` assigns stable IDs to detections across frames using a configurable backend.
6. `src.events` evaluates spatial and temporal event logic against configured polygon and line zones in image space or normalized ground-plane space.
7. `src.utils.draw` overlays boxes, track IDs, polygon zones, tripwires, and active alert banners.
8. `src.io.logger` writes structured JSON events, `src.io.recorder` saves snapshots/pre-post clips, and `src.io.health` persists camera status.
9. `src.api.app` persists alerts to SQLite and exposes filtered history, camera summaries, detail, health, and stats endpoints.
10. `src.api.dashboard` provides a Streamlit operator dashboard on top of the same database.
11. `scripts/render_metrics_report.py` turns evaluation JSON into an HTML metrics dashboard report.

## Modules

- `src/inference/detector.py`: detector interface and Ultralytics-backed implementation
- `src/inference/tracker.py`: tracker abstractions plus `ByteTracker`, `BoTSORTTracker`, and the original `SimpleTracker`
- `src/inference/ground_plane.py`: image-to-world homography utilities for perspective-aware zone reasoning
- `src/inference/motion.py`: feature-based global motion compensation for mild camera jitter
- `src/inference/multi_camera.py`: concurrent orchestration for multiple camera configs
- `src/inference/pipeline.py`: orchestration of decode, inference, event logic, rendering, and outputs
- `src/events/zones.py`: polygon utilities and image/world-space point-in-zone checks
- `src/events/intrusion.py`: zone-entry event logic with cooldown suppression
- `src/events/loitering.py`: dwell-time logic for tracked subjects
- `src/events/line_crossing.py`: direction-aware tripwire logic
- `src/events/wrong_way.py`: motion direction checks inside configured areas
- `src/events/after_hours.py`: occupancy checks against local operating hours
- `src/events/vehicle_zone.py`: vehicle detection inside pedestrian-only regions
- `src/events/abandoned_object.py`: unattended asset detection using stationary-item tracking plus nearest-owner distance checks
- `src/io/video.py`: video capture management and source parsing
- `src/io/recorder.py`: annotated output, pre/post alert clip capture, metadata sidecars, and duplicate suppression
- `src/io/health.py`: camera/read/reconnect health state persistence
- `src/io/logger.py`: JSONL event sink
- `src/api/app.py`: FastAPI app with SQLite-backed `/health`, `/cameras`, `/alerts`, `/alerts/{event_id}`, `/stats`, and `/ingest`
- `src/api/storage.py`: SQLite alert persistence and query/filter helpers
- `src/api/dashboard.py`: Streamlit operator dashboard backed by the alert database
- `Dockerfile` and `docker-compose.yml`: packaged API, dashboard, and pipeline services

## Current scope

The current implementation is focused on the core pipeline:

- Detection uses an off-the-shelf YOLO model
- Tracking defaults to a ByteTrack-style two-stage association pipeline, with an optional appearance-aware BoT-SORT-style mode
- Event logic covers intrusion, loitering, line crossing, wrong-way motion, after-hours occupancy, vehicle zone violations, and abandoned-object detection
- Perspective-aware zone reasoning can operate in normalized ground-plane coordinates when per-camera homography calibration is available
- Multi-camera configs share one central alert API while keeping per-camera evidence and health artifacts isolated on disk

That gives the repo a clean deployment story while keeping the codebase small enough to extend.
