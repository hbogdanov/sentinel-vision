# Sentinel Vision Architecture

## Data flow

1. `src.main` loads YAML configuration and opens the requested source.
2. `src.inference.detector` runs object detection on each frame.
3. `src.inference.tracker` assigns stable IDs to detections across frames.
4. `src.events` evaluates spatial and temporal event logic against configured polygon zones.
5. `src.utils.draw` overlays boxes, track IDs, zones, and active alert banners.
6. `src.io.logger` writes structured JSON events and `src.io.recorder` saves snapshots and alert clips.
7. `src.api.app` can optionally receive forwarded alerts for a lightweight dashboard.

## Modules

- `src/inference/detector.py`: detector interface and Ultralytics-backed implementation
- `src/inference/tracker.py`: lightweight IoU tracker for the MVP
- `src/inference/pipeline.py`: orchestration of decode, inference, event logic, rendering, and outputs
- `src/events/zones.py`: polygon utilities and point-in-zone checks
- `src/events/intrusion.py`: zone-entry event logic with cooldown suppression
- `src/events/loitering.py`: dwell-time logic for tracked subjects
- `src/io/video.py`: video capture management and source parsing
- `src/io/recorder.py`: annotated output, snapshots, and buffered alert clip writing
- `src/io/logger.py`: JSONL event sink
- `src/api/app.py`: minimal FastAPI app for alert ingestion/history

## MVP boundaries

The current implementation optimizes for a fast, reproducible demo:

- Detection uses an off-the-shelf YOLO model
- Tracking uses a built-in tracker rather than ByteTrack or DeepSORT
- Event logic is focused on intrusion and loitering
- Outputs are local-first with optional dashboard forwarding

That gives the repo a clean deployment story while keeping the codebase small enough to extend.
