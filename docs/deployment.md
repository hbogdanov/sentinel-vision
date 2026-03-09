# Deployment Notes

## Docker

Build the image:

```bash
docker build -t sentinel-vision .
```

Run the API:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" sentinel-vision \
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Run the pipeline with runtime overrides:

```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/configs:/app/configs" \
  -e SENTINEL_CONFIG=configs/default.yaml \
  -e SENTINEL_PROFILE=edge_cpu \
  -e SENTINEL_DEVICE=cpu \
  -e SENTINEL_MODEL_PATH=yolo11n.pt \
  sentinel-vision /app/docker/start_pipeline.sh
```

GPU profile example:

```bash
docker compose --profile gpu up --build pipeline-gpu api dashboard
```

The GPU profile only selects a CUDA-oriented config and device flag. The host still needs a working NVIDIA driver/runtime path for containers.

## Config Profiles

Available sample overlays:

- `configs/profiles/edge_cpu.yaml`
- `configs/profiles/edge_gpu.yaml`
- `configs/profiles/low_latency.yaml`

Use them from the CLI:

```bash
python -m src.main --config configs/default.yaml --profile edge_cpu
python -m src.main --config configs/default.yaml --profile edge_gpu --device cuda --model yolo11s.pt
```

Profile resolution accepts either a name under `configs/profiles/` or a direct YAML path.

## CPU And GPU Selection

- `--device cpu` forces CPU inference
- `--device cuda` or `--device cuda:0` targets NVIDIA GPUs
- `--model` overrides the configured model path without editing YAML

The container startup script reads the same values from:

- `SENTINEL_DEVICE`
- `SENTINEL_MODEL_PATH`
- `SENTINEL_CONFIG`
- `SENTINEL_PROFILE`
- `SENTINEL_SOURCE`

## systemd

Example unit for a single-camera edge node:

```ini
[Unit]
Description=Sentinel Vision Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/sentinel-vision
Environment="SENTINEL_ALERTS_DB_PATH=/opt/sentinel-vision/data/outputs/alerts.db"
ExecStart=/opt/sentinel-vision/.venv/bin/python -m src.main --config configs/default.yaml --profile edge_cpu
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

For API and dashboard processes, use separate units or run them under Compose if the host already uses containers.

## Startup Notes

- Mount `data/` persistently so alert clips, SQLite state, and camera health survive restarts.
- Use one namespaced config per deployment environment rather than editing `configs/default.yaml` in place.
- In multi-camera mode, outputs are automatically namespaced by `camera_id`, but all alerts still feed the same central SQLite/API service.
