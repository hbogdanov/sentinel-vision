# Reproducibility

This document explains exactly how to reproduce the benchmark outputs in this repository and what inputs those results depend on.

## Result provenance

There are two distinct reproduction paths:

1. `Score-only reproduction`
   Uses the checked-in prediction JSON files under `data/eval/predictions/`.
   This reproduces the checked-in metrics in `docs/results.md` and `data/eval/results/latest.json`.

2. `Live inference reproduction`
   Runs the pipeline over the benchmark videos, writes fresh prediction JSON files, and then scores them.
   This is the path to use for CPU/GPU runtime comparisons or when detector/tracker changes are made.

## Checked-in result package

The current checked-in benchmark summary was produced from:

- Manifest: [benchmark_manifest.json](c:\Users\Ivan\sentinel-vision\data\eval\benchmark_manifest.json)
- Ground truth: [annotations](c:\Users\Ivan\sentinel-vision\data\eval\annotations)
- Predictions: [predictions](c:\Users\Ivan\sentinel-vision\data\eval\predictions)
- Scoring script: [evaluate_events.py](c:\Users\Ivan\sentinel-vision\scripts\evaluate_events.py)
- Report renderer: [render_metrics_report.py](c:\Users\Ivan\sentinel-vision\scripts\render_metrics_report.py)

Because these checked-in results are scored from stored prediction JSON, the following are `not applicable` for the current checked-in `docs/results.md` artifact:

- Hardware used
- GPU type
- Detector runtime FPS
- Exact live model weights execution environment

Those only apply when you run the live inference path below.

## Exact commands

### Score-only reproduction

From the repo root:

```bash
python -m scripts.evaluate_events --manifest data/eval/benchmark_manifest.json --output-json data/eval/results/latest.json --output-markdown docs/results.md
python -m scripts.render_metrics_report --input-json data/eval/results/latest.json --output-html docs/results_dashboard.html
```

### Live inference reproduction on CPU

```bash
python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest.json --config configs/default.yaml --device cpu --model yolo11n.pt --predictions-dir data/eval/predictions_cpu --output-json data/eval/results/cpu_latest.json --output-markdown docs/results_cpu.md
python -m scripts.render_metrics_report --input-json data/eval/results/cpu_latest.json --output-html docs/results_cpu.html
```

### Live inference reproduction on GPU

```bash
python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest.json --config configs/default.yaml --device cuda:0 --model yolo11n.pt --predictions-dir data/eval/predictions_gpu --output-json data/eval/results/gpu_latest.json --output-markdown docs/results_gpu.md
python -m scripts.render_metrics_report --input-json data/eval/results/gpu_latest.json --output-html docs/results_gpu.html
```

## Model and config

The default live benchmark command uses:

- Model path: `yolo11n.pt`
- Detector wrapper: `Ultralytics YOLO` via [detector.py](c:\Users\Ivan\sentinel-vision\src\inference\detector.py)
- Base config: [default.yaml](c:\Users\Ivan\sentinel-vision\configs\default.yaml)
- Tracker: `bytetrack`
- Detection confidence: `0.35`
- Default classes: `person`, `car`, `truck`, `bus`, `backpack`, `handbag`, `suitcase`

If you change the model path, device, profile, or config overrides, the results are a different benchmark run and should be reported separately.

## Manifest explanation

Each video entry in the manifest defines five things:

1. The clip being evaluated via `video_path`
2. The expected ground truth via `ground_truth`
3. The prediction file to score via `predictions`
4. Coverage metadata such as `scene_types`, `challenge_tags`, and `subject_classes`
5. Optional clip-specific runtime settings via `config_override`

That means the manifest is the benchmark contract. If the manifest changes, the benchmark changed.

## Hardware reporting template

For any live inference result you publish, include at least:

- CPU model
- GPU model
- RAM
- OS
- Python version
- `--device` value used
- model path used
- config path used
- manifest path used

Example format:

```text
Hardware:
- CPU: Intel Core i7-12700H
- GPU: NVIDIA RTX 3060 Laptop GPU
- RAM: 32 GB
- OS: Windows 11

Run settings:
- Command: python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest.json --config configs/default.yaml --device cuda:0 --model yolo11n.pt
- Config: configs/default.yaml
- Manifest: data/eval/benchmark_manifest.json
- Model: yolo11n.pt
```

## Reporting rule

If you quote benchmark numbers in the README or a portfolio, include:

- the command used
- whether the run was score-only or live inference
- the model path
- the config path
- the hardware

Otherwise the result is not reproducible enough for engineering review.

## Video asset changes

If you re-encode any benchmark clip to reduce repository size, treat that as an input change to the benchmark:

```bash
python -m scripts.compress_benchmark_videos --input-dir data/eval/videos --output-dir data/eval/videos_compressed --crf 28 --preset slow --max-width 960
python -m scripts.run_benchmark --manifest data/eval/benchmark_manifest_public_datasets.json --config configs/default.yaml --profile edge_cpu --device cpu --model models/yolo11n.pt --predictions-dir data/eval/predictions_public --output-json data/eval/results/public_dataset_cpu.json --output-markdown docs/results_public_dataset_cpu.md
python -m scripts.render_metrics_report --input-json data/eval/results/public_dataset_cpu.json --output-html docs/results_public_dataset_cpu.html
```

On Windows, if Ultralytics cannot write to the default roaming profile directory in your shell, point it at a writable repo-local directory before running the live benchmark:

```powershell
$env:YOLO_CONFIG_DIR="$PWD\\.cache\\ultralytics"
$env:ULTRALYTICS_SETTINGS_DIR="$PWD\\.cache\\ultralytics"
```
