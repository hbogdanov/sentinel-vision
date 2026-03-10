from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_events.py"
    spec = importlib.util.spec_from_file_location("evaluate_events", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_manifest_micro_benchmark() -> None:
    module = _load_module()
    manifest_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "benchmark_manifest.json"
    manifest = module._load_json(manifest_path)
    bundles = [module._load_video_bundle(manifest_path.parent, video_spec) for video_spec in manifest["videos"]]

    results = module.evaluate_manifest(bundles, iou_threshold=0.5)

    assert results["summary"]["num_videos"] == 3
    assert results["summary"]["coverage"]["scene_types"] == ["indoor", "loading_dock", "office"]
    assert results["summary"]["detection_by_class"]["person"]["precision"] == 1.0
    assert results["summary"]["detection_by_class"]["person"]["recall"] == 0.9444
    assert results["summary"]["tracking"]["id_switches"] == 2
    assert results["summary"]["tracking"]["mota"] == 0.8333
    assert results["summary"]["tracking"]["motp"] == 1.0
    assert results["summary"]["tracking"]["idf1"] == 0.7429
    assert results["summary"]["events_by_type"]["intrusion"]["precision"] == 0.5
    assert results["summary"]["events_by_type"]["loitering"]["recall"] == 1.0
    assert results["summary"]["events_by_type"]["loitering"]["mean_alert_latency_seconds"] == 10.0
    assert results["summary"]["events_overall"]["false_alerts_per_minute"] == 2.3077
