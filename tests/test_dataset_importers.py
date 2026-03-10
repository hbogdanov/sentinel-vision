from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, relative_path: str):
    script_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_mot17_filters_visibility_and_rebases_frames(tmp_path: Path) -> None:
    module = _load_module("import_mot17", "scripts/import_mot17.py")
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "\n".join(
            [
                "1,1,10,20,30,40,1,1,0.9",
                "2,1,12,20,30,40,1,1,0.1",
                "3,2,50,60,20,30,1,2,0.9",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.convert_mot17_annotations(
        gt_path=gt_path,
        start_frame=1,
        end_frame=3,
        min_visibility=0.2,
        seqinfo_path=None,
        scale_x=1.0,
        scale_y=1.0,
    )

    assert payload["fps"] == 30.0
    assert payload["detections"] == [
        {"frame_index": 0, "track_id": 1, "class": "person", "bbox": [10.0, 20.0, 40.0, 60.0], "score": 0.9}
    ]


def test_import_visdrone_maps_supported_classes_and_rebases_frames(tmp_path: Path) -> None:
    module = _load_module("import_visdrone_mot", "scripts/import_visdrone_mot.py")
    gt_path = tmp_path / "clip.txt"
    gt_path.write_text(
        "\n".join(
            [
                "5,0,10,20,30,40,1,4,0,0",
                "6,1,50,60,20,30,1,9,0,0",
                "7,2,15,25,10,15,1,8,0,0",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.convert_visdrone_annotations(
        gt_path=gt_path,
        fps=25.0,
        start_frame=5,
        end_frame=6,
        scale_x=1.0,
        scale_y=1.0,
    )

    assert payload["fps"] == 25.0
    assert payload["detections"] == [
        {"frame_index": 0, "track_id": 1, "class": "car", "bbox": [10.0, 20.0, 40.0, 60.0], "score": 1.0},
        {"frame_index": 1, "track_id": 2, "class": "bus", "bbox": [50.0, 60.0, 70.0, 90.0], "score": 1.0},
    ]
