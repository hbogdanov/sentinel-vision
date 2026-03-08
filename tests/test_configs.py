from pathlib import Path

from src.utils.config import load_config


def test_load_config_merges_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera_id: cam_2\nzones: []\nmodel:\n  confidence: 0.5\n", encoding="utf-8")

    config = load_config(str(config_path))

    assert config["camera_id"] == "cam_2"
    assert config["model"]["confidence"] == 0.5
    assert config["tracking"]["max_age_frames"] == 30
