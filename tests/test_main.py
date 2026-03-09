from __future__ import annotations

from pathlib import Path

import pytest

from src.main import _apply_global_overrides, _resolve_profile_path


def test_apply_global_overrides_updates_model_path_and_device() -> None:
    config = {"model": {"path": "base.pt", "device": "cpu"}}

    updated = _apply_global_overrides(config, model_path="custom.pt", device="cuda:0")

    assert updated["model"]["path"] == "custom.pt"
    assert updated["model"]["device"] == "cuda:0"


def test_resolve_profile_path_accepts_explicit_path(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("model:\n  device: cpu\n", encoding="utf-8")

    assert _resolve_profile_path(str(profile_path)) == profile_path


def test_resolve_profile_path_rejects_missing_profile() -> None:
    with pytest.raises(FileNotFoundError):
        _resolve_profile_path("missing_profile_name")
