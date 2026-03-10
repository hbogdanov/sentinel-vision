from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.events.zones import PolygonZone
from src.inference.tracker import Track


@dataclass(slots=True)
class ZoneOccupancyHeatmap:
    enabled: bool
    overlay_opacity: float
    point_radius: int
    decay: float
    output_image_path: Path
    output_summary_path: Path
    _heatmap: np.ndarray | None = field(default=None, init=False, repr=False)
    _zone_mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _frame_shape: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _total_frames: int = field(default=0, init=False, repr=False)
    _zones_seen: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ZoneOccupancyHeatmap":
        return cls(
            enabled=bool(config.get("enabled", False)),
            overlay_opacity=float(config.get("overlay_opacity", 0.35)),
            point_radius=int(config.get("point_radius", 18)),
            decay=float(config.get("decay", 0.985)),
            output_image_path=Path(
                str(config.get("output_image_path", "data/outputs/zone_heatmap.png"))
            ),
            output_summary_path=Path(
                str(config.get("output_summary_path", "data/outputs/zone_heatmap.json"))
            ),
        )

    def update(
        self,
        tracks: list[Track],
        zones: list[PolygonZone],
        frame_shape: tuple[int, int],
    ) -> None:
        if not self.enabled or not zones:
            return
        self._ensure_buffers(frame_shape, zones)
        if self._heatmap is None or self._zone_mask is None:
            return

        self._total_frames += 1
        self._heatmap *= self.decay
        frame_hits = np.zeros_like(self._heatmap)

        for zone in zones:
            occupants = [track for track in tracks if zone.contains_track(track)]
            stats = self._zones_seen.setdefault(
                zone.name,
                {
                    "frames_with_occupancy": 0,
                    "peak_occupancy": 0,
                    "total_track_hits": 0,
                },
            )
            if occupants:
                stats["frames_with_occupancy"] += 1
            stats["peak_occupancy"] = max(stats["peak_occupancy"], len(occupants))
            stats["total_track_hits"] += len(occupants)
            for track in occupants:
                cx, cy = map(int, track.center)
                cv2.circle(frame_hits, (cx, cy), self.point_radius, 1.0, -1)

        if self.point_radius > 2:
            kernel = max(3, (self.point_radius // 2) * 2 + 1)
            frame_hits = cv2.GaussianBlur(frame_hits, (kernel, kernel), 0)
        self._heatmap += frame_hits
        self._heatmap *= self._zone_mask

    def build_overlay(self) -> np.ndarray | None:
        if not self.enabled or self._heatmap is None or self._zone_mask is None:
            return None
        peak = float(self._heatmap.max())
        if peak <= 1e-6:
            return None
        normalized = np.clip((self._heatmap / peak) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        return cv2.bitwise_and(colored, colored, mask=self._zone_mask.astype(np.uint8))

    def save(self) -> None:
        if not self.enabled:
            return
        overlay = self.build_overlay()
        if overlay is not None:
            self.output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.output_image_path), overlay)
        self.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_summary_path.write_text(
            json.dumps(self.summary(), indent=2),
            encoding="utf-8",
        )

    def summary(self) -> dict[str, Any]:
        peak_intensity = (
            round(float(self._heatmap.max()), 4)
            if self._heatmap is not None and self._heatmap.size
            else 0.0
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "enabled": self.enabled,
            "total_frames": self._total_frames,
            "peak_intensity": peak_intensity,
            "zones": self._zones_seen,
            "output_image_path": str(self.output_image_path).replace("\\", "/"),
        }

    def _ensure_buffers(
        self, frame_shape: tuple[int, int], zones: list[PolygonZone]
    ) -> None:
        if self._frame_shape == frame_shape and self._heatmap is not None:
            return
        height, width = frame_shape
        self._frame_shape = frame_shape
        self._heatmap = np.zeros((height, width), dtype=np.float32)
        self._zone_mask = np.zeros((height, width), dtype=np.uint8)
        for zone in zones:
            points = np.array(
                [(int(x), int(y)) for x, y in zone.points], dtype=np.int32
            )
            cv2.fillPoly(self._zone_mask, [points], 255)
