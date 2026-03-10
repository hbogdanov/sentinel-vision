from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class GroundPlaneMapper:
    matrix: np.ndarray
    enabled: bool = True

    @classmethod
    def from_correspondences(
        cls,
        image_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
    ) -> "GroundPlaneMapper":
        image = np.array(image_points, dtype=np.float32)
        world = np.array(world_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(image, world, method=0)
        if matrix is None:
            raise ValueError(
                "Failed to compute ground-plane homography from the provided correspondences."
            )
        return cls(matrix=matrix.astype(np.float32), enabled=True)

    def project_point(self, point: tuple[float, float]) -> tuple[float, float] | None:
        if not self.enabled:
            return None
        points = np.array([[[point[0], point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(points, self.matrix)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])

    def project_points(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        if not self.enabled:
            return []
        raw = np.array([[list(point)] for point in points], dtype=np.float32)
        projected = cv2.perspectiveTransform(raw, self.matrix)
        return [(float(point[0]), float(point[1])) for point in projected[:, 0, :]]
