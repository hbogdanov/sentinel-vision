from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class GlobalMotionResult:
    matrix: np.ndarray
    estimated: bool
    method: str
    match_count: int

    @classmethod
    def identity(cls, method: str = "affine") -> "GlobalMotionResult":
        if method == "homography":
            matrix = np.eye(3, dtype=np.float32)
        else:
            matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return cls(matrix=matrix, estimated=False, method=method, match_count=0)


class GlobalMotionCompensator:
    def __init__(
        self,
        enabled: bool = False,
        static_camera_assumption: bool = True,
        method: str = "affine",
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: float = 8.0,
        min_matches: int = 12,
        ransac_threshold: float = 3.0,
        smoothing_factor: float = 0.8,
    ) -> None:
        self.enabled = enabled
        self.static_camera_assumption = static_camera_assumption
        self.method = method
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.smoothing_factor = smoothing_factor
        self._previous_gray: np.ndarray | None = None
        self._smoothed_matrix: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> GlobalMotionResult:
        if not self.enabled or self.static_camera_assumption:
            self._store_frame(frame)
            self._smoothed_matrix = None
            return GlobalMotionResult.identity(method=self.method)

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None:
            self._previous_gray = current_gray
            self._smoothed_matrix = None
            return GlobalMotionResult.identity(method=self.method)

        previous_points = cv2.goodFeaturesToTrack(
            self._previous_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if previous_points is None or len(previous_points) < self.min_matches:
            self._previous_gray = current_gray
            return GlobalMotionResult.identity(method=self.method)

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_gray, current_gray, previous_points, None
        )
        if current_points is None or status is None:
            self._previous_gray = current_gray
            return GlobalMotionResult.identity(method=self.method)

        valid_mask = status.reshape(-1) == 1
        prev_valid = previous_points.reshape(-1, 2)[valid_mask]
        curr_valid = current_points.reshape(-1, 2)[valid_mask]
        if len(prev_valid) < self.min_matches:
            self._previous_gray = current_gray
            return GlobalMotionResult.identity(method=self.method)

        matrix = self._estimate_transform(curr_valid, prev_valid)
        self._previous_gray = current_gray
        if matrix is None:
            return GlobalMotionResult.identity(method=self.method)

        smoothed_matrix = self._smooth_matrix(matrix)
        return GlobalMotionResult(
            matrix=smoothed_matrix,
            estimated=True,
            method=self.method,
            match_count=int(len(prev_valid)),
        )

    def reset(self) -> None:
        self._previous_gray = None
        self._smoothed_matrix = None

    def _store_frame(self, frame: np.ndarray) -> None:
        self._previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _estimate_transform(
        self, source_points: np.ndarray, target_points: np.ndarray
    ) -> np.ndarray | None:
        if self.method == "homography":
            matrix, _ = cv2.findHomography(
                source_points,
                target_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )
            if matrix is None:
                return None
            return matrix.astype(np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if matrix is None:
            return None
        return matrix.astype(np.float32)

    def _smooth_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self._smoothed_matrix is None:
            self._smoothed_matrix = matrix
            return matrix
        alpha = float(np.clip(self.smoothing_factor, 0.0, 1.0))
        self._smoothed_matrix = (alpha * self._smoothed_matrix) + (
            (1.0 - alpha) * matrix
        )
        return self._smoothed_matrix.astype(np.float32)


def compensate_bbox(
    bbox: tuple[float, float, float, float],
    motion_transform: np.ndarray | None,
) -> tuple[float, float, float, float]:
    if motion_transform is None:
        return bbox

    x1, y1, x2, y2 = bbox
    corners = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32
    )
    if motion_transform.shape == (2, 3):
        transformed = cv2.transform(corners, motion_transform)
    elif motion_transform.shape == (3, 3):
        transformed = cv2.perspectiveTransform(corners, motion_transform)
    else:
        return bbox

    xs = transformed[:, 0, 0]
    ys = transformed[:, 0, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
