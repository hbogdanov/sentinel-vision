from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(slots=True)
class VideoSource:
    source: int | str
    capture: cv2.VideoCapture
    width: int
    height: int
    fps: float

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def reopen(self) -> bool:
        self.release()
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            return False
        self.capture = capture
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or self.fps or 30.0)
        return True

    @property
    def is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith(
            "rtsp://"
        )

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def open_video_source(source: int | str) -> VideoSource:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    return VideoSource(
        source=source, capture=capture, width=width, height=height, fps=fps
    )
