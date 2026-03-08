from __future__ import annotations

from pathlib import Path

import cv2


class AlertRecorder:
    def __init__(
        self,
        alerts_dir: str,
        annotated_video_path: str | None,
        save_annotated_video: bool,
        buffer_seconds: float,
    ) -> None:
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_video_path = Path(annotated_video_path) if annotated_video_path else None
        self.save_annotated_video = save_annotated_video
        self.buffer_seconds = buffer_seconds
        self._writer: cv2.VideoWriter | None = None

    def prepare_video_writer(self, width: int, height: int, fps: float) -> None:
        if not self.save_annotated_video or self.annotated_video_path is None:
            return
        self.annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.annotated_video_path), fourcc, fps, (width, height))

    def write_annotated_frame(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def persist_alert(self, frame, buffered_frames: list, fps: float, event_id: str) -> tuple[str, str]:
        snapshot_path = self.alerts_dir / f"{event_id}.jpg"
        clip_path = self.alerts_dir / f"{event_id}.mp4"
        cv2.imwrite(str(snapshot_path), frame)
        self._write_clip(buffered_frames, clip_path, fps)
        return str(snapshot_path).replace("\\", "/"), str(clip_path).replace("\\", "/")

    def _write_clip(self, frames: list, clip_path: Path, fps: float) -> None:
        if not frames:
            return
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for buffered in frames:
            writer.write(buffered)
        writer.release()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
