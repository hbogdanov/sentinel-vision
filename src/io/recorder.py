from __future__ import annotations

import json
import logging
import queue
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _ActiveAlertRecording:
    event: dict[str, Any]
    frames: list[Any]
    frames_remaining: int
    snapshot_path: Path
    clip_path: Path
    metadata_path: Path
    fps: float


@dataclass(slots=True)
class _FinalizeAlertTask:
    recording: _ActiveAlertRecording


class AlertRecorder:
    def __init__(
        self,
        alerts_dir: str,
        annotated_video_path: str | None,
        save_annotated_video: bool,
        buffer_seconds: float,
        post_event_seconds: float = 3.0,
        duplicate_suppression_seconds: float = 10.0,
        clip_writer_queue_size: int = 16,
    ) -> None:
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_video_path = (
            Path(annotated_video_path) if annotated_video_path else None
        )
        self.save_annotated_video = save_annotated_video
        self.buffer_seconds = buffer_seconds
        self.post_event_seconds = post_event_seconds
        self.duplicate_suppression_seconds = duplicate_suppression_seconds
        self._writer: cv2.VideoWriter | None = None
        self._pre_event_frames: deque[Any] = deque()
        self._active_recordings: list[_ActiveAlertRecording] = []
        self._last_alert_time: dict[tuple[str, str, str], datetime] = {}
        self._clip_queue: queue.Queue[_FinalizeAlertTask | None] = queue.Queue(
            maxsize=max(1, clip_writer_queue_size)
        )
        self._worker = threading.Thread(
            target=self._worker_loop, name="alert-clip-writer", daemon=True
        )
        self._worker.start()

    def prepare_video_writer(self, width: int, height: int, fps: float) -> None:
        if not self.save_annotated_video or self.annotated_video_path is None:
            return
        self.annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        safe_fps = _safe_fps(fps)
        opened = self._open_video_writer(
            self.annotated_video_path, safe_fps, width, height
        )
        if opened is None:
            LOGGER.warning(
                "Failed to open annotated video writer for %s",
                self.annotated_video_path,
            )
            return
        self._writer = opened[0]

    def write_annotated_frame(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def ingest_frame(self, frame, fps: float) -> None:
        safe_fps = _safe_fps(fps)
        maxlen = max(1, int(round(self.buffer_seconds * safe_fps)) + 1)
        if self._pre_event_frames.maxlen != maxlen:
            self._pre_event_frames = deque(self._pre_event_frames, maxlen=maxlen)
        self._pre_event_frames.append(frame.copy())

        completed: list[_ActiveAlertRecording] = []
        for recording in self._active_recordings:
            if recording.frames_remaining > 0:
                recording.frames.append(frame.copy())
                recording.frames_remaining -= 1
            if recording.frames_remaining <= 0:
                completed.append(recording)

        for recording in completed:
            self._enqueue_finalize(recording)
            self._active_recordings.remove(recording)

    def start_alert(
        self, event: dict[str, Any], frame, fps: float
    ) -> tuple[str | None, str | None, str | None]:
        if self._is_duplicate(event):
            return None, None, None

        event_key = self._event_key(event)
        event_timestamp = _parse_event_timestamp(event["timestamp"])
        self._last_alert_time[event_key] = event_timestamp

        stem = self._build_alert_stem(event)
        snapshot_path = self.alerts_dir / f"{stem}.jpg"
        clip_path = self.alerts_dir / f"{stem}.mp4"
        metadata_path = self.alerts_dir / f"{stem}.json"

        if not cv2.imwrite(str(snapshot_path), frame):
            LOGGER.warning("Failed to write alert snapshot to %s", snapshot_path)

        safe_fps = _safe_fps(fps)
        frames = list(self._pre_event_frames)
        if frames and _same_frame(frames[-1], frame):
            frames = frames[:-1]
        frames.append(frame.copy())
        frames_remaining = max(0, int(round(self.post_event_seconds * safe_fps)))
        recording = _ActiveAlertRecording(
            event=dict(event),
            frames=frames,
            frames_remaining=frames_remaining,
            snapshot_path=snapshot_path,
            clip_path=clip_path,
            metadata_path=metadata_path,
            fps=safe_fps,
        )
        if frames_remaining > 0:
            self._active_recordings.append(recording)
        else:
            self._enqueue_finalize(recording)

        return (
            str(snapshot_path).replace("\\", "/"),
            str(clip_path).replace("\\", "/"),
            str(metadata_path).replace("\\", "/"),
        )

    def close(self, fps: float | None = None) -> None:
        safe_fps = _safe_fps(fps)
        for recording in list(self._active_recordings):
            recording.fps = safe_fps
            self._enqueue_finalize(recording)
            self._active_recordings.remove(recording)
        self._clip_queue.put(None)
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            LOGGER.warning(
                "Alert clip writer did not shut down cleanly before timeout."
            )
        if self._writer is not None:
            self._writer.release()

    def _enqueue_finalize(self, recording: _ActiveAlertRecording) -> None:
        try:
            self._clip_queue.put(_FinalizeAlertTask(recording=recording), timeout=1.0)
        except queue.Full:
            LOGGER.warning(
                "Alert clip queue is full. Finalizing %s synchronously.",
                recording.clip_path.name,
            )
            self._finalize_recording(recording)

    def _worker_loop(self) -> None:
        while True:
            task = self._clip_queue.get()
            if task is None:
                self._clip_queue.task_done()
                break
            try:
                self._finalize_recording(task.recording)
            except Exception:
                LOGGER.exception(
                    "Failed to finalize alert recording for %s",
                    task.recording.clip_path,
                )
            finally:
                self._clip_queue.task_done()

    def _finalize_recording(self, recording: _ActiveAlertRecording) -> None:
        clip_result = self._write_clip(
            recording.frames, recording.clip_path, recording.fps
        )
        metadata = dict(recording.event)
        metadata["snapshot_path"] = str(recording.snapshot_path).replace("\\", "/")
        metadata["clip_path"] = str(recording.clip_path).replace("\\", "/")
        metadata["metadata_path"] = str(recording.metadata_path).replace("\\", "/")
        metadata["pre_event_seconds"] = self.buffer_seconds
        metadata["post_event_seconds"] = self.post_event_seconds
        metadata["clip_frame_count"] = len(recording.frames)
        metadata["deduplicated_by"] = ["event_type", "track_id", "zone"]
        metadata["clip_write_ok"] = clip_result["ok"]
        metadata["clip_codec"] = clip_result["codec"]
        metadata["clip_write_error"] = clip_result["error"]
        recording.metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _write_clip(
        self, frames: list, clip_path: Path, fps: float
    ) -> dict[str, str | bool | None]:
        if not frames:
            return {"ok": False, "codec": None, "error": "no_frames"}
        height, width = frames[0].shape[:2]
        opened = self._open_video_writer(clip_path, fps, width, height)
        if opened is None:
            LOGGER.warning(
                "Failed to open clip writer for %s; clip metadata will record the error.",
                clip_path,
            )
            return {"ok": False, "codec": None, "error": "codec_open_failed"}
        writer, codec = opened
        try:
            for buffered in frames:
                writer.write(buffered)
        finally:
            writer.release()
        return {"ok": True, "codec": codec, "error": None}

    def _open_video_writer(
        self, path: Path, fps: float, width: int, height: int
    ) -> tuple[cv2.VideoWriter, str] | None:
        safe_fps = _safe_fps(fps)
        for codec in ("mp4v", "avc1", "XVID", "MJPG"):
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*codec), safe_fps, (width, height)
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None

    def _is_duplicate(self, event: dict[str, Any]) -> bool:
        event_key = self._event_key(event)
        last_time = self._last_alert_time.get(event_key)
        if last_time is None:
            return False
        current_time = _parse_event_timestamp(event["timestamp"])
        return (
            current_time - last_time
        ).total_seconds() < self.duplicate_suppression_seconds

    def _event_key(self, event: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(event.get("event_type", "event")),
            str(event.get("track_id", "na")),
            str(event.get("zone", "global")),
        )

    def _build_alert_stem(self, event: dict[str, Any]) -> str:
        timestamp = _parse_event_timestamp(event["timestamp"]).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        event_type = _slugify(str(event.get("event_type", "event")))
        zone = _slugify(str(event.get("zone", "zone")))
        track_id = _slugify(str(event.get("track_id", "na")))
        event_id = _slugify(str(event.get("event_id", "evt")))
        return f"{timestamp}_{event_type}_{zone}_track{track_id}_{event_id}"


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "item"


def _parse_event_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_fps(value: float | None) -> float:
    if value is None or value <= 0:
        return 30.0
    return float(value)


def _same_frame(left, right) -> bool:
    return left.shape == right.shape and bool((left == right).all())
