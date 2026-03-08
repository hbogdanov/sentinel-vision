from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from typing import Any
from urllib import request

import cv2

from src.events.intrusion import IntrusionDetector
from src.events.loitering import LoiteringDetector
from src.events.zones import load_zones
from src.inference.detector import YoloDetector
from src.inference.tracker import SimpleTracker, Track
from src.io.logger import EventLogger
from src.io.recorder import AlertRecorder
from src.io.video import open_video_source
from src.utils.draw import draw_frame
from src.utils.timing import FpsMeter


class SentinelPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.camera_id = config["camera_id"]
        self.zones = load_zones(config["zones"])
        self.detector = YoloDetector(
            model_path=config["model"]["path"],
            confidence=float(config["model"]["confidence"]),
            classes_of_interest=config["model"].get("classes", []),
            device=config["model"].get("device", "cpu"),
        )
        tracking = config["tracking"]
        self.tracker = SimpleTracker(
            iou_threshold=float(tracking.get("iou_threshold", 0.3)),
            max_age_frames=int(tracking.get("max_age_frames", 30)),
            min_hits=int(tracking.get("min_hits", 1)),
        )
        intrusion_cfg = config["events"]["intrusion"]
        loitering_cfg = config["events"]["loitering"]
        self.intrusion = IntrusionDetector(
            enabled=bool(intrusion_cfg.get("enabled", True)),
            cooldown_seconds=float(intrusion_cfg.get("cooldown_seconds", 5)),
        )
        self.loitering = LoiteringDetector(
            enabled=bool(loitering_cfg.get("enabled", True)),
            threshold_seconds=float(loitering_cfg.get("threshold_seconds", 10)),
            cooldown_seconds=float(loitering_cfg.get("cooldown_seconds", 20)),
        )
        output_cfg = config["output"]
        self.event_logger = EventLogger(output_cfg["log_path"])
        self.recorder = AlertRecorder(
            alerts_dir=output_cfg["alerts_dir"],
            annotated_video_path=output_cfg.get("annotated_video_path"),
            save_annotated_video=bool(output_cfg.get("save_annotated_video", True)),
            buffer_seconds=float(output_cfg.get("buffer_seconds", 3)),
        )
        self.show_window = bool(output_cfg.get("show_window", False))
        self.dashboard_cfg = config.get("dashboard", {"enabled": False})

    def run(self) -> None:
        source = self.config["input"]["source"]
        with open_video_source(source) as capture:
            fps = capture.fps or 30.0
            frame_buffer: deque[Any] = deque(maxlen=max(1, int(self.recorder.buffer_seconds * fps)))
            fps_meter = FpsMeter()
            frame_index = 0
            self.recorder.prepare_video_writer(capture.width, capture.height, fps)

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp = datetime.now(timezone.utc)
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame_index=frame_index)
                events = self._evaluate_events(tracks, frame_index, timestamp, fps)
                frame_buffer.append(frame.copy())

                for event in events:
                    snapshot_path, clip_path = self.recorder.persist_alert(
                        frame=frame,
                        buffered_frames=list(frame_buffer),
                        fps=fps,
                        event_id=event["event_id"],
                    )
                    event["snapshot_path"] = snapshot_path
                    event["clip_path"] = clip_path
                    self.event_logger.write(event)
                    self._dispatch_alert(event)

                annotated = draw_frame(
                    frame=frame.copy(),
                    zones=self.zones,
                    tracks=tracks,
                    events=events,
                    fps=fps_meter.tick(),
                )
                self.recorder.write_annotated_frame(annotated)

                if self.show_window:
                    cv2.imshow("Sentinel Vision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1

            self.recorder.close()
            if self.show_window:
                cv2.destroyAllWindows()

    def _evaluate_events(
        self,
        tracks: list[Track],
        frame_index: int,
        timestamp: datetime,
        fps: float,
    ) -> list[dict[str, Any]]:
        events = []
        events.extend(
            self.intrusion.evaluate(
                tracks=tracks,
                zones=self.zones,
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.loitering.evaluate(
                tracks=tracks,
                zones=self.zones,
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        return events

    def _dispatch_alert(self, event: dict[str, Any]) -> None:
        if not self.dashboard_cfg.get("enabled", False):
            return

        endpoint = self.dashboard_cfg.get("endpoint")
        if not endpoint:
            return

        payload = json.dumps(event).encode("utf-8")
        req = request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            request.urlopen(req, timeout=float(self.dashboard_cfg.get("timeout_seconds", 1.0))).read()
        except Exception:
            return
