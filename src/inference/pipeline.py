from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

import cv2

from src.events.after_hours import AfterHoursOccupancyDetector
from src.events.intrusion import IntrusionDetector
from src.events.line_crossing import LineCrossingDetector
from src.events.loitering import LoiteringDetector
from src.events.vehicle_zone import VehicleInPedestrianZoneDetector
from src.events.wrong_way import WrongWayDetector
from src.events.zones import line_zones, load_zones, polygon_zones
from src.inference.detector import Detection, YoloDetector
from src.inference.ground_plane import GroundPlaneMapper
from src.inference.motion import GlobalMotionCompensator
from src.inference.tracker import Track, create_tracker
from src.io.logger import EventLogger
from src.io.health import CameraHealthMonitor
from src.io.recorder import AlertRecorder
from src.io.video import VideoSource, open_video_source
from src.utils.draw import draw_frame
from src.utils.timing import FpsMeter, RollingTimingStats


LOGGER = logging.getLogger(__name__)


class SentinelPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.camera_id = config["camera_id"]
        self.zones = load_zones(config["zones"])
        perspective_cfg = config.get("perspective", {"enabled": False})
        self.ground_plane = self._build_ground_plane_mapper(perspective_cfg)
        if self.ground_plane is not None:
            self._attach_projected_zone_geometry()
        self.detector = YoloDetector(
            model_path=config["model"]["path"],
            confidence=float(config["model"]["confidence"]),
            classes_of_interest=config["model"].get("classes", []),
            device=config["model"].get("device", "cpu"),
        )
        tracking = config["tracking"]
        self.tracker = create_tracker(tracking)
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
        line_crossing_cfg = config["events"]["line_crossing"]
        wrong_way_cfg = config["events"]["wrong_way"]
        after_hours_cfg = config["events"]["after_hours_occupancy"]
        vehicle_zone_cfg = config["events"]["vehicle_in_pedestrian_zone"]
        self.line_crossing = LineCrossingDetector(
            enabled=bool(line_crossing_cfg.get("enabled", True)),
            cooldown_seconds=float(line_crossing_cfg.get("cooldown_seconds", 3)),
            direction=str(line_crossing_cfg.get("direction", "any")),
        )
        self.wrong_way = WrongWayDetector(
            enabled=bool(wrong_way_cfg.get("enabled", True)),
            cooldown_seconds=float(wrong_way_cfg.get("cooldown_seconds", 10)),
            min_displacement_pixels=float(wrong_way_cfg.get("min_displacement_pixels", 25)),
            target_classes=list(wrong_way_cfg.get("target_classes", ["person", "car", "truck", "bus"])),
        )
        self.after_hours = AfterHoursOccupancyDetector(
            enabled=bool(after_hours_cfg.get("enabled", True)),
            cooldown_seconds=float(after_hours_cfg.get("cooldown_seconds", 60)),
            start_time=str(after_hours_cfg.get("start_time", "08:00")),
            end_time=str(after_hours_cfg.get("end_time", "18:00")),
            timezone_name=str(after_hours_cfg.get("timezone", "UTC")),
            target_classes=list(after_hours_cfg.get("target_classes", ["person"])),
        )
        self.vehicle_in_pedestrian_zone = VehicleInPedestrianZoneDetector(
            enabled=bool(vehicle_zone_cfg.get("enabled", True)),
            cooldown_seconds=float(vehicle_zone_cfg.get("cooldown_seconds", 5)),
            target_classes=list(
                vehicle_zone_cfg.get("target_classes", ["car", "truck", "bus", "motorcycle", "bicycle"])
            ),
        )
        output_cfg = config["output"]
        self.event_logger = EventLogger(output_cfg["log_path"])
        self.health_monitor = CameraHealthMonitor(
            camera_id=self.camera_id,
            status_path=Path(output_cfg.get("health_status_path", "data/outputs/camera_health.json")),
        )
        self.recorder = AlertRecorder(
            alerts_dir=output_cfg["alerts_dir"],
            annotated_video_path=output_cfg.get("annotated_video_path"),
            save_annotated_video=bool(output_cfg.get("save_annotated_video", True)),
            buffer_seconds=float(output_cfg.get("buffer_seconds", 3)),
            post_event_seconds=float(output_cfg.get("post_event_seconds", 3)),
            duplicate_suppression_seconds=float(output_cfg.get("duplicate_suppression_seconds", 10)),
            clip_writer_queue_size=int(output_cfg.get("clip_writer_queue_size", 16)),
        )
        self.show_window = bool(output_cfg.get("show_window", False))
        self.dashboard_cfg = config.get("dashboard", {"enabled": False})
        self.input_cfg = config["input"]
        self.runtime_cfg = config["runtime"]
        motion_cfg = self.runtime_cfg.get("motion_compensation", {})
        self.motion_compensator = GlobalMotionCompensator(
            enabled=bool(motion_cfg.get("enabled", False)),
            static_camera_assumption=bool(motion_cfg.get("static_camera_assumption", True)),
            method=str(motion_cfg.get("method", "affine")),
            max_corners=int(motion_cfg.get("max_corners", 300)),
            quality_level=float(motion_cfg.get("quality_level", 0.01)),
            min_distance=float(motion_cfg.get("min_distance", 8.0)),
            min_matches=int(motion_cfg.get("min_matches", 12)),
            ransac_threshold=float(motion_cfg.get("ransac_threshold", 3.0)),
            smoothing_factor=float(motion_cfg.get("smoothing_factor", 0.8)),
        )
        self._track_history: dict[int, deque[tuple[int, int]]] = {}
        self._dwell_entry_ts: dict[tuple[str, int], datetime] = {}
        self._active_alerts: list[dict[str, Any]] = []

    def run(self) -> None:
        source = self.config["input"]["source"]
        with open_video_source(source) as capture:
            fps = capture.fps or 30.0
            fps_meter = FpsMeter()
            stage_stats = RollingTimingStats(window_size=max(1, int(self.runtime_cfg.get("timing_log_interval_frames", 120))))
            frame_index = 0
            consecutive_read_failures = 0
            self.recorder.prepare_video_writer(capture.width, capture.height, fps)

            while True:
                with stage_stats.measure("read"):
                    ok, frame = capture.read()
                if not ok:
                    consecutive_read_failures += 1
                    self.health_monitor.mark_read_failure()
                    if self._handle_read_failure(capture, consecutive_read_failures):
                        fps = capture.fps or fps or 30.0
                        continue
                    LOGGER.warning("Stopping pipeline after %s consecutive frame read failures.", consecutive_read_failures)
                    self.health_monitor.mark_offline()
                    break
                consecutive_read_failures = 0
                self.health_monitor.mark_frame_read()

                if self._should_skip_frame(frame_index, stage_stats):
                    frame_index += 1
                    continue

                timestamp = datetime.now(timezone.utc)
                self.recorder.ingest_frame(frame, fps=fps)
                inference_frame = self._resize_for_inference(frame)
                motion_result = self.motion_compensator.update(frame)

                with stage_stats.measure("detect"):
                    try:
                        detections = self.detector.detect(inference_frame)
                    except Exception:
                        LOGGER.exception("Detector failed on frame %s. Continuing with empty detections.", frame_index)
                        self.health_monitor.mark_detector_failure()
                        detections = []
                detections = self._scale_detections_to_frame(detections, inference_frame, frame)

                with stage_stats.measure("track"):
                    tracks = self.tracker.update(
                        detections,
                        frame_index=frame_index,
                        frame=frame,
                        motion_transform=motion_result.matrix if motion_result.estimated else None,
                    )
                self._project_tracks_to_ground_plane(tracks)

                with stage_stats.measure("events"):
                    events = self._evaluate_events(tracks, frame_index, timestamp, fps)

                with stage_stats.measure("alerts"):
                    for event in events:
                        snapshot_path, clip_path, metadata_path = self.recorder.start_alert(event=event, frame=frame, fps=fps)
                        if snapshot_path is None or clip_path is None or metadata_path is None:
                            continue
                        event["snapshot_path"] = snapshot_path
                        event["clip_path"] = clip_path
                        event["metadata_path"] = metadata_path
                        self.event_logger.write(event)
                        self._dispatch_alert(event)

                with stage_stats.measure("render"):
                    self._update_track_history(tracks)
                    dwell_timers = self._update_dwell_timers(tracks, timestamp)
                    annotated = draw_frame(
                        frame=frame.copy(),
                        zones=self.zones,
                        tracks=tracks,
                        events=self._current_active_alerts(timestamp),
                        fps=fps_meter.tick(),
                        track_history=self._track_history,
                        dwell_timers=dwell_timers,
                    )
                self.recorder.write_annotated_frame(annotated)

                if self.show_window:
                    cv2.imshow("Sentinel Vision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_index > 0 and frame_index % max(1, int(self.runtime_cfg.get("timing_log_interval_frames", 120))) == 0:
                    LOGGER.info("Stage timing averages (ms): %s", stage_stats.summary_ms())

                frame_index += 1

            self.recorder.close(fps=fps)
            self.health_monitor.mark_offline()
            if self.show_window:
                cv2.destroyAllWindows()

    def _handle_read_failure(self, capture: VideoSource, consecutive_read_failures: int) -> bool:
        threshold = int(self.input_cfg.get("read_failure_threshold", 30))
        if consecutive_read_failures < max(1, threshold):
            return True
        if not capture.is_rtsp:
            return False
        reconnect_attempts = int(self.input_cfg.get("reconnect_attempts", 3))
        backoff_seconds = float(self.input_cfg.get("reconnect_backoff_seconds", 1.0))
        for attempt in range(1, reconnect_attempts + 1):
            LOGGER.warning("Frame read timeout/failure threshold hit. Reconnecting RTSP source (attempt %s/%s).", attempt, reconnect_attempts)
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)
            if capture.reopen():
                LOGGER.info("Successfully reconnected RTSP source.")
                self.health_monitor.mark_reconnect(successful=True)
                return True
            self.health_monitor.mark_reconnect(successful=False)
        return False

    def _resize_for_inference(self, frame):
        resize_width = int(self.runtime_cfg.get("resize_width", 0))
        resize_height = int(self.runtime_cfg.get("resize_height", 0))
        if resize_width <= 0 and resize_height <= 0:
            return frame
        height, width = frame.shape[:2]
        if resize_width > 0 and resize_height > 0:
            return cv2.resize(frame, (resize_width, resize_height))
        if resize_width > 0:
            scale = resize_width / max(width, 1)
            return cv2.resize(frame, (resize_width, max(1, int(height * scale))))
        scale = resize_height / max(height, 1)
        return cv2.resize(frame, (max(1, int(width * scale)), resize_height))

    def _should_skip_frame(self, frame_index: int, stage_stats: RollingTimingStats) -> bool:
        frame_skip = int(self.runtime_cfg.get("frame_skip", 0))
        if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
            return True
        if not bool(self.runtime_cfg.get("adaptive_frame_skip", False)):
            return False
        target_fps = float(self.runtime_cfg.get("target_processing_fps", 10.0))
        target_frame_seconds = 1.0 / max(target_fps, 1e-6)
        processing_seconds = sum(stage_stats.summary_ms().values()) / 1000.0
        if processing_seconds <= target_frame_seconds:
            return False
        return frame_index % 2 == 1

    def _scale_detections_to_frame(self, detections: list[Detection], inference_frame, original_frame) -> list[Detection]:
        if inference_frame.shape[:2] == original_frame.shape[:2]:
            return detections
        src_height, src_width = inference_frame.shape[:2]
        dst_height, dst_width = original_frame.shape[:2]
        scale_x = dst_width / max(src_width, 1)
        scale_y = dst_height / max(src_height, 1)
        scaled: list[Detection] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            scaled.append(
                Detection(
                    bbox=(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                    score=detection.score,
                    class_id=detection.class_id,
                    label=detection.label,
                )
            )
        return scaled

    def _build_ground_plane_mapper(self, perspective_cfg: dict[str, Any]) -> GroundPlaneMapper | None:
        if not bool(perspective_cfg.get("enabled", False)):
            return None
        return GroundPlaneMapper.from_correspondences(
            image_points=[tuple(point) for point in perspective_cfg.get("image_points", [])],
            world_points=[tuple(point) for point in perspective_cfg.get("world_points", [])],
        )

    def _attach_projected_zone_geometry(self) -> None:
        if self.ground_plane is None:
            return
        for zone in polygon_zones(self.zones):
            if zone.world_points:
                continue
            zone.world_points = self.ground_plane.project_points(zone.points)
            zone.__post_init__()

    def _project_tracks_to_ground_plane(self, tracks: list[Track]) -> None:
        if self.ground_plane is None:
            return
        for track in tracks:
            track.world_center = self.ground_plane.project_point(track.footpoint)

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
                zones=polygon_zones(self.zones, tag="restricted"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.loitering.evaluate(
                tracks=tracks,
                zones=polygon_zones(self.zones, tag="restricted"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.line_crossing.evaluate(
                tracks=tracks,
                zones=line_zones(self.zones, tag="line_crossing"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.wrong_way.evaluate(
                tracks=tracks,
                zones=polygon_zones(self.zones, tag="restricted"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.after_hours.evaluate(
                tracks=tracks,
                zones=polygon_zones(self.zones, tag="after_hours"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        events.extend(
            self.vehicle_in_pedestrian_zone.evaluate(
                tracks=tracks,
                zones=polygon_zones(self.zones, tag="pedestrian_only"),
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                camera_id=self.camera_id,
            )
        )
        occupancy = self._zone_occupancy_summary(tracks)
        for event in events:
            if event["track_id"] is not None:
                matched_track = next((track for track in tracks if track.track_id == event["track_id"]), None)
                if matched_track and matched_track.world_center is not None:
                    event["world_position"] = [round(matched_track.world_center[0], 3), round(matched_track.world_center[1], 3)]
            zone_occupancy = occupancy.get(event["zone"])
            if zone_occupancy:
                event.update(zone_occupancy)
            self._active_alerts.append({**event, "expires_at": timestamp.timestamp() + 3.0})
        return events

    def _dispatch_alert(self, event: dict[str, Any]) -> None:
        if not self.dashboard_cfg.get("enabled", False):
            return

        endpoint = self.dashboard_cfg.get("endpoint")
        if not endpoint:
            LOGGER.warning("Dashboard dispatch is enabled but no endpoint is configured.")
            return

        payload = json.dumps(event).encode("utf-8")
        req = request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            request.urlopen(req, timeout=float(self.dashboard_cfg.get("timeout_seconds", 1.0))).read()
        except Exception:
            LOGGER.exception("Failed to dispatch alert %s to dashboard endpoint %s", event.get("event_id"), endpoint)

    def _update_track_history(self, tracks: list[Track]) -> None:
        active_ids = {track.track_id for track in tracks}
        for track in tracks:
            history = self._track_history.setdefault(track.track_id, deque(maxlen=24))
            history.append((int(track.center[0]), int(track.center[1])))
        for track_id in list(self._track_history):
            if track_id not in active_ids:
                del self._track_history[track_id]

    def _update_dwell_timers(self, tracks: list[Track], timestamp: datetime) -> dict[int, float]:
        dwell_timers: dict[int, float] = {}
        active_keys: set[tuple[str, int]] = set()
        for zone in polygon_zones(self.zones, tag="restricted"):
            for track in tracks:
                if not zone.contains_track(track):
                    continue
                key = (zone.name, track.track_id)
                active_keys.add(key)
                entry_ts = self._dwell_entry_ts.setdefault(key, timestamp)
                dwell_timers[track.track_id] = max(
                    dwell_timers.get(track.track_id, 0.0),
                    (timestamp - entry_ts).total_seconds(),
                )
        for key in list(self._dwell_entry_ts):
            if key not in active_keys:
                del self._dwell_entry_ts[key]
        return dwell_timers

    def _current_active_alerts(self, timestamp: datetime) -> list[dict[str, Any]]:
        now_ts = timestamp.timestamp()
        self._active_alerts = [event for event in self._active_alerts if event.get("expires_at", 0.0) >= now_ts]
        return list(self._active_alerts)

    def _zone_occupancy_summary(self, tracks: list[Track]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for zone in polygon_zones(self.zones):
            occupants = [track for track in tracks if zone.contains_track(track)]
            zone_summary: dict[str, Any] = {"zone_occupancy_count": len(occupants)}
            if zone.world_area and zone.world_area > 0:
                zone_summary["zone_density_per_100_units2"] = round((len(occupants) / zone.world_area) * 100.0, 3)
                zone_summary["zone_world_area"] = round(zone.world_area, 3)
            summary[zone.name] = zone_summary
        return summary
