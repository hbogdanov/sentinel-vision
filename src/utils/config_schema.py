from __future__ import annotations

from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


SUPPORTED_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "backpack", "handbag", "suitcase"}


class ModelConfig(BaseModel):
    path: str = "yolo11n.pt"
    confidence: float = 0.35
    device: str = "cpu"
    classes: list[str] = Field(default_factory=lambda: ["person"])

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("model.confidence must be in the range (0, 1].")
        return value

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="model.classes")
        return value


class TrackingConfig(BaseModel):
    type: Literal["simple", "bytetrack", "botsort"] = "bytetrack"
    high_score_threshold: float = 0.5
    low_score_threshold: float = 0.1
    new_track_score_threshold: float = 0.6
    match_iou_threshold: float = 0.3
    secondary_match_iou_threshold: float = 0.15
    max_age_frames: int = 30
    min_hits: int = 2
    appearance_weight: float = 0.35
    appearance_threshold: float = 0.2

    @field_validator(
        "high_score_threshold",
        "low_score_threshold",
        "new_track_score_threshold",
        "match_iou_threshold",
        "secondary_match_iou_threshold",
        "appearance_weight",
        "appearance_threshold",
    )
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("tracking thresholds must be in the range [0, 1].")
        return value

    @field_validator("max_age_frames", "min_hits")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("tracking.max_age_frames and tracking.min_hits must be >= 1.")
        return value

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "TrackingConfig":
        if self.low_score_threshold > self.high_score_threshold:
            raise ValueError("tracking.low_score_threshold must be <= tracking.high_score_threshold.")
        return self


class IntrusionEventConfig(BaseModel):
    enabled: bool = True
    cooldown_seconds: float = 5.0

    @field_validator("cooldown_seconds")
    @classmethod
    def validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("event cooldowns must be >= 0.")
        return value


class LoiteringEventConfig(IntrusionEventConfig):
    threshold_seconds: float = 10.0

    @field_validator("threshold_seconds")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("events.loitering.threshold_seconds must be > 0.")
        return value


class LineCrossingEventConfig(IntrusionEventConfig):
    direction: Literal["any", "a_to_b", "b_to_a"] = "any"


class WrongWayEventConfig(IntrusionEventConfig):
    min_displacement_pixels: float = 25.0
    target_classes: list[str] = Field(default_factory=lambda: ["person", "car", "truck", "bus"])

    @field_validator("min_displacement_pixels")
    @classmethod
    def validate_displacement(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("events.wrong_way.min_displacement_pixels must be > 0.")
        return value

    @field_validator("target_classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="events.wrong_way.target_classes")
        return value


class AfterHoursEventConfig(IntrusionEventConfig):
    start_time: str = "08:00"
    end_time: str = "18:00"
    timezone: str = "America/New_York"
    target_classes: list[str] = Field(default_factory=lambda: ["person", "backpack", "handbag", "suitcase"])

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_hhmm(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("after-hours times must use HH:MM format.")
        hour, minute = parts
        if not hour.isdigit() or not minute.isdigit():
            raise ValueError("after-hours times must use HH:MM format.")
        if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            raise ValueError("after-hours times must use valid 24-hour values.")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"Unknown timezone '{value}'.") from exc
        return value

    @field_validator("target_classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="events.after_hours_occupancy.target_classes")
        return value


class VehicleZoneEventConfig(IntrusionEventConfig):
    target_classes: list[str] = Field(default_factory=lambda: ["car", "truck", "bus", "motorcycle", "bicycle"])

    @field_validator("target_classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="events.vehicle_in_pedestrian_zone.target_classes")
        return value


class AbandonedObjectEventConfig(IntrusionEventConfig):
    unattended_seconds: float = 20.0
    min_stationary_seconds: float = 8.0
    stationary_radius_pixels: float = 20.0
    owner_max_distance_pixels: float = 80.0
    target_classes: list[str] = Field(default_factory=lambda: ["backpack", "suitcase", "handbag", "bicycle"])
    owner_classes: list[str] = Field(default_factory=lambda: ["person"])

    @field_validator(
        "unattended_seconds",
        "min_stationary_seconds",
        "stationary_radius_pixels",
        "owner_max_distance_pixels",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("events.abandoned_object thresholds must be > 0.")
        return value

    @field_validator("target_classes")
    @classmethod
    def validate_target_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="events.abandoned_object.target_classes")
        return value

    @field_validator("owner_classes")
    @classmethod
    def validate_owner_classes(cls, value: list[str]) -> list[str]:
        _validate_supported_classes(value, field_name="events.abandoned_object.owner_classes")
        return value


class EventsConfig(BaseModel):
    intrusion: IntrusionEventConfig = Field(default_factory=IntrusionEventConfig)
    loitering: LoiteringEventConfig = Field(default_factory=LoiteringEventConfig)
    line_crossing: LineCrossingEventConfig = Field(default_factory=LineCrossingEventConfig)
    wrong_way: WrongWayEventConfig = Field(default_factory=WrongWayEventConfig)
    after_hours_occupancy: AfterHoursEventConfig = Field(default_factory=AfterHoursEventConfig)
    vehicle_in_pedestrian_zone: VehicleZoneEventConfig = Field(default_factory=VehicleZoneEventConfig)
    abandoned_object: AbandonedObjectEventConfig = Field(default_factory=AbandonedObjectEventConfig)


class InputConfig(BaseModel):
    source: int | str = 0
    read_failure_threshold: int = 30
    reconnect_attempts: int = 3
    reconnect_backoff_seconds: float = 1.0

    @field_validator("read_failure_threshold", "reconnect_attempts")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("input read/reconnect settings must be >= 0.")
        return value

    @field_validator("reconnect_backoff_seconds")
    @classmethod
    def validate_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("input.reconnect_backoff_seconds must be >= 0.")
        return value


class RuntimeConfig(BaseModel):
    class MotionCompensationConfig(BaseModel):
        enabled: bool = False
        static_camera_assumption: bool = True
        method: Literal["affine", "homography"] = "affine"
        max_corners: int = 300
        quality_level: float = 0.01
        min_distance: float = 8.0
        min_matches: int = 12
        ransac_threshold: float = 3.0
        smoothing_factor: float = 0.8

        @field_validator("max_corners", "min_matches")
        @classmethod
        def validate_positive_int(cls, value: int) -> int:
            if value < 1:
                raise ValueError("runtime.motion_compensation integer settings must be >= 1.")
            return value

        @field_validator("quality_level", "min_distance", "ransac_threshold")
        @classmethod
        def validate_positive_float(cls, value: float) -> float:
            if value <= 0:
                raise ValueError("runtime.motion_compensation positive settings must be > 0.")
            return value

        @field_validator("smoothing_factor")
        @classmethod
        def validate_smoothing_factor(cls, value: float) -> float:
            if not 0.0 <= value <= 1.0:
                raise ValueError("runtime.motion_compensation.smoothing_factor must be in [0, 1].")
            return value

    frame_skip: int = 0
    adaptive_frame_skip: bool = False
    target_processing_fps: float = 10.0
    resize_width: int = 0
    resize_height: int = 0
    timing_log_interval_frames: int = 120
    motion_compensation: MotionCompensationConfig = Field(default_factory=MotionCompensationConfig)

    @field_validator("frame_skip", "resize_width", "resize_height", "timing_log_interval_frames")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("runtime integer settings must be >= 0.")
        return value

    @field_validator("target_processing_fps")
    @classmethod
    def validate_processing_fps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("runtime.target_processing_fps must be > 0.")
        return value


class PerspectiveConfig(BaseModel):
    enabled: bool = False
    image_points: list[tuple[float, float]] = Field(default_factory=list)
    world_points: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_correspondences(self) -> "PerspectiveConfig":
        if not self.enabled:
            return self
        if len(self.image_points) < 4 or len(self.world_points) < 4:
            raise ValueError("perspective requires at least 4 image_points and 4 world_points.")
        if len(self.image_points) != len(self.world_points):
            raise ValueError("perspective image_points and world_points must have the same length.")
        _validate_points(self.image_points)
        _validate_points(self.world_points)
        return self


class CameraOverrideConfig(BaseModel):
    camera_id: str
    input: dict[str, object]
    zones: list[ZoneConfig]
    output: dict[str, object] = Field(default_factory=dict)
    dashboard: dict[str, object] = Field(default_factory=dict)
    events: dict[str, object] = Field(default_factory=dict)
    model: dict[str, object] = Field(default_factory=dict)
    tracking: dict[str, object] = Field(default_factory=dict)
    runtime: dict[str, object] = Field(default_factory=dict)
    perspective: dict[str, object] = Field(default_factory=dict)

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("camera overrides must include a non-empty camera_id.")
        return value

    @field_validator("input")
    @classmethod
    def validate_input_source(cls, value: dict[str, object]) -> dict[str, object]:
        if "source" not in value:
            raise ValueError("camera overrides must include input.source.")
        return value


class OutputConfig(BaseModel):
    show_window: bool = False
    save_annotated_video: bool = True
    annotated_video_path: str = "data/outputs/annotated.mp4"
    alerts_dir: str = "data/outputs/alerts"
    log_path: str = "data/outputs/events.jsonl"
    buffer_seconds: float = 3.0
    post_event_seconds: float = 3.0
    duplicate_suppression_seconds: float = 10.0
    clip_writer_queue_size: int = 16
    health_status_path: str = "data/outputs/camera_health.json"

    @field_validator("buffer_seconds", "post_event_seconds", "duplicate_suppression_seconds")
    @classmethod
    def validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("output clip timing values must be >= 0.")
        return value

    @field_validator("clip_writer_queue_size")
    @classmethod
    def validate_queue_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("output.clip_writer_queue_size must be >= 1.")
        return value


class DashboardConfig(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    timeout_seconds: float = 1.0

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("dashboard.timeout_seconds must be > 0.")
        return value

    @model_validator(mode="after")
    def validate_endpoint(self) -> "DashboardConfig":
        if self.enabled and not self.endpoint:
            raise ValueError("dashboard.endpoint is required when dashboard.enabled is true.")
        if self.endpoint:
            TypeAdapter(AnyHttpUrl).validate_python(self.endpoint)
        return self


class PolygonZoneConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    name: str
    points: list[tuple[float, float]]
    world_points: list[tuple[float, float]] | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("zone names must not be empty.")
        return value

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(value) < 3:
            raise ValueError("polygon zones must define at least 3 points.")
        _validate_points(value)
        return value

    @field_validator("world_points")
    @classmethod
    def validate_world_points(cls, value: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
        if value is None:
            return value
        if len(value) < 3:
            raise ValueError("polygon world_points must define at least 3 points.")
        _validate_points(value)
        return value


class LineZoneConfig(BaseModel):
    type: Literal["line"]
    name: str
    points: list[tuple[float, float]]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("zone names must not be empty.")
        return value

    @field_validator("points")
    @classmethod
    def validate_points(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(value) != 2:
            raise ValueError("line zones must define exactly 2 points.")
        _validate_points(value)
        return value


ZoneConfig = PolygonZoneConfig | LineZoneConfig


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = "default_camera"
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    perspective: PerspectiveConfig = Field(default_factory=PerspectiveConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    cameras: list[CameraOverrideConfig] = Field(default_factory=list)

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("camera_id must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_zone_names_unique(self) -> "AppConfig":
        seen: set[str] = set()
        for zone in self.zones:
            if zone.name in seen:
                raise ValueError(f"Duplicate zone name '{zone.name}' is not allowed.")
            seen.add(zone.name)
        return self


def _validate_points(points: list[tuple[float, float]]) -> None:
    for point in points:
        if len(point) != 2:
            raise ValueError("each zone point must contain exactly 2 numeric values.")
        for value in point:
            if not isinstance(value, (int, float)):
                raise ValueError("zone points must be numeric.")


def _validate_supported_classes(value: list[str], field_name: str) -> None:
    unsupported = sorted(set(value) - SUPPORTED_CLASSES)
    if unsupported:
        raise ValueError(f"{field_name} contains unsupported classes: {', '.join(unsupported)}.")
