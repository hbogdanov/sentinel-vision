from datetime import datetime, timezone

from src.events.after_hours import AfterHoursOccupancyDetector
from src.events.line_crossing import LineCrossingDetector
from src.events.vehicle_zone import VehicleInPedestrianZoneDetector
from src.events.wrong_way import WrongWayDetector
from src.events.zones import LineZone, PolygonZone
from src.inference.tracker import Track


def test_line_crossing_emits_when_track_crosses_configured_direction() -> None:
    detector = LineCrossingDetector(enabled=True, cooldown_seconds=3, direction="b_to_a")
    zone = LineZone("tripwire", (0, 5), (10, 5), tags=("line_crossing",))
    before = Track(track_id=1, bbox=(2, 1, 4, 3), label="person", score=0.9, last_seen_frame=0)
    after = Track(track_id=1, bbox=(2, 7, 4, 9), label="person", score=0.9, last_seen_frame=1)
    timestamp = datetime(2026, 3, 8, tzinfo=timezone.utc)

    first = detector.evaluate([before], [zone], 0, timestamp, 30.0, "cam_1")
    second = detector.evaluate([after], [zone], 1, timestamp, 30.0, "cam_1")

    assert first == []
    assert len(second) == 1
    assert second[0]["event_type"] == "line_crossing"


def test_wrong_way_detects_motion_against_expected_direction() -> None:
    detector = WrongWayDetector(enabled=True, cooldown_seconds=5, min_displacement_pixels=4, target_classes=["person"])
    zone = PolygonZone(
        "lane",
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        tags=("restricted",),
        metadata={"expected_direction": "right"},
    )
    start = Track(track_id=7, bbox=(12, 5, 16, 9), label="person", score=0.92, last_seen_frame=0)
    moved_left = Track(track_id=7, bbox=(4, 5, 8, 9), label="person", score=0.92, last_seen_frame=1)
    timestamp = datetime(2026, 3, 8, tzinfo=timezone.utc)

    early = detector.evaluate([start], [zone], 0, timestamp, 30.0, "cam_1")
    late = detector.evaluate([moved_left], [zone], 1, timestamp, 30.0, "cam_1")

    assert early == []
    assert len(late) == 1
    assert late[0]["event_type"] == "wrong_way"


def test_after_hours_occupancy_fires_outside_allowed_window() -> None:
    detector = AfterHoursOccupancyDetector(
        enabled=True,
        cooldown_seconds=60,
        start_time="08:00",
        end_time="18:00",
        timezone_name="America/New_York",
        target_classes=["person"],
    )
    zone = PolygonZone("lab", [(0, 0), (10, 0), (10, 10), (0, 10)], tags=("after_hours",))
    track = Track(track_id=2, bbox=(1, 1, 9, 9), label="person", score=0.95, last_seen_frame=0)
    timestamp = datetime(2026, 3, 8, 2, 0, tzinfo=timezone.utc)

    events = detector.evaluate([track], [zone], 0, timestamp, 30.0, "cam_1")

    assert len(events) == 1
    assert events[0]["event_type"] == "after_hours_occupancy"


def test_vehicle_in_pedestrian_zone_fires_on_vehicle_entry() -> None:
    detector = VehicleInPedestrianZoneDetector(enabled=True, cooldown_seconds=5, target_classes=["car"])
    zone = PolygonZone("walkway", [(0, 0), (10, 0), (10, 10), (0, 10)], tags=("pedestrian_only",))
    outside = Track(track_id=3, bbox=(20, 20, 30, 30), label="car", score=0.88, last_seen_frame=0)
    inside = Track(track_id=3, bbox=(1, 1, 9, 9), label="car", score=0.88, last_seen_frame=1)
    timestamp = datetime(2026, 3, 8, tzinfo=timezone.utc)

    first = detector.evaluate([outside], [zone], 0, timestamp, 30.0, "cam_1")
    second = detector.evaluate([inside], [zone], 1, timestamp, 30.0, "cam_1")

    assert first == []
    assert len(second) == 1
    assert second[0]["event_type"] == "vehicle_in_pedestrian_zone"
