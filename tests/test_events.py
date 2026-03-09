from datetime import datetime, timedelta, timezone

from src.events.intrusion import IntrusionDetector
from src.events.loitering import LoiteringDetector
from src.events.zones import PolygonZone
from src.inference.tracker import Track


def test_intrusion_fires_only_on_entry() -> None:
    detector = IntrusionDetector(enabled=True, cooldown_seconds=5)
    zone = PolygonZone("restricted", [(0, 0), (10, 0), (10, 10), (0, 10)])
    outside_track = Track(track_id=1, bbox=(20, 20, 30, 30), label="person", score=0.9, last_seen_frame=0)
    inside_track = Track(track_id=1, bbox=(1, 1, 9, 9), label="person", score=0.9, last_seen_frame=1)
    timestamp = datetime(2026, 3, 8, tzinfo=timezone.utc)

    first = detector.evaluate([outside_track], [zone], 0, timestamp, 30.0, "cam_1")
    second = detector.evaluate([inside_track], [zone], 1, timestamp + timedelta(seconds=1), 30.0, "cam_1")
    third = detector.evaluate([inside_track], [zone], 2, timestamp + timedelta(seconds=2), 30.0, "cam_1")

    assert first == []
    assert len(second) == 1
    assert third == []


def test_loitering_requires_dwell_time() -> None:
    detector = LoiteringDetector(enabled=True, threshold_seconds=3, cooldown_seconds=10)
    zone = PolygonZone("restricted", [(0, 0), (10, 0), (10, 10), (0, 10)])
    track = Track(track_id=5, bbox=(1, 1, 9, 9), label="person", score=0.95, last_seen_frame=0)
    start = datetime(2026, 3, 8, tzinfo=timezone.utc)

    early = detector.evaluate([track], [zone], 0, start, 30.0, "cam_1")
    late = detector.evaluate([track], [zone], 120, start + timedelta(seconds=4), 30.0, "cam_1")

    assert early == []
    assert len(late) == 1
    assert late[0]["event_type"] == "loitering"


def test_intrusion_cooldown_allows_retrigger_after_exit_and_expiry() -> None:
    detector = IntrusionDetector(enabled=True, cooldown_seconds=2)
    zone = PolygonZone("restricted", [(0, 0), (10, 0), (10, 10), (0, 10)])
    outside_track = Track(track_id=1, bbox=(20, 20, 30, 30), label="person", score=0.9, last_seen_frame=0)
    inside_track = Track(track_id=1, bbox=(1, 1, 9, 9), label="person", score=0.9, last_seen_frame=1)
    start = datetime(2026, 3, 8, tzinfo=timezone.utc)

    first = detector.evaluate([inside_track], [zone], 0, start, 30.0, "cam_1")
    still_inside = detector.evaluate([inside_track], [zone], 1, start + timedelta(seconds=1), 30.0, "cam_1")
    exited = detector.evaluate([outside_track], [zone], 2, start + timedelta(seconds=2), 30.0, "cam_1")
    retrigger = detector.evaluate([inside_track], [zone], 3, start + timedelta(seconds=3), 30.0, "cam_1")

    assert len(first) == 1
    assert still_inside == []
    assert exited == []
    assert len(retrigger) == 1


def test_loitering_uses_world_space_when_available() -> None:
    detector = LoiteringDetector(enabled=True, threshold_seconds=1, cooldown_seconds=10)
    zone = PolygonZone(
        "restricted",
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        world_points=[(0, 0), (5, 0), (5, 5), (0, 5)],
    )
    track = Track(
        track_id=5,
        bbox=(20, 20, 30, 30),
        label="person",
        score=0.95,
        last_seen_frame=0,
        world_center=(2.0, 2.0),
    )
    start = datetime(2026, 3, 8, tzinfo=timezone.utc)

    early = detector.evaluate([track], [zone], 0, start, 30.0, "cam_1")
    late = detector.evaluate([track], [zone], 30, start + timedelta(seconds=2), 30.0, "cam_1")

    assert early == []
    assert len(late) == 1
