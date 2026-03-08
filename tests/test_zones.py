from src.events.zones import PolygonZone


def test_zone_contains_center_point() -> None:
    zone = PolygonZone("restricted_lab", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert zone.contains_point((5, 5))


def test_zone_rejects_outside_point() -> None:
    zone = PolygonZone("restricted_lab", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not zone.contains_point((20, 20))
