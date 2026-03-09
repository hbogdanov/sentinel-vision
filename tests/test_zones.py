from src.events.zones import LineZone, PolygonZone, line_zones, load_zones, polygon_zones


def test_zone_contains_center_point() -> None:
    zone = PolygonZone("restricted_lab", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert zone.contains_point((5, 5))


def test_zone_rejects_outside_point() -> None:
    zone = PolygonZone("restricted_lab", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not zone.contains_point((20, 20))


def test_zone_accepts_point_on_edge() -> None:
    zone = PolygonZone("restricted_lab", [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert zone.contains_point((0, 5))


def test_load_zones_supports_polygon_and_line_tags() -> None:
    zones = load_zones(
        [
            {"name": "restricted_lab", "type": "polygon", "tags": ["restricted"], "points": [[0, 0], [10, 0], [10, 10]]},
            {"name": "tripwire", "type": "line", "tags": ["line_crossing"], "points": [[0, 5], [10, 5]]},
        ]
    )

    assert isinstance(polygon_zones(zones, tag="restricted")[0], PolygonZone)
    assert isinstance(line_zones(zones, tag="line_crossing")[0], LineZone)


def test_world_space_polygon_zone_contains_projected_track() -> None:
    zone = PolygonZone(
        "yard",
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        world_points=[(0, 0), (5, 0), (5, 5), (0, 5)],
    )

    class _Track:
        center = (50, 50)
        world_center = (2.5, 2.5)

    assert zone.contains_track(_Track())
