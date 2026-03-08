from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    Point = None
    Polygon = None


@dataclass(slots=True)
class PolygonZone:
    name: str
    points: list[tuple[float, float]]
    _polygon: object = None

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"Zone '{self.name}' must have at least 3 points.")
        self._polygon = Polygon(self.points) if Polygon is not None else None
        if self._polygon is not None and not self._polygon.is_valid:
            raise ValueError(f"Zone '{self.name}' is invalid.")

    def contains_point(self, point: tuple[float, float]) -> bool:
        if self._polygon is not None and Point is not None:
            point_geom = Point(point)
            return self._polygon.contains(point_geom) or self._polygon.touches(point_geom)
        return _point_in_polygon(point, self.points)


def load_zones(zone_configs: Iterable[dict]) -> list[PolygonZone]:
    return [PolygonZone(name=cfg["name"], points=[tuple(point) for point in cfg["points"]]) for cfg in zone_configs]


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for idx in range(n):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % n]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1)
        if intersects:
            inside = not inside
    return inside


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > 1e-6:
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= 0
