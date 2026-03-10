from __future__ import annotations

from dataclasses import dataclass, field
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
    world_points: list[tuple[float, float]] | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    _polygon: object = None
    _world_polygon: object = None

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"Zone '{self.name}' must have at least 3 points.")
        self._polygon = Polygon(self.points) if Polygon is not None else None
        if self._polygon is not None and not self._polygon.is_valid:
            raise ValueError(f"Zone '{self.name}' is invalid.")
        if self.world_points:
            if len(self.world_points) < 3:
                raise ValueError(
                    f"Zone '{self.name}' world polygon must have at least 3 points."
                )
            self._world_polygon = (
                Polygon(self.world_points) if Polygon is not None else None
            )
            if self._world_polygon is not None and not self._world_polygon.is_valid:
                raise ValueError(f"Zone '{self.name}' world polygon is invalid.")

    def contains_point(self, point: tuple[float, float], space: str = "image") -> bool:
        polygon = (
            self._world_polygon
            if space == "world" and self._world_polygon is not None
            else self._polygon
        )
        points = (
            self.world_points if space == "world" and self.world_points else self.points
        )
        if polygon is not None and Point is not None:
            point_geom = Point(point)
            return polygon.contains(point_geom) or polygon.touches(point_geom)
        return _point_in_polygon(point, points)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    @property
    def reasoning_space(self) -> str:
        return "world" if self.world_points else "image"

    @property
    def world_area(self) -> float | None:
        if self._world_polygon is not None:
            return float(self._world_polygon.area)
        if self.world_points:
            return _polygon_area(self.world_points)
        return None

    def contains_track(self, track) -> bool:
        if self.world_points and getattr(track, "world_center", None) is not None:
            return self.contains_point(track.world_center, space="world")
        return self.contains_point(track.center, space="image")


@dataclass(slots=True)
class LineZone:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def side_of_point(self, point: tuple[float, float]) -> float:
        px, py = point
        x1, y1 = self.start
        x2, y2 = self.end
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


Zone = PolygonZone | LineZone


def load_zones(zone_configs: Iterable[dict]) -> list[Zone]:
    zones: list[Zone] = []
    for cfg in zone_configs:
        zone_type = str(cfg.get("type", "polygon")).lower()
        tags = tuple(str(tag) for tag in cfg.get("tags", []))
        metadata = dict(cfg.get("metadata", {}))
        if zone_type == "line":
            points = [tuple(point) for point in cfg["points"]]
            if len(points) != 2:
                raise ValueError(
                    f"Line zone '{cfg['name']}' must have exactly 2 points."
                )
            zones.append(
                LineZone(
                    name=cfg["name"],
                    start=points[0],
                    end=points[1],
                    tags=tags,
                    metadata=metadata,
                )
            )
            continue
        world_points = cfg.get("world_points")
        zones.append(
            PolygonZone(
                name=cfg["name"],
                points=[tuple(point) for point in cfg["points"]],
                world_points=(
                    [tuple(point) for point in world_points] if world_points else None
                ),
                tags=tags,
                metadata=metadata,
            )
        )
    return zones


def polygon_zones(zones: Iterable[Zone], tag: str | None = None) -> list[PolygonZone]:
    filtered = [zone for zone in zones if isinstance(zone, PolygonZone)]
    if tag is not None:
        filtered = [zone for zone in filtered if zone.has_tag(tag)]
    return filtered


def line_zones(zones: Iterable[Zone], tag: str | None = None) -> list[LineZone]:
    filtered = [zone for zone in zones if isinstance(zone, LineZone)]
    if tag is not None:
        filtered = [zone for zone in filtered if zone.has_tag(tag)]
    return filtered


def _point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for idx in range(n):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % n]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
        )
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


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5
