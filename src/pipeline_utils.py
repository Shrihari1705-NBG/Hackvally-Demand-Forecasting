"""Small, offline-testable helpers shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import h3
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry


def parse_bool(value) -> bool:
    """Parse CSV/config boolean values without treating ``'False'`` as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Boolean value cannot be null")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Unrecognised boolean value: {value!r}")


def validate_resolution(cell: str, resolution: int) -> None:
    if not isinstance(cell, str) or not h3.is_valid_cell(cell):
        raise ValueError(f"Invalid H3 cell: {cell!r}")
    actual = h3.get_resolution(cell)
    if actual != resolution:
        raise ValueError(f"H3 cell {cell} has resolution {actual}, expected {resolution}")


def polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Return a valid Polygon/MultiPolygon or raise a useful error."""
    if geometry is None or geometry.is_empty:
        raise ValueError("Geometry is empty")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
    if not geometry.is_valid:
        raise ValueError("Geometry is invalid")
    return geometry


def h3_polygon(cell: str) -> Polygon:
    ring = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
    return Polygon(ring)


def point_in_cell_and_boundary(lat: float, lon: float, cell: str, boundary: BaseGeometry) -> bool:
    point = Point(float(lon), float(lat))
    return h3.latlng_to_cell(float(lat), float(lon), h3.get_resolution(cell)) == cell and boundary.covers(point)


def sample_point_in_cell(
    cell: str,
    operational_boundary: BaseGeometry,
    rng,
    max_attempts: int = 500,
) -> tuple[float, float]:
    """Sample a point in the clipped cell and prove both containment conditions."""
    clipped = h3_polygon(cell).intersection(operational_boundary)
    if clipped.is_empty:
        raise ValueError(f"H3 cell {cell} has an empty operational intersection")
    if not clipped.is_valid:
        clipped = clipped.buffer(0)
    if clipped.is_empty or not clipped.is_valid:
        raise ValueError(f"H3 cell {cell} intersection is invalid")
    minx, miny, maxx, maxy = clipped.bounds
    for _ in range(max_attempts):
        point = Point(float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy)))
        if not clipped.covers(point):
            continue
        lat, lon = point.y, point.x
        calculated = h3.latlng_to_cell(lat, lon, h3.get_resolution(cell))
        if calculated != cell:
            continue
        if not operational_boundary.covers(point):
            continue
        return lat, lon
    raise RuntimeError(f"Could not sample a valid point for H3 cell {cell} after {max_attempts} attempts")


def cells_intersecting_boundary(boundary: BaseGeometry, resolution: int) -> set[str]:
    """Return cells whose polygons intersect the boundary.

    H3's normal polygon fill is center-based. The implementation first uses
    that fill and then expands by a conservative local disk before testing the
    actual cell polygons, retaining boundary-intersecting partial cells while
    keeping center containment as a separate attribute.
    """
    initial = set(h3.geo_to_cells(boundary, resolution))
    candidates = set(initial)
    for cell in initial:
        candidates.update(h3.grid_disk(cell, 2))
    return {cell for cell in candidates if h3_polygon(cell).intersection(boundary).area > 0}
