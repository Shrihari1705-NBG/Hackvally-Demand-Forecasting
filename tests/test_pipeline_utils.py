import numpy as np
import pytest
import h3
from shapely.geometry import Point, Polygon, MultiPolygon

from pipeline_utils import cells_intersecting_boundary, parse_bool, polygonal_geometry, sample_point_in_cell, validate_resolution

def test_polygon_and_multipolygon_boundaries():
    polygon = Polygon([(74.5, 15.9), (74.51, 15.9), (74.51, 15.91), (74.5, 15.91)])
    assert polygonal_geometry(polygon).geom_type == "Polygon"
    multi = MultiPolygon([polygon, Polygon([(74.52, 15.9), (74.53, 15.9), (74.53, 15.91), (74.52, 15.91)])])
    assert polygonal_geometry(multi).geom_type == "MultiPolygon"

def test_invalid_geometry_is_rejected():
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    with pytest.raises(ValueError): polygonal_geometry(invalid)

def test_boolean_parser_is_explicit():
    assert parse_bool("False") is False
    assert parse_bool("true") is True
    with pytest.raises(ValueError): parse_bool("maybe")

def test_h3_cell_membership_and_boundary_sampling():
    cell = h3.latlng_to_cell(15.85, 74.5, 8)
    boundary = Polygon([(74.45, 15.80), (74.55, 15.80), (74.55, 15.90), (74.45, 15.90)])
    lat, lon = sample_point_in_cell(cell, boundary, np.random.default_rng(42))
    assert h3.latlng_to_cell(lat, lon, 8) == cell
    assert boundary.covers(Point(lon, lat))
    validate_resolution(cell, 8)

def test_boundary_intersecting_cells_include_partial_cells():
    boundary = Polygon([(74.49, 15.84), (74.505, 15.84), (74.505, 15.855), (74.49, 15.855)])
    cells = cells_intersecting_boundary(boundary, 8)
    assert cells
    assert all(h3.is_valid_cell(c) and h3.get_resolution(c) == 8 for c in cells)
