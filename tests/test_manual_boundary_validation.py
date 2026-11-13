import importlib.util
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


MODULE_PATH = Path(__file__).parents[1] / "src" / "05_validate_operational_boundary.py"
spec = importlib.util.spec_from_file_location("manual_boundary_validator", MODULE_PATH)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def write_geojson(path, geometry, crs="EPSG:4326"):
    gpd.GeoDataFrame({"source": ["test"]}, geometry=[geometry], crs=crs).to_file(path, driver="GeoJSON")


def test_valid_polygon(tmp_path):
    path = tmp_path / "boundary.geojson"
    write_geojson(path, Polygon([(74.47, 15.80), (74.62, 15.80), (74.62, 15.93), (74.47, 15.93), (74.47, 15.80)]))
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["geometry_checks_passed"] is True
    assert report["source_geometry_preserved"] is True


def test_valid_multipolygon_is_reported_as_disconnected(tmp_path):
    path = tmp_path / "boundary.geojson"
    geometry = MultiPolygon([
        Polygon([(74.47, 15.80), (74.49, 15.80), (74.49, 15.82), (74.47, 15.82), (74.47, 15.80)]),
        Polygon([(74.52, 15.87), (74.54, 15.87), (74.54, 15.89), (74.52, 15.89), (74.52, 15.87)]),
    ])
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["geometry_type"] == "MultiPolygon"
    assert report["component_count"] == 2
    assert report["unexpected_disconnected_components"] is True
    assert report["geometry_checks_passed"] is False


def test_invalid_geometry(tmp_path):
    path = tmp_path / "boundary.geojson"
    write_geojson(path, Polygon([(74.47, 15.80), (74.58, 15.93), (74.47, 15.93), (74.58, 15.80), (74.47, 15.80)]))
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["invalid_geometry"] is True
    assert report["geometry_checks_passed"] is False
    assert report["geometry_repair_performed"] is True


def test_missing_input_file(tmp_path):
    report = validator.validate_boundary(tmp_path / "missing.geojson", tmp_path, write_maps=False)
    assert report["source_file_exists"] is False
    assert validator.approval_ready(report) is False


def test_wrong_crs(tmp_path):
    path = tmp_path / "boundary.geojson"
    write_geojson(path, Polygon([(8300000, 1750000), (8310000, 1750000), (8310000, 1760000), (8300000, 1760000), (8300000, 1750000)]), "EPSG:3857")
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["crs_verified"] is False
    assert report["geometry_checks_passed"] is False


def test_anchor_outside_blocks_approval(tmp_path):
    path = tmp_path / "boundary.geojson"
    write_geojson(path, Polygon([(74.50, 15.84), (74.52, 15.84), (74.52, 15.87), (74.50, 15.87), (74.50, 15.84)]))
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["anchor_coverage_passed"] is False
    assert validator.approval_ready(report) is False


def test_successful_validation_still_requires_explicit_approval(tmp_path):
    path = tmp_path / "boundary.geojson"
    write_geojson(path, Polygon([(74.47, 15.80), (74.62, 15.80), (74.62, 15.93), (74.47, 15.93), (74.47, 15.80)]))
    report = validator.validate_boundary(path, tmp_path, write_maps=False)
    assert report["geometry_checks_passed"] is True
    assert report["anchor_coverage_passed"] is True
    assert report["operational_coverage_reviewed"] is False
    assert report["boundary_approved"] is False
    assert validator.approval_ready(report) is True
