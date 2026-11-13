import importlib.util
import json
from pathlib import Path

import h3
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "06_generate_synthetic_geography.py"
spec = importlib.util.spec_from_file_location("synthetic_geography", MODULE_PATH)
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)


def test_anchor_conversion_and_resolution():
    anchors = synthetic.load_anchors()
    cells = [h3.latlng_to_cell(a["center_lat"], a["center_lon"], 8) for a in anchors]
    assert len(cells) == 8
    assert all(h3.is_valid_cell(cell) and h3.get_resolution(cell) == 8 for cell in cells)


def test_generation_is_unique_and_deterministic():
    first = synthetic.generate_synthetic_geography(write_outputs=False)
    second = synthetic.generate_synthetic_geography(write_outputs=False)
    first_ids = [record["h3_index"] for record in first["records"]]
    second_ids = [record["h3_index"] for record in second["records"]]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first["metadata"]["random_seed"] == 42


def test_region_assignment_and_tie_breaking():
    anchors = [{"region_id": "R02", "region_name": "B", "center_lat": 0.0, "center_lon": 0.0},
               {"region_id": "R01", "region_name": "A", "center_lat": 0.0, "center_lon": 0.0}]
    region_id, distance = synthetic.assign_nearest_anchor(0.0, 0.0, anchors)
    assert region_id == "R01"
    assert distance == 0.0


def test_polygon_geometry_and_output_schema(tmp_path):
    result = synthetic.generate_synthetic_geography(output_dir=tmp_path, write_maps=False)
    assert all(record["geometry"].is_valid for record in result["records"])
    geojson = json.loads((tmp_path / "synthetic_h3_cells.geojson").read_text())
    assert geojson["type"] == "FeatureCollection"
    required = {"h3_index", "region_id", "center_lat", "center_lon", "is_synthetic", "geography_mode"}
    assert required <= set(geojson["features"][0]["properties"])


def test_missing_anchor_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        synthetic.load_anchors(tmp_path / "missing.json")


def test_invalid_coordinates(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps({"regions": [{"region_id": f"R{i:02d}", "region_name": "x", "center_lat": 100, "center_lon": 74} for i in range(1, 9)]}))
    with pytest.raises(ValueError, match="Invalid coordinates"):
        synthetic.load_anchors(path)


def test_multiple_disconnected_components_are_reported():
    result = synthetic.generate_synthetic_geography(write_outputs=False)
    assert result["metadata"]["connected_component_count"] >= 1
    assert "connected_component_sizes" in result["metadata"]

