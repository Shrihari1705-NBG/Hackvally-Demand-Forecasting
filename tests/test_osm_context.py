import importlib.util
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon


MODULE_PATH = Path(__file__).parents[1] / "src" / "07_extract_osm_context.py"
spec = importlib.util.spec_from_file_location("osm_context", MODULE_PATH)
osm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osm)

CELLS = Path(__file__).parents[1] / "data/processed/synthetic_h3_cells.geojson"


def config(tmp_path, enabled=True):
    return {"metric_crs": "EPSG:32643", "osm_context": {"enabled": enabled, "required": False,
            "retrieval_timeout_seconds": 1, "max_retries": 2, "cache_enabled": True,
            "cache_path": str(tmp_path / "cache.geojson"), "include_original_tags": True}}


def fake_features():
    cell = osm.load_cells(CELLS).iloc[0]
    center = Point(float(cell.center_lon), float(cell.center_lat))
    outside = Point(74.9, 16.2)
    polygon = Polygon([(center.x - .001, center.y - .001), (center.x + .001, center.y - .001),
                       (center.x + .001, center.y + .001), (center.x - .001, center.y + .001), (center.x - .001, center.y - .001)])
    return gpd.GeoDataFrame([
        {"element": "node", "id": 1, "amenity": "hospital", "name": "Test Hospital", "geometry": center},
        {"element": "way", "id": 2, "landuse": "industrial", "name": "Test Factory", "geometry": polygon},
        {"element": "node", "id": 3, "amenity": "restaurant", "name": None, "geometry": outside},
    ], crs="EPSG:4326")


def test_extraction_disabled_zero_fills_cells(tmp_path):
    result = osm.extract_osm_context(config(tmp_path, enabled=False), cells_path=CELLS, output_dir=tmp_path, write_outputs=False)
    assert result["metadata"]["context_source_status"] == "disabled"
    assert len(result["aggregation"]) == 94
    assert result["aggregation"]["context_data_available"].eq(False).all()


def test_mocked_extraction_assigns_points_and_polygons(tmp_path):
    result = osm.extract_osm_context(config(tmp_path), cells_path=CELLS, output_dir=tmp_path,
                                     fetcher=lambda: fake_features(), write_outputs=False)
    assert result["metadata"]["context_source_status"] == "success"
    assert len(result["pois"]) == 1
    assert len(result["context"]) == 1
    assert result["metadata"]["excluded_feature_count"] == 1
    assert result["pois"][0]["h3_index"] in set(osm.load_cells(CELLS).h3_index)
    assert result["context"][0]["osm_element_type"] == "way"
    assert result["aggregation"]["context_data_available"].eq(True).all()


@pytest.mark.parametrize("error", [TimeoutError("timeout"), RuntimeError("HTTP 502"), RuntimeError("HTTP 429")])
def test_overpass_failures_are_recorded(error, tmp_path):
    def fail():
        raise error
    result = osm.extract_osm_context(config(tmp_path), cells_path=CELLS, output_dir=tmp_path, fetcher=fail, write_outputs=False)
    assert result["metadata"]["context_source_status"] == "unavailable"
    assert result["metadata"]["errors"]
    assert len(result["aggregation"]) == 94


def test_cache_reuse_avoids_fetch(tmp_path):
    cache = tmp_path / "cache.geojson"
    fake_features().to_file(cache, driver="GeoJSON")
    cfg = config(tmp_path)
    result = osm.extract_osm_context(cfg, cells_path=CELLS, output_dir=tmp_path, write_outputs=False)
    assert result["metadata"]["context_source_status"] == "cache_reused"
    assert result["metadata"]["cache_status"] == "reused"


def test_h3_ids_are_preserved_and_zero_counts_exist(tmp_path):
    result = osm.extract_osm_context(config(tmp_path), cells_path=CELLS,
                                     output_dir=tmp_path, fetcher=lambda: fake_features(), write_outputs=True)
    original = set(osm.load_cells(CELLS).h3_index)
    assert set(result["aggregation"].h3_index) == original
    assert (result["aggregation"]["total_poi_count"] == 0).any()
    written = pd.read_csv(tmp_path / "h3_poi_context.csv")
    assert len(written) == 94
    assert "context_data_available" in written.columns

