"""Optional OpenStreetMap POI/context enrichment for synthetic H3 geography."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import geopandas as gpd
import h3
import pandas as pd
try:
    import osmnx as ox
except ImportError:  # OSMnx is optional when disabled or when a cache is reused.
    ox = None
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

from pipeline_utils import validate_resolution

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_CELLS = ROOT / "data/processed/synthetic_h3_cells.geojson"

CATEGORY_FIELDS = ["transport", "healthcare", "education", "commercial", "food_hospitality", "industrial", "recreation"]
POI_COLUMNS = ["osm_element_type", "osm_element_id", "osm_feature_id", "category", "subcategory", "name", "latitude", "longitude", "h3_index", "source", "retrieval_timestamp", "is_synthetic", "osm_tags_json"]
OSM_TAGS = {
    "amenity": ["hospital", "clinic", "doctors", "pharmacy", "school", "college", "university", "bus_station",
                "marketplace", "restaurant", "cafe", "fast_food", "pub", "bar", "hotel", "department_store", "market"],
    "shop": True,
    "shopping": True,
    "railway": ["station", "halt", "tram_stop", "subway_entrance"],
    "highway": ["bus_stop"],
    "public_transport": ["station", "stop_position", "platform"],
    "aeroway": ["aerodrome", "terminal"],
    "man_made": ["works", "warehouse"],
    "tourism": ["attraction", "hotel", "museum", "theme_park"],
    "leisure": ["park", "sports_centre", "stadium", "pitch", "fitness_centre", "track", "water_park"],
    "landuse": ["industrial", "commercial", "retail"],
    "building": ["industrial", "warehouse", "commercial", "retail", "school", "hospital"],
}


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    return str(value)


def _tags_json(row: pd.Series, include_original_tags: bool) -> str:
    if not include_original_tags:
        return "{}"
    return json.dumps({str(k): _json_value(v) for k, v in row.items() if k != "geometry"}, ensure_ascii=False, sort_keys=True)


def load_cells(path: Path | str = DEFAULT_CELLS) -> gpd.GeoDataFrame:
    cells = gpd.read_file(path)
    if cells.crs is None or cells.crs.to_epsg() != 4326:
        raise ValueError("Synthetic H3 geography must have an explicit EPSG:4326 CRS")
    required = {"h3_index", "region_id", "region_name", "center_lat", "center_lon"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Synthetic H3 geography is missing columns: {sorted(missing)}")
    if cells["h3_index"].duplicated().any():
        raise ValueError("Synthetic H3 geography contains duplicate H3 IDs")
    for cell in cells["h3_index"]:
        validate_resolution(str(cell), 8)
    return cells


def classify_feature(row: pd.Series) -> tuple[str | None, str | None]:
    def value(key):
        item = row.get(key)
        return str(item).lower() if item is not None and str(item).lower() != "nan" else ""
    amenity, shop, railway = value("amenity"), value("shop"), value("railway")
    highway, public_transport = value("highway"), value("public_transport")
    tourism, leisure = value("tourism"), value("leisure")
    landuse, building = value("landuse"), value("building")
    aeroway, man_made = value("aeroway"), value("man_made")
    if railway in {"station", "halt", "tram_stop", "subway_entrance"} or highway == "bus_stop" or public_transport in {"station", "stop_position", "platform"} or amenity == "bus_station" or aeroway in {"aerodrome", "terminal"}:
        return "transport", railway or highway or public_transport or amenity or aeroway
    if amenity in {"hospital", "clinic", "doctors", "pharmacy"}:
        return "healthcare", amenity
    if amenity in {"school", "college", "university"}:
        return "education", amenity
    if shop or amenity in {"marketplace", "department_store", "market"}:
        return "commercial", shop or amenity
    if amenity in {"restaurant", "cafe", "fast_food", "pub", "bar"} or tourism == "hotel":
        return "food_hospitality", amenity or tourism
    if landuse == "industrial" or building in {"industrial", "warehouse"} or man_made in {"works", "warehouse"}:
        return "industrial", landuse or building or man_made
    if leisure in {"park", "sports_centre", "stadium", "pitch", "fitness_centre", "track", "water_park"} or tourism in {"attraction", "museum", "theme_park"}:
        return "recreation", leisure or tourism
    if landuse in {"commercial", "retail"} or building in {"commercial", "retail"}:
        return "commercial", landuse or building
    return None, None


def _feature_identity(row: pd.Series) -> tuple[str, str]:
    element_type = str(row.get("element", row.get("osm_element_type", "unknown")))
    element_id = row.get("id", row.get("osm_element_id", row.get("osm_id", "unknown")))
    return element_type, str(_json_value(element_id))


def _write_empty_geojson(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [], "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}, indent=2) + "\n", encoding="utf-8")


def _cache_read(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path).to_crs(4326)


def _retrieve(bbox: tuple[float, float, float, float], tags: dict, settings: dict, fetcher: Callable | None = None) -> tuple[gpd.GeoDataFrame, dict]:
    timeout = int(settings.get("retrieval_timeout_seconds", 60))
    retries = int(settings.get("max_retries", 2))
    if fetcher is None and ox is None:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), {"status": "unavailable", "attempts": 0, "errors": ["OSMnx is not installed"]}
    fetcher = fetcher or (lambda: ox.features_from_bbox(bbox=bbox, tags=tags))
    errors = []
    for attempt in range(retries + 1):
        try:
            ox.settings.timeout = timeout
            data = fetcher()
            if data is None:
                data = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            if data.crs is None:
                data = data.set_crs(4326)
            else:
                data = data.to_crs(4326)
            return data, {"status": "success", "attempts": attempt + 1, "errors": errors}
        except Exception as exc:  # network/Overpass errors are captured for reproducibility
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), {"status": "unavailable", "attempts": retries + 1, "errors": errors}


def _assign_features(raw: gpd.GeoDataFrame, eligible: set[str], retrieval_timestamp: str, include_tags: bool) -> tuple[list[dict], list[dict], dict]:
    pois, context, excluded = [], [], 0
    for _, row in raw.reset_index(drop=False).iterrows():
        geometry = row.get("geometry")
        if geometry is None or geometry.is_empty:
            excluded += 1
            continue
        category, subcategory = classify_feature(row)
        if not category:
            continue
        representative = geometry if geometry.geom_type == "Point" else geometry.representative_point()
        cell = h3.latlng_to_cell(float(representative.y), float(representative.x), 8)
        if cell not in eligible:
            excluded += 1
            continue
        element_type, element_id = _feature_identity(row)
        record = {"osm_element_type": element_type, "osm_element_id": element_id,
                  "osm_feature_id": f"{element_type}/{element_id}", "category": category, "subcategory": subcategory,
                  "name": row.get("name") if pd.notna(row.get("name")) else None,
                  "latitude": float(representative.y), "longitude": float(representative.x), "h3_index": cell,
                  "source": "OpenStreetMap", "retrieval_timestamp": retrieval_timestamp, "is_synthetic": False,
                  "osm_tags_json": _tags_json(row, include_tags), "geometry": geometry}
        (pois if geometry.geom_type in {"Point", "MultiPoint"} else context).append(record)
    return pois, context, {"excluded_feature_count": excluded}


def _aggregate(cells: gpd.GeoDataFrame, pois: list[dict], context: list[dict], status: str) -> pd.DataFrame:
    counts = {cell: {f"{category}_count": 0 for category in CATEGORY_FIELDS} | {"total_poi_count": 0, "context_feature_count": 0} for cell in cells["h3_index"]}
    for item in pois:
        counts[item["h3_index"]][f"{item['category']}_count"] += 1
        counts[item["h3_index"]]["total_poi_count"] += 1
    for item in context:
        counts[item["h3_index"]]["context_feature_count"] += 1
    rows = []
    for _, cell in cells.sort_values("h3_index").iterrows():
        row = {"h3_index": cell["h3_index"], "region_id": cell["region_id"], "region_name": cell["region_name"],
               "context_data_available": status in {"success", "cache_reused"}, "context_source_status": status}
        row.update(counts[cell["h3_index"]])
        rows.append(row)
    return pd.DataFrame(rows)


def extract_osm_context(config: dict | None = None, cells_path: Path | str = DEFAULT_CELLS, output_dir: Path | str | None = None,
                        fetcher: Callable | None = None, write_outputs: bool = True) -> dict:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    settings = config.get("osm_context", {})
    cells = load_cells(cells_path)
    output_dir = Path(output_dir or ROOT / "data/processed")
    raw_dir = ROOT / "data/raw" if output_dir == ROOT / "data/processed" else output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    eligible = set(cells["h3_index"].astype(str))
    bounds = tuple(gpd.GeoSeries(cells.geometry, crs=4326).total_bounds.tolist())
    status, errors, cache_status = "disabled", [], "not_used"
    raw = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    cache_path = ROOT / settings.get("cache_path", "data/cache/osm_context_features.geojson")
    if settings.get("enabled", False):
        if settings.get("cache_enabled", True) and cache_path.exists() and fetcher is None:
            raw, status, cache_status = _cache_read(cache_path), "cache_reused", "reused"
        else:
            raw, result = _retrieve(bounds, OSM_TAGS, settings, fetcher=fetcher)
            status, errors = result["status"], result["errors"]
            if status == "success" and settings.get("cache_enabled", True):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                raw.to_file(cache_path, driver="GeoJSON")
                cache_status = "written"
    pois, context, assignment = _assign_features(raw, eligible, retrieval_timestamp, settings.get("include_original_tags", True))
    aggregation = _aggregate(cells, pois, context, status)
    metadata = {"source": "OpenStreetMap", "enabled": bool(settings.get("enabled", False)), "required": bool(settings.get("required", False)),
                "context_source_status": status, "cache_status": cache_status, "cache_path": str(cache_path),
                "retrieval_timestamp": retrieval_timestamp, "retrieval_bounds_wgs84": {"min_lon": bounds[0], "min_lat": bounds[1], "max_lon": bounds[2], "max_lat": bounds[3]},
                "h3_resolution": 8, "synthetic_cell_count": len(cells), "raw_feature_count": len(raw), "poi_feature_count": len(pois),
                "context_feature_count": len(context), "assigned_feature_count": len(pois) + len(context),
                "excluded_feature_count": assignment["excluded_feature_count"], "errors": errors,
                "category_counts": {category: sum(1 for item in pois if item["category"] == category) for category in CATEGORY_FIELDS},
                "temporal_note": "OSM features are current static contextual inputs, not historical observations for the 180-day demand period.",
                "odbl_attribution": "© OpenStreetMap contributors; data used under the Open Database License (ODbL)."}
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        poi_path, context_path = raw_dir / "osm_pois.geojson", raw_dir / "osm_context_features.geojson"
        for path, records in ((poi_path, pois), (context_path, context)):
            if records:
                gpd.GeoDataFrame(records, geometry="geometry", crs=4326).to_file(path, driver="GeoJSON")
            else:
                _write_empty_geojson(path)
        pd.DataFrame([{k: v for k, v in item.items() if k != "geometry"} for item in pois], columns=POI_COLUMNS).to_csv(output_dir / "osm_poi_features.csv", index=False)
        aggregation.to_csv(output_dir / "h3_poi_context.csv", index=False)
        (output_dir / "osm_context_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        lines = ["# OSM Context Validation", "", f"- Status: `{status}`", f"- Cache: `{cache_status}`", f"- H3 cells: `{len(cells)}`",
                 f"- Retrieved raw features: `{len(raw)}`", f"- Point POIs assigned: `{len(pois)}`", f"- Polygon context features assigned: `{len(context)}`",
                 f"- Features excluded outside synthetic coverage or invalid: `{assignment['excluded_feature_count']}`", "", "## Category counts", ""]
        lines += [f"- {category}: {metadata['category_counts'][category]}" for category in CATEGORY_FIELDS]
        lines += ["", "Cells with no POIs are zero-filled but do not imply proven absence of facilities.", "", metadata["odbl_attribution"]]
        if errors:
            lines += ["", "## Retrieval errors", ""] + [f"- {error}" for error in errors]
        (output_dir / "osm_context_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if settings.get("required", False) and status not in {"success", "cache_reused"}:
        raise RuntimeError("OSM context is required but unavailable: " + "; ".join(errors or [status]))
    return {"metadata": metadata, "pois": pois, "context": context, "aggregation": aggregation}


def main() -> int:
    parser = argparse.ArgumentParser(description="Optionally enrich synthetic H3 geography with OSM context.")
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    args = parser.parse_args()
    result = extract_osm_context(cells_path=args.cells)
    print(json.dumps(result["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
