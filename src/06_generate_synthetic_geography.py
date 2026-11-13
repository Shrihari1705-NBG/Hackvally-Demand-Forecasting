"""Generate deterministic synthetic geography from H3 neighborhoods around anchors.

The output is an analytical simulation geography. It is intentionally not an
administrative or manually approved operational boundary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, mapping

from pipeline_utils import validate_resolution

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGIONS = ROOT / "data/processed/kasucab_region_definitions.json"


def _read_json(path: Path) -> dict:
    for encoding in ("utf-8", "cp1252"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode JSON file: {path}")


def load_anchors(path: Path | str = DEFAULT_REGIONS) -> list[dict]:
    data = _read_json(Path(path))
    regions = data.get("regions", data) if isinstance(data, dict) else data
    if not isinstance(regions, list) or len(regions) != 8:
        raise ValueError(f"Expected exactly eight region anchors, found {len(regions) if isinstance(regions, list) else 0}")
    required = {"region_id", "region_name", "center_lat", "center_lon"}
    result = []
    seen = set()
    for region in regions:
        missing = required - set(region)
        if missing:
            raise ValueError(f"Anchor {region.get('region_id', '<unknown>')} is missing {sorted(missing)}")
        region_id = str(region["region_id"])
        if region_id in seen:
            raise ValueError(f"Duplicate region_id: {region_id}")
        seen.add(region_id)
        lat, lon = float(region["center_lat"]), float(region["center_lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(f"Invalid coordinates for {region_id}: ({lat}, {lon})")
        result.append({"region_id": region_id, "region_name": str(region["region_name"]),
                       "center_lat": lat, "center_lon": lon})
    return sorted(result, key=lambda x: x["region_id"])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(a)))


def assign_nearest_anchor(center_lat: float, center_lon: float, anchors: list[dict]) -> tuple[str, float]:
    """Return the nearest region, resolving equal distances by region_id."""
    distances = [(_haversine_km(center_lat, center_lon, a["center_lat"], a["center_lon"]), a["region_id"]) for a in anchors]
    return min(distances, key=lambda item: (item[0], item[1]))[1], min(distances, key=lambda item: (item[0], item[1]))[0]


def cell_polygon(cell: str) -> Polygon:
    validate_resolution(cell, 8)
    ring = [(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]
    ring.append(ring[0])
    polygon = Polygon(ring)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError(f"Invalid polygon generated for H3 cell {cell}")
    return polygon


def connected_components(cells: set[str]) -> list[list[str]]:
    remaining = set(cells)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            current = queue.popleft()
            neighbors = set(h3.grid_disk(current, 1)) & remaining
            remaining -= neighbors
            queue.extend(sorted(neighbors))
            component.extend(sorted(neighbors))
        components.append(sorted(component))
    return sorted(components, key=lambda c: c[0])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _render_png(records: list[dict], anchors: list[dict], path: Path) -> None:
    colors = {region["region_id"]: plt.cm.tab10(i % 10) for i, region in enumerate(anchors)}
    fig, ax = plt.subplots(figsize=(12, 10), dpi=180)
    for record in records:
        gpd.GeoSeries([record["geometry"]], crs="EPSG:4326").plot(ax=ax, facecolor=colors[record["region_id"]],
                                                                     edgecolor="white", linewidth=0.25, alpha=0.55)
    for region in anchors:
        ax.scatter(region["center_lon"], region["center_lat"], color=colors[region["region_id"]], edgecolor="black", s=35, zorder=4)
        ax.annotate(region["region_id"], (region["center_lon"], region["center_lat"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("SYNTHETIC OPERATIONAL GEOGRAPHY – NOT AN OFFICIAL ADMINISTRATIVE BOUNDARY", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _render_html(records: list[dict], anchors: list[dict], path: Path) -> None:
    features = []
    for record in records:
        props = {k: v for k, v in record.items() if k != "geometry"}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(record["geometry"])})
    anchor_features = [{"type": "Feature", "properties": {"region_id": a["region_id"], "region_name": a["region_name"]},
                        "geometry": {"type": "Point", "coordinates": [a["center_lon"], a["center_lat"]]}} for a in anchors]
    payload = json.dumps({"cells": {"type": "FeatureCollection", "features": features},
                          "anchors": {"type": "FeatureCollection", "features": anchor_features}})
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>KasuCab Synthetic Geography</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>#map{{height:92vh}} .warning{{font:700 16px sans-serif;color:#9b1c1c;padding:8px}}</style></head><body>
<div class="warning">SYNTHETIC OPERATIONAL GEOGRAPHY – NOT AN OFFICIAL ADMINISTRATIVE BOUNDARY</div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const data={payload}; const map=L.map('map');
const cells=L.geoJSON(data.cells,{{style:f=>{{const colors={{R01:'#1f77b4',R02:'#ff7f0e',R03:'#2ca02c',R04:'#d62728',R05:'#9467bd',R06:'#8c564b',R07:'#e377c2',R08:'#7f7f7f'}};return {{color:colors[f.properties.region_id]||'#333',weight:.5,fillOpacity:.5}}}},onEachFeature:(f,l)=>l.bindTooltip(f.properties.h3_index+' — '+f.properties.region_name)}});
const anchors=L.geoJSON(data.anchors,{{pointToLayer:(f,ll)=>L.circleMarker(ll,{{radius:6,color:'#000',fillColor:'#ffeb3b',fillOpacity:1}}),onEachFeature:(f,l)=>l.bindTooltip(f.properties.region_id+' — '+f.properties.region_name)}});
cells.addTo(map); anchors.addTo(map); L.control.layers({{}},{{'H3 cells':cells,'Region anchors':anchors}}).addTo(map); map.fitBounds(cells.getBounds());
</script></body></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def generate_synthetic_geography(config: dict | None = None, region_path: Path | str = DEFAULT_REGIONS,
                                 output_dir: Path | str | None = None, write_outputs: bool = True,
                                 write_maps: bool = True) -> dict:
    config = config or _read_json(ROOT / "config.json")
    geography = config.get("geography", {})
    mode = geography.get("mode")
    if mode != "synthetic_operational":
        raise ValueError("Synthetic geography requires geography.mode=synthetic_operational")
    resolution = int(geography.get("h3_resolution", -1))
    if resolution != 8:
        raise ValueError(f"Synthetic geography is locked to H3 resolution 8, got {resolution}")
    radius = int(geography.get("neighborhood_radius", -1))
    if radius < 0:
        raise ValueError("neighborhood_radius must be non-negative")
    if geography.get("region_assignment_method") != "nearest_anchor":
        raise ValueError("Only region_assignment_method=nearest_anchor is implemented")
    anchors = load_anchors(region_path)
    anchor_cells = {a["region_id"]: h3.latlng_to_cell(a["center_lat"], a["center_lon"], resolution) for a in anchors}
    for cell in anchor_cells.values():
        validate_resolution(cell, resolution)
    cells_by_anchor = {region_id: set(h3.grid_disk(cell, radius)) for region_id, cell in anchor_cells.items()}
    all_cells = set().union(*cells_by_anchor.values())
    if not all_cells:
        raise ValueError("No synthetic H3 cells were generated")
    anchor_by_id = {a["region_id"]: a for a in anchors}
    records, mapping_rows = [], []
    assignment_counts = Counter()
    overlap_counts = Counter()
    for cell in sorted(all_cells):
        validate_resolution(cell, resolution)
        center_lat, center_lon = h3.cell_to_latlng(cell)
        region_id, distance_km = assign_nearest_anchor(center_lat, center_lon, anchors)
        members = sorted(region_id for region_id, cell_set in cells_by_anchor.items() if cell in cell_set)
        for member in members:
            mapping_rows.append({"h3_index": cell, "anchor_region_id": member, "anchor_region_name": anchor_by_id[member]["region_name"],
                                 "anchor_h3_index": anchor_cells[member], "neighborhood_radius": radius})
        if len(members) > 1:
            overlap_counts["cells_shared_by_multiple_anchor_neighborhoods"] += 1
        assignment_counts[region_id] += 1
        records.append({"h3_index": cell, "resolution": resolution, "region_id": region_id,
                        "region_name": anchor_by_id[region_id]["region_name"], "center_lat": center_lat, "center_lon": center_lon,
                        "distance_to_region_anchor_km": distance_km, "anchor_h3_index": anchor_cells[region_id],
                        "is_synthetic": True, "geography_mode": mode, "geometry": cell_polygon(cell)})
    components = connected_components(all_cells)
    bounds = gpd.GeoSeries([r["geometry"] for r in records], crs="EPSG:4326").total_bounds.tolist()
    area_km2 = float(gpd.GeoSeries([r["geometry"] for r in records], crs="EPSG:4326").to_crs(config.get("metric_crs", "EPSG:32643")).union_all().area / 1_000_000)
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "geography_mode": mode,
                "boundary_type": geography.get("boundary_type"), "source_method": geography.get("source_method"),
                "h3_resolution": resolution, "neighborhood_radius": radius, "random_seed": geography.get("random_seed", 42),
                "region_assignment_method": geography.get("region_assignment_method"), "require_official_boundary": False,
                "anchor_count": len(anchors), "unique_anchor_h3_cells": len(set(anchor_cells.values())),
                "unique_generated_h3_cells": len(records), "anchor_h3_indices": anchor_cells,
                "cells_per_region": dict(sorted(assignment_counts.items())),
                "overlap_statistics": {"shared_cell_count": overlap_counts["cells_shared_by_multiple_anchor_neighborhoods"],
                                       "anchor_neighborhood_cell_counts": {k: len(v) for k, v in sorted(cells_by_anchor.items())}},
                "bounds_wgs84": {"min_lon": bounds[0], "min_lat": bounds[1], "max_lon": bounds[2], "max_lat": bounds[3]},
                "approximate_total_coverage_area_km2": area_km2, "connected_component_count": len(components),
                "connected_component_sizes": [len(c) for c in components], "is_disconnected": len(components) > 1,
                "official_boundary_claim": False}
    result = {"records": records, "mapping_rows": mapping_rows, "anchors": anchors, "metadata": metadata}
    if write_outputs:
        output_dir = Path(output_dir or ROOT / "data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(output_dir / "synthetic_h3_cells.geojson", driver="GeoJSON")
        gdf.drop(columns="geometry").to_csv(output_dir / "synthetic_h3_cells.csv", index=False)
        import pandas as pd
        pd.DataFrame(mapping_rows).sort_values(["h3_index", "anchor_region_id"]).to_csv(output_dir / "synthetic_region_mapping.csv", index=False)
        _write_json(output_dir / "synthetic_geography_metadata.json", metadata)
        lines = ["# Synthetic Geography Validation", "", "**SYNTHETIC OPERATIONAL GEOGRAPHY – NOT AN OFFICIAL ADMINISTRATIVE BOUNDARY**", "",
                 f"- Mode: `{mode}`", f"- H3 resolution: `{resolution}`", f"- Neighborhood radius: `{radius}`",
                 f"- Input anchors: `{len(anchors)}`", f"- Unique anchor H3 cells: `{metadata['unique_anchor_h3_cells']}`",
                 f"- Unique generated H3 cells: `{len(records)}`", f"- Shared neighborhood cells: `{metadata['overlap_statistics']['shared_cell_count']}`",
                 f"- Connected components: `{len(components)}`", f"- Approximate coverage area: `{area_km2:.3f}` km²", "", "## Cells per analytical region", ""]
        lines += [f"- `{rid}`: {count}" for rid, count in sorted(assignment_counts.items())]
        lines += ["", "The regions are analytical anchor assignments, not official boundaries. Disconnected H3 components are retained and reported."]
        (output_dir / "synthetic_geography_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if write_maps:
            _render_png(records, anchors, output_dir / "synthetic_geography_preview.png")
            _render_html(records, anchors, output_dir / "synthetic_geography_preview.html")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic H3 synthetic geography from eight anchors.")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS)
    args = parser.parse_args()
    result = generate_synthetic_geography(region_path=args.regions)
    print(json.dumps(result["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
