"""Validate a manually supplied KasuCab operational study boundary.

This module never creates a boundary.  The input GeoJSON is preserved and any
repair is performed only on a derived in-memory geometry and validation copy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
from shapely.validation import explain_validity

try:
    from shapely import make_valid
except ImportError:  # pragma: no cover - supported Shapely versions provide it
    make_valid = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/kasucab_operational_boundary.geojson"
DEFAULT_REGIONS = ROOT / "data/processed/kasucab_region_definitions.json"
DEFAULT_OUTPUT = ROOT / "data/processed"


def _epsg(value: str) -> str:
    return str(value).upper().replace("::", ":")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _load_regions(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Preserve compatibility with the existing Phase 2B report generated
        # on a Windows code page; the coordinates and identifiers are unchanged.
        text = path.read_text(encoding="cp1252")
    data = json.loads(text)
    return data.get("regions", data) if isinstance(data, dict) else data


def _repair_geometry(geometry):
    if geometry.is_valid:
        return geometry, False, None
    if make_valid is not None:
        repaired = make_valid(geometry)
        return repaired, True, "shapely.make_valid"
    return geometry.buffer(0), True, "buffer(0)"


def _bounds_dict(geometry) -> dict | None:
    if geometry is None or geometry.is_empty:
        return None
    minx, miny, maxx, maxy = geometry.bounds
    return {"min_lon": minx, "min_lat": miny, "max_lon": maxx, "max_lat": maxy}


def _metric_distance(point: Point, geometry, metric_crs: str) -> float:
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    px, py = transformer.transform(point.x, point.y)
    metric_point = Point(px, py)
    metric_geometry = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    return float(metric_point.distance(metric_geometry.boundary if metric_geometry.covers(metric_point) else metric_geometry))


def _render_png(geometry, anchors: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    gpd.GeoSeries([geometry], crs="EPSG:4326").boundary.plot(ax=ax, color="#b22222", linewidth=2)
    for anchor in anchors:
        p = Point(anchor["reference_longitude"], anchor["reference_latitude"])
        ax.scatter([p.x], [p.y], s=28, color="#1769aa", zorder=3)
        ax.annotate(anchor["region_id"], (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title + "\nDRAFT – NOT APPROVED", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _render_html(geometry, anchors: list[dict], path: Path, title: str) -> None:
    feature = {"type": "Feature", "properties": {"name": title}, "geometry": mapping(geometry)}
    points = [{"type": "Feature", "properties": {"name": a["region_name"], "region_id": a["region_id"]},
               "geometry": {"type": "Point", "coordinates": [a["reference_longitude"], a["reference_latitude"]]}}
              for a in anchors]
    payload = json.dumps({"boundary": feature, "anchors": {"type": "FeatureCollection", "features": points}})
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>#map{{height:92vh}} .status{{font:700 16px sans-serif;padding:8px;color:#9b1c1c}}</style></head>
<body><div class="status">{title} — DRAFT – NOT APPROVED</div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const data={payload}; const map=L.map('map');
const boundary=L.geoJSON(data.boundary,{{style:{{color:'#b22222',weight:3,fillOpacity:.12}}}}).bindPopup('{title}');
const anchors=L.geoJSON(data.anchors,{{pointToLayer:(f,ll)=>L.circleMarker(ll,{{radius:6,color:'#1769aa',fillOpacity:.9}}),onEachFeature:(f,l)=>l.bindTooltip(f.properties.region_id+' — '+f.properties.name)}});
boundary.addTo(map); anchors.addTo(map); L.control.layers({{}},{{'Operational boundary':boundary,'Region anchors':anchors}}).addTo(map); map.fitBounds(boundary.getBounds());
</script></body></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def approval_ready(report: dict) -> bool:
    return bool(report.get("source_file_exists") and report.get("geometry_checks_passed") and
                report.get("anchor_coverage_passed") and not report.get("approval_blockers"))


def validate_boundary(input_path: Path | str = DEFAULT_INPUT, output_dir: Path | str = DEFAULT_OUTPUT,
                      region_definitions_path: Path | str = DEFAULT_REGIONS, write_outputs: bool = True,
                      write_maps: bool = True, config: dict | None = None) -> dict:
    input_path, output_dir, region_definitions_path = map(Path, (input_path, output_dir, region_definitions_path))
    report = {
        "boundary_type": "operational_study_area", "boundary_name": "KasuCab Belagavi Urban Service Area",
        "source_method": "manual_digitization", "administrative_boundary_claim": False,
        "coordinate_reference_system": "EPSG:4326", "input_path": str(input_path),
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_exists": input_path.exists(), "source_geometry_preserved": True,
        "geometry_repair_performed": False, "geometry_repair_method": None,
        "geometry_checks_passed": False, "operational_coverage_reviewed": False,
        "boundary_approved": False, "approval_blockers": [], "anchors": [],
        "reference_layers": [],
    }
    if not input_path.exists():
        report["approval_blockers"].append(f"Manual boundary input is missing: {input_path}")
        if write_outputs:
            _write_reports(report, output_dir)
        return report

    try:
        source = gpd.read_file(input_path)
    except Exception as exc:
        report["read_error"] = str(exc)
        report["approval_blockers"].append("The GeoJSON could not be read.")
        if write_outputs:
            _write_reports(report, output_dir)
        return report

    report["source_feature_count"] = len(source)
    report["source_geometry_types"] = sorted(source.geometry.geom_type.dropna().unique().tolist())
    report["crs_verified"] = source.crs is not None and _epsg(source.crs.to_string()) == "EPSG:4326"
    report["source_crs"] = source.crs.to_string() if source.crs else None
    if not report["crs_verified"]:
        report["approval_blockers"].append("Input CRS must be explicitly EPSG:4326; it is not reprojected automatically.")

    geometries = list(source.geometry)
    non_polygonal = [g.geom_type for g in geometries if g is not None and not g.is_empty and g.geom_type not in ("Polygon", "MultiPolygon")]
    report["empty_geometry"] = any(g is None or g.is_empty for g in geometries)
    report["non_polygonal_geometry_types"] = non_polygonal
    report["invalid_geometry"] = any(g is not None and not g.is_empty and not g.is_valid for g in geometries)
    report["self_intersections"] = [explain_validity(g) for g in geometries if g is not None and not g.is_valid]
    report["source_bounds_wgs84"] = _bounds_dict(unary_union([g for g in geometries if g is not None and not g.is_empty])) if geometries else None
    if report["empty_geometry"]:
        report["approval_blockers"].append("Input contains empty or null geometries.")
    if non_polygonal:
        report["approval_blockers"].append("Input contains geometry types other than Polygon or MultiPolygon.")
    if report["invalid_geometry"]:
        report["approval_blockers"].append("Input contains invalid geometry; inspect the reported self-intersection details.")

    derived = unary_union([g for g in geometries if g is not None and not g.is_empty])
    repaired, repaired_flag, repair_method = _repair_geometry(derived)
    report["geometry_repair_performed"] = repaired_flag
    report["geometry_repair_method"] = repair_method
    report["geometry_type"] = repaired.geom_type
    report["component_count"] = len(repaired.geoms) if repaired.geom_type == "MultiPolygon" else 1
    report["unexpected_disconnected_components"] = report["component_count"] > 1
    report["bounds_wgs84"] = _bounds_dict(repaired)
    b = report["bounds_wgs84"]
    report["bounds_sanity_passed"] = bool(b and 74.3 <= b["min_lon"] <= 74.6 and 74.6 <= b["max_lon"] <= 74.9 and
                                           15.7 <= b["min_lat"] <= 16.0 and 15.9 <= b["max_lat"] <= 16.3)
    if not report["bounds_sanity_passed"]:
        report["approval_blockers"].append("Bounds are outside the configured Belagavi sanity range; geographical correctness requires review.")
    if report["unexpected_disconnected_components"]:
        report["approval_blockers"].append("Multiple disconnected components were detected; confirm this is intentional.")
    report["metric_crs"] = (config or {}).get("metric_crs", "EPSG:32643")
    report["area_km2"] = float(gpd.GeoSeries([repaired], crs="EPSG:4326").to_crs(report["metric_crs"]).area.iloc[0] / 1_000_000)
    report["geometry_checks_passed"] = bool(report["crs_verified"] and not report["empty_geometry"] and not non_polygonal and
                                             not report["invalid_geometry"] and repaired.is_valid and report["bounds_sanity_passed"] and
                                             not report["unexpected_disconnected_components"])

    try:
        regions = _load_regions(region_definitions_path)
    except Exception as exc:
        regions, report["region_definitions_error"] = [], str(exc)
        report["approval_blockers"].append("Logical region definitions could not be loaded.")
    metric_geometry = gpd.GeoSeries([repaired], crs="EPSG:4326").to_crs(report["metric_crs"]).iloc[0]
    transformer = Transformer.from_crs("EPSG:4326", report["metric_crs"], always_xy=True)
    for region in regions:
        lon, lat = float(region["center_lon"]), float(region["center_lat"])
        point = Point(lon, lat)
        mx, my = transformer.transform(lon, lat)
        mpoint = Point(mx, my)
        inside = bool(repaired.covers(point))
        distance = float(mpoint.distance(metric_geometry.boundary if inside else metric_geometry))
        report["anchors"].append({"region_id": region["region_id"], "region_name": region["region_name"],
                                  "reference_longitude": lon, "reference_latitude": lat,
                                  "point_inside_boundary": inside, "distance_to_boundary_m": distance,
                                  "coverage_status": "covered" if inside else "outside"})
    report["anchor_coverage_passed"] = bool(report["anchors"]) and all(a["point_inside_boundary"] for a in report["anchors"])
    if not report["anchor_coverage_passed"]:
        report["approval_blockers"].append("One or more logical-region reference anchors are outside the supplied boundary.")
    report["normalized_copy"] = str(output_dir / "kasucab_operational_boundary_normalized.geojson")
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"boundary_name": [report["boundary_name"]], "geometry": [repaired]}, crs="EPSG:4326").to_file(
            report["normalized_copy"], driver="GeoJSON")
        if write_maps:
            _render_png(repaired, report["anchors"], output_dir / "operational_boundary_preview.png", report["boundary_name"])
            _render_html(repaired, report["anchors"], output_dir / "operational_boundary_preview.html", report["boundary_name"])
        _write_reports(report, output_dir)
    return report


def _write_reports(report: dict, output_dir: Path) -> None:
    _write_json(output_dir / "operational_boundary_validation.json", report)
    lines = [f"# {report['boundary_name']}", "", "**DRAFT – NOT APPROVED**", "",
             f"- Boundary type: `{report['boundary_type']}`", f"- Source method: `{report['source_method']}`",
             f"- Geometry checks passed: `{report.get('geometry_checks_passed', False)}`",
             f"- Anchor coverage passed: `{report.get('anchor_coverage_passed', False)}`",
             f"- Operational coverage reviewed: `{report.get('operational_coverage_reviewed', False)}`",
             f"- Boundary approved: `{report.get('boundary_approved', False)}`", "",
             "## Geometry", "", f"- Geometry type: `{report.get('geometry_type')}`",
             f"- Area: `{report.get('area_km2')}` km²", f"- Component count: `{report.get('component_count')}`",
             f"- CRS verified: `{report.get('crs_verified', False)}`", f"- Repair performed: `{report.get('geometry_repair_performed', False)}`"]
    lines += ["", "## Anchor coverage", "", "| Region | Inside | Distance (m) | Status |", "|---|---:|---:|---|"]
    lines += [f"| {a['region_id']} — {a['region_name']} | {a['point_inside_boundary']} | {a['distance_to_boundary_m']:.2f} | {a['coverage_status']} |" for a in report.get("anchors", [])]
    lines += ["", "## Approval blockers", ""] + [f"- {x}" for x in report.get("approval_blockers", [])]
    (output_dir / "operational_boundary_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the manually supplied operational boundary only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    report = validate_boundary(args.input)
    print(json.dumps({"geometry_checks_passed": report.get("geometry_checks_passed", False),
                      "anchor_coverage_passed": report.get("anchor_coverage_passed", False),
                      "approval_blockers": report.get("approval_blockers", [])}, indent=2))
    return 0 if report.get("geometry_checks_passed") and report.get("anchor_coverage_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
