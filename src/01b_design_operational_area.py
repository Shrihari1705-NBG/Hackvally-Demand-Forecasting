"""Phase 2B: derive a clearly labelled KasuCab operational study area draft.

This stage is deliberately separate from official administrative-boundary extraction.
It uses OSM urban land-use polygons as source evidence and named-place anchors only
to test coverage. The retrieval envelope is not itself the study boundary.
"""
from datetime import datetime, timezone
from pathlib import Path
import json, time, html
import requests
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon, LineString, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/"config.json").read_text()); O=C["output"]
RAW=ROOT/"data"/"raw"; PROC=ROOT/"data"/"processed"; CAND=RAW/"boundary_candidates"
RAW.mkdir(parents=True,exist_ok=True); PROC.mkdir(parents=True,exist_ok=True); CAND.mkdir(parents=True,exist_ok=True)
UTC=datetime.now(timezone.utc).isoformat()
UA="KasuCab-Belagavi-Phase2B/1.0 (operational study area research)"
NOM="https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS=["https://overpass.kumi.systems/api/interpreter","https://overpass.private.coffee/api/interpreter","https://overpass-api.de/api/interpreter"]

REGIONS=[
 {"region_id":"R01","region_name":"Central Business District","description":"Core commercial area around Khade Bazar, Ganpat Galli, College Road and Kirloskar Road.","anchor_query":"Khade Bazar, Belagavi, Karnataka, India","reference_landmarks":["Khade Bazar","Ganpat Galli","College Road","Kirloskar Road"]},
 {"region_id":"R02","region_name":"Transport Hub","description":"Railway Station, Central Bus Stand and Kadamba Bus Stand surroundings.","anchor_query":"Belagavi Railway Station, Karnataka, India","reference_landmarks":["Belagavi Railway Station","Central Bus Stand","Kadamba Bus Stand"]},
 {"region_id":"R03","region_name":"Tilakwadi–Camp","description":"Tilakwadi, Camp and Cantonment/Fort surroundings.","anchor_query":"Tilakwadi, Belagavi, Karnataka, India","reference_landmarks":["Tilakwadi","Camp","Belgaum Fort","Belagavi Cantonment"]},
 {"region_id":"R04","region_name":"Shahapur","description":"Shahapur urban neighbourhood reference area.","anchor_query":"Shahapur, Belagavi, Karnataka, India","reference_landmarks":["Shahapur"]},
 {"region_id":"R05","region_name":"Angol–Majagaon","description":"Angol and Majagaon urban neighbourhood reference area.","anchor_query":"Angol, Belagavi, Karnataka, India","reference_landmarks":["Angol","Majagaon"]},
 {"region_id":"R06","region_name":"Shivabasav Nagar–Medical Hub","description":"Shivabasav Nagar and KLE Hospital vicinity.","anchor_query":"KLE Hospital Belagavi, Karnataka, India","reference_landmarks":["Shivabasav Nagar","KLE Hospital"]},
 {"region_id":"R07","region_name":"Udyambag Industrial Area","description":"Udyambag industrial and employment area.","anchor_query":"Udyambag Industrial Area, Belagavi, Karnataka, India","reference_landmarks":["Udyambag Industrial Area"]},
 {"region_id":"R08","region_name":"Kanabargi and Outer Residential Area","description":"Kanabargi and adjacent outer residential service area. OSM currently resolves the local spelling through Kanbargi Pond.","anchor_query":"Kanbargi Pond, Belagavi, Karnataka, India","reference_landmarks":["Kanabargi/Kanbargi","Kanbargi Pond"]}
]

def get_json(url, params=None, data=None):
    r=requests.get(url,params=params,headers={"User-Agent":UA},timeout=120) if data is None else requests.post(url,data=data,headers={"User-Agent":UA},timeout=180)
    r.raise_for_status(); return r.json()

anchors=[]; failures=[]
for region in REGIONS:
    try:
        results=get_json(NOM,params={"q":region["anchor_query"],"format":"jsonv2","addressdetails":1,"limit":1})
        if not results: raise ValueError("No Nominatim result")
        item=results[0]; region["center_lat"]=float(item["lat"]); region["center_lon"]=float(item["lon"]); region["geocoder_display_name"]=item.get("display_name"); region["source_feature"]={"osm_type":item.get("osm_type"),"osm_id":item.get("osm_id"),"class":item.get("class"),"type":item.get("type")}; anchors.append(region)
    except Exception as exc:
        region["error"]=str(exc); failures.append({"region_id":region["region_id"],"query":region["anchor_query"],"error":str(exc)})
    time.sleep(1.1)

if len(anchors)<8:
    raise SystemExit("STOP: not all eight logical-region anchors could be located; no draft boundary created.")

# This bbox is only an OSM query envelope, not the operational boundary.
minx=min(r["center_lon"] for r in anchors)-0.025; maxx=max(r["center_lon"] for r in anchors)+0.025
miny=min(r["center_lat"] for r in anchors)-0.025; maxy=max(r["center_lat"] for r in anchors)+0.025
bbox=(miny,minx,maxy,maxx)
queries={
 "landuse":f'''[out:json][timeout:120];(way["landuse"~"^(residential|commercial|retail|industrial)$"]({miny},{minx},{maxy},{maxx});relation["landuse"~"^(residential|commercial|retail|industrial)$"]({miny},{minx},{maxy},{maxx}););out tags center geom;''',
 "roads":f'''[out:json][timeout:120];way["highway"~"^(primary|secondary|tertiary|residential)$"]["name"]({miny},{minx},{maxy},{maxx});out tags center geom;''',
 "references":"[out:json][timeout:60];way(id:1119941070,138292670);out tags center geom;"
}
payloads={}; overpass_errors=[]
for label,q in queries.items():
    for endpoint in OVERPASS_ENDPOINTS:
        try: payloads[label]=get_json(endpoint,data=q); payloads[label]["endpoint"]=endpoint; break
        except Exception as exc: overpass_errors.append({"query":label,"endpoint":endpoint,"error":str(exc)})
    if label=="landuse" and label not in payloads: raise SystemExit("STOP: land-use Overpass query failed; no operational draft created. "+json.dumps(overpass_errors))

features=[]; urban_polys=[]; roads=[]; references=[]
def geom_from_element(e):
    if not e.get("geometry"): return None
    coords=[(p["lon"],p["lat"]) for p in e["geometry"]]
    if e.get("type")=="way":
        if len(coords)>=4 and coords[0]==coords[-1]: return Polygon(coords)
        return LineString(coords)
    return None
elements=[]
for item in payloads.values(): elements.extend(item.get("elements",[]))
for e in elements:
    tags=e.get("tags",{}); g=geom_from_element(e)
    if e.get("type")=="node" and e.get("lat") is not None: g=Point(e["lon"],e["lat"])
    if g is None or g.is_empty: continue
    tag_text=" ".join(str(tags.get(k,"")) for k in ("name","landuse","place","highway"))
    kind="urban_landuse" if tags.get("landuse") in {"residential","commercial","retail","industrial"} and g.geom_type in {"Polygon","MultiPolygon"} else "road_or_named_feature"
    if e.get("id") in {1119941070,138292670}: kind="unverified_reference"
    rec={"source":"OpenStreetMap Overpass","osm_element_type":e.get("type"),"osm_id":e.get("id"),"name":tags.get("name"),"tags":tags,"feature_kind":kind,"geometry":g}
    features.append(rec)
    if kind=="urban_landuse": urban_polys.append(g)
    elif kind=="unverified_reference": references.append(rec)
    elif g.geom_type in {"LineString","MultiLineString"}: roads.append(rec)

if not urban_polys: raise SystemExit("STOP: no qualifying OSM urban land-use polygons were returned; no draft created.")
urban_union=unary_union(urban_polys)
if not urban_union.is_valid: urban_union=urban_union.buffer(0)
# A documented morphological connection around selected urban land-use polygons,
# not a circle/rectangle and not a convex hull of POIs.
urban_metric=gpd.GeoSeries([urban_union],crs=4326).to_crs(C["metric_crs"]).iloc[0]
road_lines=[f["geometry"] for f in roads if f["geometry"].geom_type in {"LineString","MultiLineString"}]
road_metric=gpd.GeoSeries(road_lines,crs=4326).to_crs(C["metric_crs"]).union_all() if road_lines else None
road_corridor=road_metric.buffer(35) if road_metric is not None else None
evidence=urban_metric.union(road_corridor) if road_corridor is not None else urban_metric
draft=gpd.GeoSeries([evidence.buffer(150).buffer(-150)],crs=C["metric_crs"]).to_crs(4326).iloc[0]
if not draft.is_valid: draft=draft.buffer(0)
if draft.is_empty or not draft.is_valid: raise SystemExit("STOP: derived operational geometry is empty or invalid.")

def covers_anchor(region): return draft.covers(Point(region["center_lon"],region["center_lat"]))
for r in anchors: r["covered_by_draft"]=bool(covers_anchor(r)); r["boundary_or_inclusion_method"]="OSM named anchor checked against union of selected urban land-use polygons with 150 m morphological connection"; r["confidence_level"]="Medium" if r["covered_by_draft"] else "Low"
missing=[r["region_id"] for r in anchors if not r["covered_by_draft"]]
component_count=len(getattr(draft,"geoms",[draft])) if draft.geom_type=="MultiPolygon" else 1
draft_gdf=gpd.GeoDataFrame([{"boundary_type":"operational_study_area","boundary_name":"KasuCab Belagavi Urban Service Area","status":"DRAFT – NOT APPROVED","derivation":"Union of OSM residential/commercial/retail/industrial polygons, followed by documented 150 m morphological connection; retrieval bbox is not the boundary","component_count":component_count}],geometry=[draft],crs=4326)
draft_gdf.to_file(ROOT/O["operational_boundary_draft"],driver="GeoJSON")

source_out=ROOT/O["operational_source_features"]
source_rows=[]
for f in features:
    row={k:v for k,v in f.items() if k not in {"geometry","tags"}}
    row["tags_json"]=json.dumps(f.get("tags",{}),ensure_ascii=False)
    source_rows.append(row)
gpd.GeoDataFrame(source_rows,geometry=[f["geometry"] for f in features],crs=4326).to_file(source_out,driver="GeoJSON")

defs={"boundary_type":"operational_study_area","boundary_name":"KasuCab Belagavi Urban Service Area","city":"Belagavi, Karnataka, India","status":"DRAFT – NOT APPROVED","retrieval_timestamp_utc":UTC,"regions":[{k:v for k,v in r.items() if k!="error"} for r in anchors]}
(ROOT/O["region_definitions"]).write_text(json.dumps(defs,indent=2,ensure_ascii=False))

# Generate a GeoJSON layer only for point-based analytical regions; no invented region polygons.
region_points=gpd.GeoDataFrame([{k:v for k,v in r.items() if k not in {"center_lat","center_lon"}} for r in anchors],geometry=[Point(r["center_lon"],r["center_lat"]) for r in anchors],crs=4326)

def geojson_obj(gdf): return json.loads(gdf.to_json())
draft_json=geojson_obj(draft_gdf); points_json=geojson_obj(region_points)
ref_gdf=gpd.GeoDataFrame([{k:v for k,v in f.items() if k!="geometry"} for f in references],geometry=[f["geometry"] for f in references],crs=4326) if references else None
ref_json=geojson_obj(ref_gdf) if ref_gdf is not None and len(ref_gdf) else {"type":"FeatureCollection","features":[]}
roads_gdf=gpd.GeoDataFrame([{k:v for k,v in f.items() if k!="geometry"} for f in roads],geometry=[f["geometry"] for f in roads],crs=4326) if roads else None
roads_json=geojson_obj(roads_gdf) if roads_gdf is not None and len(roads_gdf) else {"type":"FeatureCollection","features":[]}
html_path=ROOT/O["operational_preview_html"]
html_path.write_text('''<!doctype html><html><head><meta charset="utf-8"><title>KasuCab Belagavi Operational Study Area – DRAFT</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/><style>#map{height:92vh} .draft{color:#b00;font-weight:bold}</style></head><body><div class="draft">DRAFT – NOT APPROVED. This is an operational study area, not an official administrative boundary.</div><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const draft='''+json.dumps(draft_json)+''';const regions='''+json.dumps(points_json)+''';const refs='''+json.dumps(ref_json)+''';const roads='''+json.dumps(roads_json)+''';const map=L.map('map').setView([15.85,74.50],12);const d=L.geoJSON(draft,{style:{color:'#d62728',weight:3,fillOpacity:.12}}).bindPopup('DRAFT – NOT APPROVED');const r=L.geoJSON(regions,{pointToLayer:(f,latlng)=>L.circleMarker(latlng,{radius:6,color:'#1f77b4'}),onEachFeature:(f,l)=>l.bindPopup(f.properties.region_name)});const u=L.geoJSON(refs,{style:{color:'#9467bd',dashArray:'6 4',fillOpacity:.12},onEachFeature:(f,l)=>l.bindPopup('Unverified OSM reference: '+(f.properties.name||''))});const rd=L.geoJSON(roads,{style:{color:'#999',weight:1,opacity:.35}});d.addTo(map);r.addTo(map);u.addTo(map);rd.addTo(map);L.control.layers({'Draft perimeter':d,'OSM roads':rd,'Logical-region anchors':r,'Unverified Cantonment/Fort references':u}).addTo(map);map.fitBounds(d.getBounds());</script></body></html>''',encoding="utf-8")

fig,ax=plt.subplots(figsize=(12,10),dpi=180)
gpd.GeoSeries([draft],crs=4326).plot(ax=ax,facecolor="#d62728",alpha=.12,edgecolor="#d62728",linewidth=2,label="DRAFT – NOT APPROVED")
if roads: gpd.GeoDataFrame(roads,geometry="geometry",crs=4326).plot(ax=ax,color="#999999",linewidth=.25,alpha=.25)
for r in anchors: ax.scatter(r["center_lon"],r["center_lat"],s=30,label=r["region_id"]+" "+r["region_name"])
for f in references: gpd.GeoSeries([f["geometry"]],crs=4326).plot(ax=ax,facecolor="none",edgecolor="#9467bd",linestyle="--",linewidth=1.5,label="Unverified Cantonment/Fort")
ax.set_title("KasuCab Belagavi Urban Service Area\nDRAFT – NOT APPROVED"); ax.set_xlabel("Longitude (EPSG:4326)"); ax.set_ylabel("Latitude (EPSG:4326)"); ax.legend(fontsize=7,loc="best"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(ROOT/O["operational_preview_png"]); plt.close(fig)

metric=gpd.GeoSeries([draft],crs=4326).to_crs(C["metric_crs"])
report={"boundary_type":"operational_study_area","boundary_name":"KasuCab Belagavi Urban Service Area","geometry_checks_passed":bool(not draft.is_empty and draft.is_valid and draft.geom_type in {"Polygon","MultiPolygon"} and not missing),"operational_coverage_reviewed":False,"boundary_approved":False,"status":"DRAFT – NOT APPROVED","derivation_method":"Union of OSM residential, commercial, retail and industrial land-use polygons with a 35 m metric corridor around named OSM roads, followed by a documented 150 m metric morphological connection. The retrieval envelope is not the boundary.","osm_source":"OpenStreetMap Overpass","retrieval_timestamp_utc":UTC,"retrieval_bbox_wgs84":[minx,miny,maxx,maxy],"geometry_type":draft.geom_type,"component_count":component_count,"area_km2":float(metric.area.iloc[0]/1e6),"bounds_wgs84":list(draft.bounds),"regions_total":8,"regions_covered":8-len(missing),"missing_regions":missing,"uncertain_perimeter_sections":["Outer Kanabargi edge","Udyambag industrial transition","Cantonment/Fort surroundings"],"excluded_areas":["Belagavi District and taluka outside the OSM urban land-use evidence","Areas not connected to selected urban land-use/road evidence","Administrative status of Cantonment/Fort polygons"],"unverified_reference_layers":[{"osm_id":f["osm_id"],"name":f["name"],"tags":f["tags"]} for f in references],"failures":failures+overpass_errors,"note":"Geometry validity does not establish geographical correctness. Manual operational-coverage review is required."}
(ROOT/O["operational_validation"]).write_text(json.dumps(report,indent=2))
md="# KasuCab Belagavi Urban Service Area\n\n**DRAFT – NOT APPROVED**\n\n"
md+=f"- boundary_type: `{report['boundary_type']}`\n- geometry_checks_passed: `{report['geometry_checks_passed']}`\n- operational_coverage_reviewed: `false`\n- boundary_approved: `false`\n- area_km2: `{report['area_km2']:.3f}`\n- geometry_type: `{report['geometry_type']}`\n- components: `{component_count}`\n\n"
md+="## Derivation\n\n"+report["derivation_method"]+"\n\nThe OSM retrieval envelope was used only to query source evidence; it is not an administrative or operational boundary. No POI convex hull, arbitrary circle, rectangle, district boundary, or official administrative claim was used.\n\n## Logical regions\n\n"
for r in anchors: md+=f"- **{r['region_id']} — {r['region_name']}**: ({r['center_lat']:.6f}, {r['center_lon']:.6f}); covered_by_draft=`{r['covered_by_draft']}`; confidence=`{r['confidence_level']}`; landmarks: {', '.join(r['reference_landmarks'])}.\n"
md+="\n## Manual review items\n\n- Review the outer Kanabargi, Udyambag transition, and Cantonment/Fort perimeter segments.\n- Confirm the draft is suitable as a KasuCab simulation study area; it is not an official municipal or Cantonment boundary.\n- Inspect the interactive map layer toggles and the source feature GeoJSON.\n"
(ROOT/O["operational_validation_markdown"]).write_text(md,encoding="utf-8")
print(json.dumps({"status":report["status"],"geometry_checks_passed":report["geometry_checks_passed"],"area_km2":report["area_km2"],"regions_covered":report["regions_covered"],"components":component_count},indent=2))
