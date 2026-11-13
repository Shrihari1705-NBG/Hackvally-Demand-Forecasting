"""Phase 2: discover, validate, and preview Belagavi boundaries only."""
from datetime import datetime, timezone
from pathlib import Path
import json, time

import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from shapely.geometry import shape, Point
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"config.json").read_text())
OUT=CFG["output"]
RAW=ROOT/"data"/"raw"; PROCESSED=ROOT/"data"/"processed"
RAW.mkdir(parents=True,exist_ok=True); PROCESSED.mkdir(parents=True,exist_ok=True)
USER_AGENT="KasuCab-Belagavi-Phase2/1.0 (boundary research; contact project owner)"
NOMINATIM="https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS=["https://overpass-api.de/api/interpreter","https://overpass.kumi.systems/api/interpreter","https://overpass.private.coffee/api/interpreter"]
RETRIEVED=datetime.now(timezone.utc).isoformat()

def request_json(url, *, params=None, data=None):
    response=requests.get(url,params=params,headers={"User-Agent":USER_AGENT},timeout=120) if data is None else requests.post(url,data=data,headers={"User-Agent":USER_AGENT},timeout=180)
    response.raise_for_status(); return response.json()

def nominatim_search(query):
    return request_json(NOMINATIM,params={"q":query,"format":"jsonv2","polygon_geojson":1,"addressdetails":1,"extratags":1,"namedetails":1,"limit":10})

def candidate_record(item, query):
    tags=item.get("extratags") or {}
    address=item.get("address") or {}
    geom=shape(item["geojson"]) if item.get("geojson") else None
    return {"source":"Nominatim","source_query":query,"retrieval_timestamp_utc":RETRIEVED,
      "osm_element_type":item.get("osm_type"),"osm_id":item.get("osm_id"),"relation_id":item.get("osm_id") if item.get("osm_type")=="relation" else None,
      "name":item.get("name"),"official_name":tags.get("official_name"),"local_name":tags.get("alt_name") or tags.get("loc_name"),
      "display_name":item.get("display_name"),"boundary":tags.get("boundary"),"admin_level":tags.get("admin_level") or address.get("state_district"),
      "other_osm_tags":tags,"geometry_type":geom.geom_type if geom else None,"bounds":list(geom.bounds) if geom else None,
      "area_km2":None,"geometry":geom}

def run_overpass():
    query='''[out:json][timeout:180];
    (relation["boundary"="administrative"](15.70,74.30,16.05,74.65);
     way["boundary"="administrative"](15.70,74.30,16.05,74.65);
     nwr["name"~"Belagavi|Belgaum|Cantonment|Fort",i](15.70,74.30,16.05,74.65););
    out tags center geom;'''
    errors=[]
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload=request_json(endpoint,data=query); payload["endpoint"]=endpoint; return payload
        except Exception as exc: errors.append({"endpoint":endpoint,"error":str(exc)})
    return {"error":errors,"elements":[]}

def overpass_records(payload):
    records=[]
    for e in payload.get("elements",[]):
        tags=e.get("tags",{}); text=" ".join(str(tags.get(k,"")) for k in ("name","official_name","alt_name","loc_name"))
        if not any(x in text.lower() for x in ("belagavi","belgaum","cantonment","fort")): continue
        geom=None
        if e.get("type")=="way" and e.get("geometry"):
            from shapely.geometry import LineString, Polygon
            coords=[(p["lon"],p["lat"]) for p in e["geometry"]]
            try: geom=Polygon(coords) if coords[0]==coords[-1] and len(coords)>=4 else LineString(coords)
            except Exception: geom=None
        records.append({"source":"Overpass","osm_element_type":e.get("type"),"osm_id":e.get("id"),"relation_id":e.get("id") if e.get("type")=="relation" else None,"name":tags.get("name"),"official_name":tags.get("official_name"),"local_name":tags.get("alt_name") or tags.get("loc_name"),"boundary":tags.get("boundary"),"admin_level":tags.get("admin_level"),"other_osm_tags":tags,"center":e.get("center"),"geometry_type":geom.geom_type if geom else None,"bounds":list(geom.bounds) if geom else None,"area_km2":float(gpd.GeoSeries([geom],crs=4326).to_crs(CFG["metric_crs"]).area.iloc[0]/1e6) if geom and geom.geom_type in ("Polygon","MultiPolygon") else None})
    return records

queries=["Belagavi City Corporation, Karnataka, India","Belgaum City Corporation, Karnataka, India","Belagavi Cantonment, Karnataka, India","Belgaum Cantonment, Karnataka, India","Belagavi Cantonment Fort Zone, Karnataka, India","Belgaum Cantonment Fort, Karnataka, India"]
candidates=[]; failures=[]
for query in queries:
    try:
        for item in nominatim_search(query): candidates.append(candidate_record(item,query))
    except Exception as exc: failures.append({"source":"Nominatim","query":query,"error":str(exc)})
    time.sleep(1.1)
overpass=run_overpass()
if overpass.get("error"): failures.append({"source":"Overpass","error":overpass["error"]})
overpass_meta=overpass_records(overpass)

def label(c):
    text=" ".join(str(c.get(k) or "") for k in ("name","official_name","local_name","display_name")).lower()
    if "cantonment" in text or "cantt" in text: return "cantonment"
    if "city corporation" in text or "mahanagara" in text: return "city_corporation"
    return "other"

for c in candidates: c["candidate_role"]=label(c)
city=[c for c in candidates if c["candidate_role"]=="city_corporation" and c.get("geometry")]
cantt=[c for c in candidates if c["candidate_role"]=="cantonment" and c.get("geometry")]

def choose(items, role):
    # Require an administrative relation with polygon geometry; locality/place matches are rejected.
    eligible=[x for x in items if x.get("osm_element_type")=="relation" and x.get("geometry_type") in ("Polygon","MultiPolygon") and (x.get("boundary")=="administrative" or x.get("other_osm_tags",{}).get("boundary")=="administrative")]
    if not eligible: return None, "No confidently identifiable administrative relation polygon"
    # Prefer exact name, then the largest candidate; this remains pending manual approval.
    eligible.sort(key=lambda x:("corporation" in str(x.get("name","")).lower(), x.get("geometry").area),reverse=True)
    return eligible[0], None

city_sel,city_reject=choose(city,"city_corporation"); cantt_sel,cantt_reject=choose(cantt,"cantonment")
selected=[]; rejected=[]
for group,chosen,reason in ((city,city_sel,city_reject),(cantt,cantt_sel,cantt_reject)):
    for c in group:
        if c is chosen: selected.append(c)
        else: rejected.append({"candidate":{k:v for k,v in c.items() if k!="geometry"},"reason":reason or "Not selected; lower-confidence or duplicate candidate"})
if city_sel is None: rejected.append({"candidate_role":"city_corporation","reason":city_reject})
if cantt_sel is None: rejected.append({"candidate_role":"cantonment","reason":cantt_reject})
for c in overpass_meta:
    tags=c.get("other_osm_tags",{}); name=str(c.get("name") or "")
    if c.get("osm_element_type")=="relation" and tags.get("boundary")=="administrative":
        rejected.append({"candidate":c,"reason":"Administrative relation is Belagavi District or taluku, not the City Corporation or Cantonment."})
    elif c.get("osm_id")==1119941070:
        rejected.append({"candidate":c,"reason":"Named Cantonment polygon is tagged landuse=military, not boundary=administrative; retained as a strong candidate for manual review, not selected automatically."})
    elif c.get("osm_id")==511114422:
        rejected.append({"candidate":c,"reason":"Named City Corporation polygon is tagged building=public and is the corporation office footprint, not the city boundary."})
    elif "fort" in name.lower() and c.get("geometry_type")=="Polygon":
        rejected.append({"candidate":c,"reason":"Historic Fort polygon/landmark; no evidence it represents the Cantonment Fort Zone administrative boundary."})

def validate_source(c):
    g=c["geometry"]; issues=[]; repair="not_needed"
    if g is None or g.is_empty: issues.append("empty")
    if g is not None and not g.is_valid: issues.append("invalid")
    if g is not None and not g.is_valid:
        repaired=g.buffer(0); repair="buffer(0)"; g=repaired
    if g is None or g.is_empty or not g.is_valid: issues.append("unrepairable")
    c["geometry"]=g; c["geometry_repair"]=repair; c["geometry_checks_passed"]=not issues
    c["geometry_issues"]=issues
    if g is not None and not g.is_empty: c["area_km2"]=float(gpd.GeoSeries([g],crs=4326).to_crs(CFG["metric_crs"]).area.iloc[0]/1e6)
    return c

if city_sel: validate_source(city_sel)
if cantt_sel: validate_source(cantt_sel)
for c in (city_sel,cantt_sel):
    if c and c.get("geometry_checks_passed"):
        gpd.GeoDataFrame([{k:v for k,v in c.items() if k not in {"geometry","other_osm_tags"}}],geometry=[c["geometry"]],crs=4326).to_file(ROOT/(OUT["city_boundary"] if c is city_sel else OUT["cantonment_boundary"]),driver="GeoJSON")

# Investigate whether the cantonment is represented by named separate zones.
zone_candidates=[c for c in candidates if "fort" in " ".join(str(c.get(k) or "") for k in ("name","official_name","local_name","display_name")).lower()]
zone_features=[]
if zone_candidates:
    for z in zone_candidates:
        if z.get("geometry"): zone_features.append(z)
if cantt_sel and cantt_sel.get("geometry") and not zone_features:
    zone_features=[cantt_sel]
if zone_features:
    zg=gpd.GeoDataFrame([{"zone_label":"Fort Zone" if "fort" in str(z.get("name")).lower() else "Main/combined cantonment","osm_id":z.get("osm_id"),"osm_element_type":z.get("osm_element_type")} for z in zone_features],geometry=[z["geometry"] for z in zone_features],crs=4326)
    zg.to_file(ROOT/OUT["cantonment_zones"],driver="GeoJSON")

merged=None; overlap_km2=None; suspicious=[]
if city_sel and cantt_sel and city_sel.get("geometry_checks_passed") and cantt_sel.get("geometry_checks_passed"):
    city_geom=city_sel["geometry"]; cantt_geom=cantt_sel["geometry"]; merged=unary_union([city_geom,cantt_geom])
    overlap_km2=float(gpd.GeoSeries([city_geom.intersection(cantt_geom)],crs=4326).to_crs(CFG["metric_crs"]).area.iloc[0]/1e6)
    if overlap_km2>0: suspicious.append(f"Source boundaries overlap by {overlap_km2:.6f} km²; retained in union and not removed.")
    gpd.GeoDataFrame([{"name":"Belagavi operational coverage (derived union)","source":"Derived union of selected OSM candidate geometries","boundary_status":"Pending manual administrative verification"}],geometry=[merged],crs=4326).to_file(ROOT/OUT["boundary"],driver="GeoJSON")

def components(g): return len(getattr(g,"geoms",[g])) if g else None
def bounds(g): return list(g.bounds) if g else None
def safe(v): return {k:v2 for k,v2 in v.items() if k!="geometry"} if v else None

map_path=ROOT/OUT["preview_png"]
fig,ax=plt.subplots(figsize=(12,10),dpi=180)
if city_sel and city_sel.get("geometry"): gpd.GeoSeries([city_sel["geometry"]],crs=4326).plot(ax=ax,facecolor="none",edgecolor="#1f77b4",linewidth=2,label="City Corporation")
if cantt_sel and cantt_sel.get("geometry"): gpd.GeoSeries([cantt_sel["geometry"]],crs=4326).plot(ax=ax,facecolor="none",edgecolor="#d62728",linewidth=2,label="Cantonment")
if merged: gpd.GeoSeries([merged],crs=4326).plot(ax=ax,facecolor="#2ca02c",alpha=.15,edgecolor="#2ca02c",linewidth=2,label="Merged operational coverage")
for c in candidates:
    if c.get("geometry") and c is not city_sel and c is not cantt_sel:
        gpd.GeoSeries([c["geometry"]],crs=4326).plot(ax=ax,facecolor="none",edgecolor="#777777",linestyle="--",linewidth=1,label="Rejected named candidate")
ax.scatter([74.508],[15.849],marker="*",s=100,c="black",label="City centre reference")
ax.set_title("Belagavi boundary candidates and derived operational coverage\nPending manual administrative verification")
ax.set_xlabel("Longitude (EPSG:4326)"); ax.set_ylabel("Latitude (EPSG:4326)"); ax.set_xlim(74.30,74.65); ax.set_ylim(15.70,16.05); ax.legend(); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(map_path); plt.close(fig)

report={"retrieval_timestamp_utc":RETRIEVED,"geometry_checks_passed":bool(city_sel and cantt_sel and city_sel.get("geometry_checks_passed") and cantt_sel.get("geometry_checks_passed") and merged is not None),"administrative_boundary_verified":False,"boundary_approved":False,"official_references":{"city_corporation":"https://belagavicitycorp.org/WebSiteBGM/WardInfromation.aspx","cantonment_city_map":"https://belgaum.cantt.gov.in/city-map/","cantonment_about":"https://belgaum.cantt.gov.in/about-board/","limitations":"The Cantonment city-map URL did not expose a downloadable boundary in this run; official pages are comparison references only."},"selected_city_corporation":safe(city_sel),"selected_cantonment":safe(cantt_sel),"cantonment_zone_investigation":{"zone_candidates":[safe(z) for z in zone_candidates],"representation":"separate candidate(s) found" if zone_candidates else "No separate Fort Zone geometry identified; selected cantonment geometry retained for Main/Fort coverage review"},"rejected_candidates":rejected,"overpass_candidates":overpass_meta,"request_failures":failures,"merged":{"geometry_type":merged.geom_type if merged else None,"bounds":bounds(merged),"area_km2":float(gpd.GeoSeries([merged],crs=4326).to_crs(CFG["metric_crs"]).area.iloc[0]/1e6) if merged else None,"component_count":components(merged),"overlap_area_km2":overlap_km2,"suspicious_findings":suspicious},"confidence_assessment":"Pending manual inspection; automated geometry checks do not establish legal administrative correctness."}
(PROCESSED/Path(OUT["boundary_candidates"]).name).write_text(json.dumps({"nominatim_candidates":[safe(c) for c in candidates],"overpass_candidates":overpass_meta,"failures":failures},indent=2))
(PROCESSED/Path(OUT["validation"]).name).write_text(json.dumps(report,indent=2))
md="# Belagavi boundary validation\n\n"
md+=f"- geometry_checks_passed: `{report['geometry_checks_passed']}`\n- administrative_boundary_verified: `false`\n- boundary_approved: `false`\n\n"
md+="## Official reference comparison\n\nThe City Corporation ward page and Cantonment Board pages were retained as comparison references. The supplied Cantonment city-map URL did not expose a downloadable boundary during this run. These references do not by themselves prove the OSM geometry is legally correct.\n\n"
md+="## Selected sources\n\n"
for title,obj in (("City Corporation",city_sel),("Cantonment",cantt_sel)):
    md+=f"### {title}\n\n"
    md+=json.dumps(safe(obj),indent=2)+"\n\n"
md+="## Merged coverage\n\n"+json.dumps(report["merged"],indent=2)+"\n\n"
md+="## Rejected candidates\n\n"+json.dumps(rejected,indent=2)+"\n\n"
md+="## Manual verification required\n\nConfirm that the selected City Corporation geometry covers the full corporation rather than only a locality, and that the Cantonment geometry includes both Main Cantonment and Fort Zone. Do not set `boundary_approved` automatically.\n"
(PROCESSED/Path(OUT["validation_markdown"]).name).write_text(md)
print(json.dumps({"geometry_checks_passed":report["geometry_checks_passed"],"city_area_km2":city_sel.get("area_km2") if city_sel else None,"cantonment_area_km2":cantt_sel.get("area_km2") if cantt_sel else None,"merged_area_km2":report["merged"]["area_km2"],"overlap_area_km2":overlap_km2,"preview":str(map_path)},indent=2))
