from pathlib import Path
import json
import geopandas as gpd
import osmnx as ox

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/"config.json").read_text())
bp=ROOT/C["output"]["boundary"]; poi_out=ROOT/C["output"]["pois"]; context_out=ROOT/C["output"]["context"]
if not bp.exists(): raise SystemExit("Run 01_extract_geography.py first")
poly=gpd.read_file(bp).to_crs(4326).geometry.union_all()
ox.settings.timeout=240

def extract(tags, out, dataset):
    print(f"Querying OSM Overpass for {dataset} features...")
    g=ox.features_from_polygon(poly,tags)
    if g.empty: raise SystemExit(f"No OSM {dataset} features returned; check Overpass availability.")
    g=g.reset_index(); g=g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    rp=g.geometry.representative_point(); g["lat"]=rp.y; g["lon"]=rp.x
    if "element" in g: g["osm_type"]=g["element"].astype(str)
    if "id" in g: g["osm_id"]=g["id"].astype(str)
    g["dataset_type"]=dataset
    tag_columns=[x for x in g.columns if x not in {"geometry","lat","lon"}]
    g["osm_tags_json"]=g[tag_columns].apply(lambda row: json.dumps({k:(v if isinstance(v,(str,int,float,bool)) or v is None else str(v)) for k,v in row.items()}, default=str), axis=1)
    cols=[x for x in ["osm_type","osm_id","name","dataset_type","amenity","shop","office","leisure","tourism","public_transport","railway","highway","landuse","building","lat","lon","osm_tags_json","geometry"] if x in g]
    out.parent.mkdir(parents=True,exist_ok=True); g[cols].to_file(out,driver="GeoJSON")
    print(f"Saved {len(g):,} OSM {dataset} features: {out}")

extract(C["osm_poi_tags"],poi_out,"poi")
extract(C["osm_context_tags"],context_out,"context")
