from pathlib import Path
import json, geopandas as gpd, pandas as pd, h3
from shapely.geometry import Polygon, Point
from pipeline_utils import cells_intersecting_boundary, parse_bool, polygonal_geometry, validate_resolution
ROOT=Path(__file__).resolve().parents[1]; C=json.loads((ROOT/"config.json").read_text())
poly=polygonal_geometry(gpd.read_file(ROOT/C["output"]["boundary"]).to_crs(4326).geometry.union_all())
res=int(C["h3_resolution"]); cells=sorted(h3.geo_to_cells(poly,res))
cells=sorted(cells_intersecting_boundary(poly,res))
if not cells: raise SystemExit("No H3 cells; inspect boundary.")
records=[]
for cell in cells:
    validate_resolution(cell,res)
    ring=[(lng,lat) for lat,lng in h3.cell_to_boundary(cell)]
    ring.append(ring[0]); geom=Polygon(ring).intersection(poly)
    if geom.is_empty: raise SystemExit(f"Empty intersection for H3 cell {cell}")
    if not geom.is_valid: geom=geom.buffer(0)
    if geom.is_empty or not geom.is_valid: raise SystemExit(f"Invalid clipped geometry for H3 cell {cell}")
    lat,lon=h3.cell_to_latlng(cell)
    records.append({"h3_index":cell,"resolution":res,"center_lat":lat,"center_lon":lon,
      "cell_area_km2":h3.cell_area(cell,unit="km^2"),
      "center_inside_boundary":bool(poly.covers(Point(lon,lat))),
      "boundary_intersects":True,"geometry":geom})
gdf=gpd.GeoDataFrame(records,geometry="geometry",crs=4326)
out=ROOT/C["output"]["h3"]; out.parent.mkdir(parents=True,exist_ok=True)
gdf.to_file(out,driver="GeoJSON")
gdf.drop(columns="geometry").to_csv(out.with_suffix(".csv"),index=False)
print(f"Created {len(gdf):,} H3 resolution-{res} cells.")
