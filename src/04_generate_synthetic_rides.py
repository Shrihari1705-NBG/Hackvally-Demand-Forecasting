from pathlib import Path
import json, numpy as np, pandas as pd, geopandas as gpd, h3
from pipeline_utils import parse_bool, polygonal_geometry, sample_point_in_cell, validate_resolution
ROOT=Path(__file__).resolve().parents[1]; C=json.loads((ROOT/"config.json").read_text())
rng=np.random.default_rng(int(C["random_seed"])); tz=C["timezone"]; res=int(C["h3_resolution"])
boundary=gpd.read_file(ROOT/C["output"]["boundary"]).to_crs(4326); poly=polygonal_geometry(boundary.geometry.union_all())
pois=gpd.read_file(ROOT/C["output"]["pois"]).to_crs(4326)
cells=pd.read_csv((ROOT/C["output"]["h3"]).with_suffix(".csv"))
cells=cells[cells.boundary_intersects.map(parse_bool)].copy()
ids=cells.h3_index.astype(str).tolist()
if not ids: raise SystemExit("No eligible H3 cells.")
centers={r.h3_index:(float(r.center_lat),float(r.center_lon)) for r in cells.itertuples()}
poi_counts={}
for r in pois.itertuples():
    cell=h3.latlng_to_cell(float(r.lat),float(r.lon),res); poi_counts[cell]=poi_counts.get(cell,0)+1
context=gpd.read_file(ROOT/C["output"]["context"]).to_crs(4326)
context_counts={}
for r in context.itertuples():
    cell=h3.latlng_to_cell(float(r.lat),float(r.lon),res); context_counts[cell]=context_counts.get(cell,0)+1
weights=np.array([1+np.log1p(poi_counts.get(c,0))+0.25*np.log1p(context_counts.get(c,0)) for c in ids],float); weights/=weights.sum()
end=pd.Timestamp(C["end_date"],tz=tz).normalize()+pd.Timedelta(days=1)
start=end-pd.Timedelta(days=int(C["history_days"]))
hours=pd.date_range(start,end,freq="h",inclusive="left")
base=int(C["base_requests_per_hour"])
hf=[.25,.15,.10,.08,.10,.25,.55,.85,.95,.80,.70,.75,.85,.80,.72,.78,.95,1.25,1.45,1.35,1.10,.85,.60,.40]
wf={0:1,1:1,2:1,3:1,4:1.08,5:1.18,6:.92}
rows=[]; rid=0
for ts in hours:
    h=ts.hour; dow=ts.dayofweek
    bump=1.15 if dow in (5,6) and 11<=h<=21 else 1.0
    if dow<5 and h in (7,8,9,17,18,19): bump*=1.12
    n=int(rng.poisson(max(0,base*hf[h]*wf[dow]*rng.lognormal(0,.16)*bump)))
    if not n: continue
    selected=rng.choice(len(ids),size=n,p=weights)
    for ix in selected:
        cell=ids[int(ix)]; lat,lon=sample_point_in_cell(cell,poly,rng)
        validate_resolution(cell,res)
        if h3.latlng_to_cell(lat,lon,res) != cell: raise AssertionError("Pickup H3 mismatch")
        dest=ids[int(rng.choice(len(ids),p=weights))]; dlat,dlon=sample_point_in_cell(dest,poly,rng)
        if h3.latlng_to_cell(dlat,dlon,res) != dest: raise AssertionError("Destination H3 mismatch")
        rid+=1
        rows.append({"ride_id":f"SYN-{rid:09d}",
         "requested_at_local":(ts+pd.Timedelta(minutes=int(rng.integers(60)))).isoformat(),
          "pickup_lat":lat,"pickup_lon":lon,"pickup_h3":cell,
          "destination_lat":dlat,"destination_lon":dlon,"destination_h3":dest,
         "status":rng.choice(["completed","cancelled","rejected"],p=[.78,.12,.10]),
         "is_synthetic":True,"synthetic_profile":"Belagavi simulation v1"})
rides=pd.DataFrame(rows)
if rides.empty: raise SystemExit("No rides generated; increase base_requests_per_hour.")
rp=ROOT/C["output"]["rides"]; rp.parent.mkdir(parents=True,exist_ok=True); rides.to_csv(rp,index=False)
rides["requested_at_local"]=pd.to_datetime(rides.requested_at_local,utc=True).dt.tz_convert(tz)
rides["hour_start_local"]=rides.requested_at_local.dt.floor("h")
counts=rides.groupby(["pickup_h3","hour_start_local"]).size().rename("request_count").reset_index()
full=pd.MultiIndex.from_product([ids,hours],names=["pickup_h3","hour_start_local"]).to_frame(index=False)
hourly=full.merge(counts,on=["pickup_h3","hour_start_local"],how="left")
hourly["request_count"]=hourly.request_count.fillna(0).astype(int)
hourly["is_synthetic"]=True; hourly["h3_resolution"]=res
hp=ROOT/C["output"]["hourly"]; hourly.to_csv(hp,index=False)
print(f"{len(rides):,} synthetic rides; {len(ids):,} cells; {len(hours):,} hours.")
print("Rides:",rp); print("Hourly panel:",hp)
