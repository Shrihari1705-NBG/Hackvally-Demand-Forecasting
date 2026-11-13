# KasuCab — Belagavi geography and synthetic demand pipeline

## Locked setup
- Belagavi City Corporation + Belgaum Cantonment
- H3 resolution 8
- 180 days; end date configured as 12 Nov 2026
- OSM POIs only
- Asia/Kolkata timezone; random seed 42

## Run (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py geography
```
Requires internet access to Nominatim and OpenStreetMap Overpass. Do not repeatedly hammer public OSM services; they may throttle requests. Geography extraction does not approve the boundary automatically. After manual inspection, use `python run_pipeline.py approve-boundary`, then run downstream stages explicitly or use `python run_pipeline.py all --force`.

## Generated outputs
- `data/raw/belagavi_city_corporation_plus_cantonment.geojson`
- `data/processed/boundary_validation.json`
- `data/raw/belagavi_osm_pois.geojson`
- `data/raw/belagavi_osm_context.geojson`
- `data/processed/belagavi_h3_r8.geojson` and `.csv`
- `data/processed/synthetic_rides_180d.csv`
- `data/processed/hourly_h3_demand_180d.csv`

## Boundary validation is a required human step
The script geocodes two separate OSM candidates and unions their polygons. It checks geometry validity, approximate area (broad 5–250 km² sanity range), and Belagavi-like coordinate bounds. These checks do not prove legal administrative correctness. Inspect the matched OSM names and output polygon in QGIS/geojson.io and compare against official maps before calling it an authoritative City Corporation + Cantonment boundary. If either geocoder result is wrong or missing, the script stops; it does not substitute a guessed radius or bounding box.

Official comparison references:
- City Corporation ward information: https://belagavicitycorp.org/WebSiteBGM/WardInfromation.aspx
- Cantonment city map: https://belgaum.cantt.gov.in/city-map/

## Synthetic data assumptions
The generator creates individual requests and then aggregates pickups into a complete H3-cell × hour panel, including zero-demand hours. Demand is weighted by OSM feature density and shaped with configurable hour-of-day, weekday, commute and weekend multipliers plus seeded random variation. Request outcomes are sampled as completed/cancelled/rejected; all valid requests count toward demand regardless of outcome.

**This is synthetic simulation data, not observed Belagavi demand.** `base_requests_per_hour` is an adjustable demonstration scale, not a calibrated real-world rate. Do not report model accuracy on it as real-world performance. Replace/calibrate assumptions when KasuCab has genuine request logs. Driver supply is intentionally excluded from demand labels and should be joined later for dispatch decisions.

## Data conventions
- `pickup_h3` is the demand aggregation key; destination has its own H3 key.
- H3 cell inclusion uses centroid containment; output polygon geometries are clipped to the operational boundary.
- H3 grid distance is not road ETA. Use a routing engine for real travel time.
- OSM data attribution: © OpenStreetMap contributors; ODbL: https://www.openstreetmap.org/copyright
