from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, subprocess, sys

ROOT=Path(__file__).resolve().parent
STAGES={"geography":"src/01_extract_geography.py","operational-area":"src/01b_design_operational_area.py","validate-boundary":"src/05_validate_operational_boundary.py","synthetic-geography":"src/06_generate_synthetic_geography.py","osm-context":"src/07_extract_osm_context.py","synthetic-rides":"src/08_generate_synthetic_rides.py","aggregate-hourly-demand":"src/09_aggregate_hourly_demand.py","train-demand-model":"src/12_train_demand_model.py","pois":"src/02_extract_pois.py","h3":"src/03_build_h3_grid.py","rides":"src/04_generate_synthetic_rides.py"}
OUTPUTS={"geography":["boundary","validation"],"operational-area":["operational_boundary_draft","logical_regions","region_definitions","operational_validation","operational_validation_markdown","operational_preview_png","operational_preview_html","operational_source_features"],"validate-boundary":["operational_validation","operational_validation_markdown","operational_preview_png","operational_preview_html","normalized_operational_boundary"],"synthetic-geography":["synthetic_h3_cells","synthetic_h3_cells_csv","synthetic_region_mapping","synthetic_geography_metadata","synthetic_geography_validation","synthetic_geography_preview_png","synthetic_geography_preview_html"],"osm-context":["osm_pois","osm_context_features","osm_poi_features_csv","h3_poi_context","osm_context_metadata","osm_context_validation"],"synthetic-rides":["synthetic_rides","synthetic_ride_metadata","synthetic_ride_report","synthetic_rides_sample","synthetic_ride_summary_png"],"aggregate-hourly-demand":["hourly_h3_demand","hourly_demand_metadata","hourly_demand_validation","hourly_demand_summary_png"],"train-demand-model":["forecasting_features","forecasting_feature_metadata","forecast_model","forecast_feature_schema","forecast_model_metadata","forecast_evaluation_metrics","forecast_region_metrics","forecast_hourly_metrics","forecast_diagnostics","forecast_report"],"pois":["pois","context"],"h3":["h3"],"rides":["rides","hourly"]}
REQUIRED=["geopandas","shapely","h3","pandas","numpy","pyproj","pyogrio"]

def check_dependencies(require_osm=False, require_ml=False):
    required=REQUIRED + (["osmnx"] if require_osm else []) + (["sklearn","joblib"] if require_ml else [])
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing: raise SystemExit("Missing dependencies: "+", ".join(missing)+". Install requirements.txt first.")

def read_config(): return json.loads((ROOT/"config.json").read_text())

def require_approval():
    c=read_config()
    required=("geometry_checks_passed","operational_coverage_reviewed","boundary_approved")
    if not all(c.get(key) is True for key in required):
        raise SystemExit("STOP: geometry_checks_passed, operational_coverage_reviewed and boundary_approved must all be explicitly true. Validate and manually inspect the boundary first.")

def approve_boundary():
    c=read_config(); report_path=ROOT/c["output"]["operational_validation"]
    if not report_path.exists(): raise SystemExit("No manual operational-boundary validation report exists. Run `python run_pipeline.py validate-boundary` first.")
    report=json.loads(report_path.read_text())
    if report.get("geometry_checks_passed") is not True: raise SystemExit("Geometry checks have not passed.")
    if report.get("anchor_coverage_passed") is not True: raise SystemExit("Not all eight logical-region anchors are covered.")
    if report.get("approval_blockers"): raise SystemExit("Approval is blocked by the validation report: " + "; ".join(report["approval_blockers"]))
    answer=input("Type APPROVE to attest that you manually reviewed the operational boundary: ").strip()
    if answer != "APPROVE": raise SystemExit("Approval cancelled; config.json was not changed.")
    c["geometry_checks_passed"]=True
    c["operational_coverage_reviewed"]=True
    c["boundary_approved"]=True
    c["boundary_approval_date"]=datetime.now(timezone.utc).isoformat()
    c["boundary_version"]=c.get("boundary_version") or "manual-"+datetime.now(timezone.utc).strftime("%Y%m%d")
    c["boundary_approval"]={"geometry_checks_passed":True,"administrative_boundary_verified":False,"boundary_approved":False}
    (ROOT/"config.json").write_text(json.dumps(c,indent=2)+"\n")
    print("Operational boundary approved in config.json by explicit user confirmation.")

parser=argparse.ArgumentParser(description="Run KasuCab pipeline stages safely.")
parser.add_argument("stage",nargs="?",choices=["all",*STAGES,"approve-boundary"],default="all")
parser.add_argument("--force",action="store_true",help="Allow overwriting existing stage outputs")
args=parser.parse_args()
if args.stage=="approve-boundary": approve_boundary(); raise SystemExit(0)
current_config=read_config()
osm_enabled=current_config.get("osm_context", {}).get("enabled", False)
osm_cache=ROOT/current_config.get("osm_context", {}).get("cache_path", "data/cache/osm_context_features.geojson")
requires_osm=(args.stage == "geography" or args.stage == "all" or
              (args.stage == "osm-context" and osm_enabled and not osm_cache.exists()))
check_dependencies(require_osm=requires_osm, require_ml=args.stage == "train-demand-model")
if args.stage=="validate-boundary":
    print("\n--- stage: validate-boundary ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage validate-boundary failed with exit code {exc.returncode}")
    raise SystemExit(0)
if args.stage=="synthetic-geography":
    c=read_config()
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage synthetic-geography would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print("\n--- stage: synthetic-geography ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage synthetic-geography failed with exit code {exc.returncode}")
    missing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if not (ROOT/c["output"][key]).exists()]
    if missing: raise SystemExit("Stage synthetic-geography did not create expected outputs: " + ", ".join(missing))
    print("Synthetic geography outputs validated and written.")
    raise SystemExit(0)
if args.stage=="osm-context":
    c=read_config()
    settings=c.get("osm_context", {})
    if settings.get("enabled") is False:
        print("OSM context is disabled; writing zero-filled context outputs with status=disabled.")
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage osm-context would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print("\n--- stage: osm-context ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage osm-context failed with exit code {exc.returncode}")
    missing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if not (ROOT/c["output"][key]).exists()]
    if missing: raise SystemExit("Stage osm-context did not create expected outputs: " + ", ".join(missing))
    print("OSM context outputs validated and written.")
    raise SystemExit(0)
if args.stage=="synthetic-rides":
    c=read_config()
    geography_path=ROOT/c["output"]["synthetic_h3_cells_csv"]
    if not geography_path.exists(): raise SystemExit("Synthetic geography is missing. Run `python run_pipeline.py synthetic-geography` first.")
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage synthetic-rides would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print("\n--- stage: synthetic-rides ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage synthetic-rides failed with exit code {exc.returncode}")
    missing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if not (ROOT/c["output"][key]).exists()]
    if missing: raise SystemExit("Stage synthetic-rides did not create expected outputs: " + ", ".join(missing))
    print("Synthetic ride outputs validated and written; no hourly aggregation was run.")
    raise SystemExit(0)
if args.stage=="aggregate-hourly-demand":
    c=read_config()
    for key in ("synthetic_h3_cells_csv","synthetic_rides"):
        if not (ROOT/c["output"][key]).exists(): raise SystemExit(f"Required input is missing: {ROOT/c['output'][key]}")
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage aggregate-hourly-demand would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print("\n--- stage: aggregate-hourly-demand ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage aggregate-hourly-demand failed with exit code {exc.returncode}")
    missing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if not (ROOT/c["output"][key]).exists()]
    if missing: raise SystemExit("Stage aggregate-hourly-demand did not create expected outputs: " + ", ".join(missing))
    print("Hourly demand panel validated and written; no ML training was run.")
    raise SystemExit(0)
if args.stage=="train-demand-model":
    c=read_config()
    hourly_path=ROOT/c["output"]["hourly_h3_demand"]
    if not hourly_path.exists(): raise SystemExit("Hourly demand input is missing. Run `python run_pipeline.py aggregate-hourly-demand` first.")
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage train-demand-model would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print("\n--- stage: train-demand-model ---",flush=True)
    try:
        subprocess.run([sys.executable,str(ROOT/"src/10_build_forecasting_features.py")],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/STAGES[args.stage])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage train-demand-model failed with exit code {exc.returncode}")
    missing=[str(ROOT/c["output"][key]) for key in OUTPUTS[args.stage] if not (ROOT/c["output"][key]).exists()]
    if missing: raise SystemExit("Stage train-demand-model did not create expected outputs: " + ", ".join(missing))
    print("Demand model artifacts validated; no API or frontend integration was run.")
    raise SystemExit(0)
if args.stage not in {"geography","operational-area"}: require_approval()
for name in (list(STAGES) if args.stage=="all" else [args.stage]):
    if name not in {"geography","operational-area","validate-boundary"}: require_approval()
    c=read_config()
    existing=[str(ROOT/c["output"][key]) for key in OUTPUTS[name] if (ROOT/c["output"][key]).exists()]
    if existing and not args.force: raise SystemExit(f"Stage {name} would overwrite existing outputs: {', '.join(existing)}. Re-run with --force explicitly.")
    print(f"\n--- stage: {name} ---",flush=True)
    try: subprocess.run([sys.executable,str(ROOT/STAGES[name])],cwd=ROOT,check=True)
    except subprocess.CalledProcessError as exc: raise SystemExit(f"Stage {name} failed with exit code {exc.returncode}")
print("\nPipeline stage(s) completed.")
