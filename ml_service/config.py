"""Configuration and safe repository paths for the inference service."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "demand_forecasting"
MODEL_PATH = MODEL_DIR / "model.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
GEOGRAPHY_PATH = ROOT / "data" / "processed" / "synthetic_h3_cells.csv"
HOURLY_PATH = ROOT / "data" / "processed" / "hourly_h3_demand.csv"
GEOGRAPHY_GEOJSON_PATH = ROOT / "data" / "processed" / "synthetic_h3_cells.geojson"
TIMEZONE = "Asia/Kolkata"
EXPECTED_GEOGRAPHY_MODE = "synthetic_operational"


def allowed_origins() -> list[str]:
    raw = os.getenv("KASUCAB_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
