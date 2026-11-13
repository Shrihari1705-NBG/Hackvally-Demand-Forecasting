"""Load the Phase 7A artifact and produce non-negative demand predictions."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/demand_forecasting/model.joblib"
DEFAULT_SCHEMA = ROOT / "models/demand_forecasting/feature_schema.json"


def predict_demand(feature_frame: pd.DataFrame, model_path: Path | str = DEFAULT_MODEL,
                   schema_path: Path | str = DEFAULT_SCHEMA) -> pd.DataFrame:
    model_path, schema_path = Path(model_path), Path(schema_path)
    if not model_path.exists() or not schema_path.exists():
        raise FileNotFoundError("Saved demand model and feature schema are required")
    artifact = joblib.load(model_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    missing = set(schema["feature_columns"]) - set(feature_frame.columns)
    if missing:
        raise ValueError(f"Prediction feature frame is missing columns: {sorted(missing)}")
    predictions = np.maximum(0.0, artifact["pipeline"].predict(feature_frame[schema["feature_columns"]]))
    if len(predictions) != len(feature_frame):
        raise ValueError("Prediction row count does not match input row count")
    output = pd.DataFrame({"h3_index": feature_frame["h3_index"].astype(str).to_numpy(),
                          "timestamp": feature_frame["timestamp"].to_numpy(),
                          "predicted_demand": predictions})
    return output

