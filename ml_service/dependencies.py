"""Application dependencies and startup model bundle."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib
import pandas as pd

from . import config

LOGGER = logging.getLogger("kasucab.ml_service")


@dataclass
class ModelBundle:
    artifact: dict
    schema: dict
    metadata: dict
    metrics: dict
    geography: pd.DataFrame
    geography_geojson: dict | None

    @property
    def pipeline(self):
        return self.artifact["pipeline"]


def load_model_bundle() -> ModelBundle:
    required = [config.MODEL_PATH, config.SCHEMA_PATH, config.METADATA_PATH, config.GEOGRAPHY_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inference artifacts are missing: " + ", ".join(missing))
    artifact = joblib.load(config.MODEL_PATH)
    schema = config.load_json(config.SCHEMA_PATH)
    metadata = config.load_json(config.METADATA_PATH)
    metrics = config.load_json(config.METRICS_PATH) if config.METRICS_PATH.exists() else {}
    if "pipeline" not in artifact or not schema.get("feature_columns"):
        raise ValueError("Model artifact or feature schema is invalid")
    if list(artifact.get("feature_columns", [])) != list(schema["feature_columns"]):
        raise ValueError("Model artifact feature schema does not match feature_schema.json")
    geography = pd.read_csv(config.GEOGRAPHY_PATH, dtype={"h3_index": str, "region_id": str})
    required_geo = {"h3_index", "region_id", "region_name", "center_lat", "center_lon", "geography_mode"}
    if not required_geo.issubset(geography.columns) or geography["h3_index"].duplicated().any():
        raise ValueError("Synthetic geography metadata is missing required unique cell fields")
    if len(geography) != 94 or set(geography["geography_mode"].astype(str)) != {config.EXPECTED_GEOGRAPHY_MODE}:
        raise ValueError("Synthetic geography metadata does not match the expected 94-cell geography")
    geojson = None
    if config.GEOGRAPHY_GEOJSON_PATH.exists():
        geojson = json.loads(config.GEOGRAPHY_GEOJSON_PATH.read_text(encoding="utf-8"))
    LOGGER.info("Loaded model version %s with %d geography cells", metadata.get("model_version", "unknown"), len(geography))
    return ModelBundle(artifact, schema, metadata, metrics, geography, geojson)
