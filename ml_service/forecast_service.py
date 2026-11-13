"""Inference orchestration for forecast, map and region responses."""
from __future__ import annotations

import math
import time

import pandas as pd

from .dependencies import ModelBundle
from .feature_builder import build_features
from .schemas import ForecastRequest


def predict(request: ForecastRequest, bundle: ModelBundle) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    started = time.perf_counter()
    frame, forecast_ts = build_features(request, bundle)
    predictions = bundle.pipeline.predict(frame[bundle.schema["feature_columns"]])
    frame["predicted_demand"] = [max(0.0, float(value)) if math.isfinite(float(value)) else 0.0 for value in predictions]
    return frame, forecast_ts, time.perf_counter() - started


def prediction_records(frame: pd.DataFrame) -> list[dict]:
    return frame[["h3_index", "region_id", "region_name", "predicted_demand"]].to_dict("records")
