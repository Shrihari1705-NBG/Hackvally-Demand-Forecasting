"""Build the exact Phase 7A model features from validated 168-hour histories."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config
from .dependencies import ModelBundle
from .schemas import ForecastRequest


def _normalise_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware and use Asia/Kolkata")
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 19800:
        raise ValueError("Timestamps must use Asia/Kolkata (+05:30)")
    return timestamp.tz_convert(config.TIMEZONE)


def _calendar(timestamp: pd.Timestamp) -> dict:
    hour = int(timestamp.hour)
    dow = int(timestamp.dayofweek)
    return {
        "hour": hour, "day_of_week": dow, "is_weekend": bool(dow >= 5),
        "day_of_month": int(timestamp.day), "month": int(timestamp.month),
        "week_of_year": int(timestamp.isocalendar().week),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "day_of_week_sin": float(np.sin(2 * np.pi * dow / 7)),
        "day_of_week_cos": float(np.cos(2 * np.pi * dow / 7)),
    }


def build_features(request: ForecastRequest, bundle: ModelBundle) -> tuple[pd.DataFrame, pd.Timestamp]:
    forecast_ts = _normalise_timestamp(request.forecast_timestamp)
    if request.geography_mode != config.EXPECTED_GEOGRAPHY_MODE:
        raise ValueError(f"Unsupported geography_mode: {request.geography_mode}")
    if len(request.cell_history) != len(bundle.geography):
        raise ValueError(f"A full-map forecast requires all {len(bundle.geography)} known H3 cells")
    if len({cell.h3_index for cell in request.cell_history}) != len(request.cell_history):
        raise ValueError("Duplicate H3 cells are not allowed")
    geo = bundle.geography.set_index("h3_index")
    rows = []
    expected_latest = forecast_ts - timedelta(hours=1)
    for cell in request.cell_history:
        if cell.h3_index not in geo.index:
            raise ValueError(f"Unknown H3 cell: {cell.h3_index}")
        reference = geo.loc[cell.h3_index]
        if str(cell.region_id) != str(reference.region_id) or str(cell.region_name) != str(reference.region_name):
            raise ValueError(f"Region metadata does not match synthetic geography for {cell.h3_index}")
        if abs(cell.center_lat - float(reference.center_lat)) > 1e-6 or abs(cell.center_lon - float(reference.center_lon)) > 1e-6:
            raise ValueError(f"Center coordinates do not match synthetic geography for {cell.h3_index}")
        history = [_normalise_timestamp(obs.timestamp) for obs in cell.history]
        if any(history[i] - history[i - 1] != pd.Timedelta(hours=1) for i in range(1, len(history))):
            raise ValueError(f"History for {cell.h3_index} is not 168 consecutive hourly observations")
        if history[-1] != expected_latest:
            raise ValueError(f"History for {cell.h3_index} must end immediately before forecast_timestamp")
        values = np.asarray([obs.demand_count for obs in cell.history], dtype=float)
        row = {"h3_index": cell.h3_index, "region_id": cell.region_id, "region_name": cell.region_name,
               "center_lat": float(reference.center_lat),
               "center_lon": float(reference.center_lon), "demand_lag_1": values[-1],
               "demand_lag_24": values[-24], "demand_lag_168": values[0],
               "demand_rolling_mean_3h": float(values[-3:].mean()),
               "demand_rolling_mean_24h": float(values[-24:].mean()),
               "demand_rolling_mean_168h": float(values.mean()),
               "context_data_available": False}
        row.update(_calendar(forecast_ts))
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = list(bundle.schema["feature_columns"])
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Saved model requires unavailable inference features: {missing}")
    if frame[expected].isna().any().any():
        raise ValueError("Inference feature frame contains missing values")
    for column in bundle.schema.get("numeric_features", []):
        if not pd.api.types.is_numeric_dtype(frame[column]) or not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Inference feature has an invalid numeric dtype or value: {column}")
    for column in bundle.schema.get("categorical_features", []):
        if not pd.api.types.is_string_dtype(frame[column]) and not pd.api.types.is_object_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"Inference feature has an invalid categorical dtype: {column}")
    return frame, forecast_ts
