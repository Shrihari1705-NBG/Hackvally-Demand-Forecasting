"""Build leakage-safe one-hour-ahead forecasting features."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def build_forecasting_features(config: dict | None = None, hourly_path: Path | str | None = None,
                               context_path: Path | str | None = None, strict: bool = True) -> tuple[pd.DataFrame, dict]:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    forecasting = config.get("forecasting", {})
    target = forecasting.get("target_column", "demand_count")
    timezone_name = forecasting.get("timezone", config.get("timezone", "Asia/Kolkata"))
    hourly_path = Path(hourly_path or ROOT / config["output"]["hourly_h3_demand"])
    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly demand input is missing: {hourly_path}")
    frame = pd.read_csv(hourly_path)
    required = {"h3_index", "hourly_interval", target, "region_id", "region_name", "center_lat", "center_lon", "is_synthetic", "geography_mode",
                "hour", "day_of_week", "is_weekend", "day_of_month", "month", "week_of_year", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Hourly dataset is missing feature columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["hourly_interval"], utc=True).dt.tz_convert(timezone_name)
    frame = frame.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    historical_start = frame["timestamp"].min().isoformat()
    historical_end = frame["timestamp"].max().isoformat()
    expected_cells = frame["h3_index"].nunique()
    expected_intervals = frame["timestamp"].nunique()
    if strict and (expected_cells != 94 or expected_intervals != 4320):
        raise ValueError(f"Expected 94 cells and 4320 intervals, found {expected_cells} and {expected_intervals}")
    if frame.groupby("h3_index")["timestamp"].nunique().nunique() != 1:
        raise ValueError("Every H3 cell must have a complete equal-length hourly sequence")
    if frame[["h3_index", "timestamp"]].duplicated().any():
        raise ValueError("Duplicate H3/timestamp rows in hourly input")
    lag_hours = [int(x) for x in forecasting.get("lag_hours", [1, 24, 168])]
    rolling_windows = [int(x) for x in forecasting.get("rolling_windows", [3, 24, 168])]
    group = frame.groupby("h3_index", sort=False)[target]
    for lag in lag_hours:
        frame[f"demand_lag_{lag}"] = group.shift(lag)
    for window in rolling_windows:
        frame[f"demand_rolling_mean_{window}h"] = group.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean())
    lag_features = [f"demand_lag_{lag}" for lag in lag_hours]
    rolling_features = [f"demand_rolling_mean_{window}h" for window in rolling_windows]
    model_features = ["hour", "day_of_week", "is_weekend", "day_of_month", "month", "week_of_year", "hour_sin", "hour_cos",
                      "day_of_week_sin", "day_of_week_cos", "h3_index", "region_id", "center_lat", "center_lon"] + lag_features + rolling_features
    if "context_data_available" in frame.columns:
        model_features.append("context_data_available")
    poi_numeric = [c for c in frame.columns if c.endswith("_count") and c not in {target}]
    if frame.get("context_source_status", pd.Series(dtype=str)).astype(str).isin(["success", "cache_reused"]).any():
        model_features += poi_numeric
    model_features = list(dict.fromkeys(model_features))
    removed = int(frame[lag_features + rolling_features].isna().any(axis=1).sum())
    features = frame.dropna(subset=lag_features + rolling_features).copy()
    feature_frame = features[["h3_index", "timestamp", target, "region_id", "region_name", "center_lat", "center_lon", "is_synthetic", "geography_mode"] + model_features].copy()
    # Remove duplicate columns introduced by the geography/feature lists.
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
    timestamps = pd.DatetimeIndex(feature_frame["timestamp"].drop_duplicates().sort_values())
    fractions = [float(forecasting.get("train_fraction", .70)), float(forecasting.get("validation_fraction", .15)), float(forecasting.get("test_fraction", .15))]
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Forecasting train/validation/test fractions must sum to 1")
    train_end = int(len(timestamps) * fractions[0])
    validation_end = train_end + int(len(timestamps) * fractions[1])
    if train_end <= 0 or validation_end >= len(timestamps):
        raise ValueError("Forecasting split fractions produce an empty partition")
    split_map = {"train": (timestamps[0], timestamps[train_end - 1]), "validation": (timestamps[train_end], timestamps[validation_end - 1]),
                 "test": (timestamps[validation_end], timestamps[-1])}
    feature_frame["split"] = np.select([feature_frame.timestamp <= split_map["train"][1], feature_frame.timestamp <= split_map["validation"][1]], ["train", "validation"], default="test")
    if strict and set(feature_frame.groupby("timestamp").split.nunique()) != {1}:
        raise ValueError("A timestamp was assigned to multiple chronological partitions")
    metadata = {"target_column": target, "timezone": timezone_name, "historical_start": historical_start, "historical_end": historical_end,
                "forecast_horizon_hours": int(forecasting.get("forecast_horizon_hours", 1)),
                "lag_hours": lag_hours, "rolling_windows": rolling_windows, "feature_columns": model_features,
                "categorical_features": [c for c in ["h3_index", "region_id", "context_data_available"] if c in model_features],
                "numeric_features": [c for c in model_features if c not in {"h3_index", "region_id", "context_data_available"}],
                "rows_before_history_trim": int(len(frame)), "rows_removed_insufficient_history": removed, "rows_after_history_trim": int(len(feature_frame)),
                "cells": int(expected_cells), "hourly_intervals_before_trim": int(expected_intervals),
                "split_timestamps": {name: {"start": bounds[0].isoformat(), "end": bounds[1].isoformat(),
                                             "unique_timestamps": int((feature_frame.split == name).sum() / expected_cells)} for name, bounds in split_map.items()},
                "split_rows": {name: int((feature_frame.split == name).sum()) for name in ["train", "validation", "test"]},
                "leakage_excluded_fields": ["demand_count", "ride_id", "request_outcome", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon", "destination_h3_index", "is_valid_demand"],
                "no_lag_or_rolling_future_fill": True}
    return feature_frame, metadata


def save_features(features: pd.DataFrame, metadata: dict, config: dict | None = None) -> None:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]["forecasting_features"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    features.to_csv(temp, index=False)
    os.replace(temp, output)
    meta_path = ROOT / config["output"]["forecasting_feature_metadata"]
    meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    features, metadata = build_forecasting_features(config)
    save_features(features, metadata, config)
    print(json.dumps(metadata, indent=2))
