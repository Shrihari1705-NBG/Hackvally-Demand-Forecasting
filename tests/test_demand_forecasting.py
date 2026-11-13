import importlib.util
from pathlib import Path

import pandas as pd
import pytest


FEATURE_PATH = Path(__file__).parents[1] / "src" / "10_build_forecasting_features.py"
spec = importlib.util.spec_from_file_location("forecast_features", FEATURE_PATH)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def fixture_panel(tmp_path):
    rows = []
    cells = [("cell-a", "R01"), ("cell-b", "R02")]
    timestamps = pd.date_range("2026-01-01", periods=180, freq="h", tz="Asia/Kolkata")
    for cell, region in cells:
        for i, timestamp in enumerate(timestamps):
            rows.append({"h3_index": cell, "hourly_interval": timestamp.isoformat(), "demand_count": i % 5,
                         "region_id": region, "region_name": region, "center_lat": 15.8, "center_lon": 74.5,
                         "is_synthetic": True, "geography_mode": "synthetic_operational", "hour": timestamp.hour,
                         "day_of_week": timestamp.dayofweek, "is_weekend": timestamp.dayofweek >= 5, "day_of_month": timestamp.day,
                         "month": timestamp.month, "week_of_year": int(timestamp.isocalendar().week), "hour_sin": 0.0, "hour_cos": 1.0,
                         "day_of_week_sin": 0.0, "day_of_week_cos": 1.0})
    path = tmp_path / "hourly.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_lags_and_rolling_use_only_prior_values(tmp_path):
    path = fixture_panel(tmp_path)
    config = {"timezone": "Asia/Kolkata", "end_date": "2026-01-08", "forecasting": {"lag_hours": [1, 24, 168], "rolling_windows": [3, 24, 168], "train_fraction": .7, "validation_fraction": .15, "test_fraction": .15}}
    frame, metadata = features.build_forecasting_features(config, path, strict=False)
    row = frame[(frame.h3_index == "cell-a") & (frame.timestamp == pd.Timestamp("2026-01-08 00:00:00", tz="Asia/Kolkata"))].iloc[0]
    assert row.demand_lag_1 == 2
    assert row.demand_lag_24 == 4
    assert row.demand_lag_168 == 0
    assert row.demand_rolling_mean_3h == pytest.approx((0 + 1 + 2) / 3)
    assert metadata["rows_removed_insufficient_history"] == 336


def test_no_target_or_future_features(tmp_path):
    frame, metadata = features.build_forecasting_features({"timezone": "Asia/Kolkata", "end_date": "2026-01-08", "forecasting": {"lag_hours": [1, 24, 168], "rolling_windows": [3, 24, 168], "train_fraction": .7, "validation_fraction": .15, "test_fraction": .15}}, fixture_panel(tmp_path), strict=False)
    assert "demand_count" not in metadata["feature_columns"]
    assert metadata["no_lag_or_rolling_future_fill"] is True
    assert set(frame.split) == {"train", "validation", "test"}
    assert frame.groupby("timestamp").split.nunique().max() == 1


def test_missing_input_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.build_forecasting_features({}, tmp_path / "missing.csv", strict=False)
