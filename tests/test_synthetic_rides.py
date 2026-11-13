import importlib.util
from pathlib import Path

import h3
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "08_generate_synthetic_rides.py"
spec = importlib.util.spec_from_file_location("synthetic_rides", MODULE_PATH)
rides = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rides)

CELLS = Path(__file__).parents[1] / "data/processed/synthetic_h3_cells.csv"


def small_config():
    return {"end_date": "2026-11-12", "timezone": "Asia/Kolkata", "random_seed": 42,
            "synthetic_rides": {"enabled": True, "historical_days": 180, "random_seed": 42,
                "base_requests_per_cell_hour": 0.05, "max_point_sampling_attempts": 500,
                "destination_distance_scale_km": 6.0, "same_region_destination_bonus": 2.0}}


def test_deterministic_generation_and_unique_ids():
    first = rides.generate_synthetic_rides(small_config(), write_outputs=False)["rides"]
    second = rides.generate_synthetic_rides(small_config(), write_outputs=False)["rides"]
    assert first.equals(second)
    assert first.ride_id.is_unique


def test_h3_membership_and_nonnegative_distance():
    frame = rides.generate_synthetic_rides(small_config(), write_outputs=False)["rides"]
    assert all(h3.latlng_to_cell(lat, lon, 8) == cell for lat, lon, cell in zip(frame.pickup_lat, frame.pickup_lon, frame.pickup_h3_index))
    assert all(h3.latlng_to_cell(lat, lon, 8) == cell for lat, lon, cell in zip(frame.destination_lat, frame.destination_lon, frame.destination_h3_index))
    assert (frame.trip_distance_km_geodesic >= 0).all()


def test_timezone_period_and_outcomes():
    frame = rides.generate_synthetic_rides(small_config(), write_outputs=False)["rides"]
    timestamps = pd.to_datetime(frame.request_timestamp, utc=True).dt.tz_convert("Asia/Kolkata")
    assert str(timestamps.dt.tz) in {"Asia/Calcutta", "Asia/Kolkata"}
    assert timestamps.dt.date.min().isoformat() == "2026-05-17"
    assert timestamps.dt.date.max().isoformat() == "2026-11-12"
    assert set(frame.request_outcome) <= set(rides.OUTCOMES)


def test_missing_optional_context_does_not_block():
    result = rides.generate_synthetic_rides(small_config(), write_outputs=False)
    assert result["metadata"]["osm_context_used"] is False


def test_invalid_geography_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"h3_index": ["not-a-cell"], "region_id": ["R01"], "region_name": ["x"], "center_lat": [0], "center_lon": [0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        rides.load_cells(bad)


def test_empty_geography_is_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["h3_index", "region_id", "region_name", "center_lat", "center_lon"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="empty"):
        rides.load_cells(empty)
