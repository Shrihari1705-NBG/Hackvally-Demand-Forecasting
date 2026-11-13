import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "src" / "09_aggregate_hourly_demand.py"
spec = importlib.util.spec_from_file_location("hourly_demand", MODULE_PATH)
hourly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hourly)

CELLS = Path(__file__).parents[1] / "data/processed/synthetic_h3_cells.csv"


def fixtures(tmp_path):
    geography = pd.read_csv(CELLS).head(2)
    geography.to_csv(tmp_path / "cells.csv", index=False)
    rows = []
    for cell in geography.itertuples():
        rows.extend([
            {"request_timestamp": "2026-11-11T09:00:00+05:30", "pickup_h3_index": cell.h3_index, "is_valid_demand": True, "request_outcome": "completed"},
            {"request_timestamp": "2026-11-11T09:59:59+05:30", "pickup_h3_index": cell.h3_index, "is_valid_demand": True, "request_outcome": "cancelled"},
            {"request_timestamp": "2026-11-11T10:00:00+05:30", "pickup_h3_index": cell.h3_index, "is_valid_demand": True, "request_outcome": "rejected"},
            {"request_timestamp": "2026-11-11T10:30:00+05:30", "pickup_h3_index": cell.h3_index, "is_valid_demand": False, "request_outcome": "expired"},
        ])
    rides = tmp_path / "rides.csv"
    pd.DataFrame(rows).to_csv(rides, index=False)
    config = {"end_date": "2026-11-12", "history_days": 2, "timezone": "Asia/Kolkata"}
    return config, rides, tmp_path / "cells.csv", geography


def test_full_cartesian_panel_and_conservation(tmp_path):
    config, rides, cells, _ = fixtures(tmp_path)
    result = hourly.aggregate_hourly_demand(config, rides, cells, output_dir=tmp_path, write_outputs=False, strict=False)
    panel = result["panel"]
    assert len(panel) == 2 * 2 * 24
    assert panel.demand_count.sum() == 6
    assert (panel.demand_count == 0).any()
    assert panel[["h3_index", "hourly_interval"]].duplicated().sum() == 0


def test_hour_boundary_and_pickup_not_destination(tmp_path):
    config, rides, cells, geography = fixtures(tmp_path)
    frame = pd.read_csv(rides)
    frame["destination_h3_index"] = geography.iloc[1].h3_index
    frame.to_csv(rides, index=False)
    panel = hourly.aggregate_hourly_demand(config, rides, cells, write_outputs=False, strict=False)["panel"]
    counts = panel[(panel.date == "2026-11-11") & (panel.hour.isin([9, 10]))].groupby(["h3_index", "hour"]).demand_count.sum()
    assert set(counts.tolist()) >= {1, 2}


def test_all_outcomes_are_counted_and_invalid_excluded(tmp_path):
    config, rides, cells, _ = fixtures(tmp_path)
    panel = hourly.aggregate_hourly_demand(config, rides, cells, write_outputs=False, strict=False)["panel"]
    assert panel.demand_count.sum() == 6


def test_missing_context_is_allowed(tmp_path):
    config, rides, cells, _ = fixtures(tmp_path)
    result = hourly.aggregate_hourly_demand(config, rides, cells, context_path=tmp_path / "missing_context.csv", write_outputs=False, strict=False)
    assert result["context_status"] == "missing"
    assert len(result["panel"]) == 96


def test_missing_ride_input_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        hourly.aggregate_hourly_demand({"end_date": "2026-11-12", "history_days": 2, "timezone": "Asia/Kolkata"}, tmp_path / "missing.csv", CELLS, write_outputs=False, strict=False)

