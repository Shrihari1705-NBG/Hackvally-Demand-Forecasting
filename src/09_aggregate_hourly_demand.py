"""Aggregate event-level synthetic rides into a complete hourly H3 panel."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h3

from pipeline_utils import parse_bool, validate_resolution

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
RIDE_REQUIRED = {"request_timestamp", "pickup_h3_index", "is_valid_demand", "request_outcome"}
GEO_REQUIRED = {"h3_index", "region_id", "region_name", "center_lat", "center_lon", "is_synthetic", "geography_mode"}
LEAKAGE_FIELDS = ["ride_id", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon", "destination_h3_index",
                  "destination_region_id", "request_outcome", "is_valid_demand"]


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_geography(path: Path | str, strict: bool = True) -> pd.DataFrame:
    geography = pd.read_csv(path)
    missing = GEO_REQUIRED - set(geography.columns)
    if missing:
        raise ValueError(f"Synthetic geography is missing columns: {sorted(missing)}")
    if geography.empty or geography.h3_index.duplicated().any():
        raise ValueError("Synthetic geography must be non-empty with unique H3 IDs")
    if strict and len(geography) != 94:
        raise ValueError(f"Expected 94 synthetic H3 cells, found {len(geography)}")
    for cell in geography.h3_index.astype(str):
        validate_resolution(cell, 8)
    if geography.region_id.nunique() != 8 and strict:
        raise ValueError("Synthetic geography must contain exactly eight analytical regions")
    return geography.sort_values("h3_index").reset_index(drop=True)


def _period(config: dict, strict: bool) -> tuple[pd.DatetimeIndex, date, date, str]:
    days = int(config.get("history_days", config.get("synthetic_rides", {}).get("historical_days", 180)))
    if strict and days != 180:
        raise ValueError(f"Phase 6A expects 180 historical days, found {days}")
    end = date.fromisoformat(str(config["end_date"]))
    start = end - timedelta(days=days - 1)
    timezone_name = config.get("timezone", "Asia/Kolkata")
    ZoneInfo(timezone_name)
    intervals = pd.date_range(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                              inclusive="left", freq="h", tz=timezone_name)
    return intervals, start, end, timezone_name


def _valid_mask(rides: pd.DataFrame) -> pd.Series:
    try:
        return rides["is_valid_demand"].map(parse_bool)
    except ValueError as exc:
        raise ValueError(f"Invalid is_valid_demand value in ride input: {exc}") from exc


def load_rides(path: Path | str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Synthetic ride input is missing: {path}")
    rides = pd.read_csv(path)
    missing = RIDE_REQUIRED - set(rides.columns)
    if missing:
        raise ValueError(f"Synthetic ride input is missing columns: {sorted(missing)}")
    rides["_valid_mask"] = _valid_mask(rides)
    rides["_request_timestamp_parsed"] = pd.to_datetime(rides["request_timestamp"], utc=True, errors="coerce")
    if rides["_request_timestamp_parsed"].isna().any():
        raise ValueError("Synthetic ride input contains invalid request_timestamp values")
    return rides


def _add_time_features(panel: pd.DataFrame, intervals: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(intervals)
    panel["timestamp"] = timestamps
    panel["date"] = timestamps.strftime("%Y-%m-%d")
    panel["hour"] = timestamps.hour.astype("int8")
    panel["day_of_week"] = timestamps.dayofweek.astype("int8")
    panel["day_name"] = timestamps.day_name()
    panel["is_weekend"] = (timestamps.dayofweek >= 5)
    panel["day_of_month"] = timestamps.day.astype("int8")
    panel["month"] = timestamps.month.astype("int8")
    panel["week_of_year"] = timestamps.isocalendar().week.astype("int16").to_numpy()
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"].astype(float) / 24.0)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"].astype(float) / 24.0)
    panel["day_of_week_sin"] = np.sin(2 * np.pi * panel["day_of_week"].astype(float) / 7.0)
    panel["day_of_week_cos"] = np.cos(2 * np.pi * panel["day_of_week"].astype(float) / 7.0)
    return panel


def validate_panel(panel: pd.DataFrame, geography: pd.DataFrame, intervals: pd.DatetimeIndex, valid_source_count: int,
                   timezone_name: str, strict: bool = True) -> dict:
    expected = len(geography) * len(intervals)
    if len(panel) != expected:
        raise ValueError(f"Panel row count {len(panel)} does not equal expected {expected}")
    if panel[["h3_index", "hourly_interval"]].duplicated().any():
        raise ValueError("Duplicate H3/hourly_interval combinations detected")
    if panel.h3_index.nunique() != len(geography) or panel.hourly_interval.nunique() != len(intervals):
        raise ValueError("Panel does not cover every geography cell and hourly interval")
    if panel.demand_count.isna().any() or (panel.demand_count < 0).any() or not np.issubdtype(panel.demand_count.dtype, np.integer):
        raise ValueError("demand_count must be non-negative integers without missing values")
    if int(panel.demand_count.sum()) != valid_source_count:
        raise ValueError(f"Demand conservation failed: panel={int(panel.demand_count.sum())}, source={valid_source_count}")
    parsed = pd.to_datetime(panel.hourly_interval, utc=True).dt.tz_convert(timezone_name)
    expected_local = pd.DatetimeIndex(intervals)
    if set(parsed) != set(expected_local):
        raise ValueError("Hourly intervals are missing or outside the configured period")
    if panel.region_id.isna().any() or panel.region_name.isna().any():
        raise ValueError("Geography join is incomplete")
    return {"row_count": int(len(panel)), "expected_row_count": int(expected), "unique_h3_cells": int(panel.h3_index.nunique()),
            "unique_hourly_intervals": int(panel.hourly_interval.nunique()), "total_demand": int(panel.demand_count.sum()),
            "zero_demand_cell_hours": int((panel.demand_count == 0).sum()), "nonzero_demand_cell_hours": int((panel.demand_count > 0).sum()),
            "min_demand": int(panel.demand_count.min()), "max_demand": int(panel.demand_count.max()),
            "mean_demand": float(panel.demand_count.mean()), "median_demand": float(panel.demand_count.median()),
            "demand_by_region": {str(k): int(v) for k, v in panel.groupby("region_id").demand_count.sum().items()},
            "demand_by_hour": {str(k): int(v) for k, v in panel.groupby("hour").demand_count.sum().items()},
            "demand_by_weekday": {str(k): int(v) for k, v in panel.groupby("day_of_week").demand_count.sum().items()},
            "timezone": timezone_name, "valid_source_count": int(valid_source_count), "conservation_passed": True}


def _summary_plot(panel: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=160)
    panel.groupby("hour").demand_count.sum().reindex(range(24), fill_value=0).plot.bar(ax=axes[0, 0], title="Demand by hour")
    panel.groupby("day_of_week").demand_count.sum().reindex(range(7), fill_value=0).plot.bar(ax=axes[0, 1], title="Demand by weekday")
    panel.groupby("region_id").demand_count.sum().sort_index().plot.bar(ax=axes[0, 2], title="Demand by region")
    panel.groupby("date").demand_count.sum().plot(ax=axes[1, 0], title="Daily demand")
    panel.groupby("h3_index").demand_count.sum().plot.hist(ax=axes[1, 1], bins=20, title="Demand distribution across cells")
    panel.assign(state=np.where(panel.demand_count.eq(0), "zero", "non-zero")).groupby("state").size().plot.bar(ax=axes[1, 2], title="Zero vs non-zero cell-hours")
    fig.suptitle("SYNTHETIC DEMAND DATA – NOT REAL-WORLD OBSERVATIONS", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def aggregate_hourly_demand(config: dict | None = None, ride_path: Path | str | None = None,
                            geography_path: Path | str | None = None, context_path: Path | str | None = None,
                            output_dir: Path | str | None = None, write_outputs: bool = True, strict: bool = True) -> dict:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    intervals, start, end, timezone_name = _period(config, strict)
    generation_metadata_path = ROOT / config.get("output", {}).get("synthetic_ride_metadata", "data/processed/synthetic_ride_generation_metadata.json")
    if strict and generation_metadata_path.exists():
        generation_metadata = json.loads(generation_metadata_path.read_text(encoding="utf-8"))
        if generation_metadata.get("historical_start") != start.isoformat() or generation_metadata.get("historical_end") != end.isoformat():
            raise ValueError("Configured aggregation period does not match synthetic ride-generation metadata")
    ride_path = Path(ride_path or ROOT / config["output"]["synthetic_rides"])
    geography_path = Path(geography_path or ROOT / config["output"]["synthetic_h3_cells_csv"])
    geography = load_geography(geography_path, strict=strict)
    rides = load_rides(ride_path)
    valid = rides[rides["_valid_mask"]].copy()
    valid["local_timestamp"] = valid["_request_timestamp_parsed"].dt.tz_convert(timezone_name)
    valid["hourly_interval"] = valid["local_timestamp"].dt.floor("h")
    valid["pickup_h3_index"] = valid["pickup_h3_index"].astype(str)
    eligible = set(geography.h3_index.astype(str))
    invalid_cells = ~valid.pickup_h3_index.isin(eligible)
    if invalid_cells.any():
        raise ValueError(f"{int(invalid_cells.sum())} valid rides reference H3 cells outside geography")
    grouped = valid.groupby(["pickup_h3_index", "hourly_interval"], sort=True).size().rename("demand_count")
    full_index = pd.MultiIndex.from_product([geography.h3_index.astype(str).tolist(), intervals], names=["h3_index", "hourly_interval"])
    panel = grouped.reindex(full_index, fill_value=0).rename_axis(index=["h3_index", "hourly_interval"]).reset_index()
    panel["demand_count"] = panel["demand_count"].astype("int64")
    panel = panel.merge(geography[["h3_index", "region_id", "region_name", "center_lat", "center_lon", "is_synthetic", "geography_mode"]], on="h3_index", how="left", validate="many_to_one")
    panel = _add_time_features(panel, pd.DatetimeIndex(panel["hourly_interval"]))
    panel["hourly_interval"] = pd.DatetimeIndex(panel["hourly_interval"])
    context_status = "missing"
    context_columns = []
    if context_path is None:
        context_path = ROOT / config["output"].get("h3_poi_context", "data/processed/h3_poi_context.csv")
    context_path = Path(context_path)
    if context_path.exists():
        context = pd.read_csv(context_path)
        if context.h3_index.duplicated().any():
            raise ValueError("POI context must be one-to-one on h3_index")
        if not set(geography.h3_index.astype(str)).issubset(set(context.h3_index.astype(str))):
            raise ValueError("POI context does not cover every geography H3 cell")
        context = context.drop(columns=[c for c in ["region_id", "region_name"] if c in context.columns])
        context_columns = [c for c in context.columns if c != "h3_index"]
        panel = panel.merge(context, on="h3_index", how="left", validate="many_to_one")
        context_status = str(context["context_source_status"].iloc[0]) if "context_source_status" in context.columns and not context.empty else "available"
    quality = validate_panel(panel, geography, intervals, len(valid), timezone_name, strict=strict)
    metadata = {"generator": "phase6a-v1", "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "historical_start": start.isoformat(), "historical_end": end.isoformat(), "historical_days": len(pd.date_range(start, end)),
                "timezone": timezone_name, "interval_convention": "hourly_interval is the local start of [hour, hour+1)",
                "target": "demand_count = valid requests grouped by pickup_h3_index and local hourly_interval",
                "valid_outcomes_included": ["completed", "cancelled", "rejected", "expired"],
                "geography_source": str(geography_path), "context_source_status": context_status, "context_columns": context_columns,
                "target_protection": {"excluded_from_model_table": LEAKAGE_FIELDS, "no_lag_or_rolling_features": True},
                "quality": quality}
    result = {"panel": panel, "metadata": metadata, "source_valid_count": len(valid), "context_status": context_status}
    if write_outputs:
        output_dir = Path(output_dir or ROOT / "data/processed")
        _atomic_write_dataframe(panel, output_dir / "hourly_h3_demand.csv")
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = output_dir / "hourly_demand_metadata.json.tmp"
        temp.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temp, output_dir / "hourly_demand_metadata.json")
        report = ["# Hourly H3 Demand Validation", "", "**SYNTHETIC DEMAND DATA – NOT REAL-WORLD OBSERVATIONS**", "",
                  f"- Rows: `{quality['row_count']:,}` / expected `{quality['expected_row_count']:,}`", f"- Unique H3 cells: `{quality['unique_h3_cells']}`",
                  f"- Hourly intervals: `{quality['unique_hourly_intervals']}`", f"- Total demand: `{quality['total_demand']:,}`",
                  f"- Valid source rides: `{quality['valid_source_count']:,}`", f"- Conservation passed: `{quality['conservation_passed']}`",
                  f"- Zero-demand cell-hours: `{quality['zero_demand_cell_hours']:,}`", f"- Non-zero cell-hours: `{quality['nonzero_demand_cell_hours']:,}`",
                  f"- Demand min/mean/median/max: `{quality['min_demand']}` / `{quality['mean_demand']:.4f}` / `{quality['median_demand']}` / `{quality['max_demand']}`",
                  f"- Timezone: `{timezone_name}`", f"- POI context status: `{context_status}`", "", "## Leakage protection", "",
                  "The model table does not copy ride IDs, coordinates, destinations, outcomes, validity flags, passenger identifiers, or lag/rolling features.",
                  "Zero POI counts, when present, do not prove absence of facilities.", "", "## Demand by region", ""]
        report += [f"- `{k}`: {v:,}" for k, v in sorted(quality["demand_by_region"].items())]
        report += ["", "## Demand by hour", ""] + [f"- `{int(k):02d}:00`: {v:,}" for k, v in sorted(quality["demand_by_hour"].items(), key=lambda item: int(item[0]))]
        report += ["", "## Demand by weekday", ""] + [f"- `{k}`: {v:,}" for k, v in sorted(quality["demand_by_weekday"].items(), key=lambda item: int(item[0]))]
        (output_dir / "hourly_demand_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _summary_plot(panel, output_dir / "hourly_demand_summary.png")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate synthetic rides into a complete hourly H3 demand panel.")
    args = parser.parse_args()
    result = aggregate_hourly_demand()
    print(json.dumps(result["metadata"]["quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
