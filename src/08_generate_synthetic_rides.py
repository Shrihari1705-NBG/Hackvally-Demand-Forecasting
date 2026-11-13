"""Generate reproducible individual synthetic ride-request records.

This stage creates event-level simulation data only. It does not aggregate
hourly demand, model driver supply, estimate road travel time, or claim to
represent observed Belagavi demand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from pipeline_utils import h3_polygon, validate_resolution

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_CELLS = ROOT / "data/processed/synthetic_h3_cells.csv"
SIMULATION_VERSION = "phase5a-v1"
OUTCOMES = ("completed", "cancelled", "rejected", "expired")
RIDE_COLUMNS = ["ride_id", "request_timestamp", "request_date", "request_hour", "day_of_week", "is_weekend",
                "pickup_lat", "pickup_lon", "pickup_h3_index", "pickup_region_id", "pickup_region_name",
                "destination_lat", "destination_lon", "destination_h3_index", "destination_region_id",
                "destination_region_name", "trip_distance_km_geodesic", "request_outcome", "is_valid_demand",
                "is_synthetic", "simulation_version", "random_seed"]

DEFAULT_REGION_PATTERNS = {
    "R01": {"baseline_multiplier": 1.35, "morning_peak": 1.20, "evening_peak": 1.35, "business_hours": 1.35},
    "R02": {"baseline_multiplier": 1.25, "morning_peak": 1.65, "evening_peak": 1.65, "business_hours": 1.00},
    "R03": {"baseline_multiplier": 1.00, "morning_peak": 1.15, "evening_peak": 1.20, "business_hours": 1.05},
    "R04": {"baseline_multiplier": 0.85, "morning_peak": 1.25, "evening_peak": 1.25, "business_hours": 0.90},
    "R05": {"baseline_multiplier": 1.00, "morning_peak": 1.45, "evening_peak": 1.40, "business_hours": 0.95},
    "R06": {"baseline_multiplier": 1.10, "morning_peak": 1.10, "evening_peak": 1.15, "business_hours": 1.10},
    "R07": {"baseline_multiplier": 0.90, "morning_peak": 1.75, "evening_peak": 1.55, "business_hours": 0.85},
    "R08": {"baseline_multiplier": 0.90, "morning_peak": 1.30, "evening_peak": 1.30, "business_hours": 0.95},
}


def load_cells(path: Path | str = DEFAULT_CELLS) -> pd.DataFrame:
    cells = pd.read_csv(path)
    required = {"h3_index", "region_id", "region_name", "center_lat", "center_lon"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Synthetic geography is missing columns: {sorted(missing)}")
    if cells.empty:
        raise ValueError("Synthetic geography is empty")
    if cells["h3_index"].duplicated().any():
        raise ValueError("Synthetic geography contains duplicate H3 IDs")
    if len(cells) != 94:
        raise ValueError(f"Expected the locked 94-cell synthetic geography, found {len(cells)}")
    if cells["region_id"].nunique() != 8:
        raise ValueError("Synthetic geography must contain eight analytical region assignments")
    for cell in cells["h3_index"].astype(str):
        validate_resolution(cell, 8)
    return cells.sort_values("h3_index").reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(a)))


def sample_point_in_cell(cell: str, rng: np.random.Generator, max_attempts: int = 500) -> tuple[float, float]:
    polygon = h3_polygon(cell)
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_attempts):
        point = Point(float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy)))
        if polygon.covers(point):
            calculated = h3.latlng_to_cell(point.y, point.x, 8)
            if calculated == cell:
                return float(point.y), float(point.x)
    raise RuntimeError(f"Could not generate a valid coordinate inside H3 cell {cell} after {max_attempts} attempts")


def hourly_multiplier(region_id: str, hour: int, weekday: int, settings: dict) -> float:
    pattern = DEFAULT_REGION_PATTERNS[region_id]
    value = 1.0
    if settings.get("include_hourly_patterns", True):
        if 7 <= hour <= 10:
            value *= pattern["morning_peak"]
        elif 17 <= hour <= 21:
            value *= pattern["evening_peak"]
        elif 10 <= hour <= 16:
            value *= pattern["business_hours"]
        elif 0 <= hour <= 5:
            value *= 0.45
        else:
            value *= 0.80
    if settings.get("include_weekend_effects", True) and weekday >= 5:
        value *= 0.78 if region_id in {"R01", "R02", "R07"} else 0.88
    return max(0.0, value)


def _date_range(end_date_value: str, days: int) -> list[date]:
    end = date.fromisoformat(str(end_date_value))
    if days <= 0:
        raise ValueError("historical_days must be positive")
    start = end - timedelta(days=days - 1)
    return [start + timedelta(days=i) for i in range(days)]


def _destination_probabilities(pickup: pd.Series, cells: pd.DataFrame, settings: dict) -> np.ndarray:
    scale = float(settings.get("destination_distance_scale_km", 6.0))
    bonus = float(settings.get("same_region_destination_bonus", 2.0))
    if scale <= 0:
        raise ValueError("destination_distance_scale_km must be positive")
    distances = np.array([_haversine_km(pickup.center_lat, pickup.center_lon, row.center_lat, row.center_lon) for row in cells.itertuples()])
    weights = np.exp(-distances / scale)
    weights *= np.where(cells.region_id.to_numpy() == pickup.region_id, bonus, 1.0)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Destination probabilities are invalid")
    return weights / total


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_rides(frame: pd.DataFrame, cells: pd.DataFrame, timezone_name: str, start: date, end: date) -> dict:
    required = set(RIDE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ride output is missing columns: {sorted(missing)}")
    eligible = set(cells.h3_index.astype(str))
    region_map = dict(zip(cells.region_id, cells.region_name))
    if not frame.ride_id.is_unique:
        raise ValueError("Ride IDs are not unique")
    if frame.empty:
        raise ValueError("No synthetic rides were generated")
    timestamps = pd.to_datetime(frame.request_timestamp, utc=True).dt.tz_convert(timezone_name)
    if timestamps.dt.date.min() < start or timestamps.dt.date.max() > end:
        raise ValueError("Ride timestamps fall outside the configured historical period")
    if frame.request_date.astype(str).tolist() != timestamps.dt.strftime("%Y-%m-%d").tolist():
        raise ValueError("request_date is inconsistent with request_timestamp")
    if frame.request_hour.astype(int).tolist() != timestamps.dt.hour.tolist() or frame.day_of_week.astype(int).tolist() != timestamps.dt.weekday.tolist():
        raise ValueError("request_hour or day_of_week is inconsistent with request_timestamp")
    if frame.is_weekend.astype(bool).tolist() != timestamps.dt.weekday.ge(5).tolist():
        raise ValueError("is_weekend is inconsistent with request_timestamp")
    if not frame.pickup_h3_index.astype(str).isin(eligible).all() or not frame.destination_h3_index.astype(str).isin(eligible).all():
        raise ValueError("Ride contains an H3 index outside synthetic geography")
    for side in ("pickup", "destination"):
        calculated = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(frame[f"{side}_lat"], frame[f"{side}_lon"])]
        if calculated != frame[f"{side}_h3_index"].astype(str).tolist():
            raise ValueError(f"{side} coordinates do not map back to their stored H3 cells")
        if not frame[f"{side}_region_id"].isin(region_map).all():
            raise ValueError(f"{side} region IDs are invalid")
        if not (frame[f"{side}_region_name"] == frame[f"{side}_region_id"].map(region_map)).all():
            raise ValueError(f"{side} region names do not match region IDs")
    if (frame.trip_distance_km_geodesic < 0).any() or not frame.request_outcome.isin(OUTCOMES).all():
        raise ValueError("Ride distances or outcomes are invalid")
    if not frame.is_valid_demand.astype(bool).all() or not frame.is_synthetic.astype(bool).all():
        raise ValueError("All generated rides must be valid synthetic demand records")
    if not (frame.random_seed == frame.random_seed.iloc[0]).all() or not (frame.simulation_version == SIMULATION_VERSION).all():
        raise ValueError("Simulation provenance is inconsistent")
    return {"row_count": len(frame), "unique_ride_ids": int(frame.ride_id.nunique()),
            "requests_by_region": frame.groupby("pickup_region_id").size().to_dict(),
            "requests_by_hour": frame.groupby("request_hour").size().to_dict(),
            "requests_by_weekday": frame.groupby("day_of_week").size().to_dict(),
            "requests_by_outcome": frame.groupby("request_outcome").size().to_dict(),
            "min_distance_km": float(frame.trip_distance_km_geodesic.min()), "max_distance_km": float(frame.trip_distance_km_geodesic.max())}


def _render_summary(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=160)
    frame.groupby("request_hour").size().reindex(range(24), fill_value=0).plot.bar(ax=axes[0, 0], title="Synthetic requests by hour")
    frame.groupby("day_of_week").size().reindex(range(7), fill_value=0).plot.bar(ax=axes[0, 1], title="Synthetic requests by weekday")
    frame.groupby("pickup_region_id").size().sort_index().plot.bar(ax=axes[0, 2], title="Synthetic requests by region")
    frame.groupby("request_date").size().plot(ax=axes[1, 0], title="Synthetic daily request volume")
    frame.groupby("request_outcome").size().plot.bar(ax=axes[1, 1], title="Synthetic request outcomes")
    axes[1, 2].axis("off")
    fig.suptitle("SYNTHETIC RIDE REQUESTS – SIMULATION OUTPUT, NOT OBSERVED DEMAND", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def generate_synthetic_rides(config: dict | None = None, cells_path: Path | str = DEFAULT_CELLS,
                             output_dir: Path | str | None = None, write_outputs: bool = True) -> dict:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    settings = config.get("synthetic_rides", {})
    if settings.get("enabled") is False:
        raise ValueError("synthetic_rides.enabled is false")
    timezone_name = settings.get("timezone", config.get("timezone", "Asia/Kolkata"))
    ZoneInfo(timezone_name)
    days = int(settings.get("historical_days", config.get("history_days", 180)))
    if days != 180:
        raise ValueError(f"Phase 5A is locked to 180 historical days, got {days}")
    seed = int(settings.get("random_seed", config.get("random_seed", 42)))
    base = float(settings.get("base_requests_per_cell_hour", 0.35))
    if base < 0:
        raise ValueError("base_requests_per_cell_hour must be non-negative")
    cells = load_cells(cells_path)
    dates = _date_range(config.get("end_date"), days)
    tz = ZoneInfo(timezone_name)
    rng = np.random.default_rng(seed)
    region_patterns = settings.get("region_patterns", DEFAULT_REGION_PATTERNS)
    if set(region_patterns) != set(DEFAULT_REGION_PATTERNS):
        raise ValueError("region_patterns must define all eight analytical regions")
    if region_patterns != DEFAULT_REGION_PATTERNS:
        patterns = DEFAULT_REGION_PATTERNS | region_patterns
    else:
        patterns = DEFAULT_REGION_PATTERNS
    # Cell variation and date shocks are sampled once in stable sorted order.
    cell_variation = {cell: float(rng.lognormal(mean=0.0, sigma=0.12)) for cell in cells.h3_index}
    date_shocks = {}
    for current_date in dates:
        date_shocks[current_date] = float(rng.lognormal(0.0, 0.06))
        if settings.get("include_random_shocks", True) and rng.random() < 0.035:
            date_shocks[current_date] *= float(rng.uniform(1.25, 1.75))
    outcome_probs = settings.get("outcome_probabilities", {"completed": .78, "cancelled": .12, "rejected": .07, "expired": .03})
    if set(outcome_probs) != set(OUTCOMES) or abs(sum(float(v) for v in outcome_probs.values()) - 1.0) > 1e-8:
        raise ValueError("outcome_probabilities must contain completed, cancelled, rejected and expired summing to 1")
    records = []
    ride_number = 1
    for current_date in dates:
        weekday = current_date.weekday()
        for hour in range(24):
            for cell in cells.itertuples(index=False):
                pattern = patterns[cell.region_id]
                expected = base * float(pattern["baseline_multiplier"]) * hourly_multiplier(cell.region_id, hour, weekday, settings)
                expected *= date_shocks[current_date] * cell_variation[cell.h3_index]
                count = int(rng.poisson(max(0.0, expected))) if expected > 0 else 0
                if count == 0:
                    continue
                destination_probabilities = _destination_probabilities(cell, cells, settings)
                for _ in range(count):
                    pickup_lat, pickup_lon = sample_point_in_cell(cell.h3_index, rng, int(settings.get("max_point_sampling_attempts", 500)))
                    destination_index = int(rng.choice(len(cells), p=destination_probabilities))
                    destination = cells.iloc[destination_index]
                    destination_lat, destination_lon = sample_point_in_cell(destination.h3_index, rng, int(settings.get("max_point_sampling_attempts", 500)))
                    instant = datetime.combine(current_date, time(hour=hour), tzinfo=tz) + timedelta(seconds=int(rng.integers(0, 3600)))
                    outcome = str(rng.choice(OUTCOMES, p=[float(outcome_probs[name]) for name in OUTCOMES]))
                    records.append({"ride_id": f"ride_{ride_number:09d}", "request_timestamp": instant.isoformat(),
                                    "request_date": current_date.isoformat(), "request_hour": hour, "day_of_week": weekday,
                                    "is_weekend": weekday >= 5, "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
                                    "pickup_h3_index": cell.h3_index, "pickup_region_id": cell.region_id, "pickup_region_name": cell.region_name,
                                    "destination_lat": destination_lat, "destination_lon": destination_lon,
                                    "destination_h3_index": destination.h3_index, "destination_region_id": destination.region_id,
                                    "destination_region_name": destination.region_name,
                                    "trip_distance_km_geodesic": _haversine_km(pickup_lat, pickup_lon, destination_lat, destination_lon),
                                    "request_outcome": outcome, "is_valid_demand": True, "is_synthetic": True,
                                    "simulation_version": SIMULATION_VERSION, "random_seed": seed})
                    ride_number += 1
    frame = pd.DataFrame(records, columns=RIDE_COLUMNS)
    if frame.empty:
        raise ValueError("No rides generated; increase the configurable base request rate")
    quality = validate_rides(frame, cells, timezone_name, dates[0], dates[-1])
    all_cell_hours = pd.MultiIndex.from_product([cells.h3_index.astype(str).tolist(), [d.isoformat() for d in dates], range(24)],
                                                names=["pickup_h3_index", "request_date", "request_hour"])
    cell_hour = frame.groupby(["pickup_h3_index", "request_date", "request_hour"]).size().reindex(all_cell_hours, fill_value=0)
    quality.update({"mean_requests_per_cell_hour": float(cell_hour.mean()), "median_requests_per_cell_hour": float(cell_hour.median()),
                    "max_requests_per_cell_hour": int(cell_hour.max()), "min_requests_per_cell_hour": int(cell_hour.min()),
                    "historical_start": dates[0].isoformat(), "historical_end": dates[-1].isoformat(),
                    "timezone": timezone_name, "osm_context_used": False})
    metadata = {"generator": SIMULATION_VERSION, "generated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
                "random_seed": seed, "configuration_snapshot": settings, "history_days": days,
                "historical_start": dates[0].isoformat(), "historical_end": dates[-1].isoformat(), "timezone": timezone_name,
                "timestamp_convention": "request_timestamp is a timezone-aware event timestamp within the local request hour",
                "demand_target": "valid synthetic ride-request attempts; all generated outcomes remain valid demand",
                "synthetic_assumption_notice": "Region and temporal patterns are configurable simulation assumptions, not measured Belagavi demand.",
                "osm_context_note": "OSM context is optional and was not used as a demand driver; unavailable context is not interpreted as facility absence.",
                "quality": quality}
    if write_outputs:
        output_dir = Path(output_dir or ROOT / "data/processed")
        _atomic_write_dataframe(frame, output_dir / "synthetic_rides.csv")
        _atomic_write_dataframe(frame.head(1000), output_dir / "synthetic_rides_sample.csv")
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_dir / "synthetic_ride_generation_metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir / "synthetic_ride_generation_metadata.json")
        report = ["# Synthetic Ride Generation Report", "", "**SYNTHETIC SIMULATION — NOT OBSERVED BELAGAVI DEMAND**", "",
                  f"- Generator: `{SIMULATION_VERSION}`", f"- Valid ride requests: `{len(frame):,}`", f"- Historical period: `{dates[0]}` to `{dates[-1]}`",
                  f"- Timezone: `{timezone_name}`", f"- Random seed: `{seed}`", f"- Mean requests per populated cell-hour: `{quality['mean_requests_per_cell_hour']:.3f}`",
                  f"- Median requests per populated cell-hour: `{quality['median_requests_per_cell_hour']:.3f}`", f"- Maximum requests per cell-hour: `{quality['max_requests_per_cell_hour']}`", "", "## Requests by region", ""]
        report += [f"- `{k}`: {v:,}" for k, v in sorted(quality["requests_by_region"].items())]
        report += ["", "## Requests by hour", ""] + [f"- `{k:02d}:00`: {v:,}" for k, v in sorted(quality["requests_by_hour"].items())]
        report += ["", "## Requests by weekday", ""] + [f"- `{k}`: {v:,}" for k, v in sorted(quality["requests_by_weekday"].items())]
        report += ["", "## Outcomes", ""] + [f"- `{k}`: {v:,}" for k, v in sorted(quality["requests_by_outcome"].items())]
        report += ["", "Region, hourly, weekend and shock patterns are configurable synthetic assumptions. No actual ride observations or official geography are represented."]
        (output_dir / "synthetic_ride_generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _render_summary(frame, output_dir / "synthetic_ride_generation_summary.png")
    return {"rides": frame, "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate individual synthetic ride requests only.")
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    args = parser.parse_args()
    result = generate_synthetic_rides(cells_path=args.cells)
    print(json.dumps(result["metadata"]["quality"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
