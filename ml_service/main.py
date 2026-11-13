"""Standalone FastAPI service for one-hour-ahead synthetic demand forecasts."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .dependencies import ModelBundle, load_model_bundle
from .forecast_service import prediction_records, predict
from .schemas import (ForecastRequest, ForecastResponse, HealthResponse, MapPrediction, MapResponse,
                      ModelInfoResponse, RegionResponse, RegionSummary)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("kasucab.ml_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_model_bundle()
    LOGGER.info("Demand model startup validation passed")
    yield
    app.state.bundle = None


app = FastAPI(title="KasuCab Demand Forecasting Service", version="phase8a-v1", lifespan=lifespan,
             description="One-hour-ahead forecasts from synthetic KasuCab geography. Results are not live Belagavi observations.")
app.add_middleware(CORSMiddleware, allow_origins=config.allowed_origins(), allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def bundle(request: Request) -> ModelBundle:
    loaded = getattr(request.app.state, "bundle", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Demand model is not loaded")
    return loaded


def _base_response(request: ForecastRequest, loaded: ModelBundle, frame: pd.DataFrame, timestamp: pd.Timestamp, source: str) -> dict:
    return {"forecast_timestamp": timestamp.isoformat(), "forecast_horizon_hours": 1,
            "model_version": loaded.metadata.get("model_version", "unknown"), "data_source": source,
            "is_synthetic": True, "predictions": prediction_records(frame)}


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = bundle(request)
    return {"status": "ok", "model_loaded": True, "model_version": loaded.metadata.get("model_version", "unknown"),
            "model_type": loaded.metadata.get("model_type", "unknown"),
            "synthetic_data_warning": "Model and forecasts use synthetic geography and synthetic ride requests; they are not live Belagavi demand."}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request):
    loaded = bundle(request)
    return {"model_version": loaded.metadata.get("model_version", "unknown"), "model_type": loaded.metadata.get("model_type", "unknown"),
            "forecast_horizon_hours": int(loaded.metadata.get("training_configuration", {}).get("forecast_horizon_hours", 1)),
            "historical_start": loaded.metadata.get("historical_start", ""), "historical_end": loaded.metadata.get("historical_end", ""),
            "h3_cells": int(loaded.metadata.get("cells", len(loaded.geography))), "feature_count": len(loaded.schema["feature_columns"]),
            "evaluation_metrics": loaded.metrics, "data_classification": "synthetic_experimental_not_live_demand"}


@app.post("/forecast/next-hour", response_model=ForecastResponse)
def next_hour(payload: ForecastRequest, request: Request):
    try:
        loaded = bundle(request); frame, timestamp, elapsed = predict(payload, loaded)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    LOGGER.info("forecast timestamp=%s cells=%d elapsed_ms=%.2f model=%s", timestamp.isoformat(), len(frame), elapsed * 1000, loaded.metadata.get("model_version"))
    return _base_response(payload, loaded, frame, timestamp, "synthetic_request")


def _demo_payload(loaded: ModelBundle) -> ForecastRequest:
    if not config.HOURLY_PATH.exists():
        raise ValueError("Synthetic hourly history is unavailable")
    hourly = pd.read_csv(config.HOURLY_PATH)
    hourly["timestamp"] = pd.to_datetime(hourly["hourly_interval"], utc=True).dt.tz_convert(config.TIMEZONE)
    latest = hourly["timestamp"].max()
    required = pd.date_range(end=latest, periods=168, freq="h", tz=config.TIMEZONE)
    geo = loaded.geography
    cells = []
    for _, row in geo.iterrows():
        history = hourly[(hourly.h3_index.astype(str) == str(row.h3_index)) & hourly.timestamp.isin(required)].sort_values("timestamp")
        if len(history) != 168:
            raise ValueError(f"Insufficient complete history for {row.h3_index}")
        cells.append({"h3_index": str(row.h3_index), "region_id": str(row.region_id), "region_name": str(row.region_name),
                      "center_lat": float(row.center_lat), "center_lon": float(row.center_lon),
                      "history": [{"timestamp": ts.isoformat(), "demand_count": int(d)} for ts, d in zip(history.timestamp, history.demand_count)]})
    return ForecastRequest(forecast_timestamp=(latest + pd.Timedelta(hours=1)).to_pydatetime(), geography_mode=config.EXPECTED_GEOGRAPHY_MODE, cell_history=cells)


@app.get("/forecast/demo-next-hour", response_model=ForecastResponse)
def demo_next_hour(request: Request):
    loaded = bundle(request)
    try:
        payload = _demo_payload(loaded); frame, timestamp, elapsed = predict(payload, loaded)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    LOGGER.info("demo forecast timestamp=%s cells=%d elapsed_ms=%.2f", timestamp.isoformat(), len(frame), elapsed * 1000)
    return _base_response(payload, loaded, frame, timestamp, "synthetic_demo")


def _geojson_polygons(loaded: ModelBundle) -> dict[str, dict]:
    result = {}
    for feature in (loaded.geography_geojson or {}).get("features", []):
        props = feature.get("properties", {})
        if props.get("h3_index"):
            result[str(props["h3_index"])] = feature.get("geometry")
    return result


@app.post("/forecast/map", response_model=MapResponse)
def forecast_map(payload: ForecastRequest, request: Request):
    loaded = bundle(request)
    try:
        frame, timestamp, elapsed = predict(payload, loaded)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    polygons = _geojson_polygons(loaded)
    records = []
    for row in frame.itertuples():
        records.append({"h3_index": row.h3_index, "region_id": row.region_id, "region_name": row.region_name,
                        "predicted_demand": row.predicted_demand, "center_lat": row.center_lat, "center_lon": row.center_lon,
                        "polygon": polygons.get(row.h3_index)})
    base = _base_response(payload, loaded, frame, timestamp, "synthetic_request"); base["predictions"] = records
    return base


@app.post("/forecast/regions", response_model=RegionResponse)
def forecast_regions(payload: ForecastRequest, request: Request):
    loaded = bundle(request)
    try:
        frame, timestamp, elapsed = predict(payload, loaded)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summaries = []
    for (region_id, region_name), group in frame.groupby(["region_id", "region_name"], sort=True):
        highest = group.loc[group.predicted_demand.idxmax()]
        summaries.append(RegionSummary(region_id=region_id, region_name=region_name, cell_count=len(group),
                                       predicted_demand_total=float(group.predicted_demand.sum()), predicted_demand_mean=float(group.predicted_demand.mean()),
                                       highest_demand_cell=str(highest.h3_index), highest_cell_predicted_demand=float(highest.predicted_demand)))
    base = _base_response(payload, loaded, frame, timestamp, "synthetic_request"); base["predictions"] = summaries
    return base
