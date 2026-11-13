"""Validated API request and response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class HistoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime
    demand_count: StrictInt = Field(..., ge=0)


class CellHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    h3_index: str = Field(..., min_length=1)
    region_id: str = Field(..., min_length=1)
    region_name: str = Field(..., min_length=1)
    center_lat: float = Field(..., ge=-90, le=90)
    center_lon: float = Field(..., ge=-180, le=180)
    history: list[HistoryObservation] = Field(..., min_length=168, max_length=168)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_timestamp: datetime
    geography_mode: str = Field(..., min_length=1)
    cell_history: list[CellHistory] = Field(..., min_length=1, max_length=94)


class Prediction(BaseModel):
    h3_index: str
    region_id: str
    region_name: str
    predicted_demand: float = Field(..., ge=0)


class ForecastResponse(BaseModel):
    forecast_timestamp: datetime
    forecast_horizon_hours: int
    model_version: str
    data_source: str
    is_synthetic: bool
    predictions: list[Prediction]


class MapPrediction(Prediction):
    center_lat: float
    center_lon: float
    polygon: dict[str, Any] | None = None


class MapResponse(ForecastResponse):
    predictions: list[MapPrediction]


class RegionSummary(BaseModel):
    region_id: str
    region_name: str
    cell_count: int
    predicted_demand_total: float
    predicted_demand_mean: float
    highest_demand_cell: str
    highest_cell_predicted_demand: float


class RegionResponse(ForecastResponse):
    predictions: list[RegionSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_type: str
    synthetic_data_warning: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    forecast_horizon_hours: int
    historical_start: str
    historical_end: str
    h3_cells: int
    feature_count: int
    evaluation_metrics: dict[str, Any]
    data_classification: str
