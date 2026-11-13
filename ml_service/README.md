# KasuCab Phase 8A ML service

This service exposes one-hour-ahead forecasts from the saved Phase 7A model. It loads the model once at startup and validates all requests against the 94-cell synthetic geography and the saved feature schema.

The model was trained on synthetic geography and synthetic ride requests. Predictions are experimental synthetic outputs, not live or observed Belagavi demand.

## Start

Install the project requirements, then run from the repository root:

The project declares CPython 3.11 as its supported runtime and the saved artifact was validated with scikit-learn 1.9.1. Keep scikit-learn below version 2 when deploying this artifact.

```powershell
uvicorn ml_service.main:app --host 127.0.0.1 --port 8001 --reload
```

Production-style local command:

```powershell
uvicorn ml_service.main:app --host 127.0.0.1 --port 8001
```

Configure CORS with a comma-separated environment variable; the default allows only localhost development origins:

```powershell
$env:KASUCAB_ALLOWED_ORIGINS = "http://localhost:3000,http://localhost:5173"
```

## Endpoints

- `GET /health`
- `GET /model-info`
- `POST /forecast/next-hour`
- `GET /forecast/demo-next-hour`
- `POST /forecast/map`
- `POST /forecast/regions`
- `/docs`

The POST request requires all 94 cells, each with exactly 168 consecutive, timezone-aware Asia/Kolkata observations ending one hour before `forecast_timestamp`. See the OpenAPI page for the complete schema.

Example demo request:

```powershell
curl http://127.0.0.1:8001/forecast/demo-next-hour
```

Synthetic demo response shape:

```json
{
  "forecast_timestamp": "2026-11-13T00:00:00+05:30",
  "forecast_horizon_hours": 1,
  "model_version": "phase7a-v1",
  "data_source": "synthetic_demo",
  "is_synthetic": true,
  "predictions": [{"h3_index": "...", "region_id": "R01", "region_name": "Central Business District", "predicted_demand": 0.42}]
}
```

The example is illustrative synthetic output; it is not a live Belagavi observation.
