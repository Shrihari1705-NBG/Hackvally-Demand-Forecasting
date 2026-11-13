"""Offline API tests using the existing saved model and synthetic history."""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from ml_service.dependencies import load_model_bundle
from ml_service.main import _demo_payload, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def payload():
    return _demo_payload(load_model_bundle()).model_dump(mode="json")


def test_startup_health_and_model_info(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert "synthetic" in health.json()["synthetic_data_warning"].lower()
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["h3_cells"] == 94
    assert info.json()["feature_count"] == 21


def test_valid_next_hour_forecast_and_flags(client, payload):
    response = client.post("/forecast/next-hour", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 94
    assert body["is_synthetic"] is True
    assert body["data_source"] == "synthetic_request"
    assert all(p["predicted_demand"] >= 0 for p in body["predictions"])


def test_demo_replay_map_and_region_summary(client):
    demo = client.get("/forecast/demo-next-hour")
    assert demo.status_code == 200
    assert len(demo.json()["predictions"]) == 94
    assert demo.json()["data_source"] == "synthetic_demo"
    payload = _demo_payload(load_model_bundle()).model_dump(mode="json")
    map_response = client.post("/forecast/map", json=payload)
    assert map_response.status_code == 200
    assert len(map_response.json()["predictions"]) == 94
    assert "center_lat" in map_response.json()["predictions"][0]
    regions = client.post("/forecast/regions", json=payload)
    assert regions.status_code == 200
    assert len(regions.json()["predictions"]) == 8


@pytest.mark.parametrize("mutator", [
    lambda p: p["forecast_timestamp"].replace("+05:30", ""),
    lambda p: p["forecast_timestamp"].replace("+05:30", "Z"),
])
def test_invalid_or_wrong_timezone_timestamp_rejected(client, payload, mutator):
    invalid = copy.deepcopy(payload)
    invalid["forecast_timestamp"] = mutator(invalid)
    assert client.post("/forecast/next-hour", json=invalid).status_code == 422


def test_duplicate_unknown_missing_and_nonconsecutive_cells_rejected(client, payload):
    duplicate = copy.deepcopy(payload)
    duplicate["cell_history"][1]["h3_index"] = duplicate["cell_history"][0]["h3_index"]
    assert client.post("/forecast/next-hour", json=duplicate).status_code == 422

    unknown = copy.deepcopy(payload)
    unknown["cell_history"][0]["h3_index"] = "not_an_h3_cell"
    assert client.post("/forecast/next-hour", json=unknown).status_code == 422

    missing = copy.deepcopy(payload)
    missing["cell_history"] = missing["cell_history"][:-1]
    assert client.post("/forecast/next-hour", json=missing).status_code == 422

    nonconsecutive = copy.deepcopy(payload)
    nonconsecutive["cell_history"][0]["history"][10]["timestamp"] = nonconsecutive["cell_history"][0]["history"][9]["timestamp"]
    assert client.post("/forecast/next-hour", json=nonconsecutive).status_code == 422


def test_negative_history_and_metadata_mismatch_rejected(client, payload):
    negative = copy.deepcopy(payload)
    negative["cell_history"][0]["history"][0]["demand_count"] = -1
    assert client.post("/forecast/next-hour", json=negative).status_code == 422

    wrong_region = copy.deepcopy(payload)
    wrong_region["cell_history"][0]["region_id"] = "R99"
    assert client.post("/forecast/next-hour", json=wrong_region).status_code == 422
