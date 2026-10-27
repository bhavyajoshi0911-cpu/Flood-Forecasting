import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("flask_api_app", MODULE_PATH)
flask_api_app = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(flask_api_app)


@pytest.fixture
def client():
    application = flask_api_app.create_app({"TESTING": True, "ENABLE_CORS": False})
    return application.test_client()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"
    assert response.get_json()["model_loaded"] is True


def test_live_data_success(monkeypatch, client):
    monkeypatch.setattr(flask_api_app, "load_model_contract", lambda: (object(), {"feature_names": ["rainfall"]}))
    monkeypatch.setattr(flask_api_app, "load_config", lambda: type("Config", (), {"field_mapping": {"rainfall": "rainfall"}, "expected_units": {"rainfall": "mm"}, "expected_station": "G1"})())
    monkeypatch.setattr(flask_api_app, "fetch_live_data", lambda config: {"timestamp": "2026-09-17T10:00:00+00:00", "station_id": "G1", "observations": {"rainfall": 10}, "units": {"rainfall": "mm"}})
    monkeypatch.setattr(flask_api_app, "normalize_live_data", lambda *args: {"timestamp": "2026-09-17T10:00:00+00:00", "station_id": "G1", "rainfall": 10.0})
    response = client.get("/api/live-data")
    assert response.status_code == 200
    assert response.get_json()["data"]["rainfall"] == 10.0


def test_predict_manual_success(monkeypatch, client):
    monkeypatch.setattr(flask_api_app, "load_model_contract", lambda: (object(), {"feature_names": ["rainfall"]}))
    monkeypatch.setattr(flask_api_app, "predict_flood", lambda features: {"prediction": "No Flood", "flood_probability": 10.0, "no_flood_probability": 90.0, "risk_level": "LOW"})
    response = client.post("/api/predict", json={"features": {"rainfall": 4.0}})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "No Flood"


def test_predict_missing_features(client):
    response = client.post("/api/predict", json={"features": {}})
    assert response.status_code == 400
    assert response.get_json()["missing_features"]


def test_predict_invalid_json(client):
    response = client.post("/api/predict", data="not-json", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_live_provider_failure(monkeypatch, client):
    monkeypatch.setattr(flask_api_app, "predict_live", lambda: (_ for _ in ()).throw(flask_api_app.LiveDataError("provider request failed")))
    response = client.post("/api/predict", json={})
    assert response.status_code == 502
    assert response.get_json()["message"] == "Unable to generate live prediction"
    assert response.get_json()["reason"] == "provider request failed"


def test_model_loading_failure(monkeypatch, client):
    monkeypatch.setattr(flask_api_app, "load_model_contract", lambda: (_ for _ in ()).throw(flask_api_app.LiveDataError("model unavailable")))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is False


def test_unknown_endpoint(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["status"] == "error"
