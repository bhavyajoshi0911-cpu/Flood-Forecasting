import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import deferred as deferred_routes
from app.api.routes import live as live_routes

client = TestClient(app)


def test_health_and_status_endpoints_are_available(monkeypatch):
    monkeypatch.delenv("LIVE_DATA_PROVIDER_URL", raising=False)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["dataset_configured"] is True
    assert health.json()["model_configured"] is True
    assert client.get("/api/dataset/quality").json()["status"] == "warning"
    assert client.get("/api/model/info").json()["status"] == "available"
    assert client.get("/api/live/status").json()["status"] == "not_configured"


def test_live_status_does_not_expose_provider_url(monkeypatch):
    monkeypatch.setenv("LIVE_DATA_PROVIDER_URL", "https://provider.example/live?token=private")
    monkeypatch.delenv("LIVE_DATA_API_KEY", raising=False)

    response = client.get("/api/live/status")

    assert response.status_code == 200
    assert response.json()["source"] is None
    assert "provider.example" not in response.text
    assert "private" not in response.text


def test_live_status_marks_incomplete_provider_contract(monkeypatch):
    monkeypatch.setenv("LIVE_DATA_PROVIDER_URL", "https://provider.example/live")
    monkeypatch.delenv("LIVE_DATA_STATION", raising=False)
    monkeypatch.delenv("LIVE_DATA_FIELD_MAPPING", raising=False)
    monkeypatch.delenv("LIVE_DATA_EXPECTED_UNITS", raising=False)

    response = client.get("/api/live/status")

    assert response.status_code == 200
    assert response.json()["status"] == "incomplete"
    assert "field mappings are missing" in response.json()["message"]


def test_cwc_water_level_alerts_endpoint_returns_source_records(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"stationCode": "cwc-123", "status": "WARNING", "trend": "FALLING", "value": 12.5},
                {"stationCode": "cwc-456", "status": "DANGER", "trend": None, "value": 20.0},
            ]

    def fake_get(url, headers, timeout):
        assert url == live_routes.CWC_WATER_LEVEL_ALERTS_URL
        assert headers["Accept"] == "application/json"
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(live_routes.requests, "get", fake_get)

    response = client.get("/api/live/cwc-water-level-alerts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "Central Water Commission Flood Forecasting portal"
    assert payload["source_timestamp"] is None
    assert payload["total"] == 2
    assert payload["alerts"] == [
        {"station_code": "cwc-123", "status": "WARNING", "trend": "FALLING"},
        {"station_code": "cwc-456", "status": "DANGER", "trend": None},
    ]
    assert "value" not in response.text


def test_cwc_water_level_alerts_endpoint_rejects_unexpected_payload(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"alerts": []}

    monkeypatch.setattr(live_routes.requests, "get", lambda *args, **kwargs: FakeResponse())

    response = client.get("/api/live/cwc-water-level-alerts")

    assert response.status_code == 502
    assert "unexpected response" in response.json()["detail"].lower()


def test_model_metrics_endpoint_exposes_real_model_metrics():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    metrics = payload["metrics"]
    assert metrics["accuracy"] > 0
    assert metrics["model_name"] == "Decision Tree"
    assert metrics["training_period"]
    assert metrics["confusion_matrix"]
    assert metrics["roc_curve"]["fpr"]
    assert metrics["roc_curve"]["tpr"]


def test_prediction_history_returns_only_saved_predictions(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'history.sqlite3'}")
    monkeypatch.setattr(deferred_routes, "_retrospective_test_predictions", lambda: [])
    monkeypatch.setattr(deferred_routes, "predict_flood", lambda features: {
        "prediction": "Flood",
        "prediction_value": 1,
        "flood_probability": 80.0,
        "risk_level": "HIGH",
        "model_name": "Test Model",
    })
    monkeypatch.setattr(deferred_routes, "_load_shap_summary", lambda records: {"individual_values": [{}]})
    monkeypatch.setattr(deferred_routes, "_load_model_feature_names", lambda: ["Station", "rainfall"])

    empty_history = client.get("/api/predictions")
    assert empty_history.status_code == 200
    assert empty_history.json()["records"] == []

    forecast = client.post("/api/forecast/manual", json={"features": {"Station": "G1", "rainfall": 25}})
    response = client.get("/api/predictions")

    assert forecast.status_code == 200
    assert forecast.json()["prediction_id"] == 1
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["records"][0]["prediction_type"] == "manual"
    assert payload["records"][0]["risk_label"] == "HIGH"
    assert payload["records"][0]["probability"] == 0.8
    assert payload["records"][0]["model_version"] == "Test Model"
    assert payload["records"][0]["timestamp"].endswith("+00:00")
    assert payload["retrospective_records"] == []


def test_prediction_history_returns_separate_retrospective_model_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'retrospective.sqlite3'}")

    response = client.get("/api/predictions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["records"] == []
    assert len(payload["retrospective_records"]) == 34
    sample = payload["retrospective_records"][0]
    assert sample["prediction_type"] == "retrospective test"
    assert sample["data_source"] == "held-out test features"
    assert sample["station_id"]
    assert sample["timestamp"] == "2020-10-15"
    assert 0 <= sample["probability"] <= 1


def test_alerts_endpoint_returns_cwc_source_records(monkeypatch):
    monkeypatch.setattr(live_routes, "cwc_water_level_alerts", lambda: {
        "source": "Central Water Commission Flood Forecasting portal",
        "source_url": "https://example.test/cwc",
        "fetched_at": "2026-10-31T00:00:00+00:00",
        "alerts": [{"station_code": "station-1", "status": "WARNING", "trend": "FALLING"}],
    })

    response = client.get("/api/alerts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["alerts"][0]["station_code"] == "station-1"
    assert payload["alerts"][0]["severity"] == "WARNING"
    assert "not generated by this project's model" in payload["message"]


def test_shap_summary_is_available_and_live_forecast_requires_provider():
    summary = client.get("/api/shap/summary")
    assert summary.status_code == 200
    assert summary.json()["status"] == "available"
    assert summary.json()["global_importance"]

    forecast = client.post("/api/forecast/live", json={})
    assert forecast.status_code == 503
    assert "live forecast unavailable" in forecast.json()["detail"].lower()


def test_manual_forecast_rejects_incomplete_feature_vectors():
    response = client.post(
        "/api/forecast/manual",
        json={"features": {"rainfall": 12.5, "river_discharge": 100, "water_level": 4}},
    )
    assert response.status_code == 400
    assert "missing required feature" in response.json()["detail"].lower()


def test_historical_records_do_not_claim_observed_labels_as_predictions():
    response = client.get("/api/historical?page=1&page_size=1")
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record["flood_observed"] in (0, 1)
    assert record["predicted_flood"] is None