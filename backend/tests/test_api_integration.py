import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app

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


@pytest.mark.parametrize("path", ["/api/predictions", "/api/alerts"])
def test_unimplemented_workflows_report_unavailable(path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"


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