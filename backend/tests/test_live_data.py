from datetime import datetime, timedelta, timezone

import requests
import pytest

import live_data


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request_kwargs = None

    def get(self, *args, **kwargs):
        self.request_kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def payload():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "station_id": "G1",
        "observations": {"rainfall": 12.5},
        "units": {"rainfall": "mm"},
    }


def config():
    return live_data.LiveDataConfig(
        provider_url="https://provider.example/live",
        api_key="secret-value",
        field_mapping={"rainfall": "rainfall"},
        expected_units={"rainfall": "mm"},
        expected_station="G1",
    )


def test_successful_api_response():
    result = live_data.fetch_live_data(config(), FakeSession(FakeResponse(payload())))
    assert result["station_id"] == "G1"
    assert result["observations"]["rainfall"] == 12.5


def test_api_timeout():
    with pytest.raises(live_data.LiveDataError, match="timed out"):
        live_data.fetch_live_data(config(), FakeSession(error=requests.Timeout()))


def test_invalid_api_response():
    response = FakeResponse([])
    with pytest.raises(live_data.LiveDataError, match="response must be a JSON object"):
        live_data.fetch_live_data(config(), FakeSession(response))


def test_missing_required_field():
    with pytest.raises(live_data.LiveDataError, match="Missing required live feature: river_level"):
        live_data.normalize_live_data(payload(), ["rainfall", "river_level"], {"rainfall": "rainfall"}, {"rainfall": "mm"}, "G1")


def test_invalid_numeric_value():
    invalid = payload()
    invalid["observations"]["rainfall"] = "not-a-number"
    with pytest.raises(live_data.LiveDataError, match="must be numeric"):
        live_data.normalize_live_data(invalid, ["rainfall"], {"rainfall": "rainfall"}, {"rainfall": "mm"}, "G1")


def test_api_key_is_optional_for_public_providers(monkeypatch):
    monkeypatch.setenv("LIVE_DATA_PROVIDER_URL", "https://provider.example/live")
    monkeypatch.delenv("LIVE_DATA_API_KEY", raising=False)
    assert live_data.load_config().api_key == ""


def test_public_provider_request_omits_empty_api_key_header():
    provider_config = config()
    provider_config = live_data.LiveDataConfig(
        provider_url=provider_config.provider_url,
        api_key="",
        field_mapping=provider_config.field_mapping,
        expected_units=provider_config.expected_units,
        expected_station=provider_config.expected_station,
    )
    session = FakeSession(FakeResponse(payload()))

    live_data.fetch_live_data(provider_config, session)

    assert "X-API-Key" not in session.request_kwargs["headers"]


def test_stale_live_observation_is_rejected():
    current = datetime.now(timezone.utc)
    old_payload = payload()
    old_payload["timestamp"] = (current - timedelta(days=2)).isoformat()
    with pytest.raises(live_data.LiveDataError, match="stale"):
        live_data.normalize_live_data(
            old_payload,
            ["rainfall"],
            {"rainfall": "rainfall"},
            {"rainfall": "mm"},
            "G1",
            max_age_seconds=86400,
            now=current,
        )


def test_future_live_observation_is_rejected():
    current = datetime.now(timezone.utc)
    future_payload = payload()
    future_payload["timestamp"] = (current + timedelta(hours=1)).isoformat()
    with pytest.raises(live_data.LiveDataError, match="future"):
        live_data.normalize_live_data(
            future_payload,
            ["rainfall"],
            {"rainfall": "rainfall"},
            {"rainfall": "mm"},
            "G1",
            now=current,
        )


def test_feature_mismatch():
    with pytest.raises(live_data.LiveDataError, match="missing features: river_level"):
        live_data.validate_model_features({"rainfall": 1.0}, ["rainfall", "river_level"])


def test_successful_live_data_prediction(monkeypatch):
    monkeypatch.setattr(live_data, "load_model_contract", lambda: (object(), {"feature_names": ["rainfall"]}))
    monkeypatch.setattr(live_data, "predict_flood", lambda values: {
        "prediction": "No Flood",
        "prediction_value": 0,
        "flood_probability": 12.5,
        "no_flood_probability": 87.5,
        "risk_level": "LOW",
        "forecast_horizon": "Next 1 day",
    })
    result = live_data.predict_live(config(), FakeSession(FakeResponse(payload())))
    assert result["status"] == "success"
    assert result["prediction"] == "No Flood"
    assert result["flood_probability"] == 12.5
