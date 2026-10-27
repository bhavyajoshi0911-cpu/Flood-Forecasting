"""Fetch validated live observations and pass them to the saved model.

The configured provider must return the complete engineered feature vector used
by the model. This is intentional: the current model requires historical lag,
rolling, and change features, which cannot be invented from one observation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from .predict import PredictionError, load_prediction_assets, predict_flood
except ImportError:  # Running as: python live_data.py
    from predict import PredictionError, load_prediction_assets, predict_flood


load_dotenv()
LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 10.0


class LiveDataError(ValueError):
    """Raised when live data cannot safely be used for prediction."""


@dataclass(frozen=True)
class LiveDataConfig:
    provider_url: str
    api_key: str
    field_mapping: dict[str, str]
    expected_units: dict[str, str]
    expected_station: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_age_seconds: float = 86400.0


def _json_env(name: str, default: dict[str, str] | None = None) -> dict[str, str]:
    value = os.getenv(name, "").strip()
    if not value:
        return default or {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise LiveDataError(f"{name} must contain valid JSON") from error
    if not isinstance(parsed, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()):
        raise LiveDataError(f"{name} must be a JSON object with string keys and values")
    return parsed


def load_config() -> LiveDataConfig:
    """Load provider settings without logging secrets."""

    load_dotenv()
    provider_url = os.getenv("LIVE_DATA_PROVIDER_URL", "").strip()
    api_key = os.getenv("LIVE_DATA_API_KEY", "").strip()
    if not provider_url:
        raise LiveDataError("LIVE_DATA_PROVIDER_URL is not configured")
    if not provider_url.startswith(("http://", "https://")):
        raise LiveDataError("LIVE_DATA_PROVIDER_URL must start with http:// or https://")
    try:
        timeout = float(os.getenv("LIVE_DATA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    except ValueError as error:
        raise LiveDataError("LIVE_DATA_TIMEOUT_SECONDS must be numeric") from error
    if timeout <= 0:
        raise LiveDataError("LIVE_DATA_TIMEOUT_SECONDS must be positive")
    try:
        max_age = float(os.getenv("LIVE_DATA_MAX_AGE_SECONDS", "86400"))
    except ValueError as error:
        raise LiveDataError("LIVE_DATA_MAX_AGE_SECONDS must be numeric") from error
    if not np.isfinite(max_age) or max_age <= 0:
        raise LiveDataError("LIVE_DATA_MAX_AGE_SECONDS must be positive and finite")
    return LiveDataConfig(
        provider_url=provider_url,
        api_key=api_key,
        field_mapping=_json_env("LIVE_DATA_FIELD_MAPPING"),
        expected_units=_json_env("LIVE_DATA_EXPECTED_UNITS"),
        expected_station=os.getenv("LIVE_DATA_STATION", "").strip() or None,
        timeout_seconds=timeout,
        max_age_seconds=max_age,
    )


def load_model_contract() -> tuple[Any, dict[str, Any]]:
    """Load the saved pipeline and its actual feature contract."""

    try:
        model, metadata = load_prediction_assets()
    except PredictionError as error:
        raise LiveDataError(str(error)) from error
    features = metadata.get("feature_names")
    if not isinstance(features, list) or not features or not all(isinstance(feature, str) for feature in features):
        raise LiveDataError("Model metadata does not define a valid feature_names list")
    return model, metadata


def _parse_timestamp(value: Any) -> str:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise LiveDataError("Live response timestamp is invalid") from error
    if timestamp.tzinfo is None:
        raise LiveDataError("Live response timestamp must include a timezone")
    return timestamp.astimezone(timezone.utc).isoformat()


def fetch_live_data(config: LiveDataConfig | None = None, session: requests.Session | Any = requests) -> dict[str, Any]:
    """Fetch the provider's JSON contract without exposing the API key."""

    config = config or load_config()
    LOGGER.info("Starting live-data request to configured provider")
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["X-API-Key"] = config.api_key
    try:
        response = session.get(
            config.provider_url,
            headers=headers,
            timeout=config.timeout_seconds,
        )
        if response.status_code == 429:
            raise LiveDataError("Live provider rate limit reached")
        response.raise_for_status()
        payload = response.json()
    except LiveDataError:
        LOGGER.warning("Live-data provider rejected the request")
        raise
    except requests.Timeout as error:
        LOGGER.error("Live-data provider timed out")
        raise LiveDataError("Live-data provider request timed out") from error
    except requests.ConnectionError as error:
        LOGGER.error("Could not connect to live-data provider")
        raise LiveDataError("Could not connect to live-data provider") from error
    except requests.HTTPError as error:
        LOGGER.error("Live-data provider returned an HTTP error")
        raise LiveDataError(f"Live-data provider HTTP error: {error.response.status_code if error.response is not None else 'unknown'}") from error
    except ValueError as error:
        LOGGER.error("Live-data provider returned invalid JSON")
        raise LiveDataError("Live-data provider returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise LiveDataError("Live-data response must be a JSON object")
    LOGGER.info("Live-data request completed successfully")
    return payload


def normalize_live_data(raw_data: Mapping[str, Any], required_features: list[str], field_mapping: Mapping[str, str], expected_units: Mapping[str, str] | None = None, expected_station: str | None = None, max_age_seconds: float = 86400.0, now: datetime | None = None) -> dict[str, Any]:
    """Normalize and validate a provider payload into the model feature schema.

    Expected response shape::

        {"timestamp": "...", "station_id": "...",
         "observations": {"provider_field": value},
         "units": {"provider_field": "unit"}}
    """

    if not isinstance(raw_data, Mapping):
        raise LiveDataError("Live response must be a JSON object")
    if not np.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise LiveDataError("Live-data maximum age must be positive and finite")
    timestamp = _parse_timestamp(raw_data.get("timestamp"))
    timestamp_value = datetime.fromisoformat(timestamp)
    current_time = now or datetime.now(timezone.utc)
    if timestamp_value > current_time + timedelta(minutes=5):
        raise LiveDataError("Live response timestamp is in the future")
    if current_time - timestamp_value > timedelta(seconds=max_age_seconds):
        raise LiveDataError("Live response is stale")
    station = str(raw_data.get("station_id", "")).strip()
    if not station:
        raise LiveDataError("Live response is missing station_id")
    if expected_station and station != expected_station:
        raise LiveDataError(f"Station mismatch: expected {expected_station}, received {station}")
    observations = raw_data.get("observations")
    units = raw_data.get("units")
    if not isinstance(observations, Mapping) or not isinstance(units, Mapping):
        raise LiveDataError("Live response must contain observations and units objects")
    normalized: dict[str, Any] = {}
    for feature in required_features:
        provider_field = field_mapping.get(feature, feature)
        if provider_field not in observations or observations[provider_field] in (None, ""):
            raise LiveDataError(f"Missing required live feature: {feature}")
        unit = units.get(provider_field)
        if not unit:
            raise LiveDataError(f"Missing unit for live feature: {feature}")
        expected_unit = (expected_units or {}).get(feature)
        if expected_unit and str(unit) != expected_unit:
            raise LiveDataError(f"Unit mismatch for {feature}: expected {expected_unit}, received {unit}")
        value = observations[provider_field]
        if isinstance(value, bool):
            raise LiveDataError(f"Live feature {feature} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise LiveDataError(f"Live feature {feature} must be numeric") from error
        if not np.isfinite(numeric):
            raise LiveDataError(f"Live feature {feature} is not finite")
        normalized[feature] = numeric
    normalized["timestamp"] = timestamp
    normalized["station_id"] = station
    LOGGER.info("Live data normalization completed successfully")
    return normalized


def validate_model_features(normalized_data: Mapping[str, Any], required_features: list[str]) -> None:
    """Ensure no model feature is silently omitted or replaced."""

    missing = [feature for feature in required_features if feature not in normalized_data]
    if missing:
        raise LiveDataError(f"Live data is insufficient for prediction; missing features: {', '.join(missing)}")
    invalid = [feature for feature in required_features if isinstance(normalized_data[feature], bool) or not np.isfinite(float(normalized_data[feature]))]
    if invalid:
        raise LiveDataError(f"Live data contains invalid model features: {', '.join(invalid)}")
    LOGGER.info("Live feature compatibility validation passed")


def predict_live(config: LiveDataConfig | None = None, session: requests.Session | Any = requests) -> dict[str, Any]:
    """Fetch, normalize, validate, and predict from the saved model."""

    _, metadata = load_model_contract()
    required_features = list(metadata["feature_names"])
    config = config or load_config()
    raw_data = fetch_live_data(config, session)
    normalized = normalize_live_data(
        raw_data,
        required_features,
        config.field_mapping,
        config.expected_units,
        config.expected_station,
        config.max_age_seconds,
    )
    validate_model_features(normalized, required_features)
    prediction_input = {feature: normalized[feature] for feature in required_features}
    try:
        prediction = predict_flood(prediction_input)
    except PredictionError as error:
        raise LiveDataError(str(error)) from error
    result = {
        "timestamp": normalized["timestamp"],
        "station_id": normalized["station_id"],
        "prediction": prediction.get("prediction"),
        "prediction_value": prediction.get("prediction_value"),
        "flood_probability": prediction.get("flood_probability"),
        "no_flood_probability": prediction.get("no_flood_probability"),
        "risk_level": prediction.get("risk_level"),
        "forecast_horizon": prediction.get("forecast_horizon"),
        "data_source": "live_api",
        "status": "success",
    }
    LOGGER.info("Live prediction generated")
    return result


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
    try:
        result = predict_live()
    except LiveDataError as error:
        LOGGER.error("Live prediction unavailable: %s", error)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())