"""Flask REST API for live flood-data and prediction integration."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .live_data import (
        LiveDataError,
        fetch_live_data,
        load_config,
        load_model_contract,
        normalize_live_data,
        predict_live,
    )
    from .predict import PredictionError, predict_flood
except ImportError:  # Running as: python backend/app.py
    from live_data import LiveDataError, fetch_live_data, load_config, load_model_contract, normalize_live_data, predict_live
    from predict import PredictionError, predict_flood


load_dotenv()
LOGGER = logging.getLogger(__name__)


def _json_error(message: str, status_code: int, **details: Any):
    payload: dict[str, Any] = {"status": "error", "message": message}
    payload.update(details)
    return jsonify(payload), status_code


def _provider_error_status(error: LiveDataError) -> int:
    message = str(error).lower()
    if "not configured" in message or "model" in message:
        return 503
    return 502


def _frontend_origins() -> list[str]:
    configured = os.getenv("FRONTEND_URL", "http://localhost:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create a configured Flask application without loading secrets into responses."""

    load_dotenv()
    application = Flask(__name__)
    application.config.from_mapping(
        MAX_CONTENT_LENGTH=64 * 1024,
        JSON_SORT_KEYS=False,
    )
    if test_config:
        application.config.update(test_config)
    if application.config.get("ENABLE_CORS", True):
        CORS(application, origins=application.config.get("FRONTEND_ORIGINS", _frontend_origins()))

    @application.before_request
    def log_request() -> None:
        LOGGER.info("API request %s %s", request.method, request.path)

    @application.get("/api/health")
    def health():
        try:
            load_model_contract()
            model_loaded = True
        except LiveDataError:
            model_loaded = False
        live_data_configured = bool(os.getenv("LIVE_DATA_API_KEY", "").strip() and os.getenv("LIVE_DATA_PROVIDER_URL", "").strip())
        return jsonify({
            "status": "success",
            "message": "Flood Forecasting API is running",
            "model_loaded": model_loaded,
            "live_data_configured": live_data_configured,
        })

    @application.get("/api/live-data")
    def live_data_endpoint():
        try:
            model, metadata = load_model_contract()
            config = load_config()
            raw_data = fetch_live_data(config)
            normalized = normalize_live_data(raw_data, list(metadata["feature_names"]), config.field_mapping, config.expected_units, config.expected_station)
            data = {feature: normalized[feature] for feature in metadata["feature_names"]}
            return jsonify({"status": "success", "timestamp": normalized["timestamp"], "station_id": normalized["station_id"], "data": data, "source": "live_api"})
        except LiveDataError as error:
            LOGGER.warning("Live-data endpoint unavailable: %s", error)
            return _json_error("Unable to fetch live data", _provider_error_status(error), reason=str(error))

    @application.post("/api/predict")
    def predict_endpoint():
        if request.data and not request.is_json:
            return _json_error("Request body must contain valid JSON", 400)
        payload = request.get_json(silent=True)
        if request.data and request.is_json and payload is None:
            return _json_error("Request body must contain valid JSON", 400)
        if payload is None or payload == {}:
            try:
                result = predict_live()
                return jsonify(result)
            except LiveDataError as error:
                LOGGER.warning("Live prediction unavailable: %s", error)
                return _json_error("Unable to generate live prediction", _provider_error_status(error), reason=str(error))
        if not isinstance(payload, dict):
            return _json_error("JSON request body must be an object", 400)
        features = payload.get("features", payload)
        if not isinstance(features, dict):
            return _json_error("features must be a JSON object", 400)
        try:
            _, metadata = load_model_contract()
            required = list(metadata["feature_names"])
            missing = [feature for feature in required if feature not in features]
            if missing:
                return _json_error("Required model features are missing", 400, missing_features=missing)
            unexpected = [feature for feature in features if feature not in required]
            if unexpected:
                return _json_error("Unexpected model features were supplied", 400, unexpected_features=unexpected)
            result = predict_flood(features)
            return jsonify({
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "manual",
                **result,
            })
        except PredictionError as error:
            LOGGER.warning("Manual prediction validation failed: %s", error)
            return _json_error(str(error), 400)
        except LiveDataError as error:
            LOGGER.error("Model unavailable: %s", error)
            return _json_error("Prediction model is unavailable", 503)
        except (TypeError, ValueError) as error:
            LOGGER.warning("Manual prediction input invalid: %s", error)
            return _json_error("Prediction input contains invalid values", 400)

    @application.errorhandler(404)
    def not_found(_error):
        return _json_error("Endpoint not found", 404)

    @application.errorhandler(413)
    def too_large(_error):
        return _json_error("Request payload is too large", 400)

    @application.errorhandler(Exception)
    def unexpected_error(error):
        LOGGER.exception("Unexpected API error: %s", error)
        return _json_error("Internal server error", 500)

    return application


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    LOGGER.info("Starting Flood Forecasting Flask API on %s:%s", host, port)
    app.run(host=host, port=port, debug=debug)