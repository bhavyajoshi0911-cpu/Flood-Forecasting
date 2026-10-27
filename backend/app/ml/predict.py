"""Manual and live forecast orchestration for a trained model."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .explain import ExplanationError, calculate_shap
from ..data.live_fetcher import LiveFetchResult


class ForecastError(ValueError):
    """Raised when a forecast cannot be produced with validated inputs."""


def _feature_vector(values: dict[str, Any], feature_names: list[str]) -> list[float]:
    if not feature_names:
        raise ForecastError("Model feature configuration is empty")
    vector: list[float] = []
    for feature in feature_names:
        if feature not in values or values[feature] is None or isinstance(values[feature], bool):
            raise ForecastError(f"Missing model feature: {feature}")
        try:
            vector.append(float(values[feature]))
        except (TypeError, ValueError) as error:
            raise ForecastError(f"Model feature {feature!r} is not numeric") from error
    return vector


def create_forecast(
    model: Any,
    values: dict[str, Any],
    feature_names: list[str],
    model_version: str,
    prediction_type: str,
    data_source: str,
    timestamp: str | None = None,
    data_timestamp: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict one row and attach a real SHAP explanation."""

    if prediction_type not in {"manual", "live"}:
        raise ForecastError("prediction_type must be manual or live")
    if not model_version.strip() or not data_source.strip():
        raise ForecastError("model_version and data_source are required")
    if not 0 < threshold < 1:
        raise ForecastError("Forecast threshold must be between 0 and 1")
    vector = _feature_vector(values, feature_names)
    try:
        probabilities = model.predict_proba([vector])
        probability = float(probabilities[0][1])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ForecastError(f"Model probability prediction failed: {error}") from error
    if not 0 <= probability <= 1:
        raise ForecastError("Model returned an invalid probability")
    try:
        explanation = calculate_shap(model, [vector], feature_names)
    except ExplanationError as error:
        raise ForecastError(f"SHAP explanation unavailable: {error}") from error
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "data_timestamp": data_timestamp,
        "prediction_type": prediction_type,
        "risk": "flood" if probability >= threshold else "normal",
        "prediction": int(probability >= threshold),
        "probability": probability,
        "model_version": model_version,
        "data_source": data_source,
        "explanation": explanation["individual_values"][0],
    }


def create_manual_forecast(
    model: Any, values: dict[str, Any], feature_names: list[str], model_version: str, **kwargs: Any
) -> dict[str, Any]:
    return create_forecast(model, values, feature_names, model_version, "manual", "manual input", **kwargs)


def create_live_forecast(
    model: Any,
    live_result: LiveFetchResult,
    feature_names: list[str],
    model_version: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if live_result.status != "ok" or live_result.observations is None:
        raise ForecastError(f"Live data is not usable: {live_result.status} ({live_result.message})")
    return create_forecast(
        model,
        live_result.observations,
        feature_names,
        model_version,
        "live",
        live_result.source,
        data_timestamp=live_result.data_timestamp,
        **kwargs,
    )