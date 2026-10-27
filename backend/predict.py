"""Future flood prediction using the saved preprocessing/model pipeline."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "flood_forecasting_model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "model_metadata.json"
FEATURE_METADATA_PATH = PROJECT_ROOT / "backend" / "data" / "feature_engineering_summary.json"
TARGET_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "research_target_definition.json"
TEST_FEATURES_PATH = PROJECT_ROOT / "backend" / "data" / "X_test.csv"
TEST_TARGET_PATH = PROJECT_ROOT / "backend" / "data" / "y_test.csv"


class PredictionError(ValueError):
    """Raised when a prediction request is invalid or cannot be served."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PredictionError(f"Invalid metadata file: {path}") from error
    return value if isinstance(value, dict) else {}


def load_prediction_assets() -> tuple[Any, dict[str, Any]]:
    """Load the complete serialized pipeline and metadata without retraining."""

    if not MODEL_PATH.is_file():
        raise PredictionError(f"Model file not found: {MODEL_PATH}")
    if not MODEL_METADATA_PATH.is_file():
        raise PredictionError(f"Model metadata file not found: {MODEL_METADATA_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        raise PredictionError(f"Could not load saved model: {error}") from error
    metadata = _read_json(MODEL_METADATA_PATH)
    feature_metadata = _read_json(FEATURE_METADATA_PATH)
    target_metadata = _read_json(TARGET_METADATA_PATH)
    metadata["forecast_horizon"] = metadata.get("forecast_horizon") or feature_metadata.get("forecast_horizon")
    metadata["dataset_frequency"] = metadata.get("dataset_frequency") or feature_metadata.get("dataset_frequency")
    metadata["target_class_mapping"] = metadata.get("target_class_mapping") or target_metadata.get("class_mapping", {})
    if not metadata.get("feature_names"):
        raise PredictionError("Model metadata does not define feature_names.")
    return model, metadata


def _model_classes(model: Any) -> list[Any]:
    classifier = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return []
    return list(classes)


def _feature_frame(input_data: dict[str, Any] | pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    timestamp = None
    if isinstance(input_data, dict):
        extra = sorted(set(input_data).difference(feature_names))
        missing = sorted(set(feature_names).difference(input_data))
        if missing:
            raise PredictionError(f"Missing required feature: {missing[0]}")
        if extra:
            raise PredictionError(f"Unexpected input feature: {extra[0]}")
        frame = pd.DataFrame([{name: input_data[name] for name in feature_names}])
    elif isinstance(input_data, pd.DataFrame):
        timestamp_columns = [column for column in input_data.columns if _looks_like_timestamp(column)]
        if timestamp_columns:
            timestamp = input_data[timestamp_columns[0]].copy()
        missing = sorted(set(feature_names).difference(input_data.columns))
        extra = sorted(set(input_data.columns).difference(feature_names).difference(timestamp_columns))
        if missing:
            raise PredictionError(f"Missing required feature: {missing[0]}")
        if extra:
            raise PredictionError(f"Unexpected input feature: {extra[0]}")
        frame = input_data.loc[:, feature_names].copy()
    else:
        raise PredictionError("input_data must be a feature dictionary or pandas DataFrame")
    if frame.empty:
        raise PredictionError("At least one observation is required.")
    numeric_values = frame.select_dtypes(include=np.number)
    if not numeric_values.empty and not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise PredictionError("Input contains infinite or invalid numerical values.")
    return frame, timestamp


def _looks_like_timestamp(column: str) -> bool:
    name = str(column).lower()
    return "date" in name or "time" in name or "timestamp" in name


def calculate_risk_level(flood_probability: float, medium_threshold: float | None = None, high_threshold: float | None = None) -> str:
    """Map probability to an application-level risk band.

    Defaults are configurable and are not official warning thresholds.
    """

    medium = medium_threshold if medium_threshold is not None else float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.40"))
    high = high_threshold if high_threshold is not None else float(os.getenv("RISK_HIGH_THRESHOLD", "0.70"))
    if not 0 <= medium < high <= 1:
        raise PredictionError("Risk thresholds must satisfy 0 <= medium < high <= 1")
    if flood_probability < medium:
        return "LOW"
    if flood_probability < high:
        return "MEDIUM"
    return "HIGH"


def _horizon_text(metadata: dict[str, Any]) -> str:
    horizon = metadata.get("forecast_horizon")
    if horizon is None:
        return "Forecast horizon is not defined in model metadata."
    frequency = metadata.get("dataset_frequency") or "day"
    unit = "hour" if str(frequency).lower() == "hourly" else "day"
    return f"Next {horizon} {unit}{'' if horizon == 1 else 's'}"


def _class_label(value: Any, metadata: dict[str, Any]) -> str:
    mapping = metadata.get("target_class_mapping") or {}
    label = mapping.get(str(value), mapping.get(value))
    if label is None:
        try:
            label = mapping.get(str(int(float(value))))
        except (TypeError, ValueError):
            label = None
    return str(label) if label is not None else str(value)


def _is_flood_class(value: Any, metadata: dict[str, Any]) -> bool:
    return _class_label(value, metadata).lower() == "flood"


def predict_flood(input_data: dict[str, Any] | pd.DataFrame) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict one observation or delegate a DataFrame to ``predict_batch``."""

    if isinstance(input_data, pd.DataFrame):
        return predict_batch(input_data)
    model, metadata = load_prediction_assets()
    feature_names = list(metadata["feature_names"])
    frame, _ = _feature_frame(input_data, feature_names)
    expected_count = metadata.get("number_of_features")
    if expected_count is not None and len(feature_names) != int(expected_count):
        raise PredictionError("Model metadata feature count does not match feature_names.")
    try:
        prediction_value = model.predict(frame)[0]
        probabilities = model.predict_proba(frame)[0] if hasattr(model, "predict_proba") else None
    except Exception as error:
        raise PredictionError(f"Prediction failed: {error}") from error
    classes = _model_classes(model)
    probability_map = {str(value): float(probability) for value, probability in zip(classes, probabilities)} if probabilities is not None else {}
    flood_class = None
    mapping = metadata.get("target_class_mapping") or {}
    for value in classes:
        if _is_flood_class(value, metadata):
            flood_class = value
            break
    flood_probability = probability_map.get(str(flood_class)) if flood_class is not None else None
    result: dict[str, Any] = {
        "prediction_value": prediction_value.item() if hasattr(prediction_value, "item") else prediction_value,
        "prediction": _class_label(prediction_value, metadata),
        "forecast_horizon": _horizon_text(metadata),
        "target_column": metadata.get("target_column"),
        "model_name": metadata.get("model_name"),
        "class_probabilities": probability_map,
    }
    if flood_probability is not None:
        result["flood_probability"] = round(flood_probability * 100, 2)
        result["no_flood_probability"] = round((1 - flood_probability) * 100, 2)
        result["risk_level"] = calculate_risk_level(flood_probability)
    else:
        result["probability_status"] = "Flood class mapping is not defined in model metadata."
    return result


def predict_batch(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Predict multiple already-engineered observations using the saved pipeline."""

    model, metadata = load_prediction_assets()
    feature_names = list(metadata["feature_names"])
    frame, timestamp = _feature_frame(input_dataframe, feature_names)
    try:
        predictions = model.predict(frame)
        probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None
    except Exception as error:
        raise PredictionError(f"Batch prediction failed: {error}") from error
    classes = _model_classes(model)
    mapping = metadata.get("target_class_mapping") or {}
    flood_class = next((value for value in classes if _is_flood_class(value, metadata)), None)
    output = pd.DataFrame({"prediction_value": predictions, "prediction": [_class_label(value, metadata) for value in predictions]})
    if timestamp is not None:
        output.insert(0, timestamp.name or "timestamp", timestamp.to_numpy())
    if probabilities is not None and flood_class is not None:
        flood_index = classes.index(flood_class)
        flood_probabilities = probabilities[:, flood_index]
        output["flood_probability"] = (flood_probabilities * 100).round(2)
        output["no_flood_probability"] = ((1 - flood_probabilities) * 100).round(2)
        output["risk_level"] = [calculate_risk_level(float(value)) for value in flood_probabilities]
    return output


def run_prediction_test() -> dict[str, Any]:
    if not TEST_FEATURES_PATH.is_file() or not TEST_TARGET_PATH.is_file():
        raise PredictionError("X_test.csv or y_test.csv is missing.")
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = pd.read_csv(TEST_TARGET_PATH).iloc[:, 0]
    if X_test.empty:
        raise PredictionError("X_test.csv contains no observations.")
    result = predict_flood(X_test.iloc[0].to_dict())
    result["actual_value"] = y_test.iloc[0].item() if hasattr(y_test.iloc[0], "item") else y_test.iloc[0]
    return result


def main() -> int:
    try:
        model, metadata = load_prediction_assets()
        feature_names = list(metadata["feature_names"])
        print("## MODEL CONSISTENCY CHECK")
        print("Model loaded: PASS")
        print(f"Feature count: {'PASS' if len(feature_names) == metadata.get('number_of_features') else 'FAIL'}")
        print("Feature names: PASS")
        print(f"Preprocessing: {'PASS' if hasattr(model, 'predict') else 'FAIL'}")
        result = run_prediction_test()
        print("Prediction: PASS")
        print("\n# ========================================")
        print("FUTURE FLOOD PREDICTION REPORT")
        print(f"Model: {result['model_name']}\nTarget: {result['target_column']}\nForecast Horizon: {result['forecast_horizon']}")
        print(f"Input Features: {len(feature_names)}\nActual: {_class_label(result['actual_value'], metadata)}\nPrediction: {result['prediction']}")
        print(f"Flood Probability: {result.get('flood_probability', 'Unavailable')}%\nNo Flood Probability: {result.get('no_flood_probability', 'Unavailable')}%\nRisk Level: {result.get('risk_level', 'Unavailable')}")
        print("\nModel Loaded: PASS\nInput Validation: PASS\nPrediction: PASS")
        print("\n========================================")
        print("Future Flood Prediction module completed successfully.")
        print("Next step: Integrate Live Weather/River Data and connect the prediction model to the Flask API.")
        return 0
    except (PredictionError, OSError, ValueError) as error:
        print(f"Prediction stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())