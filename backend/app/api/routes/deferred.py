"""Forecast and reporting endpoints backed by the project’s saved model artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_session_factory, init_database
from app.models.database_models import Prediction
from app.ml.explain import ExplanationError, calculate_pipeline_shap
from live_data import LiveDataError, predict_live
from predict import PredictionError, predict_flood

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_FEATURES_PATH = PROJECT_ROOT / "backend" / "data" / "X_test.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "model_metadata.json"


class ForecastPayload(BaseModel):
    features: dict[str, Any] | None = None


def _persist_prediction(
    result: dict[str, Any],
    input_features: dict[str, Any],
    prediction_type: str,
) -> int:
    """Persist an actual model result with its input and model provenance."""

    probability_percent = result.get("flood_probability")
    prediction_value = result.get("prediction_value")
    if probability_percent is None or prediction_value is None:
        raise HTTPException(
            status_code=500,
            detail="Forecast completed but cannot be recorded because its prediction class or probability is unavailable",
        )
    try:
        prediction_value = int(prediction_value)
        probability = float(probability_percent) / 100
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Forecast completed but its result could not be recorded") from error

    if not 0 <= probability <= 1:
        raise HTTPException(status_code=500, detail="Forecast completed but returned an invalid probability")

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    engine = init_database(os.getenv("DATABASE_URL") or f"sqlite:///{PROJECT_ROOT / 'backend' / 'flood_forecasting.db'}")
    session = get_session_factory(engine)()
    try:
        record = Prediction(
            timestamp=datetime.now(timezone.utc),
            prediction_type=prediction_type,
            input_features_json=json.dumps(input_features, allow_nan=False),
            prediction=prediction_value,
            probability=probability,
            risk_label=str(result.get("risk_level") or result.get("prediction") or "unknown"),
            model_version=str(result.get("model_name") or metadata.get("model_name") or "unknown"),
            dataset_version=str(metadata.get("dataset_version") or "research-dataset"),
            feature_version=f"model-features-v{metadata.get('number_of_features', len(metadata.get('feature_names', [])))}",
            data_source=str(result.get("data_source") or prediction_type),
        )
        session.add(record)
        session.commit()
        return record.id
    except (SQLAlchemyError, OSError, ValueError) as error:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Forecast completed but could not be recorded: {error}") from error
    finally:
        session.close()


def _load_model_feature_names() -> list[str]:
    if not MODEL_METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metadata is not available")
    payload = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise HTTPException(status_code=503, detail="Model feature contract is empty")
    return [str(feature) for feature in feature_names]


def _load_shap_summary(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from predict import load_prediction_assets

    model, _ = load_prediction_assets()
    feature_names = _load_model_feature_names()
    if records is None:
        if not SAMPLE_FEATURES_PATH.exists():
            raise HTTPException(status_code=503, detail="Demo feature dataset is missing")
        dataframe = pd.read_csv(SAMPLE_FEATURES_PATH, nrows=100)
        records = dataframe.loc[:, feature_names].to_dict(orient="records")
    try:
        return calculate_pipeline_shap(model, records, feature_names)
    except ExplanationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/forecast/manual")
def manual_forecast(payload: ForecastPayload) -> dict[str, Any]:
    try:
        if payload.features is None:
            raise HTTPException(status_code=400, detail="Request body must include a features object")
        result = predict_flood(payload.features)
        response = {
            "status": "success",
            "data_source": "manual",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "explanation": _load_shap_summary([payload.features])["individual_values"][0],
            **result,
        }
        response["prediction_id"] = _persist_prediction(response, payload.features, "manual")
        return response
    except HTTPException:
        raise
    except PredictionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {error}") from error


@router.post("/forecast/live")
def live_forecast() -> dict[str, Any]:
    try:
        result = predict_live()
        result["prediction_id"] = _persist_prediction(
            result,
            {"station_id": result.get("station_id"), "data_timestamp": result.get("timestamp")},
            "live",
        )
        return result
    except HTTPException:
        raise
    except LiveDataError as error:
        raise HTTPException(status_code=503, detail=f"Live forecast unavailable: {error}") from error
    except Exception as error:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=503, detail=f"Live forecast unavailable: {error}") from error


HISTORICAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_flood_dataset_with_target.csv"


@router.get("/historical")
def historical(request: Request) -> dict[str, Any]:
    if not HISTORICAL_DATA_PATH.exists():
        return {"records": [], "total": 0, "status": "unavailable", "message": "Historical dataset file is missing"}

    try:
        frame = pd.read_csv(HISTORICAL_DATA_PATH)
    except Exception as error:  # pragma: no cover - defensive fallback
        return {"records": [], "total": 0, "status": "unavailable", "message": f"Historical dataset could not be loaded: {error}"}

    if frame.empty:
        return {"records": [], "total": 0, "status": "unavailable", "message": "No historical records available"}

    query_params = request.query_params
    station_id = (query_params.get("station_id") or "").strip()
    flood_status = (query_params.get("flood_status") or "").strip()
    start_date = (query_params.get("start_date") or "").strip()
    end_date = (query_params.get("end_date") or "").strip()

    if "Station" in frame.columns:
        frame = frame.rename(columns={"Station": "station_id"})
    if "date" in frame.columns:
        frame = frame.rename(columns={"date": "timestamp"})
    if "flood_observed_research" in frame.columns:
        frame = frame.rename(columns={"flood_observed_research": "flood_observed"})
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "flood_observed" in frame.columns:
        frame["flood_observed"] = pd.to_numeric(frame["flood_observed"], errors="coerce").fillna(0).astype(int)

    if start_date:
        frame = frame[frame["timestamp"] >= start_date]
    if end_date:
        frame = frame[frame["timestamp"] <= end_date]
    if station_id:
        frame = frame[frame["station_id"].astype(str).str.contains(str(station_id), case=False, na=False)]
    if flood_status in {"0", "1"}:
        frame = frame[frame["flood_observed"] == int(flood_status)]

    frame = frame.sort_values(by="timestamp", ascending=False, kind="mergesort")

    page = max(int(query_params.get("page") or 1), 1)
    page_size = max(int(query_params.get("page_size") or 20), 1)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    page_data = frame.iloc[start_index:end_index]

    records = []
    for _, row in page_data.iterrows():
        records.append({
            "timestamp": row.get("timestamp") or "--",
            "station_id": row.get("station_id") or "--",
            "rainfall": row.get("rainfall_daily_mm"),
            "river_discharge": row.get("discharge_mean_m3_sec"),
            "water_level": row.get("river_level_mean_m"),
            "flood_observed": row.get("flood_observed"),
            "predicted_flood": row.get("predicted_flood"),
        })

    return {"records": records, "total": int(len(frame)), "page": page, "page_size": page_size, "status": "success"}


@router.post("/explain")
def explain(payload: ForecastPayload) -> dict[str, Any]:
    try:
        if payload.features is None:
            raise HTTPException(status_code=400, detail="Request body must include a features object")
        result = predict_flood(payload.features)
        summary = _load_shap_summary([payload.features])
        return {"status": "success", "prediction": result, "explanation": summary["individual_values"][0]}
    except HTTPException:
        raise
    except (PredictionError, ExplanationError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/shap/summary")
def shap_summary() -> dict[str, Any]:
    return {"status": "available", **_load_shap_summary()}


@router.get("/predictions")
def predictions(request: Request) -> dict[str, Any]:
    page = max(int(request.query_params.get("page") or 1), 1)
    page_size = min(max(int(request.query_params.get("page_size") or 20), 1), 100)
    engine = init_database(os.getenv("DATABASE_URL") or f"sqlite:///{PROJECT_ROOT / 'backend' / 'flood_forecasting.db'}")
    session = get_session_factory(engine)()
    try:
        query = session.query(Prediction).order_by(desc(Prediction.timestamp), desc(Prediction.id))
        total = query.count()
        records = query.offset((page - 1) * page_size).limit(page_size).all()
        return {
            "status": "success",
            "records": [
                {
                    "id": record.id,
                    "timestamp": (
                        record.timestamp.replace(tzinfo=timezone.utc)
                        if record.timestamp.tzinfo is None
                        else record.timestamp
                    ).isoformat(),
                    "prediction_type": record.prediction_type,
                    "risk_label": record.risk_label,
                    "probability": record.probability,
                    "data_source": record.data_source,
                    "model_version": record.model_version,
                }
                for record in records
            ],
            "retrospective_records": _retrospective_test_predictions(),
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except SQLAlchemyError as error:
        raise HTTPException(status_code=500, detail=f"Prediction history could not be loaded: {error}") from error
    finally:
        session.close()


def _retrospective_test_predictions() -> list[dict[str, Any]]:
    """Run the saved model on held-out features without treating them as live requests."""

    if not SAMPLE_FEATURES_PATH.is_file():
        raise HTTPException(status_code=503, detail="Held-out test features are not available")
    try:
        features = pd.read_csv(SAMPLE_FEATURES_PATH)
        if not {"year", "day_of_year", "Station"}.issubset(features.columns):
            raise HTTPException(
                status_code=503,
                detail="Held-out test features do not contain the date and station fields required for history",
            )
        predictions_frame = predict_flood(features)
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, ValueError, PredictionError) as error:
        raise HTTPException(status_code=503, detail=f"Held-out test predictions are unavailable: {error}") from error

    if not isinstance(predictions_frame, pd.DataFrame) or len(predictions_frame) != len(features):
        raise HTTPException(status_code=503, detail="Model returned an invalid held-out test result")

    years = pd.to_numeric(features["year"], errors="coerce")
    days = pd.to_numeric(features["day_of_year"], errors="coerce")
    if years.isna().any() or days.isna().any():
        raise HTTPException(status_code=503, detail="Held-out test dates are invalid")
    timestamps = pd.to_datetime(
        years.astype(int).astype(str) + days.astype(int).astype(str).str.zfill(3),
        format="%Y%j",
        errors="coerce",
    )
    if timestamps.isna().any():
        raise HTTPException(status_code=503, detail="Held-out test dates could not be decoded")

    records: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions_frame.to_dict(orient="records")):
        probability_percent = prediction.get("flood_probability")
        records.append(
            {
                "id": f"test-{index}",
                "timestamp": timestamps.iloc[index].strftime("%Y-%m-%d"),
                "station_id": str(features.iloc[index]["Station"]),
                "prediction_type": "retrospective test",
                "risk_label": prediction.get("risk_level") or prediction.get("prediction", "unknown"),
                "probability": (
                    float(probability_percent) / 100
                    if probability_percent is not None and pd.notna(probability_percent)
                    else None
                ),
                "data_source": "held-out test features",
                "model_version": str(metadata.get("model_name", "saved model")),
            }
        )
    return records


@router.get("/alerts")
def alerts() -> dict[str, Any]:
    """Return current CWC-reported warning statuses as source-linked alerts."""

    from .live import cwc_water_level_alerts

    feed = cwc_water_level_alerts()
    fetched_at = str(feed["fetched_at"])
    return {
        "status": "available",
        "source": feed["source"],
        "source_url": feed["source_url"],
        "fetched_at": fetched_at,
        "message": "CWC-reported station warning statuses; not generated by this project's model and not an emergency alert service.",
        "alerts": [
            {
                "id": f"cwc-{item['station_code']}",
                "severity": item["status"],
                "message": f"Station {item['station_code']} is reported as {item['status'].lower()}"
                + (f" and {item['trend'].lower()}" if item["trend"] else "")
                + ".",
                "created_at": fetched_at,
                "station_code": item["station_code"],
                "trend": item["trend"],
                "source": feed["source"],
            }
            for item in feed["alerts"]
        ],
    }