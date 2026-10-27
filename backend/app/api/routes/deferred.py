"""Forecast and reporting endpoints backed by the project’s saved model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.ml.explain import ExplanationError, calculate_pipeline_shap
from live_data import LiveDataError, predict_live
from predict import PredictionError, predict_flood

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_FEATURES_PATH = PROJECT_ROOT / "backend" / "data" / "X_test.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "backend" / "models" / "model_metadata.json"


class ForecastPayload(BaseModel):
    features: dict[str, Any] | None = None


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
        return {"status": "success", "data_source": "manual", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "explanation": _load_shap_summary([payload.features])["individual_values"][0], **result}
    except HTTPException:
        raise
    except PredictionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {error}") from error


@router.post("/forecast/live")
def live_forecast() -> dict[str, Any]:
    try:
        return predict_live()
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
def predictions() -> dict[str, str]:
    return {"status": "unavailable", "message": "Prediction history is not available yet"}


@router.get("/alerts")
def alerts() -> dict[str, str]:
    return {"status": "unavailable", "message": "Alerts are not available yet"}