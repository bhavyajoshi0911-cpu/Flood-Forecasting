"""Live-provider status routes without making an unconfigured request."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from live_data import LiveDataError, load_config, load_model_contract, validate_provider_config

router = APIRouter(prefix="/live")
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

CWC_WATER_LEVEL_ALERTS_URL = "https://ffs.india-water.gov.in/ffm/api/station-water-level-above-warning/"


def _status() -> dict[str, object]:
    if not os.getenv("LIVE_DATA_PROVIDER_URL", "").strip():
        return {"status": "not_configured", "source": None, "data_timestamp": None, "message": "No verified live provider is configured."}
    try:
        config = load_config()
        _, metadata = load_model_contract()
        validate_provider_config(config, metadata["feature_names"])
    except LiveDataError as error:
        return {
            "status": "incomplete",
            "source": None,
            "data_timestamp": None,
            "message": str(error),
        }
    return {
        "status": "configured",
        "source": None,
        "data_timestamp": None,
        "message": "Provider settings loaded; no observation is considered live until timestamp, station, units, and every model feature pass validation.",
    }


@router.get("/status")
def live_status() -> dict[str, object]:
    return _status()


@router.get("/latest")
def live_latest() -> dict[str, object]:
    status = _status()
    return {**status, "observations": None}


@router.get("/cwc-water-level-alerts")
def cwc_water_level_alerts() -> dict[str, Any]:
    """Fetch CWC's public list of stations reported above warning level."""

    try:
        response = requests.get(
            CWC_WATER_LEVEL_ALERTS_URL,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as error:
        raise HTTPException(status_code=502, detail="CWC water-level feed timed out") from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="CWC water-level feed returned invalid JSON") from error
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail="CWC water-level feed could not be reached") from error

    if not isinstance(payload, list):
        raise HTTPException(status_code=502, detail="CWC water-level feed returned an unexpected response")

    alerts: list[dict[str, str | None]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise HTTPException(status_code=502, detail=f"CWC water-level feed record {index} is invalid")
        station_code = record.get("stationCode")
        status = record.get("status")
        trend = record.get("trend")
        if not isinstance(station_code, str) or not station_code.strip() or not isinstance(status, str) or not status.strip():
            raise HTTPException(status_code=502, detail=f"CWC water-level feed record {index} is missing station or status")
        if trend is not None and not isinstance(trend, str):
            raise HTTPException(status_code=502, detail=f"CWC water-level feed record {index} has an invalid trend")
        alerts.append(
            {
                "station_code": station_code.strip(),
                "status": status.strip(),
                "trend": trend.strip() if trend else None,
            }
        )

    return {
        "status": "available",
        "source": "Central Water Commission Flood Forecasting portal",
        "source_url": CWC_WATER_LEVEL_ALERTS_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_timestamp": None,
        "station_scope": "The feed does not include station names or basin mapping.",
        "message": "These are CWC water-level warning records, not predictions from this project's model. The feed does not provide an observation timestamp or units.",
        "alerts": alerts,
        "total": len(alerts),
    }