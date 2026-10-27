"""Live-provider status routes without making an unconfigured request."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from dotenv import load_dotenv

from live_data import LiveDataError, load_config, load_model_contract

router = APIRouter(prefix="/live")
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _status() -> dict[str, object]:
    if not os.getenv("LIVE_DATA_PROVIDER_URL", "").strip():
        return {"status": "not_configured", "source": None, "data_timestamp": None, "message": "No verified live provider is configured."}
    try:
        load_config()
        load_model_contract()
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