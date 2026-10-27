"""Environment-backed application configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
METADATA_DIRECTORY = PROJECT_ROOT / "data" / "metadata"
MODEL_DIRECTORY = PROJECT_ROOT / "backend" / "models"


def cors_origins() -> list[str]:
    configured = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:4173,http://127.0.0.1:4173,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]