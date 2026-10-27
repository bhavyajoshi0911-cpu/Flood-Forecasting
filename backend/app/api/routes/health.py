"""Health and configuration status routes."""

from fastapi import APIRouter

from ...config import MODEL_DIRECTORY, RAW_DATA_DIRECTORY

router = APIRouter()

MODEL_PATH = MODEL_DIRECTORY / "flood_forecasting_model.pkl"
MODEL_METADATA_PATH = MODEL_DIRECTORY / "model_metadata.json"


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "dataset_configured": any(RAW_DATA_DIRECTORY.glob("*.csv"))
        or any(RAW_DATA_DIRECTORY.glob("*.basin")),
        "model_configured": MODEL_PATH.is_file() and MODEL_METADATA_PATH.is_file(),
    }