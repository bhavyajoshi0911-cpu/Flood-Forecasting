"""Dataset status routes."""

from functools import lru_cache
import json
from pathlib import Path

from fastapi import APIRouter

from ...config import METADATA_DIRECTORY, RAW_DATA_DIRECTORY
from ...data.loader import DatasetLoadError, load_raw_dataset, summarize_basin_files
from ...data.validator import empty_quality_report, validate_dataset

router = APIRouter(prefix="/dataset")


def _basin_quality_report() -> dict[str, object] | None:
    files = sorted(RAW_DATA_DIRECTORY.glob("*.basin"))
    if not files:
        return None

    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files
    )
    return _cached_basin_quality_report(signature)


@lru_cache(maxsize=4)
def _cached_basin_quality_report(
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, object] | None:
    if not signature:
        return None

    report_path = METADATA_DIRECTORY / "data_quality_report.json"
    try:
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached_report = None

    expected_fingerprints = [
        {
            "filename": Path(path_text).name,
            "size_bytes": size_bytes,
            "modified_time_ns": modified_time_ns,
        }
        for path_text, modified_time_ns, size_bytes in signature
    ]
    if (
        isinstance(cached_report, dict)
        and cached_report.get("path") == str(RAW_DATA_DIRECTORY)
        and cached_report.get("source_fingerprints") == expected_fingerprints
    ):
        return cached_report

    return summarize_basin_files(Path(signature[0][0]).parent)


def quality_report() -> dict[str, object]:
    basin_report = _basin_quality_report()
    if basin_report is not None:
        return basin_report
    try:
        dataset = load_raw_dataset(RAW_DATA_DIRECTORY)
        return empty_quality_report(str(RAW_DATA_DIRECTORY)) if dataset is None else validate_dataset(dataset)
    except DatasetLoadError as error:
        return {
            "status": "error",
            "path": str(RAW_DATA_DIRECTORY),
            "row_count": 0,
            "columns": [],
            "errors": [str(error)],
            "warnings": [],
        }


@router.get("/quality")
def dataset_quality() -> dict[str, object]:
    return quality_report()


@router.get("/summary")
def dataset_summary() -> dict[str, object]:
    report = quality_report()
    return {
        "status": report["status"],
        "row_count": report["row_count"],
        "columns": report["columns"],
        "path": report["path"],
    }