"""Quality checks for the minimum, source-agnostic observation contract."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .loader import LoadedDataset

REQUIRED_COLUMNS = {"timestamp", "station_id"}
MEASUREMENT_COLUMNS = ("rainfall", "river_discharge", "water_level")
METADATA_FIELDS = (
    "source_name",
    "source_url",
    "date_range",
    "station_or_geography",
    "frequency",
    "timezone",
    "units",
    "license_or_access",
)


def _parse_number(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def validate_dataset(
    dataset: LoadedDataset,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable quality findings without mutating rows."""

    rows = dataset.rows
    missing_required = sorted(REQUIRED_COLUMNS - set(dataset.columns))
    missing_values = {
        column: sum(not row.get(column, "") for row in rows)
        for column in dataset.columns
    }
    timestamp_values = [row.get("timestamp", "") for row in rows]
    invalid_timestamps = sum(_parse_timestamp(value) is None for value in timestamp_values)
    duplicate_rows = len(rows) - len({tuple(row.get(column, "") for column in dataset.columns) for row in rows})
    duplicate_keys = sum(
        count - 1
        for count in Counter(
            (row.get("timestamp", ""), row.get("station_id", "")) for row in rows
        ).values()
        if count > 1
    )
    invalid_measurements: dict[str, int] = {}
    negative_rainfall = 0
    for column in MEASUREMENT_COLUMNS:
        if column not in dataset.columns:
            continue
        invalid_measurements[column] = sum(
            bool(row.get(column, "")) and _parse_number(row.get(column, "")) is None
            for row in rows
        )
        if column == "rainfall":
            negative_rainfall = sum(
                (number := _parse_number(row.get(column, ""))) is not None and number < 0
                for row in rows
            )

    metadata_missing = []
    if metadata is not None:
        metadata_missing = [field for field in METADATA_FIELDS if not metadata.get(field)]

    errors = []
    warnings = []
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")
    if invalid_timestamps:
        errors.append(f"Invalid or missing timestamps: {invalid_timestamps}")
    if any(invalid_measurements.values()):
        errors.append(f"Invalid measurement values: {invalid_measurements}")
    if negative_rainfall:
        errors.append(f"Negative rainfall values: {negative_rainfall}")
    if duplicate_rows or duplicate_keys:
        warnings.append(f"Duplicate rows: {duplicate_rows}; duplicate station-time keys: {duplicate_keys}")
    if not rows:
        warnings.append("Dataset contains no observation rows")
    if metadata is None:
        warnings.append("No dataset metadata file supplied")
    elif metadata_missing:
        warnings.append(f"Incomplete dataset metadata: {', '.join(metadata_missing)}")

    status = "error" if errors else "warning" if warnings else "valid"
    return {
        "status": status,
        "path": str(dataset.path),
        "row_count": len(rows),
        "columns": dataset.columns,
        "missing_values": missing_values,
        "invalid_timestamps": invalid_timestamps,
        "duplicate_rows": duplicate_rows,
        "duplicate_station_time_keys": duplicate_keys,
        "invalid_measurements": invalid_measurements,
        "negative_rainfall": negative_rainfall,
        "metadata_missing": metadata_missing,
        "errors": errors,
        "warnings": warnings,
    }


def empty_quality_report(raw_directory: str) -> dict[str, Any]:
    """Return the honest report for an unconfigured raw-data directory."""

    return {
        "status": "not_configured",
        "path": raw_directory,
        "row_count": 0,
        "columns": [],
        "errors": [],
        "warnings": ["No CSV dataset has been configured in data/raw"],
    }