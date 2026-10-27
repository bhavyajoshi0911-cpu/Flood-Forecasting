"""Small, explicit preprocessing steps for validated observation rows."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable


class PreprocessingError(ValueError):
    """Raised when a transformation would require an undocumented assumption."""


def normalize_timestamp(value: str) -> str:
    """Return an ISO-8601 UTC timestamp, rejecting naive timestamps."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PreprocessingError(f"Invalid timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise PreprocessingError(f"Timestamp has no timezone: {value!r}")
    return parsed.astimezone(timezone.utc).isoformat()


def _numeric(value: str, column: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise PreprocessingError(f"{column} is not numeric: {value!r}") from error


def preprocess_rows(
    rows: Iterable[dict[str, Any]],
    numeric_columns: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize timestamps/numbers, sort rows, and remove exact station-time duplicates."""

    columns = tuple(numeric_columns)
    prepared: list[dict[str, Any]] = []
    duplicate_count = 0
    seen_keys: set[tuple[str, str]] = set()
    for source_row in rows:
        row = dict(source_row)
        row["timestamp"] = normalize_timestamp(str(row.get("timestamp", "")))
        station_id = str(row.get("station_id", "")).strip()
        if not station_id:
            raise PreprocessingError("station_id is required")
        row["station_id"] = station_id
        for column in columns:
            if column in row:
                row[column] = _numeric(row[column], column)
        key = (row["timestamp"], station_id)
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)
        prepared.append(row)

    prepared.sort(key=lambda row: (row["station_id"], row["timestamp"]))
    return prepared, {"input_rows": len(prepared) + duplicate_count, "duplicate_rows_removed": duplicate_count}


def find_daily_gaps(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Report missing calendar dates per station without filling or inventing values."""

    dates_by_station: defaultdict[str, set[date]] = defaultdict(set)
    for row in rows:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
        dates_by_station[str(row["station_id"])].add(timestamp.date())

    gaps: dict[str, list[str]] = {}
    for station_id, dates in dates_by_station.items():
        if not dates:
            gaps[station_id] = []
            continue
        missing: list[str] = []
        current = min(dates)
        while current <= max(dates):
            if current not in dates:
                missing.append(current.isoformat())
            current += timedelta(days=1)
        gaps[station_id] = missing
    return gaps


def require_documented_units(metadata: dict[str, Any], columns: Iterable[str]) -> None:
    """Require units before numeric data can be used for scientific processing."""

    units = metadata.get("units")
    if not isinstance(units, dict):
        raise PreprocessingError("Metadata must provide a units mapping")
    missing = [column for column in columns if not units.get(column)]
    if missing:
        raise PreprocessingError(f"Missing units for columns: {', '.join(missing)}")