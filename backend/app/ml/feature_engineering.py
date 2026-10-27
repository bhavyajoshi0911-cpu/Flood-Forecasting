"""Leakage-safe features derived from observations available at prediction time."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable


class FeatureEngineeringError(ValueError):
    """Raised when feature inputs do not meet the documented contract."""


def _as_number(value: Any, column: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise FeatureEngineeringError(f"{column} must be numeric, not boolean")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise FeatureEngineeringError(f"{column} must be numeric") from error


def _mean(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def build_features(
    rows: Iterable[dict[str, Any]],
    source_columns: Iterable[str],
    lag_periods: Iterable[int] = (1, 3, 7),
    rolling_windows: Iterable[int] = (3, 7),
) -> list[dict[str, Any]]:
    """Build calendar, lag, and prior-window features for each station.

    The output keeps one row per input observation. A feature at index ``i`` only
    reads source values with an index smaller than ``i`` for that station.
    """

    columns = tuple(source_columns)
    lags = tuple(lag_periods)
    windows = tuple(rolling_windows)
    if any(period < 1 for period in (*lags, *windows)):
        raise FeatureEngineeringError("Lag and rolling periods must be positive")

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row in rows:
        row = dict(source_row)
        station_id = str(row.get("station_id", "")).strip()
        if not station_id:
            raise FeatureEngineeringError("station_id is required")
        try:
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
        except (KeyError, ValueError) as error:
            raise FeatureEngineeringError("timestamp must be ISO-8601") from error
        row["station_id"] = station_id
        row["_parsed_timestamp"] = timestamp
        for column in columns:
            row[column] = _as_number(row.get(column), column)
        grouped[station_id].append(row)

    output: list[dict[str, Any]] = []
    for station_rows in grouped.values():
        station_rows.sort(key=lambda row: row["_parsed_timestamp"])
        for index, row in enumerate(station_rows):
            features: dict[str, Any] = {
                "timestamp": row["timestamp"],
                "station_id": row["station_id"],
                "month": row["_parsed_timestamp"].month,
                "day_of_year": row["_parsed_timestamp"].timetuple().tm_yday,
            }
            for column in columns:
                history = [station_rows[position].get(column) for position in range(index)]
                for period in lags:
                    features[f"{column}_lag_{period}"] = (
                        history[-period] if len(history) >= period else None
                    )
                for window in windows:
                    features[f"{column}_{window}day_mean"] = _mean(history[-window:])
            output.append(features)

    return sorted(output, key=lambda row: (row["station_id"], row["timestamp"]))


def feature_names(
    source_columns: Iterable[str],
    lag_periods: Iterable[int] = (1, 3, 7),
    rolling_windows: Iterable[int] = (3, 7),
) -> list[str]:
    """Return a deterministic feature schema for model metadata."""

    names = ["month", "day_of_year"]
    for column in source_columns:
        names.extend(f"{column}_lag_{period}" for period in lag_periods)
        names.extend(f"{column}_{window}day_mean" for window in rolling_windows)
    return names