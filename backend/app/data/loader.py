"""Load documented tabular observations without changing source files."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedDataset:
    """Rows and source information returned by the CSV loader."""

    path: Path
    columns: list[str]
    rows: list[dict[str, str]]


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be read as a supported table."""


def load_csv(path: Path) -> LoadedDataset:
    """Load a UTF-8 CSV while preserving values as strings for validation."""

    if not path.exists():
        raise DatasetLoadError(f"Dataset file does not exist: {path}")
    if not path.is_file():
        raise DatasetLoadError(f"Dataset path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise DatasetLoadError("CSV has no header row")
            columns = [column.strip() for column in reader.fieldnames]
            if any(not column for column in columns):
                raise DatasetLoadError("CSV header contains an empty column name")
            if len(set(columns)) != len(columns):
                raise DatasetLoadError("CSV header contains duplicate column names")
            rows = [
                {
                    column: (row.get(original) or "").strip()
                    for original, column in zip(reader.fieldnames, columns)
                }
                for row in reader
            ]
    except UnicodeDecodeError as error:
        raise DatasetLoadError("CSV must be UTF-8 encoded") from error
    except csv.Error as error:
        raise DatasetLoadError(f"Invalid CSV: {error}") from error

    return LoadedDataset(path=path, columns=columns, rows=rows)


def discover_csv(raw_directory: Path) -> Path | None:
    """Return the only CSV in a raw directory, or None when not configured."""

    candidates = sorted(raw_directory.glob("*.csv"))
    if len(candidates) > 1:
        raise DatasetLoadError(
            "Multiple CSV files found; select one explicitly instead of guessing"
        )
    return candidates[0] if candidates else None


def load_raw_dataset(raw_directory: Path, filename: str | None = None) -> LoadedDataset | None:
    """Load a selected CSV, returning None when no source has been configured."""

    path = raw_directory / filename if filename else discover_csv(raw_directory)
    return load_csv(path) if path else None


def summarize_basin_files(raw_directory: Path) -> dict[str, Any] | None:
    """Summarize tab-separated basin sources and validate their shared raw fields."""

    files = sorted(raw_directory.glob("*.basin"))
    if not files:
        return None

    columns: set[str] = set()
    row_count = 0
    invalid_timestamps = 0
    duplicate_rows = 0
    negative_rainfall = 0
    has_timestamps = False
    has_rainfall = False
    source_files: list[str] = []
    errors: list[str] = []
    source_fingerprints: list[dict[str, int | str]] = []
    for path in files:
        source_files.append(path.name)
        try:
            stat = path.stat()
            source_fingerprints.append(
                {
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                }
            )
        except OSError as error:
            errors.append(f"Could not inspect {path.name}: {error}")
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.reader(source, delimiter="\t")
                header = next(reader, [])
                if not header:
                    errors.append(f"{path.name} has no header row")
                    continue
                header = [column.strip() for column in header]
                columns.update(column for column in header if column)
                timestamp_column = next(
                    (index for index, column in enumerate(header) if column.lower() == "data acquisition time"),
                    None,
                )
                rainfall_column = next(
                    (index for index, column in enumerate(header) if column.lower() == "telemetry hourly rainfall (mm)"),
                    None,
                )
                has_timestamps = has_timestamps or timestamp_column is not None
                has_rainfall = has_rainfall or rainfall_column is not None
                seen_rows: set[bytes] = set()
                for row in reader:
                    if not any(cell.strip() for cell in row):
                        continue
                    row_count += 1
                    fingerprint = hashlib.blake2b(
                        "\x1f".join(row).encode("utf-8"),
                        digest_size=16,
                    ).digest()
                    if fingerprint in seen_rows:
                        duplicate_rows += 1
                    else:
                        seen_rows.add(fingerprint)
                    if timestamp_column is not None:
                        timestamp = row[timestamp_column].strip() if timestamp_column < len(row) else ""
                        if not _is_valid_basin_timestamp(timestamp):
                            invalid_timestamps += 1
                    if rainfall_column is not None and rainfall_column < len(row):
                        try:
                            rainfall = float(row[rainfall_column].strip())
                        except ValueError:
                            continue
                        if rainfall < 0:
                            negative_rainfall += 1
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            errors.append(f"Could not read {path.name}: {error}")

    return {
        "status": "error" if errors else "warning",
        "path": str(raw_directory),
        "row_count": row_count,
        "columns": sorted(columns),
        "invalid_timestamps": invalid_timestamps if has_timestamps else None,
        "duplicate_rows": duplicate_rows,
        "negative_rainfall": negative_rainfall if has_rainfall else None,
        "source_files": source_files,
        "source_fingerprints": source_fingerprints,
        "errors": errors,
        "warnings": ["Raw .basin sources are validated per source file; joined CSV-schema validation is not applied."],
    }


def _is_valid_basin_timestamp(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        try:
            datetime.strptime(value, "%d-%m-%y %H:%M")
            return True
        except ValueError:
            return False


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load JSON metadata associated with a real dataset."""

    import json

    try:
        with metadata_path.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetLoadError(f"Could not read metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise DatasetLoadError("Dataset metadata must be a JSON object")
    return metadata