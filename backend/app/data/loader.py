"""Load documented tabular observations without changing source files."""

from __future__ import annotations

import csv
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
    """Count rows and collect headers from tab-separated basin source files."""

    files = sorted(raw_directory.glob("*.basin"))
    if not files:
        return None

    columns: set[str] = set()
    row_count = 0
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
                columns.update(column.strip() for column in header if column.strip())
                row_count += sum(1 for row in reader if any(cell.strip() for cell in row))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            errors.append(f"Could not read {path.name}: {error}")

    return {
        "status": "error" if errors else "warning",
        "path": str(raw_directory),
        "row_count": row_count,
        "columns": sorted(columns),
        "source_files": source_files,
        "source_fingerprints": source_fingerprints,
        "errors": errors,
        "warnings": ["Raw .basin sources are detected; detailed CSV-schema validation is not applied."],
    }


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