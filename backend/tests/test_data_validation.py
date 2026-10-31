import csv
import json
from pathlib import Path

from app.data.loader import LoadedDataset, summarize_basin_files
from app.data.validator import empty_quality_report, validate_dataset
from app.api.routes import dataset as dataset_routes


def make_dataset(tmp_path: Path, rows: list[dict[str, str]]) -> LoadedDataset:
    path = tmp_path / "observations.csv"
    columns = ["timestamp", "station_id", "rainfall", "water_level"]
    with path.open("w", newline="", encoding="utf-8") as source:
        writer = csv.DictWriter(source, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return LoadedDataset(path, columns, rows)


def test_empty_directory_is_explicitly_unconfigured():
    report = empty_quality_report("data/raw")
    assert report["status"] == "not_configured"
    assert report["row_count"] == 0


def test_basin_sources_report_available_quality_metrics(tmp_path):
    basin_file = tmp_path / "rainfall.basin"
    basin_file.write_text(
        "Station\tData Acquisition Time\tTelemetry Hourly Rainfall (mm)\n"
        "G1\t30-10-19 14:30\t12.5\n"
        "G1\t30-10-19 14:30\t12.5\n"
        "G2\tbad date\t-2\n",
        encoding="utf-8",
    )

    report = summarize_basin_files(tmp_path)

    assert report is not None
    assert report["status"] == "warning"
    assert report["row_count"] == 3
    assert report["source_files"] == ["rainfall.basin"]
    assert "Station" in report["columns"]
    assert report["source_fingerprints"][0]["filename"] == "rainfall.basin"
    assert report["invalid_timestamps"] == 1
    assert report["duplicate_rows"] == 1
    assert report["negative_rainfall"] == 1


def test_basin_quality_marks_unavailable_fields_as_not_applicable(tmp_path):
    basin_file = tmp_path / "river.basin"
    basin_file.write_text(
        "Station\tData Acquisition Time\nG1\t30-10-19 14:30\n",
        encoding="utf-8",
    )

    report = summarize_basin_files(tmp_path)

    assert report is not None
    assert report["invalid_timestamps"] == 0
    assert report["duplicate_rows"] == 0
    assert report["negative_rainfall"] is None


def test_quality_report_rebuilds_stale_cached_basin_report(tmp_path, monkeypatch):
    raw_directory = tmp_path / "raw"
    metadata_directory = tmp_path / "metadata"
    raw_directory.mkdir()
    metadata_directory.mkdir()
    source = raw_directory / "rainfall.basin"
    source.write_text(
        "Station\tData Acquisition Time\tTelemetry Hourly Rainfall (mm)\n"
        "G1\tbad date\t-1\n",
        encoding="utf-8",
    )
    stat = source.stat()
    cached_report = {
        "status": "warning",
        "path": str(raw_directory),
        "row_count": 1,
        "columns": [],
        "source_files": ["rainfall.basin"],
        "source_fingerprints": [
            {
                "filename": source.name,
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            }
        ],
        "errors": [],
        "warnings": [],
    }
    (metadata_directory / "data_quality_report.json").write_text(
        json.dumps(cached_report),
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_routes, "RAW_DATA_DIRECTORY", raw_directory)
    monkeypatch.setattr(dataset_routes, "METADATA_DIRECTORY", metadata_directory)

    report = dataset_routes.quality_report()

    assert report["invalid_timestamps"] == 1
    assert report["duplicate_rows"] == 0
    assert report["negative_rainfall"] == 1


def test_validator_detects_invalid_timestamp_and_negative_rainfall(tmp_path):
    dataset = make_dataset(
        tmp_path,
        [{"timestamp": "not-a-date", "station_id": "G1", "rainfall": "-2", "water_level": "4.5"}],
    )
    report = validate_dataset(dataset)
    assert report["status"] == "error"
    assert report["invalid_timestamps"] == 1
    assert report["negative_rainfall"] == 1


def test_validator_warns_when_metadata_is_incomplete(tmp_path):
    dataset = make_dataset(
        tmp_path,
        [{"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "rainfall": "2", "water_level": "4.5"}],
    )
    report = validate_dataset(dataset, {"source_name": "documented source"})
    assert report["status"] == "warning"
    assert report["metadata_missing"]