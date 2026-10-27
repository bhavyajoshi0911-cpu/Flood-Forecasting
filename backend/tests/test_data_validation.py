import csv
from pathlib import Path

from app.data.loader import LoadedDataset, summarize_basin_files
from app.data.validator import empty_quality_report, validate_dataset


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


def test_basin_sources_are_detected_but_report_limited_validation(tmp_path):
    basin_file = tmp_path / "rainfall.basin"
    basin_file.write_text("Station\tRainfall\nG1\t12.5\n", encoding="utf-8")

    report = summarize_basin_files(tmp_path)

    assert report is not None
    assert report["status"] == "warning"
    assert report["row_count"] == 1
    assert report["source_files"] == ["rainfall.basin"]
    assert "Station" in report["columns"]
    assert report["source_fingerprints"][0]["filename"] == "rainfall.basin"


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