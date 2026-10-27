import pytest

from app.data.preprocessing import (
    PreprocessingError,
    find_daily_gaps,
    normalize_timestamp,
    preprocess_rows,
    require_documented_units,
)


def test_timestamp_is_normalized_to_utc():
    assert normalize_timestamp("2026-01-01T05:30:00+05:30") == "2026-01-01T00:00:00+00:00"


def test_naive_timestamp_is_rejected():
    with pytest.raises(PreprocessingError, match="no timezone"):
        normalize_timestamp("2026-01-01T00:00:00")


def test_preprocessing_coerces_numbers_and_removes_duplicate_station_time():
    rows, report = preprocess_rows(
        [
            {"timestamp": "2026-01-02T00:00:00Z", "station_id": "G1", "rainfall": "2.5"},
            {"timestamp": "2026-01-02T00:00:00Z", "station_id": "G1", "rainfall": "2.5"},
        ],
        ["rainfall"],
    )
    assert rows[0]["rainfall"] == 2.5
    assert report["duplicate_rows_removed"] == 1


def test_daily_gaps_are_reported_without_filling():
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1"},
        {"timestamp": "2026-01-03T00:00:00+00:00", "station_id": "G1"},
    ]
    assert find_daily_gaps(rows) == {"G1": ["2026-01-02"]}


def test_units_are_required_before_processing():
    with pytest.raises(PreprocessingError, match="units"):
        require_documented_units({}, ["rainfall"])