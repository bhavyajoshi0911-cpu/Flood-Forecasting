import pytest

from app.ml.split import SplitError, split_chronologically


def rows():
    return [
        {"timestamp": f"2026-01-{day:02d}T00:00:00+00:00", "station_id": "G1"}
        for day in range(1, 11)
    ]


def test_ratio_split_is_sorted_and_disjoint():
    result = split_chronologically(rows(), train_ratio=0.6, validation_ratio=0.2)
    assert [item["timestamp"] for item in result.train] == [
        "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00",
        "2026-01-04T00:00:00+00:00", "2026-01-05T00:00:00+00:00", "2026-01-06T00:00:00+00:00",
    ]
    assert len(result.validation) == 2
    assert len(result.test) == 2
    assert result.train[-1]["timestamp"] < result.validation[0]["timestamp"] < result.test[0]["timestamp"]


def test_explicit_boundaries_are_exclusive():
    result = split_chronologically(rows(), train_end="2026-01-05T00:00:00+00:00", validation_end="2026-01-08T00:00:00+00:00")
    assert len(result.train) == 4
    assert len(result.validation) == 3
    assert len(result.test) == 3


def test_invalid_or_empty_split_is_rejected():
    with pytest.raises(SplitError, match="ratios"):
        split_chronologically(rows(), train_ratio=0.8, validation_ratio=0.3)
    with pytest.raises(SplitError, match="empty"):
        split_chronologically(rows(), train_end="2025-01-01T00:00:00+00:00", validation_end="2025-01-02T00:00:00+00:00")