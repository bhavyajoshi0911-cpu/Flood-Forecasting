import pytest

from app.ml.target import TargetDefinition, TargetDefinitionError, build_next_day_targets


def definition(mode="label", column="flood_observed"):
    return TargetDefinition(mode, column, evidence_source="authoritative event dataset")


def test_next_day_label_is_aligned_by_station_and_date():
    result = build_next_day_targets(
        [
            {"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "rainfall": 5, "flood_observed": 0},
            {"timestamp": "2026-01-02T00:00:00+00:00", "station_id": "G1", "rainfall": 8, "flood_observed": 1},
        ],
        definition(),
    )
    assert len(result) == 1
    assert result[0]["target_next_day"] == 1
    assert result[0]["rainfall"] == 5


def test_last_observation_without_future_evidence_is_excluded():
    result = build_next_day_targets(
        [{"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "flood_observed": 0}],
        definition(),
    )
    assert result == []


def test_threshold_requires_explicit_evidence():
    with pytest.raises(TargetDefinitionError, match="evidence_source"):
        TargetDefinition("threshold", "water_level", threshold=10)
    with pytest.raises(TargetDefinitionError, match="research assumption"):
        TargetDefinition("threshold", "water_level", threshold=10, evidence_source="unverified")


def test_threshold_can_be_explicit_research_assumption():
    result = build_next_day_targets(
        [
            {"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "water_level": 5},
            {"timestamp": "2026-01-02T00:00:00+00:00", "station_id": "G1", "water_level": 12},
        ],
        TargetDefinition(
            "threshold", "water_level", threshold=10,
            evidence_source="research assumption: threshold requires later authority review",
        ),
    )
    assert result[0]["target_next_day"] == 1