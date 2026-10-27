import pytest

from app.ml.feature_engineering import FeatureEngineeringError, build_features, feature_names


def test_features_use_only_prior_observations():
    rows = build_features(
        [
            {"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "rainfall": 10},
            {"timestamp": "2026-01-02T00:00:00+00:00", "station_id": "G1", "rainfall": 20},
            {"timestamp": "2026-01-03T00:00:00+00:00", "station_id": "G1", "rainfall": 30},
        ],
        ["rainfall"],
        lag_periods=(1,),
        rolling_windows=(2,),
    )
    assert rows[0]["rainfall_lag_1"] is None
    assert rows[0]["rainfall_2day_mean"] is None
    assert rows[1]["rainfall_lag_1"] == 10
    assert rows[1]["rainfall_2day_mean"] == 10
    assert rows[2]["rainfall_lag_1"] == 20
    assert rows[2]["rainfall_2day_mean"] == 15


def test_stations_do_not_share_history():
    rows = build_features(
        [
            {"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "rainfall": 10},
            {"timestamp": "2026-01-02T00:00:00+00:00", "station_id": "G2", "rainfall": 99},
            {"timestamp": "2026-01-02T00:00:00+00:00", "station_id": "G1", "rainfall": 20},
        ],
        ["rainfall"],
        lag_periods=(1,),
        rolling_windows=(),
    )
    g1_second = next(row for row in rows if row["station_id"] == "G1" and row["rainfall_lag_1"] is not None)
    assert g1_second["rainfall_lag_1"] == 10


def test_non_numeric_source_value_is_rejected():
    with pytest.raises(FeatureEngineeringError, match="numeric"):
        build_features(
            [{"timestamp": "2026-01-01T00:00:00+00:00", "station_id": "G1", "rainfall": "heavy"}],
            ["rainfall"],
        )


def test_feature_schema_is_deterministic():
    assert feature_names(["rainfall"], lag_periods=(1,), rolling_windows=(3,)) == [
        "month", "day_of_year", "rainfall_lag_1", "rainfall_3day_mean"
    ]