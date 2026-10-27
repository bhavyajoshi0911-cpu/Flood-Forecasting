import pytest

from app.ml.split import DatasetSplit
from app.ml.train import TrainingError, XGBoostConfig, train_xgboost


def test_training_config_rejects_invalid_parameters():
    with pytest.raises(TrainingError, match="positive"):
        XGBoostConfig(n_estimators=0)
    with pytest.raises(TrainingError, match="Subsample"):
        XGBoostConfig(subsample=0)


def test_training_requires_both_classes_before_model_import(tmp_path):
    rows = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "rainfall_lag_1": 1, "target_next_day": 0},
        {"timestamp": "2026-01-02T00:00:00+00:00", "rainfall_lag_1": 2, "target_next_day": 0},
    ]
    split = DatasetSplit(rows, rows, rows)
    with pytest.raises(TrainingError, match="both target classes"):
        train_xgboost(split, ["rainfall_lag_1"], tmp_path, "data-v1", "features-v1", "XGB-v1")


def test_training_rejects_missing_feature_values(tmp_path):
    row = {"timestamp": "2026-01-01T00:00:00+00:00", "rainfall_lag_1": None, "target_next_day": 0}
    split = DatasetSplit([row], [row], [row])
    with pytest.raises(TrainingError, match="missing feature"):
        train_xgboost(split, ["rainfall_lag_1"], tmp_path, "data-v1", "features-v1", "XGB-v1")