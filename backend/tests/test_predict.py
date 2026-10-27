import pytest

from app.data.live_fetcher import LiveFetchResult
from app.ml.predict import ForecastError, create_forecast, create_live_forecast


class FakeModel:
    def predict_proba(self, matrix):
        return [[0.2, 0.8]]


def test_missing_model_feature_is_rejected_before_prediction():
    with pytest.raises(ForecastError, match="Missing model feature"):
        create_forecast(FakeModel(), {}, ["rainfall"], "XGB-v1", "manual", "manual input")


def test_unusable_live_data_is_rejected():
    result = LiveFetchResult("stale", "provider", message="observation is too old")
    with pytest.raises(ForecastError, match="not usable"):
        create_live_forecast(FakeModel(), result, ["rainfall"], "XGB-v1")


def test_invalid_threshold_is_rejected():
    with pytest.raises(ForecastError, match="threshold"):
        create_forecast(FakeModel(), {"rainfall": 1}, ["rainfall"], "XGB-v1", "manual", "manual input", threshold=1)