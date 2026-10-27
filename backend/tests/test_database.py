import pytest
from datetime import datetime, timezone

sqlalchemy = pytest.importorskip("sqlalchemy")

from app.database import get_session_factory, init_database  # noqa: E402
from app.models.database_models import Base, DatasetMetadata, Prediction  # noqa: E402


def test_database_creates_required_tables_and_persists_prediction(tmp_path):
    engine = init_database(f"sqlite:///{tmp_path / 'test.sqlite3'}")
    table_names = set(Base.metadata.tables)
    assert table_names == {
        "dataset_metadata", "model_metadata", "predictions", "prediction_explanations",
        "flood_events", "live_observations", "api_status", "alerts",
    }
    session = get_session_factory(engine)()
    session.add(DatasetMetadata(dataset_version="data-v1", source_name="source", source_url="url", metadata_json="{}"))
    session.add(Prediction(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), prediction_type="manual", input_features_json="{}",
        prediction=0, probability=0.1, risk_label="normal", model_version="XGB-v1",
        dataset_version="data-v1", feature_version="features-v1", data_source="manual",
    ))
    session.commit()
    assert session.query(Prediction).count() == 1
    session.close()