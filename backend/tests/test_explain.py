import pytest

from app.ml.explain import ExplanationError, calculate_shap


def test_shap_requires_feature_names():
    with pytest.raises(ExplanationError, match="feature name"):
        calculate_shap(object(), [[1]], [])


def test_shap_rejects_shape_mismatch_before_model_call():
    with pytest.raises(ExplanationError, match="match feature_names"):
        calculate_shap(object(), [[1, 2]], ["rainfall"])


def test_shap_rejects_empty_matrix():
    with pytest.raises(ExplanationError, match="one row"):
        calculate_shap(object(), [], ["rainfall"])