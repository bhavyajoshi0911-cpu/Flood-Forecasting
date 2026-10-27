import pytest

from app.ml.evaluate import EvaluationError, evaluate_predictions


def test_evaluation_rejects_mismatched_inputs():
    with pytest.raises(EvaluationError, match="same length"):
        evaluate_predictions([0, 1], [0.2])


def test_evaluation_rejects_probability_outside_range():
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        evaluate_predictions([0], [1.2])


def test_single_class_metrics_are_explicitly_unavailable():
    try:
        evaluate_predictions([0, 0], [0.1, 0.2])
    except EvaluationError as error:
        assert "scikit-learn" in str(error)
    else:
        metrics = evaluate_predictions([0, 0], [0.1, 0.2])
        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None