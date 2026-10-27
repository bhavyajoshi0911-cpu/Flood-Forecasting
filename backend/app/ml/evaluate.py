"""Calculate test metrics and evaluation plots from actual model outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class EvaluationError(ValueError):
    """Raised when predictions cannot be evaluated as binary outputs."""


def _inputs(y_true: Iterable[Any], probabilities: Iterable[Any]) -> tuple[list[int], list[float]]:
    labels = list(y_true)
    scores = list(probabilities)
    if not labels or len(labels) != len(scores):
        raise EvaluationError("Labels and probabilities must be non-empty and the same length")
    if any(value not in (0, 1, "0", "1") for value in labels):
        raise EvaluationError("Labels must be binary")
    try:
        numeric_scores = [float(value) for value in scores]
    except (TypeError, ValueError) as error:
        raise EvaluationError("Probabilities must be numeric") from error
    if any(value < 0 or value > 1 for value in numeric_scores):
        raise EvaluationError("Probabilities must be between 0 and 1")
    return [int(value) for value in labels], numeric_scores


def evaluate_predictions(
    y_true: Iterable[Any],
    probabilities: Iterable[Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return classification metrics from a binary test prediction set."""

    if not 0 < threshold < 1:
        raise EvaluationError("Classification threshold must be between 0 and 1")
    labels, scores = _inputs(y_true, probabilities)
    predictions = [int(score >= threshold) for score in scores]
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as error:
        raise EvaluationError("scikit-learn is not installed; install backend requirements first") from error

    has_both_classes = len(set(labels)) == 2
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if has_both_classes else None,
        "pr_auc": average_precision_score(labels, scores) if has_both_classes else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "flood_precision": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "flood_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "flood_f1": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "sample_count": len(labels),
    }


def save_evaluation(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist measured metrics as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_evaluation_plots(
    y_true: Iterable[Any],
    probabilities: Iterable[Any],
    feature_importance: dict[str, float],
    output_directory: Path,
) -> list[str]:
    """Save plots from measured predictions and model-provided importances."""

    labels, scores = _inputs(y_true, probabilities)
    if not feature_importance:
        raise EvaluationError("Feature importance values are required for plotting")
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
    except ImportError as error:
        raise EvaluationError("Plot dependencies are not installed") from error

    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = [int(score >= 0.5) for score in scores]
    paths: list[str] = []
    figure, axis = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(labels, predictions, ax=axis)
    paths.append(str(output_directory / "confusion_matrix.png"))
    figure.savefig(paths[-1], bbox_inches="tight")
    plt.close(figure)
    if len(set(labels)) == 2:
        for display, filename in ((RocCurveDisplay, "roc_curve.png"), (PrecisionRecallDisplay, "precision_recall_curve.png")):
            figure, axis = plt.subplots()
            display.from_predictions(labels, scores, ax=axis)
            paths.append(str(output_directory / filename))
            figure.savefig(paths[-1], bbox_inches="tight")
            plt.close(figure)
    figure, axis = plt.subplots()
    names = list(feature_importance)
    values = [feature_importance[name] for name in names]
    axis.barh(names, values)
    axis.set_title("Feature importance")
    paths.append(str(output_directory / "feature_importance.png"))
    figure.savefig(paths[-1], bbox_inches="tight")
    plt.close(figure)
    return paths