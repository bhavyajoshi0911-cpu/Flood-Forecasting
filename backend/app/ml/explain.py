"""Real SHAP explanations for a trained tree model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class ExplanationError(ValueError):
    """Raised when a SHAP explanation cannot be computed safely."""


def _validate_matrix(matrix: Iterable[Iterable[Any]], feature_names: list[str]) -> list[list[float]]:
    if not feature_names:
        raise ExplanationError("At least one feature name is required")
    values = [list(row) for row in matrix]
    if not values:
        raise ExplanationError("At least one row is required for SHAP")
    if any(len(row) != len(feature_names) for row in values):
        raise ExplanationError("Feature rows do not match feature_names")
    try:
        numeric = [[float(value) for value in row] for row in values]
    except (TypeError, ValueError) as error:
        raise ExplanationError("SHAP feature values must be numeric") from error
    return numeric


def _shap_array(raw_values: Any) -> Any:
    """Normalize SHAP's historical list and classifier array return forms."""

    if isinstance(raw_values, list):
        if not raw_values:
            raise ExplanationError("SHAP returned no values")
        raw_values = raw_values[-1]
    if getattr(raw_values, "ndim", None) == 3:
        raw_values = raw_values[:, :, -1]
    if getattr(raw_values, "ndim", None) != 2:
        raise ExplanationError("SHAP returned an unexpected value shape")
    return raw_values


def calculate_shap(model: Any, matrix: Iterable[Iterable[Any]], feature_names: list[str]) -> dict[str, Any]:
    """Calculate global and individual SHAP values from a trained model."""

    numeric_matrix = _validate_matrix(matrix, feature_names)
    try:
        import shap
    except ImportError as error:
        raise ExplanationError("SHAP is not installed; install backend requirements first") from error

    try:
        explainer = shap.TreeExplainer(model)
        values = _shap_array(explainer.shap_values(numeric_matrix))
    except Exception as error:
        raise ExplanationError(f"SHAP calculation failed: {error}") from error
    if values.shape[1] != len(feature_names) or values.shape[0] != len(numeric_matrix):
        raise ExplanationError("SHAP output does not match the feature matrix")

    value_rows = values.tolist()
    importance = {
        feature: sum(abs(row[index]) for row in value_rows) / len(value_rows)
        for index, feature in enumerate(feature_names)
    }
    global_importance = [
        {"feature": feature, "mean_abs_shap": value}
        for feature, value in sorted(importance.items(), key=lambda item: item[1], reverse=True)
    ]
    individual = [
        {feature: row[index] for index, feature in enumerate(feature_names)}
        for row in value_rows
    ]
    return {
        "feature_names": feature_names,
        "global_importance": global_importance,
        "individual_values": individual,
        "sample_count": len(value_rows),
    }


def calculate_pipeline_shap(model: Any, records: Iterable[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    """Explain a tree classifier behind a fitted sklearn preprocessing pipeline."""

    rows = list(records)
    if not rows:
        raise ExplanationError("At least one row is required for SHAP")
    if not feature_names:
        raise ExplanationError("At least one feature name is required")
    if not hasattr(model, "named_steps"):
        raise ExplanationError("SHAP pipeline explanation requires named model steps")

    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        raise ExplanationError("SHAP pipeline is missing preprocessor or model steps")

    frame = pd.DataFrame(rows).reindex(columns=feature_names)
    if frame.isnull().any().any():
        raise ExplanationError("SHAP input contains missing feature values")
    try:
        transformed = preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        import shap
        values = _shap_array(shap.TreeExplainer(estimator).shap_values(transformed))
        transformed_names = list(preprocessor.get_feature_names_out(feature_names))
    except Exception as error:
        raise ExplanationError(f"SHAP pipeline calculation failed: {error}") from error

    if values.shape[0] != len(rows) or values.shape[1] != len(transformed_names):
        raise ExplanationError("SHAP output does not match transformed features")

    grouped_values: dict[str, list[int]] = {feature: [] for feature in feature_names}
    for transformed_index, transformed_name in enumerate(transformed_names):
        normalized_name = str(transformed_name).split("__", 1)[-1]
        candidates = [
            feature for feature in feature_names
            if normalized_name == feature or normalized_name.startswith(f"{feature}_")
        ]
        feature = max(candidates, key=len) if candidates else str(transformed_name)
        grouped_values.setdefault(feature, []).append(transformed_index)

    individual: list[dict[str, float]] = []
    for row in values.tolist():
        individual.append({
            feature: sum(row[index] for index in indices)
            for feature, indices in grouped_values.items()
            if indices
        })
    importance = {
        feature: sum(abs(row.get(feature, 0.0)) for row in individual) / len(individual)
        for feature in grouped_values
    }
    global_importance = [
        {"feature": feature, "mean_abs_shap": value}
        for feature, value in sorted(importance.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "feature_names": list(grouped_values),
        "global_importance": global_importance,
        "individual_values": individual,
        "sample_count": len(individual),
    }


def save_shap_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Persist measured SHAP results as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_shap_summary_plot(model: Any, matrix: Iterable[Iterable[Any]], feature_names: list[str], output_path: Path) -> None:
    """Save a SHAP beeswarm summary plot from an actual trained model."""

    numeric_matrix = _validate_matrix(matrix, feature_names)
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as error:
        raise ExplanationError("SHAP and matplotlib are required for plots") from error
    try:
        values = _shap_array(shap.TreeExplainer(model).shap_values(numeric_matrix))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(values, numeric_matrix, feature_names=feature_names, show=False)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    except Exception as error:
        raise ExplanationError(f"SHAP plot generation failed: {error}") from error