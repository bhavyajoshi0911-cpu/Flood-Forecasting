"""Train and version an XGBoost classifier from chronological feature splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .split import DatasetSplit


class TrainingError(ValueError):
    """Raised when training inputs are incomplete or scientifically unsafe."""


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1 or self.max_depth < 1 or self.learning_rate <= 0:
            raise TrainingError("Estimator count, depth, and learning rate must be positive")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise TrainingError("Subsample values must be in the interval (0, 1]")
        if self.min_child_weight < 0 or self.reg_alpha < 0 or self.reg_lambda < 0:
            raise TrainingError("Regularization and child weight cannot be negative")


def _matrix(rows: list[dict[str, Any]], feature_names: list[str], partition: str) -> tuple[list[list[float]], list[int]]:
    if not rows:
        raise TrainingError(f"{partition} partition is empty")
    matrix: list[list[float]] = []
    targets: list[int] = []
    for row in rows:
        values: list[float] = []
        for feature in feature_names:
            value = row.get(feature)
            if value is None or isinstance(value, bool):
                raise TrainingError(f"{partition} contains missing feature {feature!r}")
            try:
                values.append(float(value))
            except (TypeError, ValueError) as error:
                raise TrainingError(f"{partition} feature {feature!r} is not numeric") from error
        target = row.get("target_next_day")
        if target not in (0, 1, "0", "1"):
            raise TrainingError(f"{partition} target_next_day must be binary")
        matrix.append(values)
        targets.append(int(target))
    return matrix, targets


def _period(rows: list[dict[str, Any]]) -> dict[str, str]:
    timestamps = [str(row["timestamp"]) for row in rows]
    return {"start": min(timestamps), "end": max(timestamps)}


def train_xgboost(
    split: DatasetSplit,
    feature_names: list[str],
    output_directory: Path,
    dataset_version: str,
    feature_version: str,
    model_version: str,
    config: XGBoostConfig | None = None,
) -> dict[str, Any]:
    """Train a binary classifier and save its model/config/metadata artifacts."""

    if not feature_names:
        raise TrainingError("At least one feature is required")
    if not all(value.strip() for value in (dataset_version, feature_version, model_version)):
        raise TrainingError("Dataset, feature, and model versions are required")
    config = config or XGBoostConfig()
    train_x, train_y = _matrix(split.train, feature_names, "train")
    validation_x, validation_y = _matrix(split.validation, feature_names, "validation")
    _matrix(split.test, feature_names, "test")
    if len(set(train_y)) < 2:
        raise TrainingError("Training data must contain both target classes")

    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise TrainingError("XGBoost is not installed; install backend requirements first") from error

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
    )
    model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)

    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / f"{model_version}.json"
    config_path = output_directory / f"{model_version}_feature_config.json"
    metadata_path = output_directory / f"{model_version}_metadata.json"
    model.save_model(model_path)
    config_path.write_text(json.dumps({"feature_names": feature_names}, indent=2), encoding="utf-8")
    metadata = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "training_created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": asdict(config),
        "objective": "binary:logistic",
        "training_period": _period(split.train),
        "validation_period": _period(split.validation),
        "test_period": _period(split.test),
        "feature_names": feature_names,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model": str(model_path), "feature_config": str(config_path), "metadata": str(metadata_path)}