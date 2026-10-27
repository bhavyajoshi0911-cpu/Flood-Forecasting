"""Model metadata and metric status routes."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from ...config import MODEL_DIRECTORY

router = APIRouter(prefix="/model")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data"
MODEL_PATH = MODEL_DIRECTORY / "flood_forecasting_model.pkl"


def _load_json(suffix: str) -> dict[str, object] | None:
    files = sorted(MODEL_DIRECTORY.glob(f"*{suffix}"))
    if not files:
        return None
    try:
        value = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _model_is_available() -> bool:
    return MODEL_PATH.is_file() and (MODEL_DIRECTORY / "model_metadata.json").is_file()


def _metric_value(source: dict[str, object], labels: tuple[str, ...]) -> float | None:
    for label in labels:
        value = source.get(label)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _load_evaluation_artifacts() -> dict[str, object] | None:
    if not MODEL_PATH.is_file():
        return None
    x_path = DATA_DIRECTORY / "X_test.csv"
    y_path = DATA_DIRECTORY / "y_test.csv"
    if not x_path.is_file() or not y_path.is_file():
        return None
    try:
        X_test = pd.read_csv(x_path)
        y_true = pd.read_csv(y_path).iloc[:, 0].astype(int)
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X_test)
        else:
            return None
        y_pred = model.predict(X_test)
        labels = [0, 1]
        confusion = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label=1)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, probabilities)
        return {
            "confusion_matrix": confusion,
            "roc_curve": {
                "fpr": [float(value) for value in fpr.tolist()],
                "tpr": [float(value) for value in tpr.tolist()],
                "auc": float(roc_auc_score(y_true, probabilities)),
            },
            "precision_recall_curve": {
                "precision": [float(value) for value in pr_precision.tolist()],
                "recall": [float(value) for value in pr_recall.tolist()],
            },
        }
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


@router.get("/info")
def model_info() -> dict[str, object]:
    metadata = _load_json("_metadata.json")
    if not metadata or not _model_is_available():
        return {"status": "unavailable", "metadata": None}
    return {"status": "available", "metadata": metadata}


@router.get("/metrics")
def model_metrics() -> dict[str, object]:
    metadata = _load_json("_metadata.json")
    if not metadata or not _model_is_available():
        return {"status": "unavailable", "metrics": None}

    metric_block = metadata.get("metrics")
    if not isinstance(metric_block, dict):
        return {"status": "unavailable", "metrics": None}

    model_name = metadata.get("model_name") or metadata.get("model_version") or "Unknown model"
    training_period = metadata.get("training_period")
    validation_period = metadata.get("validation_period")
    test_period = metadata.get("testing_period")

    artifacts = _load_evaluation_artifacts() or {}
    payload = {
        "model_name": model_name,
        "model_version": model_name,
        "training_period": training_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "accuracy": _metric_value(metric_block, ("Accuracy", "accuracy")),
        "precision": _metric_value(metric_block, ("Precision", "precision")),
        "recall": _metric_value(metric_block, ("Recall", "recall")),
        "f1": _metric_value(metric_block, ("F1 Score", "F1 score", "f1")),
        "roc_auc": _metric_value(metric_block, ("ROC-AUC", "roc_auc")),
        "pr_auc": _metric_value(metric_block, ("PR-AUC", "pr_auc")),
        "confusion_matrix": artifacts.get("confusion_matrix"),
        "roc_curve": artifacts.get("roc_curve"),
        "precision_recall_curve": artifacts.get("precision_recall_curve"),
    }
    return {"status": "available", "metrics": payload}