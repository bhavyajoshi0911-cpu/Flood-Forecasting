"""Train and compare classification models from prepared time-based splits.

This module never reloads or randomly splits the raw dataset. It expects
``backend/data/X_train.csv``, ``X_test.csv``, ``y_train.csv``, and ``y_test.csv``
to already exist from ``data_split.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "backend" / "data"
MODEL_DIRECTORY = PROJECT_ROOT / "backend" / "models"
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "model_outputs"


class TrainingError(ValueError):
    """Raised when prepared data is unsuitable for honest model training."""


def load_prepared_data(data_directory: Path = DATA_DIRECTORY) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    paths = {name: data_directory / f"{name}.csv" for name in ("X_train", "X_test", "y_train", "y_test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepared split files are missing: " + ", ".join(missing) + ". Run data_split.py first.")
    X_train = pd.read_csv(paths["X_train"])
    X_test = pd.read_csv(paths["X_test"])
    y_train_frame = pd.read_csv(paths["y_train"])
    y_test_frame = pd.read_csv(paths["y_test"])
    if y_train_frame.shape[1] != 1 or y_test_frame.shape[1] != 1:
        raise TrainingError("Each target CSV must contain exactly one target column.")
    y_train = y_train_frame.iloc[:, 0]
    y_test = y_test_frame.iloc[:, 0]
    metadata: dict[str, Any] = {}
    metadata_path = data_directory / "split_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise TrainingError("Feature and target row counts do not match.")
    if list(X_train.columns) != list(X_test.columns):
        raise TrainingError("X_train and X_test feature columns/order do not match.")
    return X_train, X_test, y_train, y_test, metadata


def identify_problem_type(y_train: pd.Series) -> str:
    unique_count = y_train.nunique(dropna=True)
    if unique_count < 2:
        raise TrainingError("The training target must contain at least two distinct values.")
    problem_type = "binary classification" if unique_count == 2 else "multiclass classification"
    print(f"Problem type: {problem_type}\nTarget unique values: {y_train.dropna().unique().tolist()}")
    return problem_type


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Remove accidental datetime columns and reject infinite values."""

    if not X_train.columns.equals(X_test.columns):
        raise TrainingError("Training and testing feature schemas differ.")
    train = X_train.copy()
    test = X_test.copy()
    datetime_columns: list[str] = []
    for column in train.columns:
        parsed = pd.to_datetime(train[column], errors="coerce", format="mixed", dayfirst=True)
        if parsed.notna().mean() >= 0.8 and ("date" in column.lower() or "time" in column.lower() or "timestamp" in column.lower()):
            datetime_columns.append(column)
    train = train.drop(columns=datetime_columns)
    test = test.drop(columns=datetime_columns)
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    numeric_columns = train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in train.columns if column not in numeric_columns]
    if not numeric_columns and not categorical_columns:
        raise TrainingError("No usable features remain after preprocessing.")
    print(f"Removed datetime features: {datetime_columns or 'None'}")
    print(f"Numeric features: {len(numeric_columns)}; categorical features: {len(categorical_columns)}")
    return train, test, numeric_columns, categorical_columns


def _preprocessor(numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if categorical_columns:
        categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(numeric_columns: list[str], categorical_columns: list[str], imbalanced: bool) -> dict[str, Pipeline]:
    class_weight = "balanced" if imbalanced else None
    return {
        "Logistic Regression": Pipeline([("preprocessor", _preprocessor(numeric_columns, categorical_columns, True)), ("model", LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42))]),
        "Decision Tree": Pipeline([("preprocessor", _preprocessor(numeric_columns, categorical_columns, False)), ("model", DecisionTreeClassifier(random_state=42, class_weight=class_weight))]),
        "Random Forest": Pipeline([("preprocessor", _preprocessor(numeric_columns, categorical_columns, False)), ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("preprocessor", _preprocessor(numeric_columns, categorical_columns, False)), ("model", GradientBoostingClassifier(random_state=42))]),
    }


def _metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray | None, binary: bool) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, average="binary" if binary else "weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, average="binary" if binary else "weighted", zero_division=0)),
        "F1 Score": float(f1_score(y_true, predictions, average="binary" if binary else "weighted", zero_division=0)),
        "ROC-AUC": None,
    }
    if probabilities is not None:
        try:
            result["ROC-AUC"] = float(roc_auc_score(y_true, probabilities if binary else probabilities, multi_class="ovr" if not binary else "raise"))
        except ValueError:
            result["ROC-AUC"] = None
    return result


def evaluate_models(models: dict[str, Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, binary: bool) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        score_values = probabilities[:, 1] if binary and probabilities is not None else probabilities
        metrics = _metrics(y_test, predictions, score_values, binary)
        rows.append({"Model": name, **metrics})
        details[name] = {"model": model, "predictions": predictions, "probabilities": probabilities, "metrics": metrics}
        print(classification_report(y_test, predictions, zero_division=0))
    return pd.DataFrame(rows), details


def save_confusion_matrices(details: dict[str, dict[str, Any]], y_test: pd.Series) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    labels = sorted(y_test.unique().tolist(), key=str)
    for name, detail in details.items():
        matrix = confusion_matrix(y_test, detail["predictions"], labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIRECTORY / f"{_file_name(name)}_confusion_matrix.png", dpi=150)
        plt.close()


def save_roc_curve(details: dict[str, dict[str, Any]], y_test: pd.Series, binary: bool) -> None:
    if not binary:
        return
    plt.figure(figsize=(7, 6))
    plotted = False
    for name, detail in details.items():
        probabilities = detail["probabilities"]
        if probabilities is None:
            continue
        false_positive, true_positive, _ = roc_curve(y_test, probabilities[:, 1])
        auc = detail["metrics"]["ROC-AUC"]
        plt.plot(false_positive, true_positive, label=f"{name} (AUC={auc:.3f})")
        plotted = True
    if plotted:
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIRECTORY / "roc_curve_comparison.png", dpi=150)
    plt.close()


def _file_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def select_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Prioritize flood recall, then F1, ROC-AUC, precision, and accuracy."""

    ranked = comparison.copy()
    for column in ("Recall", "F1 Score", "ROC-AUC", "Precision", "Accuracy"):
        ranked[column] = ranked[column].fillna(-1)
    ranked = ranked.sort_values(["Recall", "F1 Score", "ROC-AUC", "Precision", "Accuracy"], ascending=False)
    return str(ranked.iloc[0]["Model"]), "Recall, with F1/ROC-AUC/precision/accuracy tie-breaks"


def save_feature_importance(selected_name: str, selected_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str]) -> None:
    model = selected_model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    transformed_names = selected_model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=transformed_names).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    importance.sort_values().plot.barh()
    plt.title(f"Top feature importance: {selected_name}")
    plt.xlabel("Importance (not causation)")
    plt.tight_layout()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIRECTORY / "feature_importance.png", dpi=150)
    plt.close()
    print("Top features:\n", importance.to_string())


def run_training(data_directory: Path = DATA_DIRECTORY) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, split_metadata = load_prepared_data(data_directory)
    problem_type = identify_problem_type(y_train)
    X_train, X_test, numeric_columns, categorical_columns = prepare_features(X_train, X_test)
    imbalanced = len(y_train) > 0 and y_train.value_counts(normalize=True).min() < 0.2
    if imbalanced:
        print("Warning: significant training class imbalance detected; balanced class weights will be used where supported.")
    models = build_models(numeric_columns, categorical_columns, imbalanced)
    binary = problem_type == "binary classification"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    comparison, details = evaluate_models(models, X_train, X_test, y_train, y_test, binary)
    comparison.to_csv(OUTPUT_DIRECTORY / "model_comparison.csv", index=False)
    save_confusion_matrices(details, y_test)
    save_roc_curve(details, y_test, binary)
    selected_name, selection_metric = select_model(comparison)
    selected_detail = details[selected_name]
    selected_model = selected_detail["model"]
    save_feature_importance(selected_name, selected_model, X_test, y_test, list(X_train.columns))
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / "flood_forecasting_model.pkl"
    joblib.dump(selected_model, model_path)
    reloaded = joblib.load(model_path)
    reload_predictions = reloaded.predict(X_test)
    metadata = {
        "model_name": selected_name,
        "problem_type": problem_type,
        "selection_metric": selection_metric,
        "target_column": split_metadata.get("target_column", "unknown"),
        "number_of_features": len(X_train.columns),
        "feature_names": list(X_train.columns),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metrics": selected_detail["metrics"],
        "all_model_metrics": comparison.to_dict(orient="records"),
        "forecast_horizon": split_metadata.get("forecast_horizon"),
        "training_period": {key: split_metadata.get(key) for key in ("training_start", "training_end")},
        "testing_period": {key: split_metadata.get(key) for key in ("testing_start", "testing_end")},
    }
    (MODEL_DIRECTORY / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("\n## MODEL COMPARISON\n", comparison.to_string(index=False))
    print(f"\nSelected model: {selected_name}\nSelection metric: {selection_metric}\nReason: flood misses are costly, so recall is prioritized without hiding other metrics.")
    print(f"Recall: {selected_detail['metrics']['Recall']}\nPrecision: {selected_detail['metrics']['Precision']}\nF1 Score: {selected_detail['metrics']['F1 Score']}\nAccuracy: {selected_detail['metrics']['Accuracy']}\nROC-AUC: {selected_detail['metrics']['ROC-AUC']}")
    print(f"Model save: {'PASS' if model_path.is_file() else 'FAIL'}\nModel reload: {'PASS' if reloaded is not None else 'FAIL'}\nPrediction test: {'PASS' if len(reload_predictions) == len(y_test) else 'FAIL'}")
    print("\nML Model Training and Evaluation completed successfully.")
    print("Next step: Future Flood Prediction using the saved model.")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", default=str(DATA_DIRECTORY), help="Directory containing prepared split CSVs")
    args = parser.parse_args()
    try:
        run_training(Path(args.data_directory))
    except (TrainingError, FileNotFoundError, OSError, ValueError, ImportError) as error:
        print(f"Model training stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())