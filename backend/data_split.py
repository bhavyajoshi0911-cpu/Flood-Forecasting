"""Prepare leakage-safe chronological train/test data for model training.

This module does not train a model. It reads the engineered dataset, detects
its actual timestamp and target columns, validates temporal separation, and
writes split CSVs under ``backend/data``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "backend" / "data" / "engineered_flood_dataset.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data"
SUMMARY_PATH = OUTPUT_DIRECTORY / "split_summary.json"


class DataSplitError(ValueError):
    """Raised when an engineered dataset cannot be split safely."""


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _matches(column: str, keywords: Iterable[str]) -> bool:
    name = _normalise_name(column)
    return any(keyword in name for keyword in keywords)


def load_engineered_data(input_path: str | Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Load the engineered CSV and its optional feature-engineering summary."""

    path = Path(input_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Engineered dataset not found: {path}. Run feature engineering after defining a valid flood target."
        )
    dataset = pd.read_csv(path)
    summary_path = path.with_name("feature_engineering_summary.json")
    metadata: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            metadata = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DataSplitError(f"Invalid feature-engineering metadata: {summary_path}") from error
    print(f"Dataset shape: {dataset.shape}")
    print("Column names:", list(dataset.columns))
    print("Data types:\n", dataset.dtypes.to_string())
    print("Missing values:\n", dataset.isna().sum().to_string())
    return dataset, path, metadata


def detect_datetime_column(dataset: pd.DataFrame, metadata: dict[str, Any] | None = None) -> str:
    """Use recorded metadata first, then choose the most parseable timestamp column."""

    metadata_column = (metadata or {}).get("datetime_column")
    if metadata_column in dataset.columns:
        return str(metadata_column)
    candidates: list[tuple[int, float, str]] = []
    for column in dataset.columns:
        parsed = pd.to_datetime(dataset[column], errors="coerce", format="mixed", dayfirst=True)
        rate = float(parsed.notna().mean())
        if rate >= 0.8 and (_matches(column, ("date", "time", "timestamp", "observed", "recorded", "period")) or pd.api.types.is_object_dtype(dataset[column])):
            candidates.append((int(_matches(column, ("date", "time", "timestamp"))), rate, column))
    if not candidates:
        raise DataSplitError("No valid datetime column was detected in the engineered dataset.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def detect_target_column(dataset: pd.DataFrame, metadata: dict[str, Any] | None = None) -> str:
    """Use feature-engineering metadata or detect a future target by its name."""

    metadata_target = (metadata or {}).get("future_target")
    if metadata_target in dataset.columns:
        return str(metadata_target)
    candidates = [column for column in dataset.columns if _matches(column, ("future", "flood_target", "target", "label", "class", "risk"))]
    candidates = [column for column in candidates if not _matches(column, ("horizon", "lag", "rolling", "change"))]
    if not candidates:
        raise DataSplitError("No future flood target was detected in the engineered dataset.")
    return candidates[0]


def prepare_data(dataset: pd.DataFrame, datetime_column: str, target_column: str) -> tuple[pd.DataFrame, int]:
    """Parse time, remove unusable rows, and preserve chronological order."""

    before = len(dataset)
    prepared = dataset.copy()
    prepared[datetime_column] = pd.to_datetime(prepared[datetime_column], errors="coerce", format="mixed", dayfirst=True)
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(subset=[datetime_column, target_column]).sort_values(datetime_column).reset_index(drop=True)
    removed = before - len(prepared)
    print(f"Rows before preprocessing: {before:,}\nRows after preprocessing: {len(prepared):,}\nRows removed: {removed:,}\nReason for removal: invalid timestamps, infinite values, or missing target values.")
    if prepared.empty:
        raise DataSplitError("No usable rows remain after preprocessing.")
    return prepared, removed


def separate_features_target(dataset: pd.DataFrame, datetime_column: str, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build X/y while excluding timestamps and target-derived/future columns."""

    excluded = {datetime_column, target_column, "__parsed_datetime"}
    leakage_names = ("future", "lead", "forecast", "target", "label", "flood", "risk")
    feature_columns = [
        column for column in dataset.columns
        if column not in excluded and not _matches(column, leakage_names)
    ]
    if not feature_columns:
        raise DataSplitError("No non-target input features remain after leakage filtering.")
    return dataset[feature_columns].copy(), dataset[target_column].copy(), feature_columns


def create_time_split(dataset: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split oldest observations for training and newest observations for testing."""

    if not 0 < train_fraction < 1:
        raise DataSplitError("train_fraction must be between 0 and 1")
    split_index = int(len(dataset) * train_fraction)
    if split_index < 1 or split_index >= len(dataset):
        raise DataSplitError("The dataset must contain enough rows for both splits.")
    return dataset.iloc[:split_index].copy(), dataset.iloc[split_index:].copy()


def check_target_distribution(target: pd.Series, name: str) -> dict[str, Any]:
    counts = target.value_counts(dropna=False)
    percentages = (counts / len(target) * 100).round(2)
    distribution = {str(key): {"count": int(value), "percentage": float(percentages[key])} for key, value in counts.items()}
    print(f"{name} target distribution:\n{pd.DataFrame(distribution).T.to_string()}")
    if len(counts) == 2 and len(target) > 0 and float(counts.min() / counts.max()) < 0.2:
        print(f"Warning: significant class imbalance detected in {name}.")
    return distribution


def check_data_leakage(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_column: str, datetime_column: str) -> dict[str, str]:
    target_in_features = target_column in feature_columns
    future_features = [column for column in feature_columns if _matches(column, ("future", "lead", "forecast", "target", "flood", "risk", "label"))]
    temporal_overlap = not (train[datetime_column].max() < test[datetime_column].min())
    result = {
        "target_in_features": "FAIL" if target_in_features else "PASS",
        "future_information_in_features": "FAIL" if future_features else "PASS",
        "train_test_overlap": "FAIL" if temporal_overlap else "PASS",
        "chronological_order": "PASS" if train[datetime_column].is_monotonic_increasing and test[datetime_column].is_monotonic_increasing else "FAIL",
        "random_shuffle": "PASS",
    }
    print("\n## DATA LEAKAGE CHECK")
    for check, status in result.items():
        print(f"{check.replace('_', ' ').title()}: {status}")
    return result


def save_split_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, metadata: dict[str, Any], output_directory: Path = OUTPUT_DIRECTORY) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_directory / "X_train.csv", index=False)
    X_test.to_csv(output_directory / "X_test.csv", index=False)
    y_train.rename(metadata["target_column"]).to_csv(output_directory / "y_train.csv", index=False)
    y_test.rename(metadata["target_column"]).to_csv(output_directory / "y_test.csv", index=False)
    (output_directory / "split_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def generate_split_summary(dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_column: str, datetime_column: str, leakage: dict[str, str], removed: int) -> dict[str, Any]:
    summary = {
        "original_rows": len(dataset),
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_percentage": round(len(train) / len(dataset) * 100, 2),
        "testing_percentage": round(len(test) / len(dataset) * 100, 2),
        "training_start": train[datetime_column].min().isoformat(),
        "training_end": train[datetime_column].max().isoformat(),
        "testing_start": test[datetime_column].min().isoformat(),
        "testing_end": test[datetime_column].max().isoformat(),
        "number_of_features": len(feature_columns),
        "target_column": target_column,
        "datetime_column": datetime_column,
        "rows_removed": removed,
        "data_leakage": leakage,
        "temporal_separation": leakage["train_test_overlap"] == "PASS",
    }
    print("\n## TIME-BASED SPLIT SUMMARY")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return summary


def run_split(input_path: str | Path = DEFAULT_INPUT) -> dict[str, Any]:
    dataset, source_path, metadata = load_engineered_data(input_path)
    datetime_column = detect_datetime_column(dataset, metadata)
    target_column = detect_target_column(dataset, metadata)
    print(f"Target column: {target_column}\nDate/time column: {datetime_column}")
    prepared, removed = prepare_data(dataset, datetime_column, target_column)
    train_rows, test_rows = create_time_split(prepared)
    X, y, feature_columns = separate_features_target(prepared, datetime_column, target_column)
    X_train = X.iloc[: len(train_rows)].copy()
    X_test = X.iloc[len(train_rows):].copy()
    y_train = y.iloc[: len(train_rows)].copy()
    y_test = y.iloc[len(train_rows):].copy()
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise DataSplitError("Feature and target split lengths do not match.")
    print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")
    print(f"\nTRAINING PERIOD\nStart: {train_rows[datetime_column].min()}\nEnd: {train_rows[datetime_column].max()}\n\nTESTING PERIOD\nStart: {test_rows[datetime_column].min()}\nEnd: {test_rows[datetime_column].max()}")
    leakage = check_data_leakage(train_rows, test_rows, feature_columns, target_column, datetime_column)
    if any(status == "FAIL" for status in leakage.values()):
        raise DataSplitError("Data leakage checks failed; split artifacts were not saved.")
    train_distribution = check_target_distribution(y_train, "Training")
    test_distribution = check_target_distribution(y_test, "Testing")
    summary = generate_split_summary(dataset, train_rows, test_rows, feature_columns, target_column, datetime_column, leakage, removed)
    summary.update({"source_file": str(source_path), "train_target_distribution": train_distribution, "test_target_distribution": test_distribution})
    save_split_data(X_train, X_test, y_train, y_test, {**summary, "target_column": target_column, "datetime_column": datetime_column})
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nTime-Based Train/Test Split completed successfully.")
    print("Next step: ML Model Training")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_INPUT), help="Engineered CSV path")
    args = parser.parse_args()
    try:
        run_split(args.file)
    except (DataSplitError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Time-based split stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())