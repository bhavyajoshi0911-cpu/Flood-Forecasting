"""Leakage-safe feature engineering for the flood forecasting datasets.

Run from the project root, for example:
    python backend\feature_engineering.py --file data\raw\river.basin

The source dataset is never overwritten. A final engineered file is written
only when a valid flood target is present.
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
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "engineered_flood_dataset.csv"
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "dist", "__pycache__"}


class FeatureEngineeringError(ValueError):
    """Raised when the data cannot support honest forecasting features."""


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _matches(column: str, keywords: Iterable[str]) -> bool:
    name = _normalise_name(column)
    return any(keyword in name for keyword in keywords)


def _discover_files(project_root: Path) -> list[Path]:
    return sorted(
        path
        for path in project_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".basin"}
        and not any(part.lower() in EXCLUDED_DIRECTORIES for part in path.parts)
    )


def load_dataset(file_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load one raw CSV or tab-delimited basin file without changing it."""

    if file_path:
        requested = Path(file_path)
        path = requested if requested.is_absolute() else PROJECT_ROOT / requested
        candidates = [path]
    else:
        candidates = _discover_files(PROJECT_ROOT)
        if len(candidates) > 1:
            names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in candidates)
            raise FeatureEngineeringError(
                f"Multiple datasets found: {names}. Select one with --file. "
                "Rainfall, water level, and discharge must be aligned before they can be merged."
            )
    if not candidates or not candidates[0].is_file():
        raise FileNotFoundError(f"Dataset file not found: {candidates[0] if candidates else 'data/raw'}")

    path = candidates[0]
    separator = "\t" if path.suffix.lower() == ".basin" else ","
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            dataset = pd.read_csv(path, sep=separator, encoding=encoding)
            print(f"Dataset file: {path}")
            print(f"Dataset shape: {dataset.shape}")
            print(f"Delimiter: {'tab' if separator == chr(9) else 'comma'}; encoding: {encoding}")
            print("Column names:", list(dataset.columns))
            print("Data types:\n", dataset.dtypes.to_string())
            return dataset, path
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            errors.append(f"{encoding}: {error}")
    raise FeatureEngineeringError("Could not read the dataset. " + " | ".join(errors))


def detect_datetime_column(dataset: pd.DataFrame) -> str | None:
    """Find the most plausible timestamp column from names and parseability."""

    candidates: list[tuple[float, int, str]] = []
    for column in dataset.columns:
        parsed = pd.to_datetime(dataset[column], errors="coerce", format="mixed", dayfirst=True)
        parse_rate = float(parsed.notna().mean())
        if parse_rate < 0.8:
            continue
        name_score = int(_matches(column, ("date", "time", "timestamp", "observed", "recorded", "period")))
        if name_score or pd.api.types.is_object_dtype(dataset[column]) or pd.api.types.is_string_dtype(dataset[column]):
            candidates.append((parse_rate, name_score, column))
    selected = max(candidates, default=(0.0, 0, ""), key=lambda item: (item[1], item[0]))[2] or None
    print(f"Datetime column: {selected or 'None detected'}")
    return selected


def prepare_datetime(dataset: pd.DataFrame, datetime_column: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse, sort, and describe timestamps while preserving the original column."""

    if not datetime_column:
        print("No valid timestamp is available; future features cannot be created.")
        return dataset.copy(), {"column": None, "start": None, "end": None, "frequency": None, "seconds": None}
    prepared = dataset.copy()
    parsed = pd.to_datetime(prepared[datetime_column], errors="coerce", format="mixed", dayfirst=True)
    prepared["__parsed_datetime"] = parsed
    prepared = prepared.dropna(subset=["__parsed_datetime"]).sort_values("__parsed_datetime").reset_index(drop=True)
    differences = prepared["__parsed_datetime"].diff().dropna().dt.total_seconds()
    positive_differences = differences[differences > 0]
    median_seconds = float(positive_differences.median()) if not positive_differences.empty else None
    frequency = _frequency_name(median_seconds)
    start = prepared["__parsed_datetime"].min().isoformat() if not prepared.empty else None
    end = prepared["__parsed_datetime"].max().isoformat() if not prepared.empty else None
    print(f"Start date: {start}\nEnd date: {end}\nDetected frequency: {frequency or 'unknown'}")
    return prepared, {"column": datetime_column, "start": start, "end": end, "frequency": frequency, "seconds": median_seconds}


def _frequency_name(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds <= 90 * 60:
        return "hourly"
    if seconds <= 36 * 60 * 60:
        return "daily"
    return "other"


def _group_columns(dataset: pd.DataFrame) -> list[str]:
    station = next((column for column in dataset.columns if _matches(column, ("station", "site", "gauge"))), None)
    return [station] if station else []


def detect_feature_columns(dataset: pd.DataFrame, datetime_column: str | None) -> dict[str, Any]:
    """Identify environmental measurements without assuming exact column names."""

    numeric = list(dataset.select_dtypes(include=np.number).columns)
    target_candidates = [column for column in dataset.columns if _matches(column, ("flood", "target", "label", "class", "risk", "event", "alert"))]
    excluded = {datetime_column, "__parsed_datetime", *target_candidates}
    identifier_keywords = ("slno", "code", "latitude", "longitude", "id")
    measured_numeric = [column for column in numeric if column not in excluded and not _matches(column, identifier_keywords)]
    rainfall = [column for column in measured_numeric if _matches(column, ("rain", "precip", "storm"))]
    water = [column for column in measured_numeric if _matches(column, ("river", "water", "level", "discharge", "stream", "flow", "gauge", "depth"))]
    weather = [column for column in measured_numeric if _matches(column, ("temperature", "temp", "humidity", "pressure", "wind", "solar", "evap", "cloud", "weather"))]
    environmental = list(dict.fromkeys(rainfall + water + weather))
    result = {
        "numerical": numeric,
        "categorical": [column for column in dataset.select_dtypes(exclude=np.number).columns if column not in {datetime_column, "__parsed_datetime"}],
        "target_candidates": target_candidates,
        "rainfall": rainfall,
        "water": water,
        "weather": weather,
        "environmental": environmental or measured_numeric,
        "group_columns": _group_columns(dataset),
    }
    print("Numerical columns:", result["numerical"])
    print("Categorical columns:", result["categorical"])
    print("Possible target columns:", target_candidates or "None detected")
    print("Detected rainfall columns:", rainfall or "None detected")
    print("Detected water/river columns:", water or "None detected")
    print("Detected weather columns:", weather or "None detected")
    return result


def _periods(frequency: str | None) -> tuple[list[int], list[int]]:
    if frequency == "hourly":
        return [1, 3, 6, 12, 24], [3, 6, 12, 24]
    if frequency == "daily":
        return [1, 3, 7, 14, 30], [3, 7, 14, 30]
    return [1, 2, 3], [3, 7]


def _grouped_shift(dataset: pd.DataFrame, column: str, periods: Iterable[int], groups: list[str]) -> pd.DataFrame:
    grouped = dataset.groupby(groups, sort=False)[column] if groups else None
    output = pd.DataFrame(index=dataset.index)
    for period in periods:
        output[f"{column}_lag_{period}"] = grouped.shift(period) if grouped is not None else dataset[column].shift(period)
    return output


def create_time_features(dataset: pd.DataFrame, datetime_info: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    if not datetime_info["column"]:
        return dataset, []
    timestamp = dataset["__parsed_datetime"]
    features = {"year": timestamp.dt.year, "month": timestamp.dt.month, "day_of_year": timestamp.dt.dayofyear, "day_of_week": timestamp.dt.dayofweek}
    if datetime_info["frequency"] == "hourly":
        features["hour"] = timestamp.dt.hour
    output = dataset.assign(**features)
    names = list(features)
    print("Time features:", names)
    return output, names


def create_lag_features(dataset: pd.DataFrame, feature_columns: Iterable[str], datetime_info: dict[str, Any], groups: list[str]) -> tuple[pd.DataFrame, list[str]]:
    lags, _ = _periods(datetime_info["frequency"])
    output = dataset.copy()
    names: list[str] = []
    for column in feature_columns:
        created = _grouped_shift(output, column, lags, groups)
        output = output.join(created)
        names.extend(created.columns)
    print(f"Lag features: {len(names)}")
    return output, names


def create_rolling_features(dataset: pd.DataFrame, feature_columns: Iterable[str], datetime_info: dict[str, Any], groups: list[str]) -> tuple[pd.DataFrame, list[str]]:
    _, windows = _periods(datetime_info["frequency"])
    output = dataset.copy()
    names: list[str] = []
    for column in feature_columns:
        if groups:
            shifted = output.groupby(groups, sort=False)[column].shift(1)
            rolling_groups = shifted.groupby([output[group] for group in groups], sort=False)
        else:
            rolling_groups = output[column].shift(1)
        for window in windows:
            name = f"{column}_rolling_{window}_mean"
            if groups:
                values = rolling_groups.rolling(window, min_periods=window).mean().reset_index(level=list(range(len(groups))), drop=True)
                output[name] = values.reindex(output.index)
            else:
                output[name] = rolling_groups.rolling(window, min_periods=window).mean()
            names.append(name)
        for window in windows:
            name = f"{column}_rolling_{window}_max"
            if groups:
                values = rolling_groups.rolling(window, min_periods=window).max().reset_index(level=list(range(len(groups))), drop=True)
                output[name] = values.reindex(output.index)
            else:
                output[name] = rolling_groups.rolling(window, min_periods=window).max()
            names.append(name)
    print(f"Rolling features: {len(names)}")
    return output, names


def create_change_features(dataset: pd.DataFrame, feature_columns: Iterable[str], groups: list[str]) -> tuple[pd.DataFrame, list[str]]:
    output = dataset.copy()
    names: list[str] = []
    for column in feature_columns:
        previous = output.groupby(groups, sort=False)[column].shift(1) if groups else output[column].shift(1)
        change_name = f"{column}_change"
        percent_name = f"{column}_pct_change"
        output[change_name] = output[column] - previous
        output[percent_name] = np.where(previous.abs() > 0, output[column].sub(previous).div(previous.abs()) * 100, np.nan)
        names.extend([change_name, percent_name])
    print(f"Change features: {len(names)}")
    return output, names


def create_future_target(dataset: pd.DataFrame, target_column: str | None, datetime_info: dict[str, Any], groups: list[str]) -> tuple[pd.DataFrame, str, int]:
    """Shift a verified current-time target into the future by an explicit horizon."""

    if not target_column:
        raise FeatureEngineeringError("No valid flood target was found. Add authoritative flood-event labels or document a defensible threshold before feature engineering.")
    horizon = 3 if datetime_info["frequency"] == "hourly" else 1
    future_name = f"future_{target_column}"
    output = dataset.copy()
    output[future_name] = output.groupby(groups, sort=False)[target_column].shift(-horizon) if groups else output[target_column].shift(-horizon)
    print(f"Original target: {target_column}\nForecast horizon: {horizon} {datetime_info['frequency'] or 'time steps'}\nFuture target: {future_name}\nTarget definition: target at the future forecast horizon; verify that the source target represents current-time status.")
    return output, future_name, horizon


def remove_invalid_rows(dataset: pd.DataFrame, required_columns: Iterable[str]) -> tuple[pd.DataFrame, int]:
    before = len(dataset)
    output = dataset.dropna(subset=list(required_columns)).copy()
    removed = before - len(output)
    print(f"Rows before engineering: {before:,}\nRows after engineering: {len(output):,}\nRows removed: {removed:,}\nReason: insufficient history, invalid timestamps, or unavailable future target.")
    return output, removed


def check_data_leakage(dataset: pd.DataFrame, feature_columns: Iterable[str], target_column: str, datetime_info: dict[str, Any]) -> dict[str, str]:
    features = list(feature_columns)
    future_feature_leakage = any(_matches(column, ("future", "lead", "forecast", "target")) for column in features)
    target_leakage = target_column in features or any(column == target_column for column in features)
    chronological = bool(datetime_info["column"]) and dataset["__parsed_datetime"].is_monotonic_increasing
    rolling_columns = [column for column in features if "rolling" in _normalise_name(column)]
    rolling_leakage = all("future" not in _normalise_name(column) and "lead" not in _normalise_name(column) for column in rolling_columns)
    result = {"future_feature_leakage": "FAIL" if future_feature_leakage else "PASS", "target_leakage": "FAIL" if target_leakage else "PASS", "chronological_order": "PASS" if chronological else "FAIL", "rolling_feature_leakage": "PASS" if rolling_leakage else "FAIL"}
    print("\n## DATA LEAKAGE CHECK")
    for label, status in result.items():
        print(f"{label.replace('_', ' ').title()}: {status}")
    return result


def save_engineered_dataset(dataset: pd.DataFrame, output_path: str | Path = OUTPUT_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.drop(columns=["__parsed_datetime"], errors="ignore").to_csv(path, index=False)
    print(f"Engineered dataset saved successfully: {path}")
    return path


def generate_feature_summary(dataset: pd.DataFrame, datetime_info: dict[str, Any], feature_groups: dict[str, list[str]], target_name: str, horizon: int, removed: int) -> dict[str, Any]:
    summary = {"dataset_frequency": datetime_info["frequency"], "datetime_column": datetime_info["column"], "original_features": feature_groups["original"], "time_features": feature_groups["time"], "lag_features": feature_groups["lag"], "rolling_features": feature_groups["rolling"], "rainfall_accumulation_features": [name for name in feature_groups["rolling"] if any(token in name for token in ("rain", "precip"))], "river_water_level_features": [name for name in feature_groups["lag"] + feature_groups["rolling"] + feature_groups["change"] if any(token in _normalise_name(name) for token in ("river", "water", "level", "discharge", "flow"))], "future_target": target_name, "forecast_horizon": horizon, "final_number_of_features": len([column for column in dataset.columns if column not in {target_name, "__parsed_datetime"}]), "final_number_of_rows": len(dataset), "rows_removed": removed}
    print("\n## FEATURE ENGINEERING SUMMARY")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    return summary


def run_feature_engineering(file_path: str | Path | None = None) -> dict[str, Any]:
    dataset, source_path = load_dataset(file_path)
    datetime_column = detect_datetime_column(dataset)
    prepared, datetime_info = prepare_datetime(dataset, datetime_column)
    if not datetime_column:
        raise FeatureEngineeringError("A valid timestamp is required for future forecasting feature creation.")
    column_types = detect_feature_columns(prepared, datetime_column)
    target_column = next(iter(column_types["target_candidates"]), None)
    if not target_column:
        raise FeatureEngineeringError("No valid flood target exists in this dataset. The current raw file contains measurements only; no engineered training dataset was saved.")

    groups = column_types["group_columns"]
    features = column_types["environmental"]
    prepared, time_features = create_time_features(prepared, datetime_info)
    prepared, lag_features = create_lag_features(prepared, features, datetime_info, groups)
    prepared, rolling_features = create_rolling_features(prepared, features, datetime_info, groups)
    prepared, change_features = create_change_features(prepared, column_types["water"], groups)
    prepared, future_target, horizon = create_future_target(prepared, target_column, datetime_info, groups)
    required = [future_target] + lag_features + rolling_features + change_features
    prepared, removed = remove_invalid_rows(prepared, required)
    feature_names = [column for column in prepared.columns if column not in {target_column, future_target, "__parsed_datetime", datetime_column}]
    leakage = check_data_leakage(prepared, feature_names, target_column, datetime_info)
    if any(status == "FAIL" for status in leakage.values()):
        raise FeatureEngineeringError("Leakage checks failed; engineered data was not saved.")
    output_path = save_engineered_dataset(prepared)
    summary = generate_feature_summary(prepared, datetime_info, {"original": features, "time": time_features, "lag": lag_features, "rolling": rolling_features, "change": change_features}, future_target, horizon, removed)
    summary.update({"source_file": str(source_path), "output_file": str(output_path), "target_unique_values": prepared[future_target].unique().tolist(), "target_distribution": prepared[future_target].value_counts(dropna=False).to_dict()})
    print(f"\nNumber of features: {len(feature_names)}\nTarget name: {future_target}\nTarget unique values: {summary['target_unique_values']}\nTarget distribution: {summary['target_distribution']}")
    split_index = int(len(prepared) * 0.8)
    print(f"Training rows: {split_index:,}\nTesting rows: {len(prepared) - split_index:,}")
    print(f"Training start: {prepared['__parsed_datetime'].iloc[0]}\nTraining end: {prepared['__parsed_datetime'].iloc[split_index - 1]}\nTesting start: {prepared['__parsed_datetime'].iloc[split_index]}\nTesting end: {prepared['__parsed_datetime'].iloc[-1]}")
    summary_path = output_path.with_name("feature_engineering_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nFeature Engineering completed successfully.")
    print("Next step: Time-Based Train/Test Split -> ML Model Training -> Model Evaluation")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Dataset path relative to the project root")
    args = parser.parse_args()
    try:
        run_feature_engineering(args.file)
    except (FeatureEngineeringError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Feature engineering stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())