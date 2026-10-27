"""Schema-aware exploratory analysis for the flood forecasting dataset.

Run from the project root with ``python backend/eda.py``. The script never
changes the source dataset; plots and a JSON summary are written to
``backend/eda_outputs``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "eda_outputs"
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "dist", "__pycache__"}


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _name_matches(column: str, keywords: Iterable[str]) -> bool:
    name = _normalise_name(column)
    return any(keyword in name for keyword in keywords)


def discover_dataset(project_root: Path = PROJECT_ROOT, requested_file: str | None = None) -> Path:
    """Find a supported tabular dataset without assuming its filename."""

    if requested_file:
        requested = Path(requested_file)
        candidates = [requested if requested.is_absolute() else project_root / requested]
    else:
        candidates = [
            path
            for path in project_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".basin"}
            if not any(part.lower() in EXCLUDED_DIRECTORIES for part in path.parts)
        ]
        candidates.sort(
            key=lambda path: (
                0 if path.parent == project_root / "data" / "raw" else 1,
                0 if path.parent == project_root / "data" / "processed" else 1,
                len(path.parts),
                str(path).lower(),
            )
        )

    existing = [path for path in candidates if path.is_file()]
    if not existing:
        searched = project_root / "data" / "raw" if not requested_file else candidates[0]
        raise FileNotFoundError(f"No CSV dataset found. Checked: {searched}")
    if len(existing) > 1 and not requested_file:
        print(f"Multiple CSV files found; using the highest-priority file: {existing[0]}")
        print("Use --file to select another dataset explicitly.")
    return existing[0]


def load_dataset(file_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load CSV or tab-delimited .basin data with encoding fallbacks."""

    dataset_path = discover_dataset(PROJECT_ROOT, file_path) if file_path else discover_dataset()
    encoding_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            separator = "\t" if dataset_path.suffix.lower() == ".basin" else ","
            dataset = pd.read_csv(dataset_path, sep=separator, encoding=encoding)
            print(f"Dataset file: {dataset_path}")
            print(f"Delimiter: {'tab' if separator == chr(9) else 'comma'}")
            print(f"CSV encoding used: {encoding}")
            return dataset, dataset_path
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            encoding_errors.append(f"{encoding}: {error}")
    raise ValueError("Could not read the CSV with supported encodings. " + " | ".join(encoding_errors))


def basic_information(dataset: pd.DataFrame) -> dict[str, Any]:
    """Print and return the basic shape, schema, memory, and summary details."""

    print("\n## BASIC DATASET INFORMATION")
    print(f"Rows: {len(dataset):,}\nColumns: {len(dataset.columns):,}")
    print("\nFirst 5 rows:\n", dataset.head().to_string())
    print("\nLast 5 rows:\n", dataset.tail().to_string())
    print("\nColumn names:\n", list(dataset.columns))
    print("\nData types:\n", dataset.dtypes.to_string())
    print(f"\nMemory usage: {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nStatistical summary:\n", dataset.describe(include="all").to_string())
    return {
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "column_names": list(dataset.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
        "memory_mb": round(float(dataset.memory_usage(deep=True).sum() / 1024**2), 3),
    }


def _numeric_view(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking object columns for analysis without altering the source frame."""

    converted = dataset.copy()
    for column in converted.columns:
        if not pd.api.types.is_numeric_dtype(converted[column]):
            numeric = pd.to_numeric(converted[column], errors="coerce")
            if numeric.notna().mean() >= 0.8:
                converted[column] = numeric
    return converted


def check_data_quality(dataset: pd.DataFrame) -> dict[str, Any]:
    """Report quality concerns without dropping, imputing, or overwriting data."""

    numeric_dataset = _numeric_view(dataset)
    missing = dataset.isna().sum()
    duplicate_rows = int(dataset.duplicated().sum())
    infinite_values = int(np.isinf(numeric_dataset.select_dtypes(include=np.number)).sum().sum())
    invalid_numerical = {
        column: int(pd.to_numeric(dataset[column], errors="coerce").isna().sum())
        for column in dataset.columns
        if not pd.api.types.is_numeric_dtype(dataset[column])
        and pd.to_numeric(dataset[column], errors="coerce").notna().sum() > 0
        and pd.to_numeric(dataset[column], errors="coerce").isna().sum() > 0
    }
    constant_columns = [column for column in dataset.columns if dataset[column].nunique(dropna=False) <= 1]
    print("\n## DATA QUALITY CHECK")
    print("Missing values:\n", missing[missing > 0].to_string() or "None")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print("Unique values:\n", dataset.nunique(dropna=True).to_string())
    print(f"Infinite values: {infinite_values:,}")
    print("Invalid numerical values:\n", invalid_numerical or "None detected")
    print("Constant columns:", constant_columns or "None")
    print("No data was deleted. Review these findings before cleaning the original source.")
    return {
        "missing_values": {column: int(count) for column, count in missing.items() if count},
        "duplicate_rows": duplicate_rows,
        "unique_values": {column: int(count) for column, count in dataset.nunique(dropna=True).items()},
        "infinite_values": infinite_values,
        "invalid_numerical_values": invalid_numerical,
        "constant_columns": constant_columns,
    }


def detect_column_types(dataset: pd.DataFrame) -> dict[str, Any]:
    """Classify columns using dtypes, names, and parseability rather than fixed names."""

    numeric_dataset = _numeric_view(dataset)
    numerical = list(numeric_dataset.select_dtypes(include=np.number).columns)
    datetime_columns: list[str] = []
    for column in dataset.columns:
        parsed = pd.to_datetime(dataset[column], errors="coerce", format="mixed", dayfirst=True)
        has_datetime_name = _name_matches(column, ("date", "time", "timestamp", "observed", "recorded", "period", "year", "month"))
        is_string_like = pd.api.types.is_object_dtype(dataset[column]) or pd.api.types.is_string_dtype(dataset[column])
        if parsed.notna().mean() >= 0.8 and (has_datetime_name or is_string_like):
            datetime_columns.append(column)
    categorical = [column for column in dataset.columns if column not in numerical and column not in datetime_columns]
    target_candidates = [column for column in dataset.columns if _name_matches(column, ("flood", "target", "label", "class", "risk", "event", "alert"))]
    weather = [column for column in dataset.columns if _name_matches(column, ("weather", "temperature", "temp", "humidity", "pressure", "wind", "solar", "evap", "cloud"))]
    rainfall = [column for column in dataset.columns if _name_matches(column, ("rain", "precip", "precipitation", "storm"))]
    river_level = [column for column in dataset.columns if _name_matches(column, ("river", "water", "level", "discharge", "stream", "flow", "gauge", "depth"))]
    result = {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_columns,
        "possible_target": target_candidates,
        "weather": weather,
        "rainfall": rainfall,
        "river_water_level": river_level,
    }
    print("\n## DETECTED COLUMN TYPES")
    for label, columns in result.items():
        print(f"{label.replace('_', ' ').title()}: {columns or 'None detected'}")
    return result


def analyze_datetime(dataset: pd.DataFrame, column_types: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Safely parse the first detected date column and return a chronologically sorted copy."""

    datetime_columns = column_types["datetime"]
    if not datetime_columns:
        print("\n## DATE/TIME ANALYSIS\nNo date/time column detected.")
        return dataset.copy(), {"column": None, "date_range": None}
    date_column = datetime_columns[0]
    analyzed = dataset.copy()
    parsed = pd.to_datetime(analyzed[date_column], errors="coerce", format="mixed", dayfirst=True)
    analyzed[date_column] = parsed
    analyzed = analyzed.sort_values(date_column, na_position="last")
    valid_dates = parsed.dropna()
    date_range = [valid_dates.min().isoformat(), valid_dates.max().isoformat()] if not valid_dates.empty else None
    print("\n## DATE/TIME ANALYSIS")
    print(f"Date column: {date_column}\nMinimum date: {date_range[0] if date_range else 'None'}\nMaximum date: {date_range[1] if date_range else 'None'}")
    if not valid_dates.empty:
        print("Records by year:\n", valid_dates.dt.year.value_counts().sort_index().to_string())
        print("Records by month:\n", valid_dates.dt.month.value_counts().sort_index().to_string())
    return analyzed, {"column": date_column, "date_range": date_range}


def _save_figure(name: str) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / name, dpi=150, bbox_inches="tight")
    plt.close()


def create_visualizations(dataset: pd.DataFrame, column_types: dict[str, Any], datetime_info: dict[str, Any]) -> list[str]:
    """Create only plots supported by the detected schema."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    numeric_dataset = _numeric_view(dataset)
    saved: list[str] = []
    numerical = column_types["numerical"]
    if numerical:
        numeric_dataset[numerical].hist(figsize=(14, max(5, 3 * ((len(numerical) + 3) // 4)),), bins=30)
        _save_figure("feature_distributions.png")
        saved.append("feature_distributions.png")
        plt.figure(figsize=(max(8, len(numerical) * 0.8), 6))
        sns.boxplot(data=numeric_dataset[numerical], orient="h")
        _save_figure("feature_boxplots.png")
        saved.append("feature_boxplots.png")

    date_column = datetime_info["column"]
    if date_column:
        for columns, filename, title in ((column_types["rainfall"], "rainfall_over_time.png", "Rainfall over time"), (column_types["river_water_level"], "river_level_over_time.png", "River/water level over time")):
            available = [column for column in columns if column in numeric_dataset]
            if available:
                axis = numeric_dataset.sort_values(date_column).plot(x=date_column, y=available, figsize=(12, 5), title=title)
                axis.set_ylabel("Value")
                _save_figure(filename)
                saved.append(filename)

    rainfall = next((column for column in column_types["rainfall"] if column in numeric_dataset), None)
    river = next((column for column in column_types["river_water_level"] if column in numeric_dataset), None)
    target = next((column for column in column_types["possible_target"] if column in numeric_dataset), None)
    if rainfall and river:
        sns.scatterplot(data=numeric_dataset, x=rainfall, y=river)
        plt.title(f"{rainfall} vs {river}")
        _save_figure("rainfall_vs_river_level.png")
        saved.append("rainfall_vs_river_level.png")
    if target:
        for feature, filename in ((rainfall, "rainfall_vs_flood_target.png"), (river, "river_level_vs_flood_target.png")):
            if feature:
                sns.scatterplot(data=numeric_dataset, x=feature, y=target)
                plt.title(f"{feature} vs {target}")
                _save_figure(filename)
                saved.append(filename)
    return saved


def analyze_target(dataset: pd.DataFrame, column_types: dict[str, Any]) -> dict[str, Any]:
    """Summarize the first likely flood target without assuming binary encoding."""

    target = next(iter(column_types["possible_target"]), None)
    if not target:
        print("\n## TARGET ANALYSIS\nNo likely target column detected.")
        return {"column": None, "class_distribution": {}}
    counts = dataset[target].value_counts(dropna=False)
    percentages = (counts / len(dataset) * 100).round(2) if len(dataset) else counts
    print(f"\n## TARGET ANALYSIS\nPossible target: {target}\nCounts:\n{counts.to_string()}\nClass percentages:\n{percentages.to_string()}")
    numeric_dataset = _numeric_view(dataset)
    numeric_features = [column for column in column_types["numerical"] if column != target]
    feature_means = numeric_dataset.groupby(target, dropna=False)[numeric_features].mean().round(3) if numeric_features else pd.DataFrame()
    print("\nAverage numeric features by target class:\n", feature_means.to_string() if not feature_means.empty else "No numeric features available")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=dataset, x=target, order=counts.index.astype(str))
    plt.title(f"Target distribution: {target}")
    plt.xticks(rotation=30, ha="right")
    _save_figure("flood_distribution.png")
    return {
        "column": target,
        "class_distribution": {str(key): int(value) for key, value in counts.items()},
        "class_percentages": {str(key): float(value) for key, value in percentages.items()},
        "feature_means_by_class": feature_means.to_dict() if not feature_means.empty else {},
    }


def correlation_analysis(dataset: pd.DataFrame, column_types: dict[str, Any]) -> dict[str, Any]:
    """Create a numerical correlation heatmap and report strong associations."""

    numeric_dataset = _numeric_view(dataset)
    numerical = [column for column in column_types["numerical"] if column in numeric_dataset]
    if len(numerical) < 2:
        print("\n## CORRELATION ANALYSIS\nAt least two numerical columns are required.")
        return {"high_correlations": []}
    correlation = numeric_dataset[numerical].corr()
    plt.figure(figsize=(max(8, len(numerical) * 0.8), max(6, len(numerical) * 0.6)))
    sns.heatmap(correlation, annot=len(numerical) <= 12, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Numerical feature correlation")
    _save_figure("correlation_heatmap.png")
    high_correlations = []
    for index, first in enumerate(numerical):
        for second in numerical[index + 1:]:
            value = correlation.loc[first, second]
            if pd.notna(value) and abs(value) >= 0.7:
                high_correlations.append({"feature_1": first, "feature_2": second, "correlation": round(float(value), 3)})
    print("\n## CORRELATION ANALYSIS\nHighly correlated features (absolute correlation >= 0.70):", high_correlations or "None detected")
    return {"high_correlations": high_correlations}


def check_outliers(dataset: pd.DataFrame, column_types: dict[str, Any]) -> dict[str, Any]:
    """Report IQR outlier counts; observations remain untouched."""

    numeric_dataset = _numeric_view(dataset)
    outliers: dict[str, int] = {}
    for column in column_types["numerical"]:
        values = numeric_dataset[column].dropna()
        if values.empty:
            continue
        first_quartile, third_quartile = values.quantile([0.25, 0.75])
        iqr = third_quartile - first_quartile
        outliers[column] = int(((values < first_quartile - 1.5 * iqr) | (values > third_quartile + 1.5 * iqr)).sum()) if iqr else 0
    print("\n## OUTLIER ANALYSIS\nPotential IQR outliers by column:\n", outliers or "None")
    return outliers


def check_data_leakage(dataset: pd.DataFrame, column_types: dict[str, Any], datetime_info: dict[str, Any]) -> list[str]:
    """Flag naming and timestamp risks that must be resolved before forecasting."""

    target = next(iter(column_types["possible_target"]), None)
    warnings: list[str] = []
    future_named = [column for column in dataset.columns if _name_matches(column, ("future", "next", "lead", "forecast", "tomorrow", "target"))]
    if future_named:
        warnings.append(f"Review future-looking or target-derived columns: {future_named}")
    if target and datetime_info["column"]:
        warnings.append(f"Confirm that '{target}' describes a future horizon, not the same timestamp as the input features.")
    if not datetime_info["column"]:
        warnings.append("No timestamp was detected; a chronological forecasting split cannot be verified.")
    print("\n## DATA LEAKAGE CHECK\n", "\n".join(f"- {warning}" for warning in warnings) or "No obvious naming risks detected; inspect feature provenance.")
    return warnings


def generate_summary(dataset: pd.DataFrame, quality: dict[str, Any], column_types: dict[str, Any], datetime_info: dict[str, Any], target_info: dict[str, Any], correlations: dict[str, Any], outliers: dict[str, Any], leakage: list[str]) -> dict[str, Any]:
    """Print and return the concise EDA summary and forecasting recommendation."""

    features = [column for column in column_types["numerical"] + column_types["categorical"] if column != target_info["column"]]
    recommendation = (f"Use measured weather/rainfall/river features as inputs and '{target_info['column']}' as the target; verify that it represents a future horizon. Create shifted lag features and past-only rolling windows next, then use a chronological train/test split." if target_info["column"] else "Confirm the flood target and forecast horizon from domain metadata before feature engineering.")
    summary = {
        "dataset_size": f"{len(dataset):,} rows x {len(dataset.columns):,} columns",
        "date_range": datetime_info["date_range"],
        "missing_values": quality["missing_values"],
        "duplicate_rows": quality["duplicate_rows"],
        "numerical_features": column_types["numerical"],
        "categorical_features": column_types["categorical"],
        "datetime_features": column_types["datetime"],
        "possible_target": target_info["column"],
        "class_distribution": target_info["class_distribution"],
        "important_correlations": correlations["high_correlations"],
        "potential_outliers": outliers,
        "potential_data_leakage": leakage,
        "recommended_features": features,
        "recommended_next_step": recommendation,
        "forecasting_horizon": "Infer from timestamp frequency and target definition; do not assume one without metadata.",
        "recommended_lag_features": "Past rainfall, river level/discharge, weather lags and shifted rolling mean/max/sum windows.",
    }
    print("\n## EDA SUMMARY")
    for label, value in summary.items():
        print(f"{label.replace('_', ' ').title()}: {value}")
    return summary


def run_eda(file_path: str | Path | None = None) -> dict[str, Any]:
    """Run the complete EDA workflow and save its JSON summary."""

    dataset, source_path = load_dataset(file_path)
    information = basic_information(dataset)
    quality = check_data_quality(dataset)
    column_types = detect_column_types(dataset)
    analyzed_dataset, datetime_info = analyze_datetime(dataset, column_types)
    saved_plots = create_visualizations(analyzed_dataset, column_types, datetime_info)
    target_info = analyze_target(analyzed_dataset, column_types)
    correlations = correlation_analysis(analyzed_dataset, column_types)
    outliers = check_outliers(analyzed_dataset, column_types)
    leakage = check_data_leakage(analyzed_dataset, column_types, datetime_info)
    summary = generate_summary(analyzed_dataset, quality, column_types, datetime_info, target_info, correlations, outliers, leakage)
    summary.update({"source_file": str(source_path), "plots": saved_plots, "basic_information": information})
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nEDA completed successfully. Next step: Feature Engineering and Time-Based Train/Test Split.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Optional CSV path, relative to the project root")
    args = parser.parse_args()
    try:
        run_eda(args.file)
    except (FileNotFoundError, OSError, ValueError, ImportError) as error:
        print(f"EDA could not be completed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())