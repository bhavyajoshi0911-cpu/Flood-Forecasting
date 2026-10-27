"""Align rainfall, river level, and discharge observations by station and day.

The raw .basin files remain unchanged. This module creates a daily merged
measurement table, but deliberately does not create a flood label: that label
requires authoritative event evidence or an explicitly documented assumption.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_flood_dataset.csv"


def load_measurements(filename: str, value_column: str) -> pd.DataFrame:
    path = RAW_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    data = pd.read_csv(path, sep="\t")
    required = {"Station", "Data Acquisition Time", value_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(
        data["Data Acquisition Time"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=["Station", "timestamp", value_column]).copy()
    data["date"] = data["timestamp"].dt.floor("D")
    return data[["Station", "date", value_column]]


def merge_measurements() -> pd.DataFrame:
    rainfall = load_measurements("rainfall.basin", "Telemetry Hourly Rainfall (mm)")
    river = load_measurements("river.basin", "River Water Level Telemetry Hourly (meter)")
    discharge = load_measurements("riverdischarge.basin", "Manual Daily River Water Discharge (m3/sec)")

    rainfall_daily = rainfall.groupby(["Station", "date"], as_index=False).agg(
        rainfall_daily_mm=("Telemetry Hourly Rainfall (mm)", "sum")
    )
    river_daily = river.groupby(["Station", "date"], as_index=False).agg(
        river_level_mean_m=("River Water Level Telemetry Hourly (meter)", "mean"),
        river_level_max_m=("River Water Level Telemetry Hourly (meter)", "max"),
    )
    discharge_daily = discharge.groupby(["Station", "date"], as_index=False).agg(
        discharge_mean_m3_sec=("Manual Daily River Water Discharge (m3/sec)", "mean")
    )

    merged = rainfall_daily.merge(river_daily, on=["Station", "date"], how="inner")
    merged = merged.merge(discharge_daily, on=["Station", "date"], how="inner")
    return merged.sort_values(["Station", "date"]).reset_index(drop=True)


def main() -> int:
    merged = merge_measurements()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset saved: {OUTPUT_PATH}")
    print(f"Shape: {merged.shape}")
    print(f"Stations: {merged['Station'].nunique()}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print("Columns:", list(merged.columns))
    print("Missing values:\n", merged.isna().sum().to_string())
    print("No flood target was created because no authoritative flood-event evidence or documented threshold is configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())