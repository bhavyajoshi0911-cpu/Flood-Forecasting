"""Create an explicitly non-authoritative flood proxy for research demonstration.

This is not an official flood-event label. It marks a day as a research flood
event when that station's daily maximum river level is at or above the 95th
percentile calculated from the oldest 80 percent of dates. The threshold is
derived from the historical training period only, then the current-day label
can be shifted to a future target by feature engineering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_flood_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "merged_flood_dataset_with_target.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "research_target_definition.json"
VALUE_COLUMN = "river_level_max_m"
TARGET_COLUMN = "flood_observed_research"


def create_research_target() -> tuple[pd.DataFrame, dict[str, object]]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Merged dataset not found: {INPUT_PATH}")
    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = {"Station", "date", VALUE_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Merged dataset is missing required columns: {sorted(missing)}")
    data = data.sort_values(["date", "Station"]).reset_index(drop=True)
    cutoff_index = max(1, int(len(data) * 0.8))
    cutoff_date = data.iloc[cutoff_index - 1]["date"]
    training_period = data[data["date"] <= cutoff_date]
    thresholds = training_period.groupby("Station")[VALUE_COLUMN].quantile(0.95)
    data["station_threshold_m"] = data["Station"].map(thresholds)
    data[TARGET_COLUMN] = (data[VALUE_COLUMN] >= data["station_threshold_m"]).astype("int8")
    data = data.dropna(subset=["station_threshold_m"])
    metadata: dict[str, object] = {
        "mode": "threshold",
        "authoritative": False,
        "evidence_source": "Research assumption: station-specific 95th percentile of river_level_max_m",
        "evidence_column": VALUE_COLUMN,
        "target_column": TARGET_COLUMN,
        "class_mapping": {"0": "No Flood", "1": "Flood"},
        "threshold_rule": "station-specific 95th percentile calculated from oldest 80 percent of dates",
        "training_cutoff_date": cutoff_date.isoformat(),
        "thresholds_m": {str(key): float(value) for key, value in thresholds.items()},
        "warning": "This proxy is not an official flood-event label or emergency warning threshold.",
    }
    return data, metadata


def main() -> int:
    data, metadata = create_research_target()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Research-target dataset saved: {OUTPUT_PATH}")
    print(f"Rows: {len(data):,}; flood proxy rows: {int(data[TARGET_COLUMN].sum()):,}")
    print(f"Training cutoff used for thresholds: {metadata['training_cutoff_date']}")
    print(f"Target: {TARGET_COLUMN}")
    print("WARNING: This is a research assumption, not an authoritative flood label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())