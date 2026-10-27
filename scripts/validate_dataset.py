"""Validate the configured raw CSV and write a quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.data.loader import DatasetLoadError, load_metadata, load_raw_dataset, summarize_basin_files  # noqa: E402
from app.data.validator import empty_quality_report, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="CSV filename inside data/raw")
    parser.add_argument("--metadata", help="JSON metadata filename inside data/metadata")
    parser.add_argument("--output", default="data/metadata/data_quality_report.json")
    args = parser.parse_args()

    raw_directory = PROJECT_ROOT / "data" / "raw"
    metadata_directory = PROJECT_ROOT / "data" / "metadata"
    try:
        basin_report = summarize_basin_files(raw_directory) if args.file is None else None
        if basin_report is not None:
            report = basin_report
        else:
            dataset = load_raw_dataset(raw_directory, args.file)
            if dataset is None:
                report = empty_quality_report(str(raw_directory))
            else:
                metadata = load_metadata(metadata_directory / args.metadata) if args.metadata else None
                report = validate_dataset(dataset, metadata)
    except DatasetLoadError as error:
        report = {"status": "error", "path": str(raw_directory), "row_count": 0, "errors": [str(error)], "warnings": []}

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"valid", "warning", "not_configured"} else 1


if __name__ == "__main__":
    raise SystemExit(main())