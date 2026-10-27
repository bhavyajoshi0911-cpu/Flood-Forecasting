"""Chronological train, validation, and test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


class SplitError(ValueError):
    """Raised when a chronological split cannot be constructed safely."""


@dataclass(frozen=True)
class DatasetSplit:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _parse_timestamp(row: dict[str, Any]) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
    except (KeyError, ValueError) as error:
        raise SplitError("Every row must contain an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise SplitError("Every timestamp must include a timezone")
    return timestamp


def _check_nonempty(split: DatasetSplit) -> DatasetSplit:
    sizes = {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)}
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        raise SplitError(f"Split would be empty: {', '.join(empty)}")
    return split


def split_chronologically(
    rows: Iterable[dict[str, Any]],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    train_end: str | None = None,
    validation_end: str | None = None,
) -> DatasetSplit:
    """Split rows without shuffling, using explicit dates or validated ratios.

    Date boundaries are exclusive: rows before ``train_end`` go to training,
    rows from ``train_end`` to ``validation_end`` go to validation, and later
    rows go to testing. Ratio splits are only a fallback until data coverage is
    known and must never be interpreted as calendar periods.
    """

    ordered = [(dict(row), _parse_timestamp(row)) for row in rows]
    ordered.sort(key=lambda item: item[1])
    if len(ordered) < 3:
        raise SplitError("At least three timestamped rows are required")

    if train_end is not None or validation_end is not None:
        if not train_end or not validation_end:
            raise SplitError("Both train_end and validation_end are required together")
        try:
            train_boundary = datetime.fromisoformat(train_end)
            validation_boundary = datetime.fromisoformat(validation_end)
        except ValueError as error:
            raise SplitError("Split boundaries must be ISO-8601 timestamps") from error
        if train_boundary.tzinfo is None or validation_boundary.tzinfo is None:
            raise SplitError("Split boundaries must include a timezone")
        if train_boundary >= validation_boundary:
            raise SplitError("train_end must be before validation_end")
        train = [row for row, timestamp in ordered if timestamp < train_boundary]
        validation = [row for row, timestamp in ordered if train_boundary <= timestamp < validation_boundary]
        test = [row for row, timestamp in ordered if timestamp >= validation_boundary]
    else:
        if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
            raise SplitError("Split ratios must be between 0 and 1")
        if train_ratio + validation_ratio >= 1:
            raise SplitError("Train and validation ratios must leave rows for testing")
        train_end_index = int(len(ordered) * train_ratio)
        validation_end_index = int(len(ordered) * (train_ratio + validation_ratio))
        train = [row for row, _ in ordered[:train_end_index]]
        validation = [row for row, _ in ordered[train_end_index:validation_end_index]]
        test = [row for row, _ in ordered[validation_end_index:]]

    return _check_nonempty(DatasetSplit(train, validation, test))