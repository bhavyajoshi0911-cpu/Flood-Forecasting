"""Explicit, auditable next-day flood target construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable


class TargetDefinitionError(ValueError):
    """Raised when target evidence is missing or not documented."""


@dataclass(frozen=True)
class TargetDefinition:
    """The evidence and rule used to create a binary target."""

    mode: str
    evidence_column: str
    threshold: float | None = None
    evidence_source: str = ""
    authoritative: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"label", "threshold"}:
            raise TargetDefinitionError("mode must be 'label' or 'threshold'")
        if not self.evidence_column:
            raise TargetDefinitionError("evidence_column is required")
        if not self.evidence_source.strip():
            raise TargetDefinitionError("evidence_source is required")
        if self.mode == "threshold" and self.threshold is None:
            raise TargetDefinitionError("threshold mode requires a threshold")
        if self.mode == "threshold" and not self.authoritative and "assumption" not in self.evidence_source.lower():
            raise TargetDefinitionError(
                "A non-authoritative threshold must be explicitly labelled as a research assumption"
            )


def _timestamp(row: dict[str, Any]) -> datetime:
    try:
        return datetime.fromisoformat(str(row["timestamp"]))
    except (KeyError, ValueError) as error:
        raise TargetDefinitionError("Rows must contain ISO-8601 timestamps") from error


def _binary_label(value: Any) -> int:
    if value in (0, "0", False):
        return 0
    if value in (1, "1", True):
        return 1
    raise TargetDefinitionError(f"Flood labels must be 0 or 1, received {value!r}")


def build_next_day_targets(
    rows: Iterable[dict[str, Any]],
    definition: TargetDefinition,
) -> list[dict[str, Any]]:
    """Attach the next calendar day's target and drop rows without target evidence.

    The returned row at time ``t`` uses only the evidence at ``t + 1`` for its
    target. Feature engineering must run separately and must not use this field.
    """

    ordered = [dict(row) for row in rows]
    by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    for row in ordered:
        station = str(row.get("station_id", "")).strip()
        if not station:
            raise TargetDefinitionError("station_id is required")
        timestamp = _timestamp(row)
        key = (station, timestamp)
        if key in by_key:
            raise TargetDefinitionError(f"Duplicate station-time row: {station} at {timestamp.isoformat()}")
        by_key[key] = row

    result: list[dict[str, Any]] = []
    for row in ordered:
        station = str(row["station_id"]).strip()
        timestamp = _timestamp(row)
        future = by_key.get((station, timestamp + timedelta(days=1)))
        if future is None or future.get(definition.evidence_column) in (None, ""):
            continue
        target_value = future[definition.evidence_column]
        if definition.mode == "label":
            target = _binary_label(target_value)
        else:
            try:
                target = int(float(target_value) >= float(definition.threshold))
            except (TypeError, ValueError) as error:
                raise TargetDefinitionError(
                    f"Target evidence column {definition.evidence_column!r} must be numeric"
                ) from error
        labelled = dict(row)
        labelled["target_next_day"] = target
        result.append(labelled)
    return sorted(result, key=lambda item: (str(item["station_id"]), str(item["timestamp"])))