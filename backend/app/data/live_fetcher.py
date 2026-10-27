"""Provider-neutral live observation fetching and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LiveFetchResult:
    status: str
    source: str
    data_timestamp: str | None = None
    station_id: str | None = None
    observations: dict[str, float] | None = None
    message: str = ""


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def parse_live_payload(
    payload: dict[str, Any],
    required_features: list[str],
    field_mapping: dict[str, str],
    expected_station: str,
    max_age: timedelta,
    now: datetime | None = None,
) -> LiveFetchResult:
    """Normalize a provider payload only when its contract checks pass.

    ``field_mapping`` maps each model feature to a provider payload key. Units
    must be supplied in ``payload['units']`` for every mapped feature.
    """

    source = str(payload.get("source", "configured provider"))
    try:
        data_timestamp = _timestamp(payload["timestamp"])
        station_id = str(payload["station_id"])
        if station_id != expected_station:
            raise ValueError(f"station mismatch: expected {expected_station}, received {station_id}")
        if data_timestamp > (now or datetime.now(timezone.utc)) + timedelta(minutes=5):
            raise ValueError("timestamp is in the future")
        if (now or datetime.now(timezone.utc)) - data_timestamp > max_age:
            return LiveFetchResult("stale", source, data_timestamp.isoformat(), station_id, message="observation is too old")
        values = payload["observations"]
        units = payload["units"]
        if not isinstance(values, dict) or not isinstance(units, dict):
            raise ValueError("observations and units must be objects")
        observations: dict[str, float] = {}
        for feature in required_features:
            provider_field = field_mapping.get(feature)
            if not provider_field:
                raise ValueError(f"no provider mapping for feature {feature}")
            if provider_field not in values or values[provider_field] in (None, ""):
                raise ValueError(f"missing provider field {provider_field}")
            if not units.get(provider_field):
                raise ValueError(f"missing unit for provider field {provider_field}")
            observations[feature] = float(values[provider_field])
    except (KeyError, TypeError, ValueError) as error:
        return LiveFetchResult("invalid", source, message=str(error))
    return LiveFetchResult("ok", source, data_timestamp.isoformat(), station_id, observations)


def fetch_live_observation(
    url: str,
    required_features: list[str],
    field_mapping: dict[str, str],
    expected_station: str,
    max_age: timedelta,
    timeout_seconds: float = 10.0,
    opener: Callable[..., Any] = urlopen,
) -> LiveFetchResult:
    """Fetch JSON from an explicitly configured provider and normalize it."""

    if not url.strip():
        return LiveFetchResult("not_configured", "", message="no live-data provider URL configured")
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "godavari-flood-forecasting/0.1"})
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return LiveFetchResult("offline", url, message=f"provider returned HTTP {error.code}")
    except (URLError, TimeoutError, OSError) as error:
        return LiveFetchResult("offline", url, message=f"provider request failed: {error}")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        return LiveFetchResult("invalid", url, message=f"provider response is not valid JSON: {error}")
    if not isinstance(payload, dict):
        return LiveFetchResult("invalid", url, message="provider response must be a JSON object")
    return parse_live_payload(payload, required_features, field_mapping, expected_station, max_age)