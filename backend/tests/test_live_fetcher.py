from datetime import datetime, timedelta, timezone

from app.data.live_fetcher import LiveFetchResult, fetch_live_observation, parse_live_payload


def payload(timestamp="2026-01-01T12:00:00+00:00"):
    return {
        "source": "documented test provider",
        "timestamp": timestamp,
        "station_id": "G1",
        "observations": {"rain": "12.5"},
        "units": {"rain": "mm"},
    }


def test_payload_is_mapped_and_normalized():
    result = parse_live_payload(payload(), ["rainfall"], {"rainfall": "rain"}, "G1", timedelta(hours=1), datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc))
    assert result.status == "ok"
    assert result.observations == {"rainfall": 12.5}


def test_stale_payload_is_not_accepted_as_current():
    result = parse_live_payload(payload(), ["rainfall"], {"rainfall": "rain"}, "G1", timedelta(minutes=10), datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc))
    assert result.status == "stale"


def test_missing_mapping_is_invalid():
    result = parse_live_payload(payload(), ["water_level"], {}, "G1", timedelta(hours=1), datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc))
    assert result.status == "invalid"


def test_blank_url_is_not_configured():
    result = fetch_live_observation("", ["rainfall"], {"rainfall": "rain"}, "G1", timedelta(hours=1))
    assert isinstance(result, LiveFetchResult)
    assert result.status == "not_configured"