# Dataset and Evidence Contract

## Current status

The repository contains three raw `.basin` files (`rainfall.basin`,
`river.basin`, and `riverdischarge.basin`) and processed/model artifacts. The
API can read the raw files and count their tab-separated rows, but it does not
yet perform detailed schema, unit, station-alignment, or domain validation for
these sources. Their upstream provenance, access terms, and license must be
verified before treating them as an approved research dataset.

The current model target is `flood_observed_research`, built from
station-specific 95th-percentile `river_level_max_m` thresholds. The target
metadata explicitly marks this as a non-authoritative research assumption. It
must not be described as an observed official flood event or warning threshold.

## Exact ML problem

Intended task: binary next-day research-target classification. The current
artifacts do not establish an operational or officially validated flood forecast.

```text
Input: observations available at prediction time t
Output: probability and class for flood risk at t + 1 day
Current research labels: 0 = below the assumed threshold, 1 = at or above the
assumed threshold
```

A multiclass target such as normal, alert, and flood is allowed only when an
authoritative event source supports it and the class rules are documented.

## Required columns for an authoritative future training table

At minimum, a usable training table must contain:

| Column | Meaning | Required evidence |
|---|---|---|
| `timestamp` | Observation time | timezone, frequency, completeness |
| `station_id` | Measurement location | station registry and basin relationship |
| `flood_observed` or an equivalent event join key | Target evidence | authoritative event definition |
| `rainfall` | Precipitation measurement | unit, gauge, accumulation period |
| `river_discharge` or `water_level` | Hydrological signal | unit, datum, station, measurement method |

The final required set will be the intersection of variables actually present, valid, aligned, and available at prediction time. A flood-event table may be joined by station and time rather than containing a direct target column.

## Optional columns

Only use these if the selected source documents them and they improve the research question without causing leakage:

- `temperature`
- `humidity`
- `soil_moisture`
- `wind_speed`
- Additional discharge or water-level measurements
- Quality flags, sensor status, and observation confidence
- Basin, sub-basin, station latitude, and station longitude

Calendar features such as month, season, and day of year can be derived from a valid timestamp. They are not substitutes for environmental measurements.

## Flood-event information

For every event or target source, record:

- event identifier and observed start/end time
- station, river, basin, and location
- event category or threshold
- threshold units, datum, and comparison rule
- source organization, document title, URL, access date, and license/access terms
- whether the record is observed, reported, modelled, or inferred

The current threshold is a research assumption. It must remain visibly
identified as such in the application and must never be presented as an
official warning threshold.

## Required metadata for every real dataset

Store a metadata record before processing:

- source name and URL
- provider and access method
- date range and retrieval date
- station or geographic coverage
- temporal frequency and timezone
- every column's unit and meaning
- missing-value and quality-flag conventions
- license or access restrictions
- dataset version or checksum where available
- transformations performed by this project

Keep raw source files unchanged in `data/raw/`. Put cleaned outputs in `data/processed/` and metadata in `data/metadata/`.

## Validation requirements

The current `.basin` API report only counts non-empty rows and collects
headers; it explicitly returns a warning because detailed validation is not
applied. Before using the data for stronger claims, validation must check
missing dates, duplicates, missing values, invalid values, negative rainfall,
impossible measurements, outliers, inconsistent units, station mismatches,
timezone issues, and target coverage. Preserve counts before and after each
cleaning action.

## Leakage controls

Only information known by the forecast cutoff may be used. In particular, do not use future rainfall, discharge, water level, flood status, or rolling values that include future rows. Lag and rolling features must be shifted so their window ends at or before the prediction time. Split chronologically after feature and target construction, with dates selected from actual coverage rather than copied from an example.

## Live-data requirements

Before an API is selected, list the final model features and map each one to a provider field, a recent historical observation, or an unavailable value. Verify station specificity, units, timezone, update frequency, rate limit, credentials, and historical availability. A weather API must not be described as a river-discharge or water-level API unless its documentation proves that capability.

A live observation is usable only when its timestamp, station, units, required fields, plausible range, and freshness pass validation. Otherwise the API status must be `offline`, `invalid`, or `stale`.

## Pending decisions

1. What are the verified source URLs, licenses, and coverage of the existing raw files?
2. Can rainfall, discharge, water level, and authoritative flood events be aligned at the same frequency?
3. What authoritative target definition should replace the research assumption?
4. Which variables are genuinely available at prediction time?
5. What chronological train, validation, and test periods are supported by validated data?
6. Can a live provider reproduce every trained feature without unsupported imputation?
