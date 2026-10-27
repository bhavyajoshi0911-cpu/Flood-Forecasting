# Architecture

## Purpose and current implementation

The repository contains a FastAPI service, React dashboard, raw `.basin` files,
processed data, and a scikit-learn Decision Tree pipeline. The binary model
target is a station-specific 95th-percentile water-level research assumption,
not an authoritative flood-event label or emergency-warning threshold. Raw-data
provenance, license, and detailed domain validation remain open.

## Implemented research flow

```text
Raw basin sources (provenance not yet verified)
        |
        v
Loader -> validator -> quality report
        |
        v
Cleaned, aligned observations
        |
        v
Leakage-safe feature engineering -> target generation
        |
        v
Chronological train / validation / test datasets
        |
        v
scikit-learn Decision Tree -> test metrics and model artifact
        |                         |
        v                         v
Prediction service          SHAP explanation service
        ^                         ^
        |                         |
Validated live observations or validated manual inputs
        |
        v
FastAPI -> SQLite prediction history and status records -> React dashboard
```

## Ownership boundaries

### Data layer

`backend/app/data/` loads and validates supported CSV observations. The API can
count the current tab-separated `.basin` files, but reports a warning because
it does not apply detailed schema and domain checks to them.

### ML layer

`backend/app/ml/` contains preprocessing, leakage-safe feature, target, split,
training, evaluation, prediction, and SHAP components. Current model metrics
and explanations describe the research dataset only.

### API layer

`backend/app/api/routes/` will expose health, dataset, model, historical, live-data, forecast, explanation, prediction-history, and alert endpoints. Pydantic schemas will validate request and response contracts.

### Persistence layer

SQLite schemas exist for dataset/model metadata, predictions, explanations,
flood events, live observations, API status, and alerts. Prediction-history and
alert routes are not wired to persistence and explicitly report unavailable.

### Frontend layer

`frontend/src/` will provide the dashboard, forecast, historical, explainability, model-performance, data-quality, API-status, prediction-history, alerts, and about views. It will display the data status and source alongside every live result.

## Request flows

### Manual forecast

Complete validated feature vector -> saved model probability -> research risk
label -> SHAP values -> response.

Manual input requires the full model feature contract. Missing values are
rejected; the service never fills them from held-out test rows.

### Live forecast

Provider client -> timeout and HTTP handling -> response normalization ->
timestamp, station, unit, range, and staleness validation -> full feature
contract -> model probability and research risk label -> response.

A failed, stale, or incomplete provider response returns an unavailable result.
Test-set data is never substituted for live data.

## Required API contract candidates

```text
GET  /api/health
GET  /api/dataset/summary
GET  /api/dataset/quality
GET  /api/model/info
GET  /api/model/metrics
GET  /api/model/features
GET  /api/model/feature-importance
GET  /api/historical
GET  /api/flood-events
GET  /api/live/status
GET  /api/live/latest
POST /api/forecast/manual
POST /api/forecast/live
POST /api/explain
GET  /api/predictions
GET  /api/predictions/{id}
GET  /api/shap/summary
GET  /api/alerts
```

The API exposes health, dataset/model status, historical research records, live
status, manual/live forecast, explanation, prediction-history, and alert routes.
Prediction-history and alert routes remain unavailable until persistence is
connected.

## Reliability and safety rules

- Use environment variables for provider credentials.
- Apply request timeouts, HTTP error handling, missing-field checks, rate-limit awareness, and provider status tracking.
- Use database access patterns that avoid string-built SQL.
- Validate all user inputs at the API boundary.
- Log operational events without keys or secrets.
- Version model, dataset, and feature definitions for every prediction.
- Expose quality failures, model unavailability, SHAP unavailability, and API outages to the UI.
