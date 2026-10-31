# Godavari River Basin Flood Forecasting

A research and demonstration application for exploring flood-risk
classification with historical hydrological data. The project includes a
FastAPI backend, a React/Vite dashboard, a saved scikit-learn model, evaluation
artifacts, and SHAP explanations.

> **Important:** This is not an official flood-warning system. Its current
> target is a research assumption, not an authoritative flood-event label or
> warning threshold. Do not use its predictions for emergency or operational
> decisions.

## Current project status

- The raw source directory contains `rainfall.basin`, `river.basin`, and
  `riverdischarge.basin`. The API can count their rows and list their columns,
  but it does not yet perform full schema, unit, station-alignment, or
  hydrological-domain validation. Source provenance and license also need to
  be verified.
- The saved Decision Tree model has 68 required input features. Its target is
  based on station-specific 95th-percentile river-level thresholds, explicitly
  documented as a non-authoritative research assumption in
  `data/metadata/research_target_definition.json`.
- Reported model metrics and SHAP values describe the current research
  artifacts only; they are not evidence of real-world forecasting performance.
- A live forecast requires a provider to supply every model feature with
  validated station, timestamp, units, and freshness. The application rejects
  missing features instead of substituting held-out test data.
- Prediction history and alerts are not connected to persistence and currently
  report that they are unavailable.

For details, see [docs/dataset.md](docs/dataset.md),
[docs/limitations.md](docs/limitations.md), and
[docs/architecture.md](docs/architecture.md).

## Project layout

```text
backend/       FastAPI application, model services, and tests
data/raw/      Original basin source files
data/processed/Processed research datasets
data/metadata/ Dataset quality and target-definition metadata
backend/models/Saved model and model metadata
frontend/      React/Vite web application
docs/          Dataset, architecture, limitations, and setup documentation
scripts/       Dataset validation utilities
```

## Run locally on Windows

Open two terminals from the project root.

### 1. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

If `backend\.venv` already exists, skip the `venv` creation step. The API is
available at <http://127.0.0.1:8000>; interactive documentation is at
<http://127.0.0.1:8000/docs>.

### 2. Start the frontend

In the second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL printed in the terminal (normally
<http://127.0.0.1:5173>). The frontend uses
`http://127.0.0.1:8000/api` by default. To use a different backend, set
`VITE_API_BASE_URL` before starting Vite, for example:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000/api"
npm.cmd run dev
```

The backend's default CORS configuration allows local frontend origins on
ports 4173 and 5173–5175. Set `CORS_ORIGINS` to a comma-separated list when
using another origin.

## Verify the project

Run the backend tests from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Build the frontend from `frontend/`:

```powershell
npm.cmd run build
```

Refresh the raw-data quality summary from the project root:

```powershell
.\backend\.venv\Scripts\python.exe .\scripts\validate_dataset.py
```

The `.basin` files are tab-separated. The generated quality report counts
non-empty rows and collects headers, and marks the result as a warning because
full validation has not been applied.

## API overview

All API paths are prefixed with `/api`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Backend, dataset-file, and model-artifact status |
| `GET /dataset/summary` | Dataset row count and available columns |
| `GET /dataset/quality` | Data-quality status, warnings, and source files |
| `GET /model/info` | Model metadata and required feature list |
| `GET /model/metrics` | Saved test metrics and evaluation curves |
| `GET /historical` | Paginated research records; accepts `page`, `page_size`, `station_id`, `start_date`, `end_date`, and `flood_status` filters |
| `GET /shap/summary` | Global feature-importance explanation |
| `GET /live/status` | Live-provider configuration status |
| `POST /forecast/manual` | Research forecast from a complete feature object |
| `POST /forecast/live` | Forecast from a validated live-provider response |
| `GET /predictions`, `GET /alerts` | Currently unavailable; persistence is not connected |

Manual forecast requests must provide all 68 fields listed by `GET
/api/model/info` under `metadata.feature_names`. No features are filled from
sample or test rows.

## Optional live-provider configuration

Live forecasts remain unavailable until a provider supplies the full trained
feature vector in the format expected by the model. A current weather reading
alone is insufficient because the model also requires lagged, rolling, change,
station, and calendar features.

Configure these values in `backend/.env`; do not commit credentials:

```dotenv
LIVE_DATA_PROVIDER_URL=https://provider.example/live
LIVE_DATA_API_KEY=
LIVE_DATA_STATION=your_station_id
LIVE_DATA_TIMEOUT_SECONDS=10
LIVE_DATA_MAX_AGE_SECONDS=86400
LIVE_DATA_FIELD_MAPPING={"model_feature_1":"provider_field_1","model_feature_2":"provider_field_2"}
LIVE_DATA_EXPECTED_UNITS={"model_feature_1":"mm","model_feature_2":"m"}
RISK_MEDIUM_THRESHOLD=0.40
RISK_HIGH_THRESHOLD=0.70
```

The feature names and provider fields above are examples only. Configure a
mapping and expected unit for every required model feature, and set
`LIVE_DATA_STATION`. A generic weather endpoint such as Open-Meteo's forecast
URL cannot directly supply this model's full river, discharge, lagged, rolling,
and change feature vector. Use a provider or adapter that returns the complete
contract below. An API key is optional for providers that do not require
authentication. The maximum observation age defaults to 24 hours; choose a
limit suitable for the provider's reporting frequency.

The provider must return a JSON object in this general shape, with all mapped
features included in `observations` and corresponding entries in `units`:

```json
{
  "timestamp": "2026-09-17T10:00:00+00:00",
  "station_id": "your_station_id",
  "observations": {
    "provider_field_1": 12.5,
    "provider_field_2": 3.4
  },
  "units": {
    "provider_field_1": "mm",
    "provider_field_2": "m"
  }
}
```

The backend checks timestamp, station, units, and feature values before
predicting. A configured URL alone does not mean that live data has been
received or validated. Provider URLs and credentials are not returned by the
live-status endpoint.

## Data, model, and safety caveats

- The current threshold-based target is a research proxy. It is not an
  official flood-event label, flood-stage threshold, or emergency warning.
- Source provenance, licensing, detailed data quality, and the model's
  applicability to future observations still require review.
- Saved test metrics use the existing chronological test partition and should
  be interpreted only in that research context.
- Risk bands are application-level categories. They must not be interpreted as
  official warning levels.
- Do not expose provider secrets in frontend variables, source code, logs, or
  screenshots.
