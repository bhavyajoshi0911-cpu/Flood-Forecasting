# Godavari River Basin Flood Forecasting

An end-to-end research and demonstration system for next-day flood-risk classification using historical environmental and hydrological data, scikit-learn, SHAP explanations, FastAPI, React, and SQLite.

## Current status

The repository includes raw basin data files, processed data, a trained
Decision Tree pipeline, evaluation artifacts, a FastAPI backend, and a React
dashboard. The current model uses a research target defined by station-specific
95th-percentile water-level thresholds. This is not an authoritative flood-event
label or warning threshold. Raw source provenance and license, full domain
validation, and a complete live mapping for every model feature remain
unverified.

The reported test metrics describe the current research dataset and split only;
they are not a real-world performance guarantee. The API does not substitute
held-out test rows for live observations or fill missing manual features with
sample values.

## Project principles

- Never present synthetic data as real observations.
- Record the source, URL, coverage, station, frequency, units, and access or license information for every real dataset.
- Define flood labels from an authoritative source or explicitly document a research assumption.
- Preserve timestamps and use chronological splits to prevent leakage.
- Mark live data as unavailable or stale when it cannot be validated.
- Keep API keys in environment variables and never expose them to the frontend.

## Planned structure

```text
backend/       FastAPI application, ML pipeline, data services, and tests
frontend/      React dashboard
 data/         raw, processed, and metadata directories
models/        versioned trained-model artifacts
notebooks/     reproducible research notebooks
docs/          architecture, dataset, and later technical documentation
scripts/       repeatable setup, validation, and training commands
```

See [docs/architecture.md](docs/architecture.md) for the data flow,
[docs/dataset.md](docs/dataset.md) for data and target caveats, and
[docs/limitations.md](docs/limitations.md) for current research constraints.

## Windows setup

Run these commands in separate terminals from the project root.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

`python -m venv .venv` creates an isolated environment, the next command installs
the backend dependencies, and Uvicorn starts the FastAPI development server.

### Frontend

Open a second terminal from the project root:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The frontend reads the backend URL from `VITE_API_BASE_URL`; it defaults to
`http://127.0.0.1:8000/api`.

## Live data integration

`backend/live_data.py` fetches a provider response, validates its timestamp,
station, units, and values, then requires the exact feature vector from
`backend/models/model_metadata.json`. It does not retrain the model or invent
missing lag and rolling values. The live forecast endpoint reports unavailable
until the provider returns every required feature.

The current model requires 68 engineered fields, including historical lag,
rolling, change, station, and calendar features. A provider that returns only a
current rainfall or water-level observation is therefore insufficient. Configure
a provider or upstream feature service that supplies the complete trained
feature vector with the same units used during training.

Set these variables in `backend/.env` (do not commit credentials):

```text
LIVE_DATA_API_KEY=your_provider_key
LIVE_DATA_PROVIDER_URL=https://provider.example/live
LIVE_DATA_STATION=your_station_id
LIVE_DATA_TIMEOUT_SECONDS=10
LIVE_DATA_MAX_AGE_SECONDS=86400
LIVE_DATA_FIELD_MAPPING={"model_feature_1":"provider_field_1","model_feature_2":"provider_field_2"}
LIVE_DATA_EXPECTED_UNITS={"model_feature_1":"mm","model_feature_2":"m"}
RISK_MEDIUM_THRESHOLD=0.40
RISK_HIGH_THRESHOLD=0.70
```

The mapping values above are illustrative only; configure every required model
feature and its expected unit before considering the provider usable.
`LIVE_DATA_API_KEY` is optional for public providers that do not require
authentication. `LIVE_DATA_MAX_AGE_SECONDS` defaults to 86400 seconds (24
hours); select a freshness limit appropriate to the provider and observation
frequency.

The following abbreviated example illustrates the response shape. A real
response must include observations and units for all 68 model features:

```json
{
	"timestamp": "2026-09-17T10:00:00+00:00",
	"station_id": "G1",
	"observations": {"provider_field": 12.5},
	"units": {"provider_field": "mm"}
}
```

Run the live prediction module from the project root:

```powershell
backend\.venv\Scripts\python.exe backend\live_data.py
```

Run the mocked live-data tests and existing suite:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Timeouts, HTTP errors, invalid JSON, missing fields, station mismatches, unit
mismatches, and incomplete model features return controlled errors. Provider
credentials and URLs are not returned by the live-status endpoint. Risk
thresholds are application-level bands and must be validated against
flood-management requirements before use.

## Development workflow

Run the backend tests from `backend` with
`.\.venv\Scripts\python.exe -m pytest -q`, and build the frontend from
`frontend` with `npm.cmd run build`.

## Known decisions still pending

- The verified provenance and license of the rainfall and hydrological sources.
- An authoritative flood-event source and threshold or category definition.
- The final required model columns and supported prediction frequency.
- The chronological train, validation, and test periods.
- A live provider that supplies the model's required variables at the required station and frequency.
- Whether the available live data is sufficient for a valid live prediction.

## Scope and limitations

This is a research and demonstration system, not an emergency warning service. A forecast must not be treated as an official flood warning. Data gaps, station mismatch, measurement uncertainty, distribution shift, and API outages must be visible to users.
