# API

Start the backend from `backend/` after installing dependencies:

```powershell
venv\Scripts\activate
uvicorn app.main:app --reload
```

Current status endpoints:

```text
GET /api/health
GET /api/dataset/summary
GET /api/dataset/quality
GET /api/model/info
GET /api/model/metrics
GET /api/live/status
GET /api/live/latest
GET /api/live/cwc-water-level-alerts
```

`GET /api/dataset/quality` reports row counts, invalid acquisition timestamps,
duplicate rows, and negative rainfall when raw `.basin` files are configured.
Quality checks are performed per source file; they do not imply that separate
rainfall, water-level, and discharge files have been joined into one table.
Metrics for fields absent from all configured sources are returned as `null`.

`GET /api/live/cwc-water-level-alerts` retrieves the Central Water Commission
Flood Forecasting portal's public list of stations reported above warning
level. This is a separate source feed, not an input to this project's model:
the response does not include source observation timestamps, units, station
names, or basin mapping. The project's model-based live forecast remains
unavailable until a provider can supply and validate the complete feature
contract.

Successful manual and configured live model forecasts are written to the
SQLite database (`backend/flood_forecasting.db` by default, configurable with
`DATABASE_URL`) and returned by `GET /api/predictions`. `GET /api/alerts`
returns the current CWC station-warning feed; it does not create alerts from
model predictions.

Workflow endpoints are registered but return HTTP 503 until real artifacts and persistence are connected:

```text
POST /api/forecast/manual
POST /api/forecast/live
GET  /api/historical
POST /api/explain
GET  /api/shap/summary
GET  /api/predictions
GET  /api/alerts
```

The API never returns fabricated observations, probabilities, metrics, or SHAP values.
