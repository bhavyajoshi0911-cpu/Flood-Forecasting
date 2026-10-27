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
```

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
