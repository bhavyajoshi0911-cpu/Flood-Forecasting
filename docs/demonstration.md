# Reproducible Demonstration

## Backend checks

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite includes API integration checks and should not skip them because
of a missing `httpx` dependency.

Start the API from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Check `http://127.0.0.1:8000/docs`. Health should detect the `.basin` source
files and model artifact. Dataset quality should report a warning because the
current `.basin` path does not apply full schema/domain validation. Live
forecasts should be unavailable unless all provider features validate.

## Frontend check

From `frontend/`, build the production bundle:

```powershell
npm.cmd run build
```

To start the development server:

```powershell
npm.cmd run dev
```

Confirm the dashboard shows the research-target caveat, labels model ROC-AUC as
a test metric rather than a flood probability, and does not invent latest
observations, forecasts, alerts, or prediction history.

## Research completion gate

A stronger research demonstration requires verified source provenance and
license, detailed quality checks, authoritative target evidence, validated
chronological evaluation, and a provider mapping for every model feature. It
must not be presented as an emergency warning service.
