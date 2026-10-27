# Development Checklist

Complete one phase, run its checks, report the result, and stop for confirmation before starting the next phase.

- [x] Phase 1: Create architecture, planning documents, and empty project structure.
- [x] Phase 2 implementation: Add the source-agnostic loader, validator, and data-quality report.
- [ ] Phase 2 data gate: Select and document authoritative real datasets before processing.
- [x] Phase 3 implementation: Add timestamp normalization, numeric coercion, duplicate handling, gap reporting, and documented-unit checks.
- [ ] Phase 3 data gate: Apply preprocessing to the selected real dataset after source metadata is available.
- [x] Phase 4 implementation: Add station-scoped lag, prior-window, calendar, and deterministic feature-schema helpers.
- [ ] Phase 4 data gate: Apply features only to verified columns from the selected real dataset.
- [x] Phase 5 implementation: Add explicit label/threshold target definitions and next-day station/date alignment.
- [ ] Phase 5 data gate: Select authoritative flood-event evidence or document a labelled research assumption.
- [x] Phase 6 implementation: Add chronological ratio/date splitting with non-empty and timezone checks.
- [ ] Phase 6 data gate: Select split periods from the actual dataset coverage.
- [x] Phase 7 implementation: Add configurable XGBoost training and versioned model/feature/metadata artifact writing.
- [ ] Phase 7 data gate: Train only after real labelled data and verified chronological splits are available.
- [x] Phase 8 implementation: Add measured binary metrics, unavailable single-class AUC handling, JSON persistence, and evaluation plot generation.
- [ ] Phase 8 data gate: Calculate and report metrics only from the real chronological test partition.
- [x] Phase 9 implementation: Add real TreeExplainer global importance, individual contributions, JSON persistence, and summary plotting.
- [ ] Phase 9 data gate: Generate SHAP results only from the trained model and verified feature matrix.
- [x] Phase 10 implementation: Add FastAPI configuration, CORS, health, dataset status, and model status endpoints.
- [ ] Phase 10 data gate: Add forecast and persistence endpoints after model/data artifacts are available.
- [x] Phase 11 implementation: Add SQLite SQLAlchemy models, explicit initialization, session factory, and persistence tests.
- [ ] Phase 11 data gate: Connect prediction and API routes to persistence after their schemas are defined.
- [x] Phase 12 implementation: Add provider-neutral live JSON fetching, field mapping, unit/station/timestamp/freshness validation, and outage statuses.
- [ ] Phase 12 data gate: Verify a real provider covers the final trained feature contract before configuring a live URL.
- [x] Phase 13 implementation: Add manual/live forecast orchestration with model validation, risk classification, timestamps, source labels, and required SHAP output.
- [ ] Phase 13 data gate: Connect forecasts to a trained artifact, verified feature configuration, and persistence after real data is available.
- [x] Phase 14 implementation: Create the React/Vite frontend shell, responsive status dashboard, and backend API service layer.
- [ ] Phase 14 data gate: Connect populated forecast, history, and explanation views after backend contracts are available.
- [x] Phase 15 implementation: Build the responsive dashboard with scientific KPIs, honest empty states, and model/data status indicators.
- [ ] Phase 15 data gate: Populate dashboard values only from validated observations and actual model forecasts.
- [x] Phase 16 implementation: Build live/manual forecast tabs, numeric input validation, loading/error states, and backend result rendering.
- [ ] Phase 16 data gate: Enable forecast results only after backend POST contracts, model artifacts, and persistence are available.
- [ ] Phase 17: Build the SHAP Explainability page using API results.
- [x] Phase 18 implementation: Add paginated historical table, date/station/flood filters, empty/error states, and validated trend-chart placeholder.
- [ ] Phase 18 data gate: Render historical rows and charts only from the backend’s validated dataset response.
- [x] Phase 19 implementation: Add model-period metadata, metric cards, evaluation-chart panels, and unavailable/error states.
- [ ] Phase 19 data gate: Render metrics and plots only from actual test evaluation artifacts.
- [x] Phase 20 implementation: Add dataset quality and live API status routes and responsive frontend status views.
- [ ] Phase 20 data gate: Populate status details from validated source/provider records after configuration.
- [x] Phase 21 implementation: Add prediction-history and research-alert views with source/model provenance and honest empty/error states.
- [ ] Phase 21 data gate: Connect views to persisted prediction and alert records after database routes are implemented.
- [x] Phase 22 implementation: Add API integration tests, explicit unavailable workflow responses, and POST CORS coverage.
- [ ] Phase 22 data gate: Run full integration workflows after real data, model, persistence, and frontend dependencies are installed.
- [x] Phase 23 implementation: Review responsive states, keyboard focus, readable typography, reduced motion, and unavailable/error messaging.
- [ ] Phase 23 data gate: Perform browser-based desktop/mobile verification after Node/npm is installed.
- [x] Phase 24 implementation: Complete technical documentation and document the reproducible scaffold demonstration.
- [ ] Phase 24 data gate: Complete the final real-data demonstration after dataset, model, provider, and browser gates pass.

## Phase completion report template

For each phase record:

1. What was created.
2. Files created or modified.
3. What each important file does.
4. Exact command to run from the correct folder.
5. Expected output.
6. How to verify it worked.
7. Problems, limitations, and assumptions.
