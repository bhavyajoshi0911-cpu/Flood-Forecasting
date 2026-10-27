# Frontend

The React/Vite interface is in `frontend/`. Run it from that folder:

```powershell
npm install
npm run dev
```

Set `VITE_API_BASE_URL` only when the backend is not at `http://localhost:8000/api`. The UI includes dashboard, forecast, historical, performance, data-quality, API-status, prediction-history, and alert views. Empty and unavailable states are intentional while data and model gates remain open.

The frontend does not receive or store external provider API keys.
