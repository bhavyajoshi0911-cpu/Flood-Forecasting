const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api").replace(/\/$/, "");

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload?.detail ?? payload?.message ?? `Backend request failed (${response.status})`;
      throw new Error(message);
    }
    return payload;
  }

  const text = await response.text().catch(() => "");
  if (!response.ok) {
    throw new Error(text || `Backend request failed (${response.status})`);
  }
  return text;
}

async function get(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  return parseResponse(response);
}

export async function getSystemStatus() {
  const [health, dataset, model] = await Promise.all([get("/health"), get("/dataset/summary"), get("/model/info")]);
  return { health, dataset, model };
}

async function post(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return parseResponse(response);
}

export function requestManualForecast(features) { return post("/forecast/manual", { features }); }
export function requestLiveForecast() { return post("/forecast/live", {}); }
export function getCwcWaterLevelAlerts() { return get("/live/cwc-water-level-alerts"); }
export function getHistorical(filters = {}) { const query = new URLSearchParams({ page: filters.page || 1, page_size: 20 }); Object.entries(filters).forEach(([key, value]) => { if (value && key !== "page") query.set(key, value); }); return get(`/historical?${query.toString()}`); }
export function getModelMetrics() { return get("/model/metrics"); }
export function getPredictions(filters = {}) { const query = new URLSearchParams({ page: filters.page || 1, page_size: 20 }); Object.entries(filters).forEach(([key, value]) => { if (value && key !== "page") query.set(key, value); }); return get(`/predictions?${query.toString()}`); }
export function getAlerts() { return get("/alerts"); }
export function getShapSummary() { return get("/shap/summary"); }
export { get };