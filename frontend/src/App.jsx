import { Activity, Droplets, Gauge, RefreshCw, ShieldAlert, Waves } from "lucide-react";
import { useEffect, useState } from "react";
import { getCwcWaterLevelAlerts, getShapSummary, getSystemStatus } from "./services/api.js";
import ForecastPage from "./pages/ForecastPage.jsx";
import HistoricalPage from "./pages/HistoricalPage.jsx";
import PerformancePage from "./pages/PerformancePage.jsx";
import QualityPage from "./pages/QualityPage.jsx";
import ApiStatusPage from "./pages/ApiStatusPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import AlertsPage from "./pages/AlertsPage.jsx";

const emptyState = { health: null, dataset: null, model: null, shap: null };

function StatusPill({ online }) {
  return <span className={`status-pill ${online ? "online" : "offline"}`}><span />{online ? "Connected" : "Unavailable"}</span>;
}

function formatMetricValue(value, digits = 0) {
  if (value == null || value === "" || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function App() {
  const [status, setStatus] = useState(emptyState);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [cwcFeed, setCwcFeed] = useState(null);
  const [cwcError, setCwcError] = useState("");
  const [shapSummary, setShapSummary] = useState(null);
  const [shapError, setShapError] = useState("");

  async function refreshCwcFeed() {
    setCwcError("");
    try {
      setCwcFeed(await getCwcWaterLevelAlerts());
    } catch (requestError) {
      setCwcFeed(null);
      setCwcError(requestError.message);
    }
  }

  async function refresh() {
    setLoading(true);
    setError("");
    const [systemResult] = await Promise.allSettled([getSystemStatus(), refreshCwcFeed()]);
    if (systemResult.status === "fulfilled") {
      setStatus(systemResult.value);
    } else {
      setStatus(emptyState);
      setError(systemResult.reason.message);
    }
    setLoading(false);
  }

  useEffect(() => {
    refresh();
    getShapSummary().then(setShapSummary).catch((requestError) => setShapError(requestError.message));
  }, []);

  const backendOnline = Boolean(status.health);
  const datasetReady = ["valid", "warning"].includes(status.dataset?.status);
  const modelReady = status.model?.status === "available";
  const datasetRows = status.dataset?.row_count ?? null;
  const featureNames = status.model?.metadata?.feature_names ?? [];
  const shapFeatures = shapSummary?.global_importance?.slice(0, 3) ?? [];
  const modelMetrics = status.model?.metadata?.metrics ?? {};
  const rocAuc = modelMetrics["ROC-AUC"];

  const cwcAlertCount = cwcFeed?.total ?? null;
  const heroTitle = cwcError
    ? "CWC warning feed unavailable"
    : cwcFeed
      ? `${cwcAlertCount} station records in CWC feed`
      : "Loading CWC warning feed";
  const heroDescription = cwcError
    ? "The official source could not be reached. Refresh to try again; no current model forecast is available."
    : cwcFeed
      ? `${cwcFeed.message} The CWC feed does not identify a Godavari-specific station set. This is separate from the local model, which has no forecast without a complete validated feature vector.`
      : "Checking the Central Water Commission's public water-level warning feed. No result is a project model forecast.";

  const rainfallValue = datasetReady ? formatMetricValue(datasetRows, 0) : "--";
  const riverValue = modelReady ? `${status.model?.metadata?.training_rows ?? "--"}` : "--";
  const waterValue = modelReady ? `${status.model?.metadata?.testing_rows ?? "--"}` : "--";
  const topFeatures = shapFeatures.map((item) => item.feature).join(", ") || featureNames.slice(0, 3).join(", ") || "No feature metadata";
  const explainabilityText = modelReady
    ? "Feature signal available"
    : "SHAP explanation unavailable.";
  const explainabilityNote = modelReady
    ? `Top features: ${topFeatures}`
    : "Explanations appear with a real model prediction.";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark"><Activity size={20} /><span>GODAVARI<br /><b>FIELD LAB</b></span></div>
        <nav aria-label="Primary navigation">
          <a className="active" href="#dashboard"><Gauge size={17} />Dashboard</a>
          <a href="#forecast">Flood Forecast</a>
          <a href="#historical">Historical Data</a>
          <a href="#explainability">SHAP Explainability</a>
          <a href="#performance">Model Performance</a>
          <a href="#quality">Data Quality</a>
          <a href="#api-status">API Status</a>
          <a href="#prediction-history">Prediction History</a>
          <a href="#alerts">Alerts</a>
        </nav>
        <div className="sidebar-note"><ShieldAlert size={16} /><span>Research and demonstration system. Not an emergency warning service.</span></div>
      </aside>

      <section className="content" id="dashboard">
        <header className="topbar">
          <div><p className="eyebrow">HYDROLOGICAL INTELLIGENCE / 01</p><h1>Godavari River Basin</h1><p className="subtitle">Flood forecasting workspace</p></div>
          <button className="refresh-button" onClick={refresh} disabled={loading} title="Refresh system status"><RefreshCw size={17} className={loading ? "spin" : ""} />Refresh</button>
        </header>

        {error && <div className="notice error"><strong>Backend unavailable</strong><span>{error}</span></div>}
        <section className="status-strip" aria-label="System status">
          <div className="status-heading"><span className="pulse" />SYSTEM STATUS</div>
          <div><small>API</small><StatusPill online={backendOnline} /></div>
          <div><small>DATASET</small><StatusPill online={datasetReady} /></div>
          <div><small>MODEL</small><StatusPill online={modelReady} /></div>
          <div><small>CWC FEED</small><StatusPill online={Boolean(cwcFeed)} /></div>
          <time>{new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })}</time>
        </section>

        <section className="hero-panel">
          <div><p className="eyebrow light">OFFICIAL CWC WARNING FEED · NOT A MODEL FORECAST</p><h2>{heroTitle}</h2><p>{heroDescription}</p></div>
          <div className="hero-value">{cwcAlertCount == null ? "--" : formatMetricValue(cwcAlertCount)}<small>{cwcAlertCount == null ? "feed records unavailable" : "records reported by CWC"}</small></div>
        </section>

        <section className="metrics-grid">
          <article className="metric-card"><Droplets size={18} /><span>RAW SOURCE ROWS</span><strong>{rainfallValue}</strong><small>{datasetReady ? "across available source files" : "no source files"}</small></article>
          <article className="metric-card"><Waves size={18} /><span>TRAINING ROWS</span><strong>{riverValue}</strong><small>{modelReady ? "model training set" : "no validated observation"}</small></article>
          <article className="metric-card"><Activity size={18} /><span>TEST ROWS</span><strong>{waterValue}</strong><small>{modelReady ? "evaluation set" : "no validated observation"}</small></article>
          <article className="metric-card"><Gauge size={18} /><span>TEST ROC-AUC</span><strong>{rocAuc == null ? "--" : `${(Number(rocAuc) * 100).toFixed(1)}%`}</strong><small>test-set ranking metric, not flood probability</small></article>
        </section>

        <section className="lower-grid">
          <article className="panel"><div className="panel-heading"><div><p className="eyebrow">LIVE SOURCE FEED</p><h3>CWC station warnings</h3></div><span className="empty-tag">{cwcFeed ? `${cwcFeed.total} RECORDS` : cwcError ? "UNAVAILABLE" : "LOADING"}</span></div>{cwcError ? <div className="empty-state"><Activity size={28} /><p>CWC warning feed unavailable</p><small>{cwcError}</small></div> : cwcFeed ? <><div className="cwc-dashboard-summary"><strong>{cwcFeed.total} station records</strong><small>{cwcFeed.message} The feed does not identify which stations are in the Godavari basin.</small></div>{cwcFeed.alerts.length ? <div className="contributor-list cwc-dashboard-list">{cwcFeed.alerts.map((alert) => <div className="contributor-row" key={alert.station_code}><span>{alert.station_code}</span><strong>{alert.status}{alert.trend ? ` · ${alert.trend.toLowerCase()}` : ""}</strong></div>)}</div> : <div className="empty-state"><p>No station warning records returned</p><small>An empty feed does not establish that there is no flood risk.</small></div>}<div className="cwc-dashboard-footer"><small>Retrieved {new Date(cwcFeed.fetched_at).toLocaleString()}. Source observation time and station mapping are unavailable.</small><a href={cwcFeed.source_url} target="_blank" rel="noreferrer">Open CWC feed</a></div></> : <div className="empty-state"><Activity size={28} /><p>Loading CWC warning records</p><small>Checking the official source feed.</small></div>}</article>
          <article className="panel" id="explainability"><div className="panel-heading"><div><p className="eyebrow">EXPLAINABILITY</p><h3>Top contributors</h3></div></div>{shapFeatures.length ? <div className="contributor-list">{shapFeatures.map((item) => <div className="contributor-row" key={item.feature}><span>{item.feature}</span><strong>{Number(item.mean_abs_shap).toFixed(3)}</strong></div>)}</div> : <div className="empty-state"><ShieldAlert size={28} /><p>{shapError ? "SHAP summary unavailable" : explainabilityText}</p><small>{shapError || explainabilityNote}</small></div>}</article>
        </section>
        <ForecastPage featureNames={featureNames} />
        <HistoricalPage />
        <PerformancePage />
        <QualityPage />
        <ApiStatusPage />
        <HistoryPage />
        <AlertsPage />
      </section>
    </main>
  );
}

export default App;
