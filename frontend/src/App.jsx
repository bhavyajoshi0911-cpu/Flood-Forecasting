import { Activity, Droplets, Gauge, RefreshCw, ShieldAlert, Waves } from "lucide-react";
import { useEffect, useState } from "react";
import { getShapSummary, getSystemStatus } from "./services/api.js";
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
  const [shapSummary, setShapSummary] = useState(null);
  const [shapError, setShapError] = useState("");

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      setStatus(await getSystemStatus());
    } catch (requestError) {
      setStatus(emptyState);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
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

  const heroTitle = !backendOnline
    ? "Backend unavailable"
    : !datasetReady
      ? "Dataset unavailable"
      : !modelReady
        ? "Model unavailable"
        : "Forecast not generated";

  const heroDescription = !backendOnline
    ? "The API is not responding; refresh to retry the connection."
    : !datasetReady
      ? "A valid raw dataset is required before any forecast can be generated."
      : !modelReady
        ? "The trained model metadata is missing or unreadable."
        : "The model and source files are available, but a forecast needs a complete, validated feature vector. The trained target is a research assumption, not an official warning.";

  const rainfallValue = datasetReady ? formatMetricValue(datasetRows, 0) : "--";
  const riverValue = modelReady ? `${status.model?.metadata?.training_rows ?? "--"}` : "--";
  const waterValue = modelReady ? `${status.model?.metadata?.testing_rows ?? "--"}` : "--";
  const topFeatures = shapFeatures.map((item) => item.feature).join(", ") || featureNames.slice(0, 3).join(", ") || "No feature metadata";
  const latestConditionText = datasetReady
    ? `${formatMetricValue(datasetRows, 0)} rows were read across the raw source files.`
    : "No validated live observations are available.";
  const latestConditionNote = datasetReady
    ? "Source files are present; detailed schema and domain validation is still required."
    : "Configure a verified provider after the trained feature contract is finalized.";
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
          <time>{new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" })}</time>
        </section>

        <section className="hero-panel">
          <div><p className="eyebrow light">CURRENT FLOOD RISK</p><h2>{heroTitle}</h2><p>{heroDescription}</p></div>
          <div className="hero-value">--<small>no current forecast</small></div>
        </section>

        <section className="metrics-grid">
          <article className="metric-card"><Droplets size={18} /><span>RAW SOURCE ROWS</span><strong>{rainfallValue}</strong><small>{datasetReady ? "across available source files" : "no source files"}</small></article>
          <article className="metric-card"><Waves size={18} /><span>TRAINING ROWS</span><strong>{riverValue}</strong><small>{modelReady ? "model training set" : "no validated observation"}</small></article>
          <article className="metric-card"><Activity size={18} /><span>TEST ROWS</span><strong>{waterValue}</strong><small>{modelReady ? "evaluation set" : "no validated observation"}</small></article>
          <article className="metric-card"><Gauge size={18} /><span>TEST ROC-AUC</span><strong>{rocAuc == null ? "--" : `${(Number(rocAuc) * 100).toFixed(1)}%`}</strong><small>test-set ranking metric, not flood probability</small></article>
        </section>

        <section className="lower-grid">
          <article className="panel"><div className="panel-heading"><div><p className="eyebrow">OBSERVATION FEED</p><h3>Source coverage</h3></div><span className="empty-tag">{datasetReady ? "SOURCE PRESENT" : "NO SOURCE"}</span></div><div className="empty-state"><Activity size={28} /><p>{latestConditionText}</p><small>{latestConditionNote}</small></div></article>
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
