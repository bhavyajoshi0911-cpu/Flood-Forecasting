import { CloudRain, LoaderCircle, Radio, Send, Waves } from "lucide-react";
import { useState } from "react";
import { getCwcWaterLevelAlerts, requestManualForecast } from "../services/api.js";

export default function ForecastPage({ featureNames = [] }) {
  const [mode, setMode] = useState("live");
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [cwcFeed, setCwcFeed] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function normalizeForecastResponse(response) {
    const rawProbability = response.flood_probability ?? response.probability ?? null;
    const probability = rawProbability == null
      ? null
      : Number(rawProbability) > 1
        ? Number(rawProbability)
        : Number(rawProbability) * 100;

    return {
      risk: String(response.risk_level ?? response.risk ?? (response.prediction === 1 || response.prediction === "flood" ? "HIGH" : "LOW")).toUpperCase(),
      probability,
      data_source: response.data_source ?? response.source ?? "--",
      model_version: response.model_name ?? response.model_version ?? "--",
      prediction: response.prediction ?? response.prediction_value ?? "--",
      forecast_horizon: response.forecast_horizon ?? "--",
      explanation: response.explanation ?? null,
    };
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setCwcFeed(null);
    setLoading(true);
    try {
      if (mode === "live") {
        setCwcFeed(await getCwcWaterLevelAlerts());
      } else {
        const response = await requestManualForecast(Object.fromEntries(featureNames.map((name) => [
          name,
          name === "Station" ? values[name] : Number(values[name]),
        ])));
        setResult(normalizeForecastResponse(response));
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="forecast-page" id="forecast">
      <div className="page-heading"><div><p className="eyebrow">FORECAST WORKSPACE / 02</p><h2 className="page-title">Flood Forecast</h2><p className="page-description">Review CWC station warnings or submit a complete feature vector for a research-only model forecast.</p></div><span className="research-label">RESEARCH MODE</span></div>
      <div className="forecast-layout">
        <form className="forecast-form" onSubmit={submit}>
          <div className="tabs" role="tablist" aria-label="Forecast type">
            <button type="button" role="tab" aria-selected={mode === "live"} className={mode === "live" ? "selected" : ""} onClick={() => { setMode("live"); setError(""); setResult(null); setCwcFeed(null); }}><Radio size={16} />CWC Live Data</button>
            <button type="button" role="tab" aria-selected={mode === "manual"} className={mode === "manual" ? "selected" : ""} onClick={() => { setMode("manual"); setError(""); setResult(null); setCwcFeed(null); }}><Send size={16} />Manual Forecast</button>
          </div>
          {mode === "live" ? <div className="form-empty"><Radio size={27} /><strong>Fetch CWC station warnings</strong><p>Retrieve the Central Water Commission's public list of stations reported above warning level. This feed is not a forecast from this project's model and does not include station names or observation timestamps.</p></div> : <div className="input-grid">
            {featureNames.map((name) => <label key={name}><span>{name}</span><input type={name === "Station" ? "text" : "number"} step={name === "Station" ? undefined : "any"} required value={values[name] ?? ""} onChange={(event) => setValues({ ...values, [name]: event.target.value })} /></label>)}
            {!featureNames.length && <p className="form-footnote">The model feature contract is not available; manual forecasting is disabled.</p>}
          </div>}
          <button className="submit-button" type="submit" disabled={loading || (mode === "manual" && !featureNames.length)}>{loading ? <LoaderCircle className="spin" size={17} /> : mode === "live" ? <Radio size={17} /> : <CloudRain size={17} />}{loading ? "Requesting..." : mode === "live" ? "Fetch CWC Warnings" : "Forecast Flood Risk"}</button>
          {mode === "manual" && <p className="form-footnote">All {featureNames.length || "required"} model features are mandatory. Manual inputs are research-only and are not live observations.</p>}
        </form>
        <div className="forecast-result" aria-live="polite">{error ? <div className="result-error"><strong>{mode === "live" ? "CWC data unavailable" : "Forecast unavailable"}</strong><span>{error}</span></div> : cwcFeed ? <div className="cwc-feed"><p className="eyebrow">OFFICIAL CWC WATER-LEVEL WARNINGS</p><strong className="result-risk">{cwcFeed.total} station{cwcFeed.total === 1 ? "" : "s"}</strong><p className="cwc-feed-note">{cwcFeed.message}</p>{cwcFeed.alerts.length ? <div className="cwc-alert-list">{cwcFeed.alerts.map((alert) => <div className="cwc-alert-row" key={alert.station_code}><strong>{alert.station_code}</strong><span>{alert.status}{alert.trend ? ` · ${alert.trend.toLowerCase()}` : ""}</span></div>)}</div> : <p className="cwc-feed-note">The source returned no station records. This does not establish that there is no flood risk.</p>}<small>Retrieved {new Date(cwcFeed.fetched_at).toLocaleString()} · Source observation time and station-to-basin mapping were not provided.</small><a href={cwcFeed.source_url} target="_blank" rel="noreferrer">Open Central Water Commission source</a></div> : result ? <><p className="eyebrow">MODEL RESPONSE · RESEARCH ONLY</p><strong className="result-risk">{result.risk ?? "--"}</strong><div className="result-details"><span>Probability <b>{result.probability == null ? "--" : `${Number(result.probability).toFixed(1)}%`}</b></span><span>Source <b>{result.data_source ?? "--"}</b></span><span>Model <b>{result.model_version ?? "--"}</b></span></div>{result.explanation && <div className="result-explanation"><p className="eyebrow">TOP SHAP SIGNALS</p>{Object.entries(result.explanation).sort(([, left], [, right]) => Math.abs(Number(right)) - Math.abs(Number(left))).slice(0, 3).map(([feature, value]) => <span key={feature}>{feature} <b>{Number(value).toFixed(3)}</b></span>)}</div>}</> : <div className="result-placeholder"><Waves size={26} /><p>{mode === "live" ? "CWC warning data will appear here" : "Forecast result will appear here"}</p><small>{mode === "live" ? "No source data has been retrieved yet." : "No prediction has been returned."}</small></div>}</div>
      </div>
    </section>
  );
}