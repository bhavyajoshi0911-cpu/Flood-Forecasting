import { CloudOff, Radio, RefreshCw, Server } from "lucide-react";
import { useEffect, useState } from "react";
import { get, getCwcWaterLevelAlerts } from "../services/api.js";

export default function ApiStatusPage() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [cwcFeed, setCwcFeed] = useState(null);
  const [cwcError, setCwcError] = useState("");
  const [loading, setLoading] = useState(false);

  async function refreshStatuses() {
    setLoading(true);
    setError("");
    setCwcError("");
    const [modelInput, cwc] = await Promise.allSettled([
      get("/live/status"),
      getCwcWaterLevelAlerts(),
    ]);
    if (modelInput.status === "fulfilled") {
      setStatus(modelInput.value);
    } else {
      setStatus(null);
      setError(modelInput.reason.message);
    }
    if (cwc.status === "fulfilled") {
      setCwcFeed(cwc.value);
    } else {
      setCwcFeed(null);
      setCwcError(cwc.reason.message);
    }
    setLoading(false);
  }

  useEffect(() => {
    refreshStatuses();
  }, []);

  const modelInputLabel = status?.status === "configured" || status?.status === "ok"
    ? "CONFIGURED"
    : status?.status === "incomplete"
      ? "INCOMPLETE"
      : "NOT CONFIGURED";
  const modelInputReady = modelInputLabel === "CONFIGURED";

  return <section className="api-page" id="api-status"><div className="page-heading"><div><p className="eyebrow">CONNECTIVITY / 07</p><h2 className="page-title">API Status</h2><p className="page-description">CWC warning data is connected separately from the local model's required input provider.</p></div><button className="refresh-button" type="button" onClick={refreshStatuses} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} />{loading ? "Checking..." : "Refresh feeds"}</button></div>{error && <div className="notice error"><strong>Model input status unavailable</strong><span>{error}</span></div>}<div className="api-cards"><article className="api-card"><Server size={20} /><div><span>BACKEND API</span><strong className="api-online">{error ? "UNAVAILABLE" : "CONNECTED"}</strong><small>Application status endpoint</small></div></article><article className="api-card"><CloudOff size={20} /><div><span>LOCAL MODEL INPUT PROVIDER</span><strong className={modelInputReady ? "api-online" : "api-offline"}>{modelInputLabel}</strong><small>{status?.message ?? "Checking model input configuration..."}</small></div></article><article className="api-card"><Radio size={20} /><div><span>CWC WATER-LEVEL WARNING FEED</span><strong className={cwcFeed ? "api-online" : "api-offline"}>{cwcError ? "UNAVAILABLE" : cwcFeed ? "CONNECTED" : "CHECKING"}</strong><small>{cwcError || (cwcFeed ? `${cwcFeed.total} station records received from the Central Water Commission.` : "Checking the official source feed...")}</small></div></article></div><div className="api-note"><Radio size={18} /><div><strong>Latest successful CWC feed retrieval</strong><p>{cwcFeed?.fetched_at ? new Date(cwcFeed.fetched_at).toLocaleString() : "--"}</p><small>{cwcFeed ? "This is when this app fetched the CWC response; the source does not include an observation timestamp. This feed is not used as input to the local model." : cwcError || "No successful CWC fetch is available yet."}</small>{cwcFeed?.source_url && <a href={cwcFeed.source_url} target="_blank" rel="noreferrer">Open CWC source</a>}</div></div></section>;
}