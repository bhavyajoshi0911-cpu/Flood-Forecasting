import { CloudOff, Radio, Server } from "lucide-react";
import { useEffect, useState } from "react";
import { get } from "../services/api.js";

export default function ApiStatusPage() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { get("/live/status").then(setStatus).catch((requestError) => setError(requestError.message)); }, []);
  return <section className="api-page" id="api-status"><div className="page-heading"><div><p className="eyebrow">CONNECTIVITY / 07</p><h2 className="page-title">API Status</h2><p className="page-description">Provider health and freshness are shown without hiding outages.</p></div></div>{error && <div className="notice error"><strong>Status unavailable</strong><span>{error}</span></div>}<div className="api-cards"><article className="api-card"><Server size={20} /><div><span>BACKEND API</span><strong className="api-online">{error ? "UNAVAILABLE" : "CONNECTED"}</strong><small>Application status endpoint</small></div></article><article className="api-card"><CloudOff size={20} /><div><span>LIVE DATA PROVIDER</span><strong className="api-offline">{status?.status === "ok" ? "CONNECTED" : status?.status === "configured" ? "CONFIGURED" : "NOT CONFIGURED"}</strong><small>{status?.message ?? "Checking provider configuration..."}</small></div></article></div><div className="api-note"><Radio size={18} /><div><strong>Latest successful fetch</strong><p>{status?.data_timestamp ?? "--"}</p><small>Live data is not considered current until timestamp and feature validation pass.</small></div></div></section>;
}