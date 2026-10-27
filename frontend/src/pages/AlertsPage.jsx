import { Bell, CircleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { getAlerts } from "../services/api.js";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getAlerts()
      .then((response) => {
        const nextAlerts = response.alerts || response.data || [];
        setAlerts(nextAlerts);
      })
      .catch((requestError) => {
        setError(requestError.message);
        setAlerts([]);
      });
  }, []);

  return <section className="alerts-page" id="alerts"><div className="page-heading"><div><p className="eyebrow">MONITORING / 09</p><h2 className="page-title">Alerts</h2><p className="page-description">Research notifications generated from configured forecast rules.</p></div><span className="research-label">DEMO ONLY</span></div>{error && <div className="notice error"><strong>Alerts unavailable</strong><span>{error}</span></div>}<div className="alerts-panel">{alerts.length ? alerts.map((alert) => <article className="alert-row" key={alert.id}><Bell size={17} /><div><strong>{alert.severity ?? "Notice"}</strong><p>{alert.message ?? "--"}</p><small>{alert.created_at ?? "--"}</small></div></article>) : <div className="empty-state"><CircleAlert size={27} /><p>{error ? "Alert records could not be loaded" : "No alerts recorded"}</p><small>Alerts will appear only when an actual stored prediction meets a configured rule.</small></div>}</div><p className="alert-disclaimer">These are research/demo notifications and are not official emergency warnings.</p></section>;
}