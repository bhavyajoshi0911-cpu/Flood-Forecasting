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

  return <section className="alerts-page" id="alerts"><div className="page-heading"><div><p className="eyebrow">MONITORING / 09</p><h2 className="page-title">CWC Station Warnings</h2><p className="page-description">Current warning statuses reported by the Central Water Commission feed.</p></div><span className="research-label">SOURCE FEED</span></div>{error && <div className="notice error"><strong>CWC warning feed unavailable</strong><span>{error}</span></div>}<div className="alerts-panel">{alerts.length ? alerts.map((alert) => <article className="alert-row" key={alert.id}><Bell size={17} /><div><strong>{alert.severity ?? "Notice"} · {alert.station_code ?? "Station"}</strong><p>{alert.message ?? "--"}</p><small>Retrieved {alert.created_at ?? "--"} · CWC source status; source observation time is not provided.</small></div></article>) : <div className="empty-state"><CircleAlert size={27} /><p>{error ? "CWC warning data could not be loaded" : "No CWC warning records returned"}</p><small>This is the live source feed, not alerts generated from project-model predictions.</small></div>}</div>{!error && <p className="alert-source-link"><a href="https://ffs.india-water.gov.in/ffm/api/station-water-level-above-warning/" target="_blank" rel="noreferrer">Open the Central Water Commission source feed</a></p>}<p className="alert-disclaimer">CWC station warnings are reproduced from the source feed. Verify station identity, basin, and timing with CWC; this project is not an emergency warning service.</p></section>;
}