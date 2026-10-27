import { Clock3 } from "lucide-react";
import { useEffect, useState } from "react";
import { getPredictions } from "../services/api.js";

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getPredictions()
      .then((result) => {
        setRecords(result?.records || result?.data || []);
      })
      .catch((requestError) => {
        setError(requestError.message);
        setRecords([]);
      });
  }, []);

  return <section className="history-page" id="prediction-history"><div className="page-heading"><div><p className="eyebrow">DECISION LOG / 08</p><h2 className="page-title">Prediction History</h2><p className="page-description">Stored forecast requests with source and model provenance.</p></div><span className="research-label">AUDIT TRAIL</span></div>{error && <div className="notice error"><strong>Prediction history unavailable</strong><span>{error}</span></div>}<div className="historical-table-wrap"><table><thead><tr><th>Date</th><th>Type</th><th>Risk</th><th>Probability</th><th>Data source</th><th>Model</th></tr></thead><tbody>{records.length ? records.map((record) => <tr key={record.id ?? record.timestamp}><td>{record.timestamp ?? "--"}</td><td>{record.prediction_type ?? "--"}</td><td>{record.risk_label ?? "--"}</td><td>{record.probability == null ? "--" : `${(record.probability * 100).toFixed(1)}%`}</td><td>{record.data_source ?? "--"}</td><td>{record.model_version ?? "--"}</td></tr>) : <tr><td colSpan="6"><div className="table-empty"><Clock3 size={24} /><p>{error ? "Stored predictions could not be loaded" : "No predictions recorded"}</p><small>Predictions will appear after a validated forecast is persisted.</small></div></td></tr>}</tbody></table></div></section>;
}