import { Clock3 } from "lucide-react";
import { useEffect, useState } from "react";
import { getPredictions } from "../services/api.js";

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [retrospectiveRecords, setRetrospectiveRecords] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getPredictions()
      .then((result) => {
        setRecords(result?.records || result?.data || []);
        setRetrospectiveRecords(result?.retrospective_records || []);
      })
      .catch((requestError) => {
        setError(requestError.message);
        setRecords([]);
        setRetrospectiveRecords([]);
      });
  }, []);

  return <section className="history-page" id="prediction-history">
    <div className="page-heading"><div><p className="eyebrow">DECISION LOG / 08</p><h2 className="page-title">Prediction History</h2><p className="page-description">Saved forecast requests and clearly separated retrospective model results.</p></div><span className="research-label">AUDIT TRAIL</span></div>
    {error && <div className="notice error"><strong>Prediction history unavailable</strong><span>{error}</span></div>}
    <div className="historical-table-wrap"><table><thead><tr><th>Date</th><th>Type</th><th>Risk</th><th>Probability</th><th>Data source</th><th>Model</th></tr></thead><tbody>{records.length ? records.map((record) => <tr key={record.id ?? record.timestamp}><td>{record.timestamp ?? "--"}</td><td>{record.prediction_type ?? "--"}</td><td>{record.risk_label ?? "--"}</td><td>{record.probability == null ? "--" : `${(record.probability * 100).toFixed(1)}%`}</td><td>{record.data_source ?? "--"}</td><td>{record.model_version ?? "--"}</td></tr>) : <tr><td colSpan="6"><div className="table-empty"><Clock3 size={24} /><p>{error ? "Stored predictions could not be loaded" : "No saved forecasts yet"}</p><small>Successful manual or configured live forecasts will appear here. Retrospective test results are listed separately below.</small></div></td></tr>}</tbody></table></div>
    <div className="page-heading retrospective-heading"><div><p className="eyebrow">MODEL EVALUATION DATA</p><h3 className="page-title">Retrospective test predictions</h3><p className="page-description">Actual outputs from running the saved model against its held-out test features. These are historical evaluation results, not current or live forecasts.</p></div><span className="research-label">{retrospectiveRecords.length} TEST ROWS</span></div>
    <div className="historical-table-wrap"><table><thead><tr><th>Test date</th><th>Station</th><th>Type</th><th>Model risk</th><th>Probability</th><th>Model</th></tr></thead><tbody>{retrospectiveRecords.length ? retrospectiveRecords.map((record) => <tr key={record.id}><td>{record.timestamp}</td><td>{record.station_id}</td><td>{record.prediction_type}</td><td>{record.risk_label}</td><td>{record.probability == null ? "--" : `${(record.probability * 100).toFixed(1)}%`}</td><td>{record.model_version}</td></tr>) : <tr><td colSpan="6"><div className="table-empty"><Clock3 size={24} /><p>{error ? "Retrospective results unavailable" : "No held-out test results available"}</p><small>Check the saved model and test feature artifacts.</small></div></td></tr>}</tbody></table></div>
    <p className="alert-disclaimer">Test predictions are generated from held-out model features only. They are not official event labels, current observations, or emergency warnings.</p>
  </section>;
}