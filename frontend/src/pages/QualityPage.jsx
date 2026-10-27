import { CircleCheck, CircleX, Database } from "lucide-react";
import { useEffect, useState } from "react";
import { get } from "../services/api.js";

export default function QualityPage() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { get("/dataset/quality").then(setReport).catch((requestError) => setError(requestError.message)); }, []);
  const status = report?.status;
  const good = status === "valid";
  return <section className="quality-page" id="quality"><div className="page-heading"><div><p className="eyebrow">DATA INTEGRITY / 06</p><h2 className="page-title">Data Quality</h2><p className="page-description">Validation findings for the configured source table.</p></div><span className={`quality-badge ${good ? "good" : "pending"}`}>{status ?? "CHECKING"}</span></div>{error && <div className="notice error"><strong>Quality report unavailable</strong><span>{error}</span></div>}<div className="quality-overview"><div><Database size={18} /><span>TOTAL RECORDS</span><strong>{report?.row_count ?? "--"}</strong></div><div><span>INVALID TIMESTAMPS</span><strong>{report?.invalid_timestamps ?? "--"}</strong></div><div><span>DUPLICATE ROWS</span><strong>{report?.duplicate_rows ?? "--"}</strong></div><div><span>NEGATIVE RAINFALL</span><strong>{report?.negative_rainfall ?? "--"}</strong></div></div><div className="quality-detail"><div className="panel-heading"><h3>Validation findings</h3></div>{report ? <div className="finding-list">{[...(report.errors || []).map((item) => ({ type: "error", item })), ...(report.warnings || []).map((item) => ({ type: "warning", item }))].map(({ type, item }) => <div className={`finding ${type}`} key={item}>{type === "error" ? <CircleX size={17} /> : <CircleCheck size={17} />}<span>{item}</span></div>)}{!(report.errors?.length || report.warnings?.length) && <div className="finding success"><CircleCheck size={17} /><span>No validation findings.</span></div>}</div> : <div className="empty-state"><p>Loading quality report...</p></div>}</div></section>;
}