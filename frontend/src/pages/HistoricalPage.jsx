import { BarChart3, ChevronLeft, ChevronRight, Filter, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { getHistorical } from "../services/api.js";

const emptyFilters = { start_date: "", end_date: "", station_id: "", flood_status: "" };

export default function HistoricalPage() {
  const [filters, setFilters] = useState(emptyFilters);
  const [page, setPage] = useState(1);
  const [records, setRecords] = useState([]);
  const [total, setTotal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function search(event, nextPage = 1) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await getHistorical({ ...filters, page: nextPage });
      setRecords(response.records || response.data || []);
      setTotal(response.total ?? null);
      setPage(nextPage);
    } catch (requestError) {
      setRecords([]);
      setTotal(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="historical-page" id="historical">
      <div className="page-heading"><div><p className="eyebrow">OBSERVATION ARCHIVE / 03</p><h2 className="page-title">Historical Data</h2><p className="page-description">Inspect source observations and the non-authoritative research target.</p></div><span className="research-label">PAGINATED</span></div>
      <form className="filter-bar" onSubmit={search}><Filter size={16} /><label>From<input type="date" value={filters.start_date} onChange={(event) => setFilters({ ...filters, start_date: event.target.value })} /></label><label>To<input type="date" value={filters.end_date} onChange={(event) => setFilters({ ...filters, end_date: event.target.value })} /></label><label>Station<input placeholder="Station ID" value={filters.station_id} onChange={(event) => setFilters({ ...filters, station_id: event.target.value })} /></label><label>Flood status<select value={filters.flood_status} onChange={(event) => setFilters({ ...filters, flood_status: event.target.value })}><option value="">All statuses</option><option value="0">No flood</option><option value="1">Flood</option></select></label><button className="filter-button" type="submit" disabled={loading}>{loading ? <LoaderCircle className="spin" size={15} /> : "Apply filters"}</button></form>
      {error && <div className="notice error"><strong>Historical data unavailable</strong><span>{error}</span></div>}
      <div className="historical-table-wrap"><table><thead><tr><th>Date</th><th>Station</th><th>Rainfall</th><th>Discharge</th><th>Water level</th><th>Research target</th><th>Model prediction</th></tr></thead><tbody>{records.length ? records.map((record) => <tr key={`${record.timestamp}-${record.station_id}`}><td>{record.timestamp ?? "--"}</td><td>{record.station_id ?? "--"}</td><td>{record.rainfall ?? "--"}</td><td>{record.river_discharge ?? "--"}</td><td>{record.water_level ?? "--"}</td><td>{record.flood_observed ?? "--"}</td><td>{record.predicted_flood ?? "--"}</td></tr>) : <tr><td colSpan="7"><div className="table-empty"><p>{loading ? "Loading observations..." : "No historical observations returned"}</p><small>{error ? "The backend request could not be completed." : "Apply a filter or connect a validated historical dataset."}</small></div></td></tr>}</tbody></table></div>
      <div className="table-footer"><span>{total == null ? "Total records: --" : `Total records: ${total}`}</span><div><button className="icon-button" title="Previous page" disabled={page <= 1 || loading} onClick={() => search(null, page - 1)}><ChevronLeft size={16} /></button><span>Page {page}</span><button className="icon-button" title="Next page" disabled={loading || !records.length || (total != null && page * 20 >= total)} onClick={() => search(null, page + 1)}><ChevronRight size={16} /></button></div></div>
      <div className="chart-empty"><BarChart3 size={21} /><div><strong>Historical trends</strong><p>Rainfall, discharge, and water-level charts will appear when the backend returns validated time-series data.</p></div></div>
    </section>
  );
}