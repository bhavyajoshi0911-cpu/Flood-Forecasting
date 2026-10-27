import { BarChart3, ChartNoAxesCombined, CircleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { getModelMetrics } from "../services/api.js";

const metricLabels = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1 score"], ["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"]];

function formatPeriod(period) {
  if (!period) return "--";
  if (typeof period === "string") return period;
  if (period.training_start && period.training_end) return `${period.training_start} → ${period.training_end}`;
  if (period.start && period.end) return `${period.start} → ${period.end}`;
  return Object.values(period).join(" → ") || "--";
}

function buildChartPoints(values, width = 320, height = 170) {
  if (!Array.isArray(values) || values.length < 2) return "";
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / Math.max(max - min, 1)) * (height - 20) - 10;
      return `${x},${y}`;
    })
    .join(" ");
}

export default function PerformancePage() {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelMetrics().then(setResponse).catch((requestError) => setError(requestError.message));
  }, []);

  const metrics = response?.metrics;
  const unavailable = error || response?.status !== "available";
  const confusionMatrix = Array.isArray(metrics?.confusion_matrix) ? metrics.confusion_matrix : null;
  const rocCurve = metrics?.roc_curve && Array.isArray(metrics.roc_curve.fpr) && Array.isArray(metrics.roc_curve.tpr) ? metrics.roc_curve : null;
  const prCurve = metrics?.precision_recall_curve && Array.isArray(metrics.precision_recall_curve.precision) && Array.isArray(metrics.precision_recall_curve.recall) ? metrics.precision_recall_curve : null;
  const rocPoints = rocCurve ? buildChartPoints(rocCurve.tpr) : "";
  const prPoints = prCurve ? buildChartPoints(prCurve.recall) : "";

  return (
    <section className="performance-page" id="performance">
      <div className="page-heading"><div><p className="eyebrow">MODEL VALIDATION / 05</p><h2 className="page-title">Model Performance</h2><p className="page-description">Measured results from the chronological test partition.</p></div><span className="research-label">NO CLAIMS WITHOUT TEST DATA</span></div>
      <div className="model-periods"><div><span>MODEL</span><strong>{metrics?.model_version ?? metrics?.model_name ?? "--"}</strong></div><div><span>TRAINING PERIOD</span><strong>{formatPeriod(metrics?.training_period)}</strong></div><div><span>VALIDATION PERIOD</span><strong>{formatPeriod(metrics?.validation_period)}</strong></div><div><span>TEST PERIOD</span><strong>{formatPeriod(metrics?.test_period)}</strong></div></div>
      {error && <div className="notice error"><strong>Metrics unavailable</strong><span>{error}</span></div>}
      <div className="metric-results">{metricLabels.map(([key, label]) => <article className="result-metric" key={key}><span>{label}</span><strong>{unavailable || metrics?.[key] == null ? "--" : `${(metrics[key] * 100).toFixed(1)}%`}</strong></article>)}</div>
      <div className="performance-charts">
        <article className="chart-panel">
          <div className="panel-heading"><div><p className="eyebrow">CLASSIFICATION</p><h3>Confusion matrix</h3></div><BarChart3 size={18} /></div>
          {confusionMatrix && confusionMatrix.length === 2 ? (
            <div className="confusion-matrix">
              <div className="confusion-header"><span>Actual ↓ / Predicted →</span><span>0</span><span>1</span></div>
              {confusionMatrix.map((row, rowIndex) => (
                <div className="confusion-row" key={rowIndex}>
                  <span>{rowIndex === 0 ? "0" : "1"}</span>
                  {row.map((value, colIndex) => <span className="confusion-cell" key={`${rowIndex}-${colIndex}`}>{value}</span>)}
                </div>
              ))}
            </div>
          ) : (
            <div className="chart-unavailable"><CircleAlert size={24} /><p>{unavailable ? "No measured test results available" : "Confusion matrix artifact unavailable"}</p></div>
          )}
        </article>
        <article className="chart-panel">
          <div className="panel-heading"><div><p className="eyebrow">RANKING QUALITY</p><h3>ROC and precision-recall</h3></div><ChartNoAxesCombined size={18} /></div>
          {rocCurve || prCurve ? (
            <div className="roc-panel">
              <svg viewBox="0 0 320 170" role="img" aria-label="ROC and precision-recall curve">
                <line x1="20" y1="150" x2="300" y2="150" stroke="#c1d4d5" strokeWidth="1" />
                <line x1="20" y1="20" x2="20" y2="150" stroke="#c1d4d5" strokeWidth="1" />
                {rocPoints && <polyline points={rocPoints} fill="none" stroke="#1f7d82" strokeWidth="2" />}
                {prPoints && <polyline points={prPoints} fill="none" stroke="#d88c5a" strokeWidth="2" />}
              </svg>
              <div className="roc-legend"><span><i className="roc-line roc-line-main" />ROC</span><span><i className="roc-line roc-line-alt" />PR</span></div>
            </div>
          ) : (
            <div className="chart-unavailable"><CircleAlert size={24} /><p>{unavailable ? "Evaluate a real chronological test set first" : "Curve artifacts unavailable"}</p></div>
          )}
        </article>
      </div>
    </section>
  );
}