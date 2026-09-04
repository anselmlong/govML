import { Fragment } from 'react';
import { CatalogResult, Insight, MapDataset } from './api';

export function hasTrainedResult(item: CatalogResult | MapDataset): boolean {
  if (item.insight) return true;
  return 'has_insight' in item && Boolean((item as MapDataset).has_insight);
}

export function fmtMetric(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return 'NA';
  return Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 0 }) : value.toPrecision(3);
}

function metricRows(insight: Insight): { label: string; value: string }[] {
  const m = insight.metrics || {};
  if (insight.task_type === 'regression') {
    return [
      { label: 'MAE', value: fmtMetric(m.MAE) },
      { label: 'R²', value: fmtMetric(m.R2) }
    ];
  }
  return [
    { label: m.roc_auc !== undefined ? 'ROC AUC' : 'F1', value: fmtMetric(m.roc_auc !== undefined ? m.roc_auc : m.f1) },
    { label: 'Accuracy', value: fmtMetric(m.acc) }
  ];
}

export default function InsightCard({ insight }: { insight: Insight }) {
  return (
    <div className={`insight-card tone-${insight.verdict_tone}`}>
      <div className="row-between">
        <p className="section-label">ML result — already trained</p>
        <span className={`verdict-pill ${insight.verdict_tone}`}>{insight.verdict_tone}</span>
      </div>
      <p className="insight-target">
        Predicts <b>{insight.target}</b> <span className="muted">({insight.task_type})</span>
      </p>
      <div className="meta-grid">
        <span>Best model</span>
        <b>{insight.best_model}</b>
        {metricRows(insight).map((row) => (
          <Fragment key={row.label}>
            <span>{row.label}</span>
            <b>{row.value}</b>
          </Fragment>
        ))}
      </div>
      {insight.verdict_text && <p className="insight-verdict">{insight.verdict_text}</p>}
      {insight.correlations?.length ? (
        <>
          <p className="section-label">Cross-dataset correlations</p>
          <ul className="reason-list">
            {insight.correlations.slice(0, 3).map((c) => (
              <li key={c.dataset_id}>
                {c.name}: r={c.r >= 0 ? '+' : ''}
                {c.r.toFixed(2)} over {c.overlap_months} months
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
