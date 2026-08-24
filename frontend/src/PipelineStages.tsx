import { useMemo } from 'react';

type StageStatus = 'pending' | 'running' | 'completed' | 'failed';

interface Stage {
  key: string;
  label: string;
}

interface Facts {
  rows?: string;
  columns?: string;
  plan?: string;
  target?: string;
  taskType?: string;
  features?: string;
  added?: string;
  metrics?: string;
  topFeatures: string[];
}

const STAGES: Stage[] = [
  { key: 'research', label: 'Research' },
  { key: 'fetch', label: 'Fetch' },
  { key: 'preprocess', label: 'Clean' },
  { key: 'target', label: 'Target' },
  { key: 'context', label: 'Context' },
  { key: 'train', label: 'Train' },
  { key: 'diagnose', label: 'Diagnose' },
  { key: 'persist', label: 'Save' },
  { key: 'report', label: 'Report' }
];

function parseStage(text: string): number | null {
  const match = text.match(/^\s*\[(\d+)\/(\d+)\]\s*(.+)$/);
  if (!match) return null;
  return Number(match[1]) - 1;
}

export default function PipelineStages({ lines, failed = false }: { lines: string[]; failed?: boolean }) {
  const parsed = useMemo(() => {
    const statuses: StageStatus[] = STAGES.map(() => 'pending');
    const facts: Facts = { topFeatures: [] };
    let current = -1;
    let inTopFeatures = false;

    for (const line of lines) {
      const idx = parseStage(line);
      if (idx !== null) {
        if (current >= 0 && statuses[current] === 'running') statuses[current] = 'completed';
        current = idx;
        statuses[current] = 'running';
        inTopFeatures = false;
      }
      if (line.includes('fetched ') && line.includes(' rows')) {
        facts.rows = line.replace(/^.*fetched\s+/, '').replace(/\s+rows.*$/, '');
      }
      const shape = line.match(/raw shape:\s*\((\d+),\s*(\d+)\)/);
      if (shape) {
        facts.rows = shape[1];
        facts.columns = shape[2];
      }
      const total = line.match(/total rows available:\s*(\d+)/);
      if (total) facts.rows = total[1];
      const plan = line.match(/^plan:\s*(.+)$/);
      if (plan) facts.plan = plan[1];
      const target = line.match(/^chosen:\s*(.+)\.$/);
      if (target) facts.target = target[1];
      const task = line.match(/^task type:\s*(.+)\.$/);
      if (task) facts.taskType = task[1];
      const features = line.match(/^features:\s*(.+)$/);
      if (features) facts.features = features[1];
      if (line.startsWith('added ') && line.includes('columns')) facts.added = line;
      if (line.startsWith('done. best:')) facts.metrics = line.replace('done. best:', '').trim();
      if (line.startsWith('top features:')) {
        inTopFeatures = true;
        continue;
      }
      if (inTopFeatures && line.trim().startsWith('-')) facts.topFeatures.push(line.trim().slice(1).trim());
      if (inTopFeatures && line.trim() && !line.trim().startsWith('-')) inTopFeatures = false;
    }

    if (failed) {
      const running = statuses.findIndex((s) => s === 'running');
      if (running >= 0) statuses[running] = 'failed';
    } else if (lines.some((line) => line.startsWith('done. best:'))) {
      statuses.fill('completed');
    }
    return { statuses, facts };
  }, [lines, failed]);

  return (
    <section className="stage-card">
      <div className="stage-grid">
        {STAGES.map((stage, index) => {
          const status = parsed.statuses[index];
          return (
            <div className={`stage ${status}`} key={stage.key}>
              <span key={status}>
                {status === 'completed' ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <path d="M2.5 6.2L4.8 8.5L9.5 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : status === 'failed' ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <path d="M3 3L9 9M9 3L3 9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                  </svg>
                ) : (
                  index + 1
                )}
              </span>
              <b>{stage.label}</b>
            </div>
          );
        })}
      </div>
      <div className="facts-grid">
        <span>Rows</span>
        <b>{parsed.facts.rows ?? 'Pending'}</b>
        <span>Columns</span>
        <b>{parsed.facts.columns ?? 'Pending'}</b>
        <span>Target</span>
        <b>{parsed.facts.target ?? 'Pending'}</b>
        <span>Task</span>
        <b>{parsed.facts.taskType ?? 'Pending'}</b>
        <span>Features</span>
        <b>{parsed.facts.features ?? 'Pending'}</b>
        <span>Metrics</span>
        <b>{parsed.facts.metrics ?? 'Pending'}</b>
      </div>
      {parsed.facts.topFeatures.length > 0 && (
        <div className="top-features">
          {parsed.facts.topFeatures.slice(0, 6).map((feature) => (
            <span key={feature}>{feature}</span>
          ))}
        </div>
      )}
    </section>
  );
}

