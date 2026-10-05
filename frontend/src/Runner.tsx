import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { friendlyError, getRun, reportUrl, RunRecord, streamRun } from './api';
import Compass from './Compass';
import PipelineStages from './PipelineStages';
import { useCountUp } from './useCountUp';

// Errors in target units (e.g. dollars) read better grouped than in
// exponent form; ratios and scores keep four significant figures.
function formatMetric(value: number | null) {
  if (value === null) return 'NA';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toPrecision(4);
}

export default function Runner() {
  const { runId = '' } = useParams();
  const [run, setRun] = useState<RunRecord | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [showRaw, setShowRaw] = useState(false);
  const [error, setError] = useState('');
  const logRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (!runId) return;
    getRun(runId).then(setRun).catch((err: unknown) => setError(friendlyError(err)));
    const source = streamRun(
      runId,
      (line) => setLines((current) => [...current, line]),
      (updated) => setRun(updated)
    );
    return () => source.close();
  }, [runId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines, showRaw]);

  const completed = run?.status === 'completed';
  const failed = run?.status === 'failed';
  const isClassification = run?.mae === null || run?.mae === undefined;
  const mae = useCountUp(completed ? run?.mae : null);
  const r2 = useCountUp(completed ? run?.r2 : null);
  const accuracy = useCountUp(completed ? run?.accuracy : null);
  const f1 = useCountUp(completed ? run?.f1 : null);

  return (
    <main className="runner-shell">
      <header className="runner-header">
        <Link to="/" className="back-link">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M8.5 2.5L4 7l4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Back
        </Link>
        <div>
          <h1>{run?.name ?? runId}</h1>
        </div>
        <span className={`status-badge ${run?.status ?? 'running'}`} role="status">
          {run?.status === 'running' && <Compass spinning size={13} />} {run?.status ?? 'loading'}
        </span>
      </header>

      {error && <div className="toast" role="alert">{error}</div>}
      <PipelineStages lines={lines} failed={failed} />

      {completed && run && (
        <section className="completion-card completed">
          <div>
            <span>Best model</span>
            <b>{run.best_model ?? 'Completed'}</b>
          </div>
          {isClassification ? (
            <>
              <div>
                <span>Accuracy</span>
                <b>{formatMetric(accuracy)}</b>
              </div>
              <div>
                <span>F1</span>
                <b>{formatMetric(f1)}</b>
              </div>
            </>
          ) : (
            <>
              <div>
                <span>MAE</span>
                <b>{formatMetric(mae)}</b>
              </div>
              <div>
                <span>R2</span>
                <b>{formatMetric(r2)}</b>
              </div>
            </>
          )}
          <a href={reportUrl(run.run_id)} target="_blank" rel="noreferrer">
            Open report
          </a>
        </section>
      )}

      <section className="log-card">
        <div className="row-between">
          <h2>Run Log</h2>
          <button onClick={() => setShowRaw((value) => !value)}>{showRaw ? 'Hide raw' : 'Show raw'}</button>
        </div>
        {showRaw ? <pre ref={logRef}>{lines.join('\n')}</pre> : <p>{lines[lines.length - 1] ?? 'Waiting for pipeline output\u2026'}</p>}
      </section>
    </main>
  );
}
