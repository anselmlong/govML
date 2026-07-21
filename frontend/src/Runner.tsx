import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getRun, RunRecord, streamRun } from './api';
import PipelineStages from './PipelineStages';

export default function Runner() {
  const { runId = '' } = useParams();
  const [run, setRun] = useState<RunRecord | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [showRaw, setShowRaw] = useState(false);
  const [error, setError] = useState('');
  const logRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (!runId) return;
    getRun(runId).then(setRun).catch((err: unknown) => setError(String(err)));
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

  return (
    <div className="runner-shell">
      <header className="runner-topbar">
        <Link to="/" className="back-link">
          ← Catalog
        </Link>
        <h1>{run?.name ?? runId}</h1>
        <span className={`status-chip ${run?.status ?? 'running'}`}>
          <span className={`status-dot ${run?.status ?? 'running'}`} />
          {run?.status ?? 'loading'}
        </span>
      </header>

      <main className="runner-body">
        {error && <div className="toast">{error}</div>}

        <section className="panel">
          <div className="panel-head">
            <h2>Pipeline</h2>
          </div>
          <div className="panel-body">
            <PipelineStages lines={lines} failed={failed} />
          </div>
        </section>

        {completed && run && (
          <section className="panel">
            <div className="panel-head">
              <h2>Result</h2>
              <a className="source-link" href={`/reports/${run.run_id}`} target="_blank" rel="noreferrer">
                Open full report ↗
              </a>
            </div>
            <div className="panel-body">
              <div className="completion-grid">
                <div className="fact">
                  <span>Best model</span>
                  <b>{run.best_model ?? 'Completed'}</b>
                </div>
                <div className="fact">
                  <span>MAE</span>
                  <b>{run.mae?.toPrecision(4) ?? '—'}</b>
                </div>
                <div className="fact">
                  <span>R²</span>
                  <b>{run.r2?.toPrecision(4) ?? '—'}</b>
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="panel">
          <div className="panel-head">
            <h2>Run log</h2>
            <button className="button" onClick={() => setShowRaw((value) => !value)}>
              {showRaw ? 'Latest line' : 'Full log'}
            </button>
          </div>
          {showRaw ? (
            <pre className="log-pre" ref={logRef}>
              {lines.join('\n')}
            </pre>
          ) : (
            <div className="panel-body">
              <p className="log-line-latest">{lines[lines.length - 1] ?? 'Waiting for pipeline output...'}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
