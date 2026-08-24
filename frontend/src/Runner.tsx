import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { friendlyError, getRun, RunRecord, streamRun } from './api';
import Compass from './Compass';
import PipelineStages from './PipelineStages';
import { useCountUp } from './useCountUp';

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
          ← Back
        </Link>
        <div>
          <h1>{run?.name ?? runId}</h1>
        </div>
        <span className={`status-badge ${run?.status ?? 'running'}`}>
          {run?.status === 'running' && <Compass spinning size={13} />} {run?.status ?? 'loading'}
        </span>
      </header>

      {error && <div className="toast">{error}</div>}
      <PipelineStages lines={lines} failed={failed} />

      {completed && run && (
        <section className="completion-card">
          <div>
            <span>Best model</span>
            <b>{run.best_model ?? 'Completed'}</b>
          </div>
          {isClassification ? (
            <>
              <div>
                <span>Accuracy</span>
                <b>{accuracy !== null ? accuracy.toPrecision(4) : 'NA'}</b>
              </div>
              <div>
                <span>F1</span>
                <b>{f1 !== null ? f1.toPrecision(4) : 'NA'}</b>
              </div>
            </>
          ) : (
            <>
              <div>
                <span>MAE</span>
                <b>{mae !== null ? mae.toPrecision(4) : 'NA'}</b>
              </div>
              <div>
                <span>R2</span>
                <b>{r2 !== null ? r2.toPrecision(4) : 'NA'}</b>
              </div>
            </>
          )}
          <a href={`/reports/${run.run_id}`} target="_blank" rel="noreferrer">
            Open report
          </a>
        </section>
      )}

      <section className="log-card">
        <div className="row-between">
          <h2>Run Log</h2>
          <button onClick={() => setShowRaw((value) => !value)}>{showRaw ? 'Hide raw' : 'Show raw'}</button>
        </div>
        {showRaw ? <pre ref={logRef}>{lines.join('\n')}</pre> : <p>{lines[lines.length - 1] ?? 'Waiting for pipeline output...'}</p>}
      </section>
    </main>
  );
}
