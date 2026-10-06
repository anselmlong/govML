import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getRuns, reportUrl, RunRecord } from './api';

export default function RunsDock() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = () => getRuns().then((data) => mounted && setRuns(data)).catch(() => undefined);
    load();
    const handle = window.setInterval(load, 3500);
    return () => {
      mounted = false;
      window.clearInterval(handle);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const running = runs.filter((run) => run.status === 'running').length;

  return (
    <div className="runs-dock">
      <button onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls="runs-menu">
        Runs <b>{runs.length}</b> {running ? <span>{running} running</span> : null}
      </button>
      {open && (
        <div className="runs-menu" id="runs-menu">
          {runs.slice(0, 8).map((run) => (
            <div key={run.run_id} className="run-row">
              <span className={`run-dot ${run.status}`} />
              <div>
                <b>{run.name}</b>
                <small>{run.status} / {new Date(run.started_at).toLocaleString()}</small>
              </div>
              <Link to={`/run/${run.run_id}`}>Log</Link>
              {run.status === 'completed' && (
                <a href={reportUrl(run.run_id)} target="_blank" rel="noreferrer">
                  Report
                </a>
              )}
            </div>
          ))}
          {!runs.length && <p className="empty-state">No runs yet. Pick a dataset and choose Run ML to start one.</p>}
        </div>
      )}
    </div>
  );
}

