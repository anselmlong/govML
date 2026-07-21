import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { getRuns, RunRecord } from './api';

export default function RunsDock() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement | null>(null);

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
    function onPointerDown(event: PointerEvent) {
      if (!anchorRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false);
    }
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  const running = runs.filter((run) => run.status === 'running').length;

  return (
    <div className="runs-anchor" ref={anchorRef}>
      <button className="statusbar-item" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span className={`status-dot ${running ? 'running' : runs.length ? 'completed' : ''}`} />
        Runs {runs.length}
        {running ? ` · ${running} running` : ''}
      </button>
      {open && (
        <div className="runs-menu" role="menu">
          {runs.slice(0, 8).map((run) => (
            <div key={run.run_id} className="run-row">
              <span className={`status-dot ${run.status}`} />
              <div>
                <b>{run.name}</b>
                <small>
                  {run.status} · {new Date(run.started_at).toLocaleString()}
                </small>
              </div>
              <Link to={`/run/${run.run_id}`}>Log</Link>
              {run.status === 'completed' && (
                <a href={`/reports/${run.run_id}`} target="_blank" rel="noreferrer">
                  Report
                </a>
              )}
            </div>
          ))}
          {!runs.length && <p className="empty-note">No runs yet. Pick a dataset and launch the pipeline.</p>}
        </div>
      )}
    </div>
  );
}
