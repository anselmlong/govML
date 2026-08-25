import { useEffect, useRef, useState } from 'react';
import Compass from './Compass';
import { CatalogBuildStatus, friendlyError, getCatalogBuildStatus, startCatalogBuild } from './api';

const STAGES: { key: 'ingest' | 'embed' | 'map'; label: string }[] = [
  { key: 'ingest', label: 'Ingest' },
  { key: 'embed', label: 'Embed' },
  { key: 'map', label: 'Map' }
];
const ORDER = STAGES.map((stage) => stage.key);

function stageState(stage: 'ingest' | 'embed' | 'map', phase: CatalogBuildStatus['phase']) {
  if (!phase) return 'pending';
  const phaseIndex = ORDER.indexOf(phase === 'done' ? 'map' : phase);
  const stageIndex = ORDER.indexOf(stage);
  if (phase === 'done') return 'completed';
  if (stageIndex < phaseIndex) return 'completed';
  if (stageIndex === phaseIndex) return 'running';
  return 'pending';
}

export default function CatalogBuild({ onComplete }: { onComplete: () => void }) {
  const [status, setStatus] = useState<CatalogBuildStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const notifiedRef = useRef(false);

  useEffect(() => {
    getCatalogBuildStatus().then(setStatus).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!status?.running) return;
    const handle = window.setInterval(() => {
      getCatalogBuildStatus().then(setStatus).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(handle);
  }, [status?.running]);

  useEffect(() => {
    if (status && !status.running && status.phase === 'done' && !notifiedRef.current) {
      notifiedRef.current = true;
      onComplete();
    }
  }, [status, onComplete]);

  async function build() {
    setStarting(true);
    setError('');
    try {
      await startCatalogBuild();
      setStatus(await getCatalogBuildStatus());
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setStarting(false);
    }
  }

  const active = Boolean(status?.running) || status?.phase === 'done';

  if (!active) {
    return (
      <div className="catalog-build">
        <p>
          govML pulls the current data.gov.sg dataset listing (about 4,600 datasets) and builds a local semantic index. This
          runs once, entirely on your machine, and takes a few minutes.
        </p>
        <button onClick={build} disabled={starting}>
          {starting && <Compass spinning size={14} />} {starting ? 'Starting' : 'Build catalog'}
        </button>
        {(error || status?.error) && <p className="build-error">{error || status?.error}</p>}
      </div>
    );
  }

  return (
    <div className="catalog-build">
      <div className="catalog-build-stages">
        {STAGES.map((stage) => {
          const state = stageState(stage.key, status?.phase ?? null);
          return (
            <div className={`stage ${state}`} key={stage.key}>
              <span>
                {state === 'completed' ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <path d="M2.5 6.2L4.8 8.5L9.5 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : state === 'running' ? (
                  <Compass spinning size={12} />
                ) : (
                  ORDER.indexOf(stage.key) + 1
                )}
              </span>
              <b>{stage.label}</b>
            </div>
          );
        })}
      </div>
      <p className="catalog-build-detail">
        {status?.phase === 'done'
          ? `Catalog built${status.total_ingested ? ` — ${status.total_ingested.toLocaleString()} datasets` : ''}.`
          : status?.detail || 'Starting…'}
      </p>
    </div>
  );
}
