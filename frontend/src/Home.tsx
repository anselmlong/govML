import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import CatalogBuild from './CatalogBuild';
import Compass from './Compass';
import MapCanvas, { MapCanvasHandle } from './MapCanvas';
import {
  askCatalog,
  CatalogResult,
  CatalogStats,
  DatasetInfo,
  friendlyError,
  getCatalogMap,
  getCatalogStats,
  getDatasetInfo,
  getSuitabilityLookup,
  MapDataset,
  searchCatalog,
  startRun,
  Suitability
} from './api';
import RunsDock from './RunsDock';
import { applyTheme, initialTheme, ThemeMode } from './theme';

type Mode = 'search' | 'ask';
type ColorMode = 'agency' | 'fit';

const PALETTE = ['#2f6e7f', '#bd5029', '#a5824a', '#5b4636', '#3f7d52', '#6b4e9e', '#2b5a8c', '#8c4b3e', '#4b7a8c', '#7a6a3a'];
const FIT_COLORS: Record<string, string> = {
  good: '#3f7d52',
  okay: '#a5824a',
  marginal: '#7c8f93',
  blocked: '#bd5029'
};

const ASK_EXAMPLES = [
  'How does HDB resale price vary by town?',
  'What datasets correlate with tourism arrivals?',
  'Which factors predict traffic accident rates?'
];

function toneLabel(suitability?: Suitability | null, fallbackScore?: number | null, fallbackTone?: string | null) {
  if (suitability) return `${suitability.tone} ${suitability.score}`;
  if (fallbackScore !== undefined && fallbackScore !== null) return `${fallbackTone ?? 'fit'} ${fallbackScore}`;
  return 'unscored';
}

function renderAnswer(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) return <b key={index}>{part.slice(2, -2)}</b>;
    return <span key={index}>{part}</span>;
  });
}

export default function Home() {
  const navigate = useNavigate();
  const mapRef = useRef<MapCanvasHandle | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);

  const [nodes, setNodes] = useState<MapDataset[]>([]);
  const [stats, setStats] = useState<CatalogStats | null>(null);
  const [mode, setMode] = useState<Mode>('search');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<CatalogResult[]>([]);
  const [answer, setAnswer] = useState('');
  const [answering, setAnswering] = useState(false);
  const [selected, setSelected] = useState<CatalogResult | null>(null);
  const [selectedInfo, setSelectedInfo] = useState<DatasetInfo | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [agencyFilter, setAgencyFilter] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>('agency');
  const [suitabilityLookup, setSuitabilityLookup] = useState<Record<string, Suitability>>({});
  const [themeMode, setThemeMode] = useState<ThemeMode>(initialTheme());
  const [maxRows, setMaxRows] = useState(0);
  const [noResearch, setNoResearch] = useState(true);
  const [noCorrelation, setNoCorrelation] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [railOpen, setRailOpen] = useState(false);
  const [loading, setLoading] = useState(true);

  const loadCatalog = useCallback(() => {
    return Promise.allSettled([getCatalogMap(), getCatalogStats(), getSuitabilityLookup()]).then(
      ([mapResult, statsResult, fitResult]) => {
        if (mapResult.status === 'fulfilled') setNodes(mapResult.value);
        if (statsResult.status === 'fulfilled') setStats(statsResult.value);
        if (fitResult.status === 'fulfilled') setSuitabilityLookup(fitResult.value);
      }
    );
  }, []);

  useEffect(() => {
    let mounted = true;
    loadCatalog()
      .catch((err: unknown) => setError(friendlyError(err)))
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, [loadCatalog]);

  const handleCatalogBuilt = useCallback(() => {
    setLoading(true);
    loadCatalog()
      .catch((err: unknown) => setError(friendlyError(err)))
      .finally(() => setLoading(false));
  }, [loadCatalog]);

  const resetView = useCallback(() => {
    mapRef.current?.resetView();
  }, []);

  // fly the camera to a dot when it's chosen from the results list so users
  // see WHERE in semantic space their pick lives
  const flyToNode = useCallback((node: MapDataset) => {
    mapRef.current?.flyTo(node);
  }, []);

  useEffect(() => {
    if (mode !== 'search') return;
    const handle = window.setTimeout(() => {
      if (query.trim().length < 2) {
        setResults([]);
        return;
      }
      searchCatalog(query, 10).then(setResults).catch((err: unknown) => setError(friendlyError(err)));
    }, 280);
    return () => window.clearTimeout(handle);
  }, [query, mode]);

  useEffect(() => {
    if (selected) closeButtonRef.current?.focus();
  }, [selected]);

  const agencyColors = useMemo(() => {
    const agencies = stats?.top_agencies.map((a) => a.agency) ?? [];
    const map = new Map<string, string>();
    agencies.forEach((agency, index) => map.set(agency, PALETTE[index % PALETTE.length]));
    return map;
  }, [stats]);

  const selectedIds = useMemo(() => new Set(results.map((r) => r.dataset_id)), [results]);

  const dotColorById = useMemo(() => {
    const map = new Map<string, string>();
    for (const node of nodes) {
      const fit = suitabilityLookup[node.dataset_id];
      const color =
        colorMode === 'fit'
          ? FIT_COLORS[fit?.tone ?? node.suitability_tone ?? 'marginal'] ?? '#737373'
          : agencyColors.get(node.agency) ?? PALETTE[Math.abs(hashCode(node.agency)) % PALETTE.length];
      map.set(node.dataset_id, color);
    }
    return map;
  }, [nodes, colorMode, agencyColors, suitabilityLookup]);

  const topFit = useMemo(
    () =>
      [...nodes]
        .filter((n) => typeof n.suitability_score === 'number')
        .sort((a, b) => (b.suitability_score ?? 0) - (a.suitability_score ?? 0))
        .slice(0, 8),
    [nodes]
  );

  function nodeColor(node: CatalogResult) {
    const fit = suitabilityLookup[node.dataset_id];
    if (colorMode === 'fit') return FIT_COLORS[fit?.tone ?? node.suitability_tone ?? 'marginal'] ?? '#737373';
    return agencyColors.get(node.agency) ?? PALETTE[Math.abs(hashCode(node.agency)) % PALETTE.length];
  }

  function hashCode(value: string) {
    return Array.from(value || 'unknown').reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) | 0, 7);
  }

  const chooseDataset = useCallback(
    async (dataset: CatalogResult, fly = false) => {
      openerRef.current = document.activeElement as HTMLElement;
      setSelected(dataset);
      setSelectedInfo(null);
      setExpanded(false);
      setDetailLoading(true);
      setError('');
      if (fly) {
        const node = nodes.find((n) => n.dataset_id === dataset.dataset_id);
        if (node) flyToNode(node);
      }
      try {
        const info = await getDatasetInfo(dataset.dataset_id);
        setSelectedInfo(info);
      } catch (err) {
        setError(friendlyError(err));
      } finally {
        setDetailLoading(false);
      }
    },
    [nodes, flyToNode]
  );

  function closeDetail() {
    setSelected(null);
    setSelectedInfo(null);
    openerRef.current?.focus();
  }

  async function submitAsk(question = query) {
    const q = question.trim();
    if (!q) return;
    setMode('ask');
    setQuery(q);
    setAnswering(true);
    setAnswer('');
    setError('');
    try {
      const res = await askCatalog(q, 6);
      setAnswer(res.answer);
      setResults(res.datasets);
      setAskResults(res.datasets);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setAnswering(false);
    }
  }

  async function launchRun() {
    if (!selected) return;
    setLaunching(true);
    setError('');
    try {
      const res = await startRun({
        resource_id: selected.dataset_id,
        name: selectedInfo?.name ?? selected.name,
        max_rows: maxRows,
        no_research: noResearch,
        no_correlation: noCorrelation
      });
      navigate(`/run/${res.run_id}`);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLaunching(false);
    }
  }

  function toggleTheme() {
    const next = themeMode === 'light' ? 'dark' : 'light';
    setThemeMode(next);
    applyTheme(next);
  }

  const searching = mode === 'search' && query.trim().length >= 2;
  const browseItems = topFit.length ? topFit : nodes.slice(0, 8);
  // switching to Ask keeps its own results; switching back to Search must not
  // show stale ask-results under a search box that hasn't re-fired yet.
  const [askResults, setAskResults] = useState<CatalogResult[]>([]);
  const listItems =
    mode === 'ask' ? askResults : searching ? results : browseItems;
  const catalogEmpty = !loading && nodes.length === 0;
  const showSkeleton = loading && !searching && listItems.length === 0;
  const info = selectedInfo;
  const fit = selected ? suitabilityLookup[selected.dataset_id] ?? info?.suitability : null;
  const description = info?.description || selected?.description || '';
  const displayedDescription = expanded || description.length < 360 ? description : `${description.slice(0, 360)}...`;

  return (
    <main className="home-shell">
      <section className="map-stage" aria-label="Semantic dataset map">
        <MapCanvas
          ref={mapRef}
          nodes={nodes}
          colorFor={dotColorById}
          agencyFilter={agencyFilter}
          selectedId={selected?.dataset_id}
          matchedIds={selectedIds}
          hasResults={results.length > 0}
          onSelect={chooseDataset}
        />
        {loading && (
          <div className="map-loading">
            <Compass spinning size={14} />
            Charting{stats?.total ? ` ${stats.total.toLocaleString()} datasets` : ''}&hellip;
          </div>
        )}
        <button className="map-compass" onClick={resetView} aria-label="Reset map view" title="Reset view">
          <Compass size={38} signature />
        </button>
        <div className="map-hint">Scroll to zoom &middot; drag to pan &middot; click a marker</div>
      </section>

      <header className="topbar">
        <a className="brand-lockup" href="/">
          <span className="brand-mark" aria-hidden="true">
            <Compass size={26} />
          </span>
          <span className="brand-copy">
            <h1>govML</h1>
            <span>Chart of Singapore&rsquo;s open data</span>
          </span>
        </a>
        <div className="nav-actions">
          <a className="attribution-link" href="https://data.gov.sg" target="_blank" rel="noreferrer">
            Data from data.gov.sg ↗
          </a>
          <button
            className="icon-button"
            onClick={toggleTheme}
            aria-label={themeMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            title={themeMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            {themeMode === 'light' ? (
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
              </svg>
            ) : (
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle cx="12" cy="12" r="4.5" stroke="currentColor" strokeWidth="1.8" />
                <path
                  d="M12 2.5v2.4M12 19.1v2.4M4.2 4.2l1.7 1.7M18.1 18.1l1.7 1.7M2.5 12h2.4M19.1 12h2.4M4.2 19.8l1.7-1.7M18.1 5.9l1.7-1.7"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            )}
          </button>
        </div>
      </header>

      <aside className={`rail ${railOpen ? 'open' : 'collapsed'}`} aria-label="Catalog controls" aria-hidden={!railOpen}>
        <button className="rail-toggle" onClick={() => setRailOpen((v) => !v)} aria-expanded={railOpen} title={railOpen ? 'Hide panel' : 'Show panel'}>
          {railOpen ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M8.5 2.5L4 7l4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M5.5 2.5L10 7l-4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
          <span>{railOpen ? 'Hide' : 'Search & filters'}</span>
        </button>
        <div className="rail-group">
        <div className="segmented">
          <span className={`indicator ${mode === 'ask' ? 'pos-1' : ''}`} aria-hidden="true" />
          <button className={mode === 'search' ? 'selected' : ''} onClick={() => setMode('search')}>
            Search
          </button>
          <button className={mode === 'ask' ? 'selected' : ''} onClick={() => setMode('ask')}>
            Ask
          </button>
        </div>

        {mode === 'search' ? (
          <div className="search-field">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="7" cy="7" r="5.25" stroke="currentColor" strokeWidth="1.5" />
              <path d="M11 11L14.5 14.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <input
              className="search-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="housing prices, traffic, tourism"
              aria-label="Search datasets"
            />
          </div>
        ) : (
          <form
            className="ask-form"
            onSubmit={(event) => {
              event.preventDefault();
              submitAsk();
            }}
          >
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask across the catalog"
              aria-label="Ask a question across the catalog"
            />
            <button type="submit" disabled={answering}>
              {answering && <Compass spinning size={14} />} {answering ? 'Thinking' : 'Ask'}
            </button>
          </form>
        )}

        {mode === 'ask' && (
          <div className="examples">
            {ASK_EXAMPLES.map((example) => (
              <button key={example} onClick={() => submitAsk(example)}>
                {example}
              </button>
            ))}
          </div>
        )}

        {answer && <div className="answer">{renderAnswer(answer)}</div>}
        </div>

        <div className="rail-divider" role="separator" />

        <div className="rail-group rail-group-quiet">
          <div className="row-between">
            <span>Color</span>
            <div className="mini-toggle">
              <span className="indicator" aria-hidden="true" />
              <button className="selected" onClick={() => setColorMode('agency')}>
                Agency
              </button>
            </div>
          </div>
        </div>

        <div className="rail-divider" role="separator" />

        <div className="rail-group">
        <p className="section-label">Agencies</p>
        <div className="chips" aria-label="Agency filter">
          <button className={!agencyFilter ? 'selected' : ''} onClick={() => setAgencyFilter(null)}>
            All
          </button>
          {stats?.top_agencies.slice(0, 10).map((agency) => (
            <button
              key={agency.agency}
              className={agencyFilter === agency.agency ? 'selected' : ''}
              style={{ borderColor: agencyColors.get(agency.agency) }}
              onClick={() => setAgencyFilter(agency.agency)}
            >
              {agency.agency || 'Unknown'} {agency.count}
            </button>
          ))}
        </div>

        <p className="section-label">
          Datasets{stats?.total ? <span className="section-count">{listItems.length} of {stats.total.toLocaleString()}</span> : null}
        </p>

        {showSkeleton ? (
          <div className="result-skeleton" aria-hidden="true">
            {[0, 1, 2, 3].map((row) => (
              <div key={row} className="skeleton-row" />
            ))}
          </div>
        ) : searching && listItems.length === 0 ? (
          <p className="empty-state">No datasets match &ldquo;{query}&rdquo;. Try a broader term, or browse by agency below.</p>
        ) : catalogEmpty ? (
          <CatalogBuild onComplete={handleCatalogBuilt} />
        ) : (
          <div className="result-list">
            {listItems.map((item, index) => (
              <button key={item.dataset_id} onClick={() => chooseDataset(item, true)}>
                <span className="manifest-index">{String(index + 1).padStart(2, '0')}</span>
                <span className="agency-dot" style={{ background: nodeColor(item) }} />
                <b>{item.name}</b>
                <small>
                  {item.agency || 'Unknown'} &nbsp;•&nbsp; {toneLabel(suitabilityLookup[item.dataset_id], item.suitability_score, item.suitability_tone)}
                </small>
              </button>
            ))}
          </div>
        )}
        </div>
      </aside>

      {selected && (
        <aside className="detail-panel" aria-label="Dataset detail" key={selected.dataset_id}>
          <button ref={closeButtonRef} className="close-button" onClick={closeDetail} aria-label="Close dataset detail">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M1 1L13 13M13 1L1 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
          <h2>{info?.name || selected.name}</h2>
          <p className="agency-line">{info?.agency || selected.agency || 'Singapore open data'}</p>
          {detailLoading && <p className="muted">Loading dataset preview...</p>}
          <div className={`fit-badge ${fit?.tone ?? 'marginal'}`}>
            <b>{fit ? fit.score : selected.suitability_score ?? 'NA'}</b>
            <span>{fit?.verdict ?? selected.suitability_tone ?? 'Not scored yet'}</span>
          </div>
          <p className="description">{displayedDescription || 'No description available.'}</p>
          {description.length > 360 && (
            <button className="text-button" onClick={() => setExpanded((value) => !value)}>
              {expanded ? 'Read less' : 'Read more'}
            </button>
          )}
          {info?.url && (
            <a className="source-link" href={info.url} target="_blank" rel="noreferrer">
              Source
            </a>
          )}
          <div className="meta-grid">
            <span>Rows</span>
            <b>{info?.row_count_total?.toLocaleString() ?? 'Unknown'}</b>
            <span>Columns</span>
            <b>{info?.column_count ?? 'Unknown'}</b>
            <span>Updated</span>
            <b>{info?.last_updated || selected.last_updated_at || 'Unknown'}</b>
            <span>Coverage</span>
            <b>{info?.coverage_start || selected.coverage_start || 'NA'} to {info?.coverage_end || selected.coverage_end || 'NA'}</b>
          </div>
          {fit?.reasons?.length ? (
            <ul className="reason-list">
              {fit.reasons.slice(0, 4).map((reason) => (
                <li key={`${reason.kind}-${reason.text}`}>{reason.text}</li>
              ))}
            </ul>
          ) : null}
          {info?.columns?.length ? (
            <div className="schema-table">
              <table>
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Kind</th>
                    <th>Null</th>
                  </tr>
                </thead>
                <tbody>
                  {info.columns.slice(0, 12).map((column) => (
                    <tr key={column.name}>
                      <td>{column.title || column.name}</td>
                      <td>{column.kind}</td>
                      <td>{column.null_pct.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          <div className="run-config">
            <label>
              Rows
              <select value={maxRows} onChange={(event) => setMaxRows(Number(event.target.value))}>
                <option value={0}>All rows up to 200,000</option>
                <option value={60000}>60,000 rows</option>
                <option value={30000}>30,000 rows</option>
                <option value={5000}>5,000 quick rows</option>
              </select>
            </label>
            <label className="check-row">
              <input type="checkbox" checked={noResearch} onChange={(event) => setNoResearch(event.target.checked)} />
              Skip research
            </label>
            <label className="check-row">
              <input type="checkbox" checked={noCorrelation} onChange={(event) => setNoCorrelation(event.target.checked)} />
              Skip correlation
            </label>
            <button className="launch-button" onClick={launchRun} disabled={launching}>
              {launching && <Compass spinning size={14} />} {launching ? 'Launching' : 'Run ML'}
            </button>
          </div>
        </aside>
      )}

      {error && <div className="toast">{error}</div>}
      <RunsDock />
    </main>
  );
}
