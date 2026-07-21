import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { select, zoom } from 'd3';
import {
  askCatalog,
  CatalogResult,
  CatalogStats,
  DatasetInfo,
  getCatalogMap,
  getCatalogStats,
  getDatasetInfo,
  getScoreStatus,
  getSuitabilityLookup,
  MapDataset,
  ScoreStatus,
  searchCatalog,
  startRun,
  startScoreAll,
  Suitability
} from './api';
import RunsDock from './RunsDock';
import { applyTheme, initialTheme, ThemeMode } from './theme';

type Mode = 'search' | 'ask';
type ColorMode = 'agency' | 'fit';

const PALETTE = ['#5e6ad2', '#26a269', '#e01b24', '#e5a50a', '#9141ac', '#1a9db8', '#c061cb', '#6fae3c', '#3584e4', '#b5835a'];
const FIT_COLORS: Record<string, string> = {
  good: '#26a269',
  okay: '#e5a50a',
  marginal: '#77767b',
  blocked: '#e01b24'
};

const FEATURED: CatalogResult[] = [
  {
    dataset_id: 'f1765b54-a209-4718-8d38-a39237f502b3',
    name: 'HDB Resale Flat Prices',
    description: 'Resale flat transactions from January 2017 onwards.',
    agency: 'Housing and Development Board'
  },
  { dataset_id: 'unemployment-rate', name: 'Unemployment Rate', description: 'Labour-market headline series.', agency: 'Ministry of Manpower' },
  { dataset_id: 'monthly-taxi-fleet', name: 'Monthly Taxi Fleet', description: 'Taxi fleet counts by month.', agency: 'Land Transport Authority' },
  {
    dataset_id: 'assessable-income-distribution',
    name: 'Assessable Income Distribution',
    description: 'Income distribution tables for analytical work.',
    agency: 'Inland Revenue Authority of Singapore'
  }
];

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
  const svgRef = useRef<SVGSVGElement | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
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
  const [scoreStatus, setScoreStatus] = useState<ScoreStatus | null>(null);
  const [themeMode, setThemeMode] = useState<ThemeMode>(initialTheme());
  const [maxRows, setMaxRows] = useState(0);
  const [noResearch, setNoResearch] = useState(true);
  const [noCorrelation, setNoCorrelation] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([getCatalogMap(), getCatalogStats(), getSuitabilityLookup(), getScoreStatus()])
      .then(([mapResult, statsResult, fitResult, scoreResult]) => {
        if (!mounted) return;
        if (mapResult.status === 'fulfilled') setNodes(mapResult.value);
        if (statsResult.status === 'fulfilled') setStats(statsResult.value);
        if (fitResult.status === 'fulfilled') setSuitabilityLookup(fitResult.value);
        if (scoreResult.status === 'fulfilled') setScoreStatus(scoreResult.value);
      })
      .catch((err: unknown) => setError(String(err)))
      .finally(() => mounted && setLoading(false));
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;
    const svg = select<SVGSVGElement, unknown>(svgRef.current);
    const behavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 22])
      .on('zoom', (event) => {
        select(gRef.current).attr('transform', event.transform.toString());
      });
    svg.call(behavior);
    return () => {
      svg.on('.zoom', null);
    };
  }, [nodes.length]);

  useEffect(() => {
    if (mode !== 'search') return;
    const handle = window.setTimeout(() => {
      if (query.trim().length < 2) {
        setResults([]);
        return;
      }
      searchCatalog(query, 10).then(setResults).catch((err: unknown) => setError(String(err)));
    }, 280);
    return () => window.clearTimeout(handle);
  }, [query, mode]);

  useEffect(() => {
    const handle = window.setInterval(() => {
      getScoreStatus().then(setScoreStatus).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(handle);
  }, []);

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
    if (colorMode === 'fit') return FIT_COLORS[fit?.tone ?? node.suitability_tone ?? 'marginal'] ?? '#77767b';
    return agencyColors.get(node.agency) ?? PALETTE[Math.abs(hashCode(node.agency)) % PALETTE.length];
  }

  function hashCode(value: string) {
    return Array.from(value || 'unknown').reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) | 0, 7);
  }

  async function chooseDataset(dataset: CatalogResult) {
    openerRef.current = document.activeElement as HTMLElement;
    setSelected(dataset);
    setSelectedInfo(null);
    setExpanded(false);
    setDetailLoading(true);
    setError('');
    try {
      const info = await getDatasetInfo(dataset.dataset_id);
      setSelectedInfo(info);
    } catch (err) {
      setError(String(err));
    } finally {
      setDetailLoading(false);
    }
  }

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
    } catch (err) {
      setError(String(err));
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
      setError(String(err));
    } finally {
      setLaunching(false);
    }
  }

  async function startScoring() {
    try {
      await startScoreAll(undefined, true);
      setScoreStatus(await getScoreStatus());
    } catch (err) {
      setError(String(err));
    }
  }

  function toggleTheme() {
    const next = themeMode === 'light' ? 'dark' : 'light';
    setThemeMode(next);
    applyTheme(next);
  }

  const listItems = results.length ? results : topFit.length ? topFit : FEATURED;
  const info = selectedInfo;
  const fit = selected ? suitabilityLookup[selected.dataset_id] ?? info?.suitability : null;
  const description = info?.description || selected?.description || '';
  const displayedDescription = expanded || description.length < 320 ? description : `${description.slice(0, 320)}...`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          govML
        </div>
        <span className="topbar-sub">Open Data ML Workbench</span>
        <div className="topbar-right">
          {stats && <span>{stats.total.toLocaleString()} datasets</span>}
          <button className="icon-button" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
            {themeMode === 'light' ? '☾' : '☀'}
          </button>
        </div>
      </header>

      <div className="main">
        <aside className="sidebar" aria-label="Catalog controls">
          <div className="segmented" role="tablist">
            <button className={mode === 'search' ? 'selected' : ''} onClick={() => setMode('search')}>
              Search
            </button>
            <button className={mode === 'ask' ? 'selected' : ''} onClick={() => setMode('ask')}>
              Ask
            </button>
          </div>

          {mode === 'search' ? (
            <input
              className="search-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search datasets..."
            />
          ) : (
            <form
              className="ask-form"
              onSubmit={(event) => {
                event.preventDefault();
                submitAsk();
              }}
            >
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask across the catalog..." />
              <button type="submit" className="button primary" disabled={answering}>
                {answering ? 'Thinking...' : 'Ask'}
              </button>
            </form>
          )}

          {mode === 'ask' && !answer && (
            <div className="examples">
              {ASK_EXAMPLES.map((example) => (
                <button key={example} onClick={() => submitAsk(example)}>
                  {example}
                </button>
              ))}
            </div>
          )}

          {answer && <div className="answer">{renderAnswer(answer)}</div>}

          <div className="section-label">
            <span>Color by</span>
            <div className="mini-toggle">
              <button className={colorMode === 'agency' ? 'selected' : ''} onClick={() => setColorMode('agency')}>
                Agency
              </button>
              <button className={colorMode === 'fit' ? 'selected' : ''} onClick={() => setColorMode('fit')}>
                ML fit
              </button>
            </div>
          </div>

          <div className="section-label">
            <span>Agencies</span>
          </div>
          <div className="agency-list" aria-label="Agency filter">
            <button className={!agencyFilter ? 'selected' : ''} onClick={() => setAgencyFilter(null)}>
              <span className="agency-swatch" style={{ background: 'var(--faint)' }} />
              <span className="agency-name">All agencies</span>
              <span className="agency-count">{stats?.total ?? ''}</span>
            </button>
            {stats?.top_agencies.slice(0, 10).map((agency) => (
              <button
                key={agency.agency}
                className={agencyFilter === agency.agency ? 'selected' : ''}
                onClick={() => setAgencyFilter(agencyFilter === agency.agency ? null : agency.agency)}
              >
                <span className="agency-swatch" style={{ background: agencyColors.get(agency.agency) }} />
                <span className="agency-name">{agency.agency || 'Unknown'}</span>
                <span className="agency-count">{agency.count}</span>
              </button>
            ))}
          </div>

          <div className="section-label">
            <span>{results.length ? 'Results' : topFit.length ? 'Best ML fit' : 'Featured'}</span>
          </div>
          <div className="result-list">
            {listItems.map((item) => (
              <button
                key={item.dataset_id}
                className={selected?.dataset_id === item.dataset_id ? 'selected' : ''}
                onClick={() => chooseDataset(item)}
              >
                <span className="result-dot" style={{ background: nodeColor(item) }} />
                <span className="result-name">{item.name}</span>
                <span className="result-meta">
                  {item.agency || 'Unknown'} · {toneLabel(suitabilityLookup[item.dataset_id], item.suitability_score, item.suitability_tone)}
                </span>
              </button>
            ))}
            {!listItems.length && <p className="empty-note">No datasets match this search yet. Try a broader term.</p>}
          </div>
        </aside>

        <section className="canvas" aria-label="Semantic dataset map">
          <svg ref={svgRef} viewBox="0 0 1000 700" role="img" aria-label="Semantic map of open datasets">
            <g ref={gRef}>
              {nodes.map((node) => {
                const filtered = agencyFilter && node.agency !== agencyFilter;
                const active = selected?.dataset_id === node.dataset_id;
                const matched = selectedIds.has(node.dataset_id);
                return (
                  <circle
                    key={node.dataset_id}
                    className={`map-dot ${active ? 'active' : ''} ${matched ? 'matched' : ''}`}
                    cx={node.x * 940 + 30}
                    cy={node.y * 640 + 30}
                    r={active ? 7 : matched ? 5 : 3.2}
                    fill={nodeColor(node)}
                    opacity={filtered ? 0.08 : matched || active || !results.length ? 0.9 : 0.25}
                    tabIndex={0}
                    onClick={() => chooseDataset(node)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') chooseDataset(node);
                    }}
                  />
                );
              })}
            </g>
          </svg>
          {loading && (
            <div className="canvas-loading">
              <span className="spinner" aria-hidden="true" />
              Loading the semantic map...
            </div>
          )}
        </section>

        {selected && (
          <aside className="inspector" aria-label="Dataset detail">
            <div className="inspector-head">
              <div>
                <h2>{info?.name || selected.name}</h2>
                <p className="inspector-agency">{info?.agency || selected.agency || 'Unknown agency'}</p>
              </div>
              <button ref={closeButtonRef} className="icon-button" onClick={closeDetail} aria-label="Close dataset detail">
                ✕
              </button>
            </div>

            <div className="inspector-body">
              {detailLoading && (
                <>
                  <div className="skeleton" style={{ height: 24, width: 140 }} />
                  <div className="skeleton" style={{ height: 60 }} />
                  <div className="skeleton" style={{ height: 120 }} />
                </>
              )}

              {!detailLoading && (
                <>
                  <span className={`fit-chip ${fit?.tone ?? ''}`}>
                    <b>{fit ? fit.score : selected.suitability_score ?? '—'}</b>
                    {fit?.verdict ?? selected.suitability_tone ?? 'Not scored yet'}
                  </span>

                  <p className="description">{displayedDescription || 'No description available.'}</p>
                  {description.length > 320 && (
                    <button className="text-button" onClick={() => setExpanded((value) => !value)}>
                      {expanded ? 'Show less' : 'Show more'}
                    </button>
                  )}

                  <div className="meta-grid">
                    <span>Rows</span>
                    <b>{info?.row_count_total?.toLocaleString() ?? 'Unknown'}</b>
                    <span>Columns</span>
                    <b>{info?.column_count ?? 'Unknown'}</b>
                    <span>Updated</span>
                    <b>{info?.last_updated || selected.last_updated_at || 'Unknown'}</b>
                    <span>Coverage</span>
                    <b>
                      {info?.coverage_start || selected.coverage_start || '—'} to {info?.coverage_end || selected.coverage_end || '—'}
                    </b>
                    {info?.url && (
                      <>
                        <span>Source</span>
                        <a className="source-link" href={info.url} target="_blank" rel="noreferrer">
                          data.gov.sg ↗
                        </a>
                      </>
                    )}
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
                              <td>
                                <span className="kind-tag">{column.kind}</span>
                              </td>
                              <td className="num">{column.null_pct.toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </>
              )}
            </div>

            <div className="inspector-foot">
              <label className="field">
                Rows to fetch
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
              <button className="button primary" onClick={launchRun} disabled={launching}>
                {launching ? 'Launching...' : 'Run ML pipeline'}
              </button>
            </div>
          </aside>
        )}
      </div>

      <footer className="statusbar">
        <span className="statusbar-item">
          ML fit scored {scoreStatus?.scored ?? 0}/{scoreStatus?.total ?? 0}
        </span>
        <button className="statusbar-item" onClick={startScoring} disabled={scoreStatus?.running}>
          {scoreStatus?.running ? 'Scoring...' : 'Score all'}
        </button>
        <span className="grow" />
        <span className="statusbar-item">Scroll to zoom · drag to pan · click a dot</span>
        <RunsDock />
      </footer>

      {error && <div className="toast">{error}</div>}
    </div>
  );
}
