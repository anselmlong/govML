export interface CatalogResult {
  dataset_id: string;
  name: string;
  description: string;
  agency: string;
  format?: string;
  status?: string;
  last_updated_at?: string;
  coverage_start?: string;
  coverage_end?: string;
  suitability_score?: number | null;
  suitability_tone?: string | null;
  score?: number;
}

export interface CatalogStats {
  total: number;
  embedded: number;
  top_agencies: { agency: string; count: number }[];
}

export interface SampleDataset extends CatalogResult {}

export interface MapDataset extends CatalogResult {
  x: number;
  y: number;
}

export interface RunRecord {
  run_id: string;
  resource_id: string;
  name: string;
  max_rows: number;
  no_research: boolean;
  no_correlation: boolean;
  started_at: string;
  completed_at: string | null;
  status: 'running' | 'completed' | 'failed';
  report_path: string;
  log_path: string;
  best_model: string | null;
  mae: number | null;
  r2: number | null;
  accuracy?: number;
  f1?: number;
}

export interface RunRequest {
  resource_id: string;
  name?: string;
  max_rows: number;
  no_research: boolean;
  no_correlation: boolean;
  force?: boolean;
}

export interface ColumnInfo {
  name: string;
  title: string;
  dtype: string;
  kind: 'numeric' | 'date' | 'text';
  non_null: number;
  null_pct: number;
  is_categorical: boolean;
  min?: number | string | null;
  max?: number | string | null;
  mean?: number | null;
  unique?: number | null;
  sample?: string[];
}

export interface Suitability {
  score: number;
  tone: 'good' | 'okay' | 'marginal' | 'blocked';
  verdict: string;
  reasons: { kind: string; text: string }[];
  recommended_targets: string[];
}

export interface DatasetInfo {
  resource_id: string;
  name: string;
  description: string;
  agency: string;
  url: string;
  format: string;
  coverage_start: string;
  coverage_end: string;
  last_updated: string;
  created_at: string;
  frequency: string;
  contact_emails: string[];
  size_bytes: number | null;
  size_human: string;
  row_count_total: number | null;
  sample_rows: Record<string, unknown>[];
  column_count: number;
  columns: ColumnInfo[];
  has_datastore: boolean;
  errors: string[];
  suitability: Suitability;
}

export interface AskResult {
  answer: string;
  datasets: CatalogResult[];
  provenance: string;
}

export interface ScoreStatus {
  total: number;
  scored: number;
  running: boolean;
  started_at?: string | null;
  completed_at?: string | null;
  last?: unknown;
}

export interface CatalogBuildStatus {
  running: boolean;
  phase: 'ingest' | 'embed' | 'map' | 'done' | null;
  detail: string | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  total_ingested: number | null;
}

export function friendlyError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  if (/failed to fetch|networkerror|load failed/i.test(raw)) {
    return "Can't reach the server. Check your connection and try again.";
  }
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === 'string') return parsed.detail;
    if (Array.isArray(parsed?.detail)) {
      return parsed.detail.map((d: { msg?: string }) => d?.msg || JSON.stringify(d)).join('; ');
    }
  } catch {
    // Not JSON; fall through to the raw message.
  }
  return raw || 'Something went wrong. Please try again.';
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export const searchCatalog = (q: string, topK = 10) =>
  getJson<CatalogResult[]>(`/api/catalog/search?q=${encodeURIComponent(q)}&top_k=${topK}`);
export const getCatalogStats = () => getJson<CatalogStats>('/api/catalog/stats');
export const getCatalogSample = (limit = 250) => getJson<SampleDataset[]>(`/api/catalog/sample?limit=${limit}`);
export const getCatalogMap = (force = false) => getJson<MapDataset[]>(`/api/catalog/map?force=${force}`);
export const getDatasetInfo = (resourceId: string) => getJson<DatasetInfo>(`/api/datasets/${encodeURIComponent(resourceId)}/info`);
export const getSuitabilityLookup = () => getJson<Record<string, Suitability>>('/api/catalog/suitability');
export const getScoreStatus = () => getJson<ScoreStatus>('/api/catalog/score-status');
export const startScoreAll = (limit?: number, resume = true, rescoreBelow?: number) => {
  const params = new URLSearchParams();
  if (limit !== undefined) params.set('limit', String(limit));
  params.set('resume', String(resume));
  if (rescoreBelow !== undefined) params.set('rescore_below', String(rescoreBelow));
  return postJson<{ started: boolean }>(`/api/catalog/score-all?${params.toString()}`);
};
export const askCatalog = (query: string, topK = 5) => postJson<AskResult>('/api/ask', { query, top_k: topK });
export const getCatalogBuildStatus = () => getJson<CatalogBuildStatus>('/api/catalog/build-status');
export const startCatalogBuild = () => postJson<{ started: boolean }>('/api/catalog/build');
export const getRuns = () => getJson<RunRecord[]>('/api/runs');
export const getRun = (runId: string) => getJson<RunRecord>(`/api/runs/${encodeURIComponent(runId)}`);
export const startRun = (request: RunRequest) => postJson<{ run_id: string; reused: boolean }>('/api/runs', request);

export function streamRun(runId: string, onLine: (line: string) => void, onStatus: (run: RunRecord) => void): EventSource {
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream`);
  source.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as { event?: string; run?: RunRecord };
      if (parsed.event === 'status' && parsed.run) {
        onStatus(parsed.run);
        source.close();
        return;
      }
    } catch {
      // Plain log line.
    }
    onLine(event.data);
  };
  return source;
}

