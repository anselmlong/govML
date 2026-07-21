# govML

govML is an on-demand machine-learning pipeline and gallery UI for Singapore
data.gov.sg datasets.

The CLI accepts a data.gov.sg datastore resource ID, fetches CKAN rows,
preprocesses columns, proposes targets, engineers context features, trains a
bank of models, diagnoses the result, persists the best model, and writes a
self-contained HTML report.

The web app wraps the pipeline with a searchable semantic catalog, D3 semantic
map, dataset preview, ML suitability scoring, run launching, log streaming,
report serving, and catalog Q&A.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-web.txt
```

Copy `.env.example` to `.env` and set your OpenAI key if you want LLM-backed
planning, embeddings, and Q&A:

```bash
OPENAI_API_KEY=sk-...
# optional overrides
#OPENAI_BASE_URL=          # any OpenAI-compatible endpoint
#OPENAI_CHAT_MODEL=gpt-4o-mini
#OPENAI_EMBED_MODEL=text-embedding-3-small
```

The project works without a key by using deterministic heuristics and local
hash embeddings, so everything stays functional offline — just less smart.

## Catalog

```bash
python catalog.py ingest --max-pages 10
python catalog.py embed
python catalog.py stats
```

`catalog.db` and `map_coords.json` are generated locally and are not source
artifacts.

## Pipeline

Default run:

```bash
python pipeline.py --max-rows 30000 --no-research
```

The default dataset is HDB resale flat prices:
`f1765b54-a209-4718-8d38-a39237f502b3`.

Main output:

- `output/report.html`
- `output/model_<best_model_slug>.pkl`

## Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

`GET /api/health` reports service, catalog, and LLM availability — point your
uptime checks there.

Set `GOVML_CORS_ORIGINS` to a comma-separated list of allowed browser origins
in production (defaults to `*` for local development).

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/reports` to the backend on port 8000.

For production, `npm run build` emits `frontend/dist`, which the backend
serves automatically at `/` when present.

## Docker

The included `Dockerfile` builds the frontend and serves the full app from a
single container:

```bash
docker build -t govml .
docker run -p 8000:8000 --env-file .env govml
```

## Tests

```bash
pytest
```
