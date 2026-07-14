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

Create a local `.env` if you want LLM-backed planning, embeddings, or Q&A:

```bash
GENAI_API_KEY=
VISA_GENAI_TOKEN=
VISA_GENAI_BASE_URL=
VISA_GENAI_MODEL=claude-sonnet-4-5-20250929
VISA_GENAI_EMBED_MODEL=text-embedding-3-small
ANTHROPIC_API_KEY=
```

The project works without these values by using deterministic heuristics and
local deterministic embeddings.

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

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/reports` to the backend on port 8000.

## Tests

```bash
pytest
```

