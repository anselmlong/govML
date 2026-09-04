# Changelog

## Unreleased

- Fixed `run_insights` table only being created by `batch_ml.py`, which made
  `/api/insights/*` and `/api/catalog/search` 500 on any deployment whose
  catalog was built without ever running the batch script; the table is now
  created by `src/catalog.py`'s shared schema setup.
- Backend now self-populates on startup: an empty catalog runs the full
  ingest → embed → map → score → train chain automatically, and a non-empty
  one resumes any unfinished scoring/training, instead of requiring a
  visitor to click "Build catalog" or someone to SSH in and run
  `batch_ml.py` by hand. New `POST /api/catalog/train-all` /
  `GET /api/catalog/train-status` mirror the existing catalog-build and
  score-all background-job pattern.
- The frontend now actually surfaces precomputed ML results: dataset detail
  view shows a full result card (target, best model, metrics, verdict,
  cross-dataset correlations) when one exists, map dots and list rows show
  a "trained" indicator, and a background training-progress note appears
  while the server is still working through the catalog.

## 0.1.0

- Rebuilt govML from the code-light specification.
- Added CKAN fetcher with parquet cache and retry behavior.
- Added nine-phase pipeline, SQLite catalog, FastAPI backend, and React/D3 UI.

