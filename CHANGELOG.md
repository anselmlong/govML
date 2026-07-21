# Changelog

## 0.2.0

- Generalized the LLM layer from the Visa-internal GenAI gateway to the public
  OpenAI API (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL` for any
  OpenAI-compatible endpoint); removed the `anthropic` and `truststore`
  dependencies.
- Production hardening: `/api/health` endpoint, configurable CORS via
  `GOVML_CORS_ORIGINS`, `.env.example`, and a single-container `Dockerfile`.
- Redesigned the frontend with a modern glass aesthetic: aurora gradient
  backdrop, gradient branding, staggered entrance animations, pulsing map
  dots and run indicators, shimmer loading, and refined light/dark themes.

## 0.1.0

- Rebuilt govML from the code-light specification.
- Added CKAN fetcher with parquet cache and retry behavior.
- Added nine-phase pipeline, SQLite catalog, FastAPI backend, and React/D3 UI.

