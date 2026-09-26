# Operator API and public gallery

The gallery's ordinary GET routes remain public. Every mutating `/api/` request, including dataset questions that may call an LLM, and forced map rebuilds (`/api/catalog/map?force=true`) require `Authorization: Bearer <GOVML_OPERATOR_TOKEN>`. With no token configured these operations return 503; an invalid or missing bearer returns 401. Set a strong random operator token in the backend environment, and keep it out of the frontend bundle and source control.

The existing browser controls do not manage operator credentials. Use a trusted operator API client or an authenticated reverse proxy that injects the header after authenticating an operator. Never have a public proxy add this token for everyone. Use HTTPS for remote operator access. The read-only gallery remains useful as a portfolio demo.

`GOVML_CORS_ORIGINS` is a comma-separated browser-origin allowlist; its default is `http://localhost:5173`. Same-origin requests need no CORS exception. CORS is not authentication.

Run input is limited to a dataset identifier, a 200-character name and 1–100,000 rows. Only one pipeline may run at a time per application process. Deploy with one worker for this limit; multiple workers/replicas need a shared job queue and quota store. This is a first boundary, not a full per-user quota or sandbox: an authenticated operator can still launch expensive work, and pipelines run with the server account's permissions.

Validation: `python -m pytest tests/test_operator_security.py` (install the repository dependencies plus pytest and httpx). No live deployment configuration is changed by this patch.
