import asyncio
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from backend.app import app, RunRequest, AskRequest, active_procs, api_start_run


def request(path, token=None, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, headers={"Authorization": token} if token else {}, **kwargs)
    return asyncio.run(send())


@pytest.mark.parametrize("path", ["/api/runs", "/api/ask", "/api/catalog/score-all", "/api/catalog/build"])
def test_costly_endpoints_fail_closed(monkeypatch, path):
    monkeypatch.delenv("GOVML_OPERATOR_TOKEN", raising=False)
    assert request(path, json={}).status_code == 503
    monkeypatch.setenv("GOVML_OPERATOR_TOKEN", "test-token")
    assert request(path, json={}).status_code == 401
    assert request(path, token="Bearer wrong", json={}).status_code == 401


def test_operator_reaches_request_validation(monkeypatch):
    monkeypatch.setenv("GOVML_OPERATOR_TOKEN", "test-token")
    assert request("/api/runs", token="Bearer test-token", json={}).status_code == 422


def test_forced_map_rebuild_requires_operator(monkeypatch):
    monkeypatch.delenv("GOVML_OPERATOR_TOKEN", raising=False)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for value in ["true", "1", "yes", "on"]:
                assert (await client.get(f"/api/catalog/map?force={value}")).status_code == 503
    asyncio.run(check())


@pytest.mark.parametrize("kwargs", [{"max_rows": 0}, {"max_rows": 100001}, {"resource_id": "https://example.org/private"}])
def test_run_bounds(kwargs):
    with pytest.raises(ValidationError):
        RunRequest(**({"resource_id": "d_example"} | kwargs))


def test_ask_bounds():
    with pytest.raises(ValidationError):
        AskRequest(query="question", top_k=1000)


def test_concurrent_run_rejected(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setitem(active_procs, "busy", Mock(poll=lambda: None))
    with pytest.raises(HTTPException) as exc:
        api_start_run(RunRequest(resource_id="d_example"))
    assert exc.value.status_code == 429
