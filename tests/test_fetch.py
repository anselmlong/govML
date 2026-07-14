from __future__ import annotations

import pytest

from src.fetch import fetch_resource


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def ok(records, total=None):
    return FakeResponse(
        200,
        {
            "success": True,
            "result": {
                "records": records,
                "total": len(records) if total is None else total,
            },
        },
    )


def test_404_raises_value_error(monkeypatch, tmp_path):
    monkeypatch.setattr("src.fetch.requests.get", lambda *a, **k: FakeResponse(404, {}))
    with pytest.raises(ValueError, match="missing-resource"):
        fetch_resource("missing-resource", cache_dir=tmp_path)


def test_413_halves_page_size_and_retries(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["limit"])
        if len(calls) == 1:
            return FakeResponse(413, {})
        return ok([{"x": 1}], total=1)

    monkeypatch.setattr("src.fetch.requests.get", fake_get)
    df = fetch_resource("r1", page_size=1000, cache_dir=tmp_path)
    assert calls[:2] == [1000, 500]
    assert df.to_dict("records") == [{"x": 1}]


def test_repeated_413_halves_until_success(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["limit"])
        if params["limit"] > 125:
            return FakeResponse(413, {})
        return ok([{"x": 1}], total=1)

    monkeypatch.setattr("src.fetch.requests.get", fake_get)
    fetch_resource("r2", page_size=1000, cache_dir=tmp_path)
    assert calls == [1000, 500, 250, 125]


def test_persistent_413_at_minimum_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("src.fetch.requests.get", lambda *a, **k: FakeResponse(413, {}))
    with pytest.raises(ValueError, match="minimum page_size=100"):
        fetch_resource("r3", page_size=100, cache_dir=tmp_path)


def test_multi_page_fetch_assembles_records(monkeypatch, tmp_path):
    def fake_get(url, params, timeout):
        offset = params["offset"]
        if offset == 0:
            return ok([{"x": 1}, {"x": 2}], total=3)
        return ok([{"x": 3}], total=3)

    monkeypatch.setattr("src.fetch.requests.get", fake_get)
    df = fetch_resource("r4", page_size=2, cache_dir=tmp_path)
    assert df["x"].tolist() == [1, 2, 3]


def test_max_rows_truncates_exactly(monkeypatch, tmp_path):
    def fake_get(url, params, timeout):
        return ok([{"x": i} for i in range(10)], total=10)

    monkeypatch.setattr("src.fetch.requests.get", fake_get)
    df = fetch_resource("r5", max_rows=3, page_size=10, cache_dir=tmp_path)
    assert len(df) == 3
    assert df["x"].tolist() == [0, 1, 2]


def test_ckan_success_false_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("src.fetch.requests.get", lambda *a, **k: FakeResponse(200, {"success": False, "error": "bad"}))
    with pytest.raises(RuntimeError, match="CKAN datastore_search failed"):
        fetch_resource("r6", cache_dir=tmp_path)

