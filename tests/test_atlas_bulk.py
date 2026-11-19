import sqlite3

import pandas as pd
import pytest

from src.atlas.bulk import fetch_one, missing_ids


class Reply:
    def __init__(self, payload=None, content=b""):
        self.payload, self.content = payload, content

    def json(self):
        return self.payload


def scripted(replies):
    calls = []

    def get(url, timeout):
        calls.append(url)
        return replies.pop(0)
    return get, calls


def test_a_rate_limited_poll_waits_then_the_csv_lands_as_all_parquet(tmp_path):
    get, calls = scripted([
        Reply({"code": 24, "name": "TOO_MANY_REQUESTS"}),
        Reply({"code": 0, "data": {"url": "https://s3/x.csv"}}),
        Reply(content=b"month,town,price\n2024-01,A,1\n2024-02,B,2\n"),
    ])
    assert fetch_one("d_x", tmp_path, get, lambda s: None, 0) == 2
    frame = pd.read_parquet(tmp_path / "d_x_all.parquet")
    assert list(frame.columns) == ["month", "town", "price"] and frame["price"].tolist() == ["1", "2"]
    assert (tmp_path / "d_x_all.meta.json").exists()
    assert not any("initiate-download" in c for c in calls), "a rate-limited poll does not re-initiate"


def test_a_table_not_yet_prepared_is_initiated_and_gives_up_after_the_attempts(tmp_path):
    get, calls = scripted([Reply({"code": 0, "data": {"status": "PENDING"}}), Reply({})] * 3)
    with pytest.raises(RuntimeError, match="no download after 3 polls"):
        fetch_one("d_y", tmp_path, get, lambda s: None, 0, attempts=3)
    assert sum("initiate-download" in c for c in calls) == 3
    assert not (tmp_path / "d_y_all.parquet").exists()


def test_missing_ids_are_csv_datasets_with_no_cache_file(tmp_path):
    catalog = tmp_path / "catalog.db"
    with sqlite3.connect(catalog) as conn:
        conn.execute("CREATE TABLE datasets (dataset_id TEXT, format TEXT)")
        conn.executemany("INSERT INTO datasets VALUES (?, ?)",
                         [("d_a", "CSV"), ("d_b", "csv"), ("d_c", "GEOJSON"), ("d_d", "CSV")])
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "d_a_10000.parquet").write_bytes(b"")
    assert missing_ids(catalog, cache) == ["d_b", "d_d"]
