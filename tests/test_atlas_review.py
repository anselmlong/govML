import json

import pandas as pd
import pytest
import requests

from src.atlas import review as R

KEY = "sk-or-SECRET-KEY-123"
PERIODS = [f"2020-{m:02d}" for m in range(1, 13)] + [f"2021-{m:02d}" for m in range(1, 13)] + ["2022-01", "2022-02"]

KEEP = {"verdict": "keep", "rationale": "Distinct markets with a stable lead.", "drop_reason": None, "headline": "Rents tend to rise before prices.",
        "explanation": "Over 2020 to 2022, rents moved before prices by 2 months, with caution.",
        "caveats": ["Short sample"], "ideas": [
            {"title": "Brainstorm", "description": "speculative", "kind": "brainstormed"},
            {"title": "Early indicator", "description": "track prices with rents", "kind": "verified"}]}
DROP = {"verdict": "drop", "rationale": "One is a part of the other.", "drop_reason": "component_of_total"}


class Resp:
    def __init__(self, status=200, content=None, usage=None, text=""):
        self.status_code, self.text = status, text
        self._body = {"choices": [{"message": {"content": content}}],
                      "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20}}

    def json(self):
        return self._body


class FakePost:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(post, model="m/test"):
    return R.Client(api_key=KEY, model=model, post=post, sleep=lambda s: None)


def make_store(tmp_path, n_cands=3):
    rows = []
    for i in range(n_cands):
        rows.append({"kind": "lead" if i % 2 else "comove", "x": f"d1:a{i}", "y": "d2:b", "freq": "M", "lag": 2 if i % 2 else 0,
                     "n_train": 20, "n_test": 10, "r_train": 0.5, "r_test": 0.4, "p_test": 0.01, "p_test_adj": 0.04,
                     "r_test_spearman": 0.35, "r_ex_covid": 0.3, "span": "2020-01..2022-02", "transform": "diff",
                     "name_overlap": 0.0, "score": 10 - i})
    pd.DataFrame(rows).to_parquet(tmp_path / "candidates.parquet")
    ids = [f"d1:a{i}" for i in range(n_cands)] + ["d2:b"]
    pd.DataFrame([{"series_id": s, "dataset_id": s.split(":")[0], "label": f"label {s}", "dims": "{}", "measure": None,
                   "unit": "Dollars", "freq": "M", "n_points": len(PERIODS), "start": PERIODS[0], "end": PERIODS[-1]}
                  for s in ids]).to_parquet(tmp_path / "series_meta.parquet")
    pd.DataFrame([{"series_id": s, "period": p, "value": float(j) + k} for k, s in enumerate(ids)
                  for j, p in enumerate(PERIODS)]).to_parquet(tmp_path / "series_points.parquet")
    common = {"before": 0.01, "after": 0.02, "duplicate_of": None}
    pd.DataFrame([{"series_id": "d2:b", "period": "2020-04", "statistic": 30.0, "p": 0.001, "direction": "slowed",
                   "size": 5.0, **common},
                  {"series_id": "d1:a0", "period": "2021-07", "statistic": 14.0, "p": 0.004, "direction": "accelerated",
                   "size": 2.0, **common},
                  {"series_id": "d1:a1", "period": "2021-07", "statistic": 40.0, "p": 0.001, "direction": "accelerated",
                   "size": 2.0, **{**common, "duplicate_of": "d1:a0"}}]
                 ).to_parquet(tmp_path / "breaks.parquet")
    return tmp_path


def break_reply():
    return {"notes": [
        {"series_id": "d2:b", "period": "2020-04", "note": "Guess: COVID circuit breaker.", "event": "COVID-19", "confidence": "known_event"},
        {"series_id": "d1:a0", "period": "2021-07", "note": "Guess only.", "event": None, "confidence": "guess"}]}


def router(replies_for_review):
    """A fake API that answers break prompts with break notes and review prompts in order."""
    queue = list(replies_for_review)

    def post(url, headers=None, json=None, timeout=None):
        post.calls.append(json)
        user = json["messages"][-1]["content"]
        body = break_reply() if user.startswith("Breaks to annotate") else queue.pop(0)
        return Resp(content=body if isinstance(body, str) else __import__("json").dumps(body))
    post.calls = []
    return post


def test_review_prompt_carries_labels_stats_lag_and_last_24_aligned_points(tmp_path):
    store = make_store(tmp_path)
    cand = pd.read_parquet(store / "candidates.parquet").to_dict("records")[1]
    meta = {r["series_id"]: r for r in pd.read_parquet(store / "series_meta.parquet").to_dict("records")}
    pts = {}
    for r in pd.read_parquet(store / "series_points.parquet").itertuples():
        pts.setdefault(r.series_id, {})[r.period] = r.value
    prompt = R.build_review_prompt(cand, meta, pts, {"d1": "Rental Index"})
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["kind"] == "lead" and payload["lag"] == 2 and "before y" in payload["meaning"]
    assert payload["x"]["label"] == "label d1:a1" and payload["x"]["dataset"] == "Rental Index"
    assert payload["y"]["dataset"] == "d2" and payload["x"]["unit"] == "Dollars"
    assert list(payload["x"]["excerpt"]) == PERIODS[-24:] and len(payload["y"]["excerpt"]) == 24
    assert payload["stats"]["r_test"] == 0.4 and payload["stats"]["p_test_adj"] == 0.04


def test_system_prompt_is_strict_and_forbids_causation_and_invented_numbers():
    s = R.REVIEW_SYSTEM
    assert "When in doubt, drop" in s and "Never claim causation" in s and "Never invent" in s
    for reason in R.DROP_REASONS:
        assert reason in s


def test_a_valid_keep_and_drop_parse_and_verified_ideas_sort_first():
    kept = R.validate_review(KEEP)
    assert [i["kind"] for i in kept["ideas"]] == ["verified", "brainstormed"]
    assert R.validate_review(DROP)["drop_reason"] == "component_of_total"


@pytest.mark.parametrize("bad", [
    {"verdict": "maybe"}, {"verdict": "drop", "drop_reason": "boring"}, {"verdict": "drop", "drop_reason": None},
    {**KEEP, "drop_reason": "same_quantity"}, {**KEEP, "headline": ""},
    {**KEEP, "ideas": [{"title": "t", "description": "d", "kind": "maybe"}]},
])
def test_invalid_replies_are_rejected(bad):
    with pytest.raises(R.ReviewError):
        R.validate_review(bad)


def test_invalid_json_is_retried_once_with_the_error_fed_back_then_succeeds(tmp_path):
    post = FakePost(Resp(content="not json {"), Resp(content=json.dumps(DROP)))
    got, hit = R.ask(client(post), tmp_path, "sys", "user", R.validate_review)
    assert got["verdict"] == "drop" and hit is False and len(post.calls) == 2
    retry_msgs = post.calls[1]["json"]["messages"]
    assert retry_msgs[-1]["role"] == "user" and "rejected" in retry_msgs[-1]["content"]


def test_two_invalid_replies_raise(tmp_path):
    post = FakePost(Resp(content="nope"), Resp(content="still nope"))
    with pytest.raises(R.ReviewError):
        R.ask(client(post), tmp_path, "sys", "user", R.validate_review)
    assert len(post.calls) == 2 and not list(tmp_path.glob("*.json"))


def test_429_and_5xx_back_off_then_succeed_and_other_4xx_fail_fast(tmp_path):
    sleeps = []
    c = R.Client(api_key=KEY, model="m", sleep=sleeps.append,
                 post=FakePost(Resp(status=429), Resp(status=503), requests.ConnectionError("x"), Resp(content="{}")))
    assert c.chat([{"role": "user", "content": "hi"}]) == "{}"
    assert sleeps == [2.0, 4.0, 8.0]
    c = R.Client(api_key=KEY, model="m", sleep=lambda s: None, post=FakePost(Resp(status=401, text=f"bad {KEY}")))
    with pytest.raises(R.ReviewError) as e:
        c.chat([])
    assert KEY not in str(e.value)


def test_request_uses_json_mode_timeout_and_bearer_auth(tmp_path):
    post = FakePost(Resp(content=json.dumps(DROP)))
    R.ask(client(post), tmp_path, "sys", "user", R.validate_review)
    call = post.calls[0]
    assert call["url"] == R.API_URL and call["timeout"] and call["json"]["response_format"] == {"type": "json_object"}
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_cache_hit_makes_no_call_and_a_different_model_misses(tmp_path):
    first = FakePost(Resp(content=json.dumps(KEEP)))
    a, _ = R.ask(client(first), tmp_path, "sys", "user", R.validate_review)
    second = FakePost()
    b, hit = R.ask(client(second), tmp_path, "sys", "user", R.validate_review)
    assert hit is True and a == b and second.calls == []
    other = FakePost(Resp(content=json.dumps(DROP)))
    c, hit = R.ask(client(other, model="m/other"), tmp_path, "sys", "user", R.validate_review)
    assert hit is False and c["verdict"] == "drop"


def test_keep_drop_split_and_error_candidates_are_never_kept(tmp_path):
    store = make_store(tmp_path)
    post = router([KEEP, DROP, "garbage", "garbage"])
    c = client(post)
    res = R.review(store, c, "m/test", top=3, log=lambda s: None)
    assert [f["x"] for f in res.findings] == ["d1:a0"]
    assert res.findings[0]["headline"] and res.findings[0]["model"] == "m/test" and res.findings[0]["r_test"] == 0.4
    assert res.dropped[0]["drop_reason"] == "component_of_total"
    assert res.dropped[1]["drop_reason"] == "review_error" and "invalid" in res.dropped[1]["review_error"]
    on_disk = json.loads((store / "findings.json").read_text())
    assert on_disk == json.loads(json.dumps(res.findings))
    assert len(json.loads((store / "dropped.json").read_text())) == 2


def test_top_n_takes_the_highest_scores(tmp_path):
    store = make_store(tmp_path)
    res = R.review(store, client(router([KEEP])), "m/test", top=1, log=lambda s: None)
    assert [f["x"] for f in res.findings] == ["d1:a0"] and res.dropped == []


def test_break_notes_are_one_per_top_break_and_guess_unless_a_known_event(tmp_path):
    store = make_store(tmp_path)
    R.review(store, client(router([KEEP, KEEP, KEEP])), "m/test", log=lambda s: None)
    notes = json.loads((store / "break_notes.json").read_text())
    assert [(n["series_id"], n["confidence"]) for n in notes] == [("d2:b", "known_event"), ("d1:a0", "guess")]
    assert R.BREAKS_SYSTEM.count('"guess"') and "unless the period clearly matches" in R.BREAKS_SYSTEM


def test_rerun_is_free_and_identical(tmp_path):
    store = make_store(tmp_path)
    R.review(store, client(router([KEEP, DROP, DROP])), "m/test", log=lambda s: None)
    before = {n: (store / n).read_text() for n in ("findings.json", "dropped.json", "break_notes.json")}
    post = FakePost()
    c = client(post)
    R.review(store, c, "m/test", log=lambda s: None)
    assert post.calls == [] and c.usage.cache_hits == 4
    assert before == {n: (store / n).read_text() for n in before}


def test_dry_run_prints_the_first_prompt_and_makes_no_network_call(tmp_path, capsys, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("network call in dry run"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert R.main(["--store", str(store), "--dry-run", "--model", "m/x"]) == 0
    out = capsys.readouterr().out
    assert "Candidate to review" in out and "label d1:a0" in out and "m/x" in out
    assert not (store / "findings.json").exists()


def test_key_is_in_no_written_file_or_log_line(tmp_path):
    failing, passing = tmp_path / "failing", tmp_path / "passing"
    failing.mkdir(), passing.mkdir()
    logs = []
    post = FakePost(*[Resp(status=500, text=KEY)] * 4, Resp(status=500), Resp(status=500), Resp(status=500), Resp(status=500))
    res = R.review(make_store(failing), client(post), "m/test", top=1, log=logs.append)
    assert res.dropped[0]["drop_reason"] == "review_error" and any("review_error" in line for line in logs)
    R.review(make_store(passing), client(router([KEEP, DROP, DROP])), "m/test", log=logs.append)
    files = [f for f in tmp_path.rglob("*") if f.is_file() and f.suffix == ".json"]
    assert files
    for f in files:
        assert KEY not in f.read_text(), f
    assert all(KEY not in line for line in logs)


def test_load_api_key_reads_env_file_without_exporting_it(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nexport OPENROUTER_API_KEY='abc123'\n")
    assert R.load_api_key(env) == "abc123"
    monkeypatch.setenv("OPENROUTER_API_KEY", "fromenv")
    assert R.load_api_key(env) == "fromenv"


def test_a_verdict_without_a_rationale_is_rejected():
    with pytest.raises(R.ReviewError):
        R.validate_review({k: v for k, v in DROP.items() if k != "rationale"})
