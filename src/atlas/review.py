"""Phase 4: an LLM pass over the findings engine's candidates.

Reads `candidates`, `breaks`, `series_meta` and `series_points` from a store directory, asks an
OpenRouter model to keep or drop each of the top candidates and to write the finding text and
project ideas, and writes `findings.json`, `dropped.json` and `break_notes.json` beside them.

    python -m src.atlas.review --store atlas [--top 60] [--model M] [--dry-run]

Replies are cached on disk under `<store>/review_cache/`, keyed by a hash of (model, prompt), so a
rerun makes no network call and returns the same answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
EXCERPT_POINTS = 24
BREAK_WINDOW = 8
TOP_BREAKS = 12
TIMEOUT_S = 90
MAX_HTTP_ATTEMPTS = 4
BACKOFF_S = 2.0
CACHE_DIR = "review_cache"
PROMPT_VERSION = "1"

DROP_REASONS = (
    "same_quantity", "component_of_total", "mechanical_identity", "trivially_obvious", "likely_artefact",
)
IDEA_KINDS = ("verified", "brainstormed")

REVIEW_SYSTEM = """\
You are the reviewing editor for govML, a public site that shows Singaporeans surprising, statistically
defensible relationships between government datasets. A statistical engine has already proposed the
candidate pair you are given. Your job is to decide whether it is worth showing, and if so to describe it.

The pair has already passed these statistical gates, so do not re-judge them from the excerpt: the lag was
chosen on the earlier 70% of the data and held on the later 30% (r_test) after a multiple-testing correction
(p_test_adj), using changes not levels with seasonality removed, and it still holds with 2020-2022 removed
(r_ex_covid) and by rank (r_test_spearman). Your job is the judgment statistics cannot make: is this one
quantity counted twice, an identity, something nobody would be surprised by, or an artefact of how the
tables are built?

Be strict. When in doubt, drop. A finding a visitor would share beats ten they would skim, and a bad
finding costs more than a missing one.

Drop the pair when it is:
- same_quantity: the two series measure the same thing, or one is a rescaled, rebased or re-cut copy of the other.
- component_of_total: one series is a part of the other (a total and one of its own components or sub-totals).
- mechanical_identity: they are linked by an accounting identity or definition (a ratio and its numerator, a stock and its flow, a price and an index built from it).
- trivially_obvious: the link is real but nobody would be surprised (rainfall and rainfall days, population and housing stock).
- likely_artefact: the link is probably produced by how the tables are built (shared base year, smoothing, one survey feeding both), or r_test is far above r_train, which suggests a single recent episode rather than a stable relationship.

Rules you must follow:
- Use only the numbers given to you. Never invent, estimate or recall figures, dates or events. Quote a number only if it appears in the input.
- Never claim causation. Say "moves with", "tends to rise before", "is associated with". For a lead relationship you may say x tends to move before y by the stated lag, and nothing stronger.
- Hedge. Mention the lag (in the series' own frequency units) and the period the relationship was observed over.
- headline: one plain sentence a curious Singaporean would share. No jargon, no statistics terms.
- explanation: 2 or 3 sentences, hedged, mentioning the lag and the period.
- caveats: a list of short strings, at least one.
- ideas: project ideas that build on this relationship. Each has title, description and kind.
  kind "verified" means the idea follows directly from the tested relationship (for example, track y using x as an early indicator).
  kind "brainstormed" means it is speculative, going beyond what was tested. Label honestly. Give at most 3 ideas in total.
- rationale: always 1 or 2 sentences saying why you kept or dropped it. This is for the editor, not the public.
- If you drop, set headline, explanation and ideas to empty values ("" and []) and put the reason in drop_reason. If you keep, drop_reason must be null.

Reply with a single JSON object and nothing else, with exactly these keys:
{"verdict": "keep" | "drop", "rationale": string,
 "drop_reason": "same_quantity" | "component_of_total" | "mechanical_identity" | "trivially_obvious" | "likely_artefact" | null,
 "headline": string, "explanation": string, "caveats": [string],
 "ideas": [{"title": string, "description": string, "kind": "verified" | "brainstormed"}]}
"""

BREAKS_SYSTEM = """\
You annotate structural breaks detected in Singapore government time series. For each break you are
given the series, the period the test flagged, whether growth sped up or slowed (direction), the mean
change per period before and after, the shift in standard deviations (size) and a few values around it.

For each break write one short note (one sentence) guessing which well-known event might coincide with it,
for example COVID-19 in 2020, property cooling measures, GST rate changes, or a change in how the series
is measured. Rules:
- Set confidence to "guess" unless the period clearly matches a well-known event. Set it to "known_event"
  only then, and name the event in "event". Say plainly in the note when it is a guess.
- Do not claim causation. Never invent numbers; use only those given.
- If nothing plausible fits, say so and set event to null, confidence "guess".

Reply with a single JSON object and nothing else:
{"notes": [{"series_id": string, "period": string, "note": string, "event": string | null, "confidence": "guess" | "known_event"}]}
with one entry per break, in the order given.
"""


class ReviewError(Exception):
    """A model reply that could not be obtained or validated."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    def add(self, u: dict[str, Any] | None) -> None:
        u = u or {}
        self.prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.completion_tokens += int(u.get("completion_tokens") or 0)


@dataclass
class Client:
    """The OpenRouter chat-completions boundary. `post` and `sleep` are injectable for tests."""

    api_key: str
    model: str
    post: Callable[..., Any] = requests.post
    sleep: Callable[[float], None] = time.sleep
    usage: Usage = field(default_factory=Usage)

    def scrub(self, text: str) -> str:
        return text.replace(self.api_key, "[redacted]") if self.api_key else text

    def chat(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-Title": "govML atlas review"}
        last = "no attempt made"
        for attempt in range(MAX_HTTP_ATTEMPTS):
            if attempt:
                self.sleep(BACKOFF_S * 2 ** (attempt - 1))
            try:
                resp = self.post(API_URL, headers=headers, json=body, timeout=TIMEOUT_S)
            except requests.RequestException as e:
                last = f"network error {type(e).__name__}"
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"http {resp.status_code}"
                continue
            if resp.status_code != 200:
                raise ReviewError(f"http {resp.status_code}: {self.scrub(resp.text[:300])}")
            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                last = "malformed response envelope"
                continue
            self.usage.add(data.get("usage"))
            self.usage.calls += 1
            if not isinstance(content, str):
                raise ReviewError("reply content is not text")
            return content
        raise ReviewError(f"gave up after {MAX_HTTP_ATTEMPTS} attempts: {last}")


def parse_json_object(text: str) -> dict[str, Any]:
    s = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    if fence:
        s = fence.group(1)
    try:
        obj = json.loads(s)
    except ValueError as e:
        raise ReviewError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ReviewError("reply is not a JSON object")
    return obj


def _text(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str):
        raise ReviewError(f"{key} must be a string")
    return v.strip()


def validate_review(obj: dict[str, Any]) -> dict[str, Any]:
    verdict = obj.get("verdict")
    if verdict not in ("keep", "drop"):
        raise ReviewError("verdict must be keep or drop")
    reason = obj.get("drop_reason")
    rationale = _text(obj, "rationale")
    if not rationale:
        raise ReviewError("every verdict needs a rationale")
    if verdict == "drop":
        if reason not in DROP_REASONS:
            raise ReviewError("a drop needs a valid drop_reason")
        return {"verdict": "drop", "drop_reason": reason, "rationale": rationale, "headline": "",
                "explanation": "", "caveats": [], "ideas": []}
    if reason is not None:
        raise ReviewError("a keep must have drop_reason null")
    headline, explanation = _text(obj, "headline"), _text(obj, "explanation")
    if not headline or not explanation:
        raise ReviewError("a keep needs a headline and an explanation")
    caveats = obj.get("caveats")
    if isinstance(caveats, str):
        caveats = [caveats]
    if not isinstance(caveats, list) or not all(isinstance(c, str) for c in caveats):
        raise ReviewError("caveats must be a list of strings")
    ideas = obj.get("ideas", [])
    if not isinstance(ideas, list):
        raise ReviewError("ideas must be a list")
    clean = []
    for i in ideas:
        if not isinstance(i, dict) or i.get("kind") not in IDEA_KINDS:
            raise ReviewError("each idea needs a kind of verified or brainstormed")
        clean.append({"title": _text(i, "title"), "description": _text(i, "description"), "kind": i["kind"]})
    clean.sort(key=lambda i: i["kind"] != "verified")
    return {"verdict": "keep", "drop_reason": None, "rationale": rationale, "headline": headline, "explanation": explanation,
            "caveats": [c.strip() for c in caveats if c.strip()], "ideas": clean}


def validate_break_notes(obj: dict[str, Any], expected: list[tuple[str, str]]) -> list[dict[str, Any]]:
    notes = obj.get("notes")
    if not isinstance(notes, list):
        raise ReviewError("notes must be a list")
    by_key = {}
    for n in notes:
        if not isinstance(n, dict):
            raise ReviewError("each note must be an object")
        if n.get("confidence") not in ("guess", "known_event"):
            raise ReviewError("confidence must be guess or known_event")
        event = n.get("event")
        if event is not None and not isinstance(event, str):
            raise ReviewError("event must be a string or null")
        by_key[(n.get("series_id"), n.get("period"))] = {
            "series_id": n["series_id"], "period": n["period"], "note": _text(n, "note"),
            "event": event, "confidence": "known_event" if n["confidence"] == "known_event" and event else "guess",
        }
    missing = [k for k in expected if k not in by_key]
    if missing:
        raise ReviewError(f"no note for {len(missing)} break(s)")
    return [by_key[k] for k in expected]


def ask(client: Client, cache_dir: Path, system: str, user: str, validate: Callable[[dict], Any]) -> tuple[Any, bool]:
    """One cached model call. Returns (validated reply, cache hit). Invalid JSON or an invalid
    schema is retried once with the error fed back; a second failure raises ReviewError."""
    key = hashlib.sha256(json.dumps([client.model, PROMPT_VERSION, system, user]).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            client.usage.cache_hits += 1
            return validate(cached), True
        except (ValueError, ReviewError):
            path.unlink(missing_ok=True)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = ""
    for attempt in range(2):
        content = client.chat(messages)
        try:
            raw = parse_json_object(content)
            result = validate(raw)
        except ReviewError as e:
            problem = str(e)
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That reply was rejected: {problem}. "
                                            "Reply again with only the corrected JSON object."},
            ]
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, ensure_ascii=False))
        return result, False
    raise ReviewError(f"invalid reply after retry: {problem}")


def _num(v: Any) -> Any:
    if isinstance(v, float):
        return None if math.isnan(v) else float(f"{v:.4g}")
    if hasattr(v, "item"):
        return _num(v.item())
    return v


def _excerpt(points: dict[str, float], periods: list[str]) -> dict[str, Any]:
    return {p: _num(points[p]) for p in periods}


def _series_block(meta: Any, points: dict[str, float], periods: list[str], dataset_names: dict[str, str]) -> dict:
    try:
        dims = json.loads(meta["dims"]) if isinstance(meta["dims"], str) else (meta["dims"] or {})
    except ValueError:
        dims = {}
    return {
        "series_id": meta["series_id"], "label": meta["label"], "unit": meta["unit"] or None, "dims": dims,
        "measure": meta["measure"] or None, "freq": meta["freq"],
        "dataset": dataset_names.get(meta["dataset_id"]) or meta["dataset_id"],
        "span": f"{meta['start']}..{meta['end']}", "excerpt": _excerpt(points, periods),
    }


STAT_KEYS = ("n_train", "n_test", "r_train", "r_test", "p_test", "p_test_adj", "r_test_spearman",
             "r_ex_covid", "span", "transform", "name_overlap", "score")


def build_review_prompt(cand: dict, metas: dict, points: dict[str, dict[str, float]],
                        dataset_names: dict[str, str]) -> str:
    """The user message for one candidate: both series, the stats, and the last aligned points."""
    x, y = cand["x"], cand["y"]
    aligned = sorted(set(points[x]) & set(points[y]))[-EXCERPT_POINTS:]
    payload = {
        "kind": cand["kind"],
        "meaning": ("x and y move together at the same time" if cand["kind"] == "comove"
                    else f"x tends to move before y, by {cand['lag']} period(s) of freq {cand['freq']}"),
        "freq": cand["freq"], "lag": _num(cand["lag"]),
        "stats": {k: _num(cand.get(k)) for k in STAT_KEYS},
        "x": _series_block(metas[x], points[x], aligned, dataset_names),
        "y": _series_block(metas[y], points[y], aligned, dataset_names),
        "excerpt_note": f"excerpts are the last {len(aligned)} periods both series share, as raw values",
    }
    return "Candidate to review:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


def top_breaks(breaks: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n strongest breaks of distinct quantities: lowest p first, then the largest statistic."""
    if breaks.empty:
        return breaks
    b = breaks[breaks["duplicate_of"].isna()]
    b = b.sort_values(["p", "statistic", "series_id"], ascending=[True, False, True], kind="stable")
    return b.head(n).reset_index(drop=True)


def build_breaks_prompt(top: pd.DataFrame, metas: dict, points: dict[str, dict[str, float]],
                        dataset_names: dict[str, str]) -> str:
    items = []
    for r in top.itertuples(index=False):
        pts = points[r.series_id]
        periods = list(pts)
        i = periods.index(r.period) if r.period in pts else len(periods) // 2
        window = periods[max(0, i - BREAK_WINDOW): i + BREAK_WINDOW]
        m = metas[r.series_id]
        items.append({
            "series_id": r.series_id, "label": m["label"], "unit": m["unit"] or None, "freq": m["freq"],
            "dataset": dataset_names.get(m["dataset_id"]) or m["dataset_id"], "break_period": r.period,
            "direction": r.direction, "before": _num(r.before), "after": _num(r.after), "size": _num(r.size),
            "p": _num(r.p),
            "values_around_break": _excerpt(pts, window),
        })
    return "Breaks to annotate:\n" + json.dumps(items, ensure_ascii=False, indent=1)


# --- boundary: environment and store ---------------------------------------------------------

def load_api_key(env_path: Path | None = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    for path in [env_path] if env_path else [Path(".env"), Path(__file__).resolve().parents[2] / ".env"]:
        if path and path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                name, sep, value = line.partition("=")
                if sep and name.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip("'\"")
    return ""


def _read_table(store: Path, name: str, required: bool = True) -> pd.DataFrame:
    pq, js = store / f"{name}.parquet", store / f"{name}.json"
    if pq.exists():
        return pd.read_parquet(pq)
    if js.exists():
        return pd.DataFrame(json.loads(js.read_text()))
    if required:
        raise SystemExit(f"missing {name}.parquet (or .json) in {store}")
    return pd.DataFrame()


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else _num(v)) for k, v in r.items()})
    return out


@dataclass
class Result:
    findings: list[dict]
    dropped: list[dict]
    break_notes: list[dict]
    usage: Usage


def load_dataset_names(store: Path, catalog: Path = Path("catalog.db")) -> dict[str, str]:
    """`<store>/dataset_names.json` when present, else the catalog's names, else none."""
    names_file = store / "dataset_names.json"
    if names_file.exists():
        return json.loads(names_file.read_text())
    if catalog.exists():
        with sqlite3.connect(catalog) as conn:
            return dict(conn.execute("SELECT dataset_id, name FROM datasets"))
    return {}


def review(store: Path, client: Client | None, model: str, top: int = 60, dry_run: bool = False,
           log: Callable[[str], None] = print) -> Result | None:
    cands = _read_table(store, "candidates").sort_values("score", ascending=False).head(top)
    meta = _read_table(store, "series_meta")
    pts_df = _read_table(store, "series_points")
    breaks = _read_table(store, "breaks", required=False)
    metas = {r["series_id"]: r for r in meta.to_dict("records")}
    points = {sid: dict(zip(g["period"], g["value"])) for sid, g in pts_df.sort_values("period").groupby("series_id")}
    dataset_names = load_dataset_names(store)
    records = _records(cands)

    if dry_run:
        if not records:
            log("no candidates")
            return None
        log(f"[model {model}]\n--- system ---\n{REVIEW_SYSTEM}\n--- user ---\n"
            + build_review_prompt(records[0], metas, points, dataset_names))
        return None

    assert client is not None
    cache_dir = store / CACHE_DIR
    findings, dropped = [], []
    for c in records:
        base = {k: c.get(k) for k in ("kind", "x", "y", "freq", "lag", *STAT_KEYS)}
        base["x_label"], base["y_label"] = metas[c["x"]]["label"], metas[c["y"]]["label"]
        prompt = build_review_prompt(c, metas, points, dataset_names)
        try:
            reply, _ = ask(client, cache_dir, REVIEW_SYSTEM, prompt, validate_review)
        except ReviewError as e:
            dropped.append({**base, "drop_reason": "review_error", "review_error": client.scrub(str(e))})
            log(f"review_error {c['x']} / {c['y']}: {client.scrub(str(e))}")
            continue
        if reply["verdict"] == "keep":
            findings.append({**base, **{k: reply[k] for k in ("headline", "explanation", "caveats", "ideas", "rationale")},
                             "model": model})
        else:
            dropped.append({**base, "drop_reason": reply["drop_reason"], "rationale": reply["rationale"]})

    notes: list[dict] = []
    if not breaks.empty:
        tb = top_breaks(breaks, TOP_BREAKS)
        expected = [(r.series_id, r.period) for r in tb.itertuples(index=False)]
        try:
            notes, _ = ask(client, cache_dir, BREAKS_SYSTEM, build_breaks_prompt(tb, metas, points, dataset_names),
                           lambda o: validate_break_notes(o, expected))
        except ReviewError as e:
            log(f"break notes failed: {client.scrub(str(e))}")

    def dump(name: str, obj: Any) -> None:
        (store / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

    dump("findings.json", findings)
    dump("dropped.json", dropped)
    dump("break_notes.json", notes)
    u = client.usage
    log(f"model {model}: {len(findings)} kept, {len(dropped)} dropped, {len(notes)} break notes; "
        f"{u.calls} api calls, {u.cache_hits} cache hits, "
        f"{u.prompt_tokens} prompt + {u.completion_tokens} completion tokens")
    return Result(findings, dropped, notes, u)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM review of atlas candidates")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--model", default=os.environ.get("GOVML_LLM_MODEL") or DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt; no network call")
    a = ap.parse_args(argv)
    if a.dry_run:
        review(a.store, None, a.model, a.top, dry_run=True)
        return 0
    key = load_api_key()
    if not key:
        print("OPENROUTER_API_KEY is not set (environment or .env)", file=sys.stderr)
        return 2
    review(a.store, Client(api_key=key, model=a.model), a.model, a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
