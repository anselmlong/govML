"""Batch ML runner: runs the pipeline over all suitable datasets and stores
insights back into catalog.db as embeddings for semantic retrieval.

Usage:
    python batch_ml.py [--tones good,okay,marginal] [--limit N] [--workers 3]

Each completed run produces one insight record:
  - run_id, dataset_id, target, task_type, best_model
  - metrics (MAE/RMSE/R2 or acc/f1), verdict_text/tone from the honest-verdict system
  - insight_text: a compact natural-language summary of what was learned,
    including cross-dataset correlations — this is what gets embedded so that
    search/ask can surface connections between datasets.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.catalog import DB_PATH, connect  # noqa: E402
from src.fetch import fetch_resource, GoneError, ThrottledError  # noqa: E402
from src.correlation import correlate_related  # noqa: E402
from src.llm import embed_texts  # noqa: E402
from src.preprocess import find_time_column  # noqa: E402
from src.train import TASK_REGRESSION, train_models  # noqa: E402
from src.agents import propose_targets  # noqa: E402

# If this fraction of a pass comes back throttled, the upstream is rate-limiting
# us systemically, not a few hot ids. Skip the serial pass-2 retry loop and leave
# the throttled sets for the next run instead of grinding for hours/day.
THROTTLE_BREAKER_RATIO = 0.5
from src.preprocess_planner import build_plan, execute_plan  # noqa: E402

INSIGHT_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL UNIQUE REFERENCES datasets(dataset_id),
    run_id TEXT,
    target TEXT,
    task_type TEXT,
    best_model TEXT,
    metrics_json TEXT,
    verdict_tone TEXT,
    verdict_text TEXT,
    insight_text TEXT NOT NULL,
    correlations_json TEXT,
    embedding BLOB,
    embedding_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def ensure_schema(db_path: Path = DB_PATH) -> None:
    conn = connect(db_path)
    conn.executescript(INSIGHT_SCHEMA)
    conn.commit()
    conn.close()


def eligible_ids(tones: list[str], db_path: Path = DB_PATH) -> list[str]:
    conn = connect(db_path)
    q = "SELECT dataset_id FROM datasets WHERE suitability_tone IN (%s) ORDER BY suitability_score DESC" % ",".join(
        "?" * len(tones)
    )
    ids = [r["dataset_id"] for r in conn.execute(q, tones)]
    conn.close()
    return ids


def fmt(v) -> str:
    try:
        x = float(v)
        return f"{x:,.4g}"
    except Exception:
        return "n/a"


def build_insight_text(name: str, target: str, chosen_task: str, best_name: str, best_metrics: dict,
                       n_rows: int, n_features: int, verdict: dict, correlations: list[dict]) -> str:
    """Compact natural-language summary of a run — the unit that gets embedded."""
    if chosen_task == TASK_REGRESSION:
        metric_line = (
            f"best model {best_name} predicts {target} with MAE={fmt(best_metrics.get('MAE'))}, "
            f"RMSE={fmt(best_metrics.get('RMSE'))}, R2={fmt(best_metrics.get('R2'))}"
        )
    else:
        primary = "roc_auc" if "roc_auc" in best_metrics else "f1"
        metric_line = (
            f"best classifier {best_name} predicts {target} with "
            f"{primary}={fmt(best_metrics.get(primary))}, accuracy={fmt(best_metrics.get('acc'))}"
        )
    parts = [
        f"ML analysis of '{name}' ({n_rows:,} rows, {n_features} features).",
        f"Target: {target} ({chosen_task}). {metric_line}.",
    ]
    vt = (verdict or {}).get("text")
    if vt:
        parts.append(f"Verdict: {vt}")
    if correlations:
        top = correlations[:3]
        corr_bits = "; ".join(
            f"'{c['name']}' r={c['r']:+.3f} over {c['overlap_months']} aligned months"
            + (" (strong)" if abs(c['r']) >= 0.7 else " (moderate)" if abs(c['r']) >= 0.4 else " (weak)")
            for c in top
        )
        parts.append(f"Cross-dataset correlations with {target}: {corr_bits}.")
    return " ".join(parts)


def _related_rows(dataset_id: str) -> list[dict]:
    """Fallback related datasets if no embeddings are stored yet."""
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT dataset_id, name FROM datasets WHERE dataset_id != ? ORDER BY name LIMIT 8",
            (dataset_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def run_one(dataset_id: str, name: str, max_rows: int = 10000, no_correlation: bool = False) -> dict | None:
    """Lightweight in-process version of pipeline.py — enough to produce insights."""
    df = fetch_resource(dataset_id, max_rows=max_rows)
    if df.empty or len(df) < 10:
        return None

    plan = build_plan(df, target=None, use_llm=False)
    clean, _notes = execute_plan(df, plan)
    if clean.empty:
        return None

    report = propose_targets(clean, use_llm=False, forced_task_type=None)
    if not report.proposals:
        return None
    chosen = report.proposals[0]

    train_result = train_models(clean, chosen.target, chosen.task_type)

    correlations: list[dict] = []
    if not no_correlation and train_result.task_type == TASK_REGRESSION:
        try:
            corr_results = correlate_related(
                clean,
                chosen.target,
                dataset_id,
                related=[{"dataset_id": r["dataset_id"], "name": r["name"]} for r in _related_rows(dataset_id)],
            )
            correlations = [
                {
                    "dataset_id": c.dataset_id,
                    "name": c.name,
                    "r": c.r,
                    "overlap_months": c.overlap_months,
                    "similarity": c.similarity,
                    "proxy_column": c.proxy_column,
                    "correlation_note": c.correlation_note,
                }
                for c in corr_results
                if c.correlation_note == "numeric-timeseries" or c.similarity is not None
            ]
        except Exception:
            correlations = []

    best = train_result.leaderboard[0]
    insight = build_insight_text(
        name=name,
        target=chosen.target,
        chosen_task=train_result.task_type,
        best_name=best.name,
        best_metrics=best.metrics,
        n_rows=len(clean),
        n_features=len(train_result.numeric_features) + len(train_result.categorical_features),
        verdict={"text": getattr(train_result, "verdict_text", "")},
        correlations=correlations,
    )

    return {
        "target": chosen.target,
        "task_type": train_result.task_type,
        "best_model": best.name,
        "metrics": best.metrics,
        "verdict_tone": getattr(train_result, "verdict_tone", "okay"),
        "verdict_text": getattr(train_result, "verdict_text", ""),
        "insight_text": insight,
        "correlations": correlations,
    }


def _saved_since(dataset_id: str, cutoff: str, db_path: Path = DB_PATH) -> bool:
    conn = connect(db_path)
    row = conn.execute(
        "SELECT updated_at FROM run_insights WHERE dataset_id=?", (dataset_id,)
    ).fetchone()
    conn.close()
    return bool(row) and str(row["updated_at"]) > cutoff


def save_insight(dataset_id: str, result: dict, run_id: str | None = None, db_path: Path = DB_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = connect(db_path)
    existing = conn.execute("SELECT id FROM run_insights WHERE dataset_id=?", (dataset_id,)).fetchone()
    if existing:
        conn.execute(
            """UPDATE run_insights SET run_id=?, target=?, task_type=?, best_model=?, metrics_json=?,
               verdict_tone=?, verdict_text=?, insight_text=?, correlations_json=?, updated_at=? WHERE dataset_id=?""",
            (
                run_id, result["target"], result["task_type"], result["best_model"],
                json.dumps(result["metrics"]), result["verdict_tone"], result["verdict_text"],
                result["insight_text"], json.dumps(result["correlations"]), now, dataset_id,
            ),
        )
    else:
        conn.execute(
            """INSERT INTO run_insights (dataset_id, run_id, target, task_type, best_model, metrics_json,
               verdict_tone, verdict_text, insight_text, correlations_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                dataset_id, run_id, result["target"], result["task_type"], result["best_model"],
                json.dumps(result["metrics"]), result["verdict_tone"], result["verdict_text"],
                result["insight_text"], json.dumps(result["correlations"]), now, now,
            ),
        )
    conn.commit()
    conn.close()


def embed_pending(batch_size: int = 64, db_path: Path = DB_PATH) -> int:
    """Embed insights that don't have an embedding yet."""
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT dataset_id, insight_text FROM run_insights WHERE embedding IS NULL"
    ).fetchall()
    conn.close()
    done = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        texts = [r["insight_text"] for r in chunk]
        res = embed_texts(texts)
        vectors = res.value or []
        if len(vectors) != len(chunk):
            continue
        conn = connect(db_path)
        for r, vec in zip(chunk, vectors):
            blob = json.dumps(vec).encode()
            conn.execute(
                "UPDATE run_insights SET embedding=?, embedding_model=? WHERE dataset_id=?",
                (blob, res.provenance, r["dataset_id"]),
            )
        conn.commit()
        conn.close()
        done += len(chunk)
        print(f"[embed] {done}/{len(rows)}")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tones", default="good,okay,marginal")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--retrain-existing", action="store_true", help="re-run datasets already in run_insights")
    ap.add_argument("--no-correlation", action="store_true", help="skip numeric cross-correlation (faster)")
    ap.add_argument("--since", default=None, help="resume: only retrain datasets last saved before this ISO timestamp")
    args = ap.parse_args()

    ensure_schema()

    if args.embed_only:
        n = embed_pending()
        print(f"embedded {n} insights")
        return 0

    tones = [t.strip() for t in args.tones.split(",")]
    ids = eligible_ids(tones)
    conn = connect(DB_PATH)
    already = {r["dataset_id"] for r in conn.execute("SELECT dataset_id FROM run_insights").fetchall()}
    conn.close()
    if args.retrain_existing:
        todo = list(ids)
    else:
        todo = [i for i in ids if i not in already]
    # resume support: in retrain mode, skip datasets already re-saved since a
    # given cutoff so OOM restarts gain ground instead of redoing from the top.
    # datasets never saved (not in `already`) must always run.
    if args.retrain_existing and args.since:
        todo = [i for i in todo if i not in already or not _saved_since(i, args.since)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(ids)} eligible, {len(already)} done, {len(todo)} to run (retrain_existing={args.retrain_existing})")

    names: dict[str, str] = {}
    conn = connect(DB_PATH)
    for r in conn.execute("SELECT dataset_id, name FROM datasets"):
        names[r["dataset_id"]] = r["name"]
    conn.close()

    t0 = time.time()
    run_done = 0  # total futures handled across passes

    def run_pass(ds_ids: list[str], workers: int, note: str) -> tuple[list[str], int, int]:
        """Run a pass over ds_ids. Returns (throttled_ids_to_retry, ok_count, fail_count).

        ThrottledError = dataset alive but upstream transiently throttling us — retry later.
        GoneError = dataset truly retired — terminal. Clean skips (empty df) are terminal."""
        nonlocal run_done
        retry_ids: list[str] = []
        w_ok = w_fail = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one, d, names.get(d, d), 10000, args.no_correlation): d
                for d in ds_ids
            }
            for fut in as_completed(futures):
                # gentle pacing: data.gov.sg rate-limits bursts aggressively
                time.sleep(0.6)
                d = futures[fut]
                run_done += 1
                try:
                    result = fut.result()
                    if result:
                        save_insight(d, result)
                        w_ok += 1
                        print(f"[{run_done}/{len(todo)}] OK  {d} ({time.time()-t0:.0f}s) {note}")
                    else:
                        w_fail += 1
                        print(f"[{run_done}/{len(todo)}] SKIP {d}: empty/too few rows {note}")
                except GoneError as exc:
                    w_fail += 1
                    print(f"[{run_done}/{len(todo)}] GONE {d}: {str(exc)[:90]} {note}")
                except ThrottledError as exc:
                    # alive but throttled — retry gently; don't drop permanently
                    retry_ids.append(d)
                    print(f"[{run_done}/{len(todo)}] RETRY {d}: throttled, alive {note}")
                except Exception as exc:
                    w_fail += 1
                    print(f"[{run_done}/{len(todo)}] FAIL {d}: {str(exc)[:120]} {note}")
        return retry_ids, w_ok, w_fail

    pass2_throttled, ok, fail = run_pass(todo, args.workers, "[pass 1]")
    # A second, gentler pass over only the throttled-but-alive ids now that the
    # first burst has drained — gives data.gov.sg's silent throttle time to clear.
    # But if the throttle is systemic (>>half the pass throttled), the whole
    # upstream is rate-limiting us, not a few hot ids. Re-hammering every id
    # serially would just burn hours/days fetcher-retrying a wall that won't
    # clear. Detect that and skip pass 2: sink what landed, leave the rest for
    # the next run (they're tracked by --since), and exit instead of spinning.
    throttle_ratio = len(pass2_throttled) / max(1, len(todo))
    pass2_still: list[str] = []
    if throttle_ratio >= THROTTLE_BREAKER_RATIO:
        print(f"\n{len(pass2_throttled)}/{len(todo)} throttled "
              f"({throttle_ratio:.0%}) = systemic upstream throttle; "
              "skipping pass 2, leaving them for next run.")
        pass2_still = pass2_throttled
    elif pass2_throttled:
        print(f"\n{len(pass2_throttled)} datasets throttled but alive. retrying gently...")
        time.sleep(5)
        pass2_still, ok2, fail2 = run_pass(
            pass2_throttled, max(1, args.workers // 2), "[pass 2 retry]"
        )
        ok += ok2
        fail += fail2
        if pass2_still:
            print(f"{len(pass2_still)} still throttled after retry — leaving for next run.")

    print(f"runs complete: {ok} ok, {fail} failed/skipped, "
          f"{len(pass2_still)} throttled. Embedding...")
    embed_pending()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
