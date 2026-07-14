#!/usr/bin/env python
"""govML nine-phase pipeline runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from src.agents import propose_targets
from src.catalog import related as catalog_related
from src.catalog import search as catalog_search
from src.context import add_context_features
from src.context_extractor import extract_context
from src.correlation import correlate_related
from src.diagnostics import classification_diagnostics, regression_diagnostics
from src.fetch import fetch_resource
from src.preprocess import build_eda_report
from src.preprocess_planner import build_plan, execute_plan
from src.report import render_report
from src.research import collect_research
from src.train import TASK_REGRESSION, train_models

DEFAULT_RESOURCE_ID = "f1765b54-a209-4718-8d38-a39237f502b3"
DEFAULT_NAME = "HDB Resale Flat Prices (Jan 2017+)"
DEFAULT_TOPIC = "Singapore HDB resale flat prices market trends"


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")[:80] or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the govML pipeline.")
    parser.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--max-rows", type=int, default=30_000, help="0 means all rows up to 200,000")
    parser.add_argument("--target")
    parser.add_argument("--no-research", action="store_true")
    parser.add_argument("--engine-fresh", action="store_true")
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--task-type", choices=["regression", "binary_classification", "multiclass_classification"])
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-ttl-days", type=int)
    parser.add_argument("--no-correlation", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    output = Path("output")
    output.mkdir(exist_ok=True)

    print("[1/9] research (last30days).")
    if args.no_research:
        research_text, research_provenance = "", "skipped"
        print("skipped by flag.")
    else:
        research_text, research_provenance = collect_research(
            args.topic,
            output,
            engine_fresh=args.engine_fresh,
            use_llm=not args.no_agent,
        )
        print(f"provenance: {research_provenance}, {len(research_text)} chars.")

    print("[2/9] fetch.")
    df_raw = fetch_resource(
        args.resource_id,
        max_rows=args.max_rows,
        refresh=args.refresh_cache,
        cache_ttl_days=args.cache_ttl_days,
    )
    print(f"raw shape: {df_raw.shape}.")

    print("[3/9] preprocess (agent-planned) + EDA.")
    plan = build_plan(df_raw, target=None, use_llm=not args.no_agent)
    print(f"plan: {plan.provenance} - {len(plan.steps)} steps.")
    for step in plan.steps:
        print(f"  - {step.op}: {step.column or ','.join(step.columns)}")
    df_clean, preprocess_notes = execute_plan(df_raw, plan)

    print("[4/9] propose target (agent).")
    target_report = propose_targets(
        df_clean,
        research_text=research_text,
        use_llm=not args.no_agent,
        forced_target=args.target,
        forced_task_type=args.task_type,
    )
    print(f"provenance: {target_report.provenance}, {len(target_report.proposals)} proposal(s).")
    for proposal in target_report.proposals:
        print(f"  - {proposal.target} [{proposal.task_type}] conf={proposal.confidence:.2f}.")
    if not target_report.proposals:
        raise RuntimeError("no target proposals available")
    chosen = target_report.proposals[0]
    print(f"chosen: {chosen.target}.")
    print(f"task type: {args.task_type or chosen.task_type}.")
    eda = build_eda_report(df_clean, chosen.target)
    if "mean" in eda.target_stats and eda.target_stats["mean"] is not None:
        print(f"eda: rows={eda.row_count}, target mean={eda.target_stats['mean']:.4g}.")
    else:
        print(f"eda: rows={eda.row_count}, target n_unique={eda.target_stats.get('n_unique', 0)}.")

    print("[5/9] context (agent-extracted facts + features).")
    context_pack = extract_context(research_text, use_llm=not args.no_agent)
    print(f"provenance: {context_pack.provenance}, {len(context_pack.facts)} fact(s).")
    df_context, context_notes = add_context_features(df_clean, context_pack)
    if context_notes:
        print("added policy_regime + mop_wave_active columns.")
        preprocess_notes.extend([f"added {note}" for note in context_notes])

    print("[6/9] train (multi-model bank).")
    train_result = train_models(df_context, chosen.target, args.task_type or chosen.task_type)

    print("[7/9] diagnose.")
    if train_result.task_type == TASK_REGRESSION:
        diag = regression_diagnostics(train_result)
        print(f"MAE {diag.mae:.4g} / MAPE {diag.mape:.4g}% / R2 {diag.r2:.4g}.")
    else:
        diag = classification_diagnostics(train_result)
        roc = train_result.leaderboard[0].metrics.get("roc_auc", float("nan"))
        print(f"accuracy {diag.accuracy:.4g} / f1_macro {diag.f1_macro:.4g} / roc_auc {roc:.4g}.")

    print("[8/9] persist best model.")
    model_name = f"model_{_slug(train_result.best_model)}.pkl"
    model_path = output / model_name
    joblib.dump(train_result.best_pipeline, model_path)
    print(f"saved -> {model_name}.")

    print("[9/9] report.")
    related = catalog_related(args.resource_id, top_k=6)
    if not related:
        related = catalog_search(args.name, top_k=6)
    print(f"related: {len(related)} nearest from catalog.")
    correlations = []
    if train_result.task_type == TASK_REGRESSION and not args.no_correlation:
        correlations = correlate_related(df_context, chosen.target, related)
        if correlations:
            print(f"correlation: {len(correlations)} datasets, top r={correlations[0].r:.3f} ({correlations[0].name}).")
        else:
            print("correlation: 0 datasets, top r=n/a (none).")
    elif args.no_correlation:
        print("skipped by flag.")

    report_path = render_report(
        dataset_name=args.name,
        resource_id=args.resource_id,
        target_report=target_report,
        eda=eda,
        preprocess_notes=preprocess_notes,
        train_result=train_result,
        diagnostics=diag,
        context_pack=context_pack,
        research_text=research_text,
        related=related,
        correlations=correlations,
        output_path=output / "report.html",
    )
    print(f"wrote -> {report_path}.")

    best = train_result.leaderboard[0]
    if train_result.task_type == TASK_REGRESSION:
        print(
            f"done. best: {best.name}  MAE={best.metrics.get('MAE', float('nan')):.4g}  "
            f"RMSE={best.metrics.get('RMSE', float('nan')):.4g}  R2={best.metrics.get('R2', float('nan')):.4g}."
        )
    else:
        primary = "roc_auc" if "roc_auc" in best.metrics else "f1"
        print(
            f"done. best: {best.name}  {primary}={best.metrics.get(primary, float('nan')):.4g}  "
            f"acc={best.metrics.get('acc', float('nan')):.4g}  f1={best.metrics.get('f1', float('nan')):.4g}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

