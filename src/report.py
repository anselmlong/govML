"""Self-contained HTML report rendering."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import markdown as markdown_lib

from .agents import TargetReport
from .context import ContextPack
from .diagnostics import ClassDiagnostics, Diagnostics
from .preprocess import EDAReport
from .train import TASK_REGRESSION, TrainResult


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.{digits}g}m"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.{digits}g}k"
    return f"{x:.{digits}g}"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _leaderboard(result: TrainResult) -> str:
    rows = []
    for run in result.leaderboard:
        metrics = " ".join(f"{k}={fmt_num(v)}" for k, v in run.metrics.items())
        rows.append(f"<tr><td>{_esc(run.name)}</td><td>{_esc(metrics)}</td><td>{run.seconds:.1f}s</td></tr>")
    return "<table><thead><tr><th>Model</th><th>Metrics</th><th>Time</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _target_section(report: TargetReport) -> str:
    items = "".join(
        f"<li><b>{_esc(p.target)}</b> <span>{_esc(p.task_type)}</span> conf={p.confidence:.2f}<br>{_esc(p.rationale)}</li>"
        for p in report.proposals
    )
    return f"<section><h2>Target Proposals</h2><p>{_esc(report.provenance)}. {_esc(report.reasoning)}</p><ul>{items}</ul></section>"


def _eda_section(eda: EDAReport) -> str:
    miss = sorted(eda.missingness.items(), key=lambda x: x[1], reverse=True)[:10]
    miss_rows = "".join(f"<tr><td>{_esc(c)}</td><td>{v * 100:.1f}%</td></tr>" for c, v in miss)
    target = " ".join(f"{_esc(k)}={_esc(fmt_num(v))}" for k, v in eda.target_stats.items())
    time = " ".join(f"{_esc(k)}={_esc(v)}" for k, v in eda.time_range.items())
    return (
        "<section><h2>Data Overview</h2>"
        f"<div class='kpis'><div><b>{eda.row_count:,}</b><span>rows</span></div><div><b>{eda.column_count:,}</b><span>columns</span></div>"
        f"<div><b>{_esc(target or 'n/a')}</b><span>target stats</span></div><div><b>{_esc(time or 'n/a')}</b><span>time range</span></div></div>"
        f"<h3>Missingness</h3><table><tbody>{miss_rows}</tbody></table></section>"
    )


def _diagnostics_section(task_type: str, diag: Diagnostics | ClassDiagnostics | None) -> str:
    if diag is None:
        return ""
    if task_type == TASK_REGRESSION and isinstance(diag, Diagnostics):
        segs = "".join(f"<tr><td>{_esc(s.segment)}</td><td>{_esc(s.value)}</td><td>{s.count}</td><td>{fmt_num(s.mae)}</td></tr>" for s in diag.segment_errors)
        return (
            "<section><h2>Diagnostics</h2>"
            f"<div class='kpis'><div><b>{fmt_num(diag.mae)}</b><span>MAE</span></div><div><b>{fmt_num(diag.rmse)}</b><span>RMSE</span></div>"
            f"<div><b>{fmt_num(diag.r2)}</b><span>R2</span></div><div><b>{fmt_num(diag.mape)}%</b><span>MAPE</span></div></div>"
            f"<p>{_esc(diag.verdict_text)}</p><img class='chart' src='{diag.residual_chart}' alt='Residual chart'>"
            f"<h3>Segment Errors</h3><table><thead><tr><th>Segment</th><th>Value</th><th>N</th><th>MAE</th></tr></thead><tbody>{segs}</tbody></table></section>"
        )
    if isinstance(diag, ClassDiagnostics):
        rows = "".join(
            f"<tr><td>{_esc(r['class'])}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['support']}</td></tr>"
            for r in diag.per_class
        )
        return (
            "<section><h2>Diagnostics</h2>"
            f"<div class='kpis'><div><b>{diag.accuracy:.3f}</b><span>accuracy</span></div><div><b>{diag.f1_macro:.3f}</b><span>f1 macro</span></div></div>"
            f"<p>{_esc(diag.verdict_text)}</p><img class='chart' src='{diag.confusion_matrix_image}' alt='Confusion matrix'>"
            f"<h3>Per-Class Metrics</h3><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{rows}</tbody></table></section>"
        )
    return ""


def _context_section(pack: ContextPack, research_text: str) -> str:
    facts = "".join(f"<li><b>{_esc(f.date)}</b> {_esc(f.label)}: {_esc(f.description)}</li>" for f in pack.facts)
    research_html = markdown_lib.markdown(research_text or "_No research text available._")
    return f"<section><h2>Context</h2><ul>{facts}</ul><div class='markdown'>{research_html}</div></section>"


def _related_section(related: list[dict[str, Any]], correlations: list[Any]) -> str:
    cards = "".join(
        f"<a class='card' href='https://data.gov.sg/datasets/{_esc(r.get('dataset_id', ''))}'><b>{_esc(r.get('name', 'Dataset'))}</b><span>{_esc(r.get('agency', ''))}</span></a>"
        for r in related
    )
    corr_rows = "".join(
        f"<tr><td>{_esc(c.name)}</td><td>{_esc(c.proxy_column)}</td><td>{c.overlap_months}</td><td>{c.r:.3f}</td></tr>"
        for c in correlations
    )
    return (
        "<section><h2>Related Datasets</h2><div class='cards'>"
        + cards
        + "</div><h3>Cross-Dataset Correlations</h3><table><thead><tr><th>Dataset</th><th>Proxy</th><th>Months</th><th>r</th></tr></thead><tbody>"
        + corr_rows
        + "</tbody></table></section>"
    )


def _importance_section(result: TrainResult) -> str:
    rows = "".join(f"<tr><td>{_esc(name)}</td><td>{value:.4g}</td></tr>" for name, value in result.feature_importance)
    return f"<section><h2>Feature Importance</h2><table><thead><tr><th>Feature</th><th>Permutation impact</th></tr></thead><tbody>{rows}</tbody></table></section>"


def render_report(
    *,
    dataset_name: str,
    resource_id: str,
    target_report: TargetReport,
    eda: EDAReport,
    preprocess_notes: list[str],
    train_result: TrainResult,
    diagnostics: Diagnostics | ClassDiagnostics | None,
    context_pack: ContextPack,
    research_text: str,
    related: list[dict[str, Any]],
    correlations: list[Any],
    output_path: str | Path = "output/report.html",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = train_result.leaderboard[0]
    if train_result.task_type == TASK_REGRESSION:
        verdict = f"{best.name} with RMSE {fmt_num(best.metrics.get('RMSE'))} and R2 {fmt_num(best.metrics.get('R2'))}"
    else:
        verdict = f"{best.name} with accuracy {fmt_num(best.metrics.get('acc'))} and F1 {fmt_num(best.metrics.get('f1'))}"
    notes = "".join(f"<li>{_esc(n)}</li>" for n in preprocess_notes)
    css = """
body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f4ef;color:#161616}
header{padding:48px 7vw;background:#0d0f12;color:white;border-bottom:6px solid #e73d32}
main{padding:30px 7vw 70px;display:grid;gap:24px}
section{background:white;border:1px solid #d8d2c7;padding:22px;border-radius:8px}
h1{font-size:clamp(34px,6vw,76px);line-height:.94;margin:0 0 16px;letter-spacing:0}
h2{margin:0 0 16px;font-size:24px} h3{margin-top:22px}
.sub{max-width:920px;color:#d7dce4;font-size:18px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpis div{border:1px solid #d8d2c7;padding:14px;background:#fbfaf7;border-radius:6px}.kpis b{display:block;font-size:24px}.kpis span{color:#68635c}
table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #e7e2d8;text-align:left;font-size:14px}
.chart{max-width:100%;border:1px solid #e7e2d8;border-radius:6px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{display:grid;gap:8px;text-decoration:none;color:#161616;border:1px solid #d8d2c7;border-radius:6px;padding:14px;background:#fbfaf7}
.markdown{line-height:1.55;max-width:980px}
"""
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(dataset_name)} - govML report</title><style>{css}</style></head>
<body><header><p>govML report / {_esc(resource_id)}</p><h1>{_esc(dataset_name)}</h1><p class="sub">{_esc(verdict)}</p></header>
<main>
<section><h2>Best Model</h2><div class="kpis"><div><b>{_esc(best.name)}</b><span>model</span></div><div><b>{_esc(train_result.target)}</b><span>target</span></div><div><b>{_esc(train_result.task_type)}</b><span>task</span></div><div><b>{_esc(train_result.split_boundary)}</b><span>split</span></div></div></section>
{_target_section(target_report)}
{_eda_section(eda)}
<section><h2>Preprocessing Notes</h2><ul>{notes}</ul></section>
<section><h2>Model Leaderboard</h2>{_leaderboard(train_result)}</section>
{_importance_section(train_result)}
{_diagnostics_section(train_result.task_type, diagnostics)}
{_context_section(context_pack, research_text)}
{_related_section(related, correlations)}
</main></body></html>"""
    out.write_text(html_doc, encoding="utf-8")
    return out

