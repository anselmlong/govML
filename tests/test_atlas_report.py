import json

from src.atlas.report import main

CAND = {"kind": "lead", "x": "a:x", "y": "b:y", "x_label": "A: x", "y_label": "B: y", "freq": "Q", "lag": 1,
        "n_test": 40, "r_train": 0.6, "r_test": 0.8, "r_ex_covid": 0.79, "p_test_adj": 0.01, "span": "2000Q1..2026Q2"}


def test_report_shows_kept_dropped_and_annotated_breaks(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    files = {
        "coverage.json": {"matched_tables": 2, "tables": 3, "series": 10},
        "candidates.json": [CAND, CAND],
        "findings.json": [{**CAND, "headline": "X tends to move before Y.", "explanation": "Hedged.",
                           "caveats": ["Short sample."], "ideas": [{"title": "Tracker", "description": "Use X.", "kind": "verified"}],
                           "rationale": "Distinct markets."}],
        "dropped.json": [{**CAND, "drop_reason": "component_of_total", "rationale": "Y is | part of X."}],
        "breaks.json": [{"series_id": "a:x", "label": "A: x", "period": "2013Q1", "statistic": 40.0, "p": 0.002,
                         "direction": "slowed", "size": 1.2}],
        "break_notes.json": [{"series_id": "a:x", "period": "2013Q1", "note": "Cooling measures.", "confidence": "guess"}],
    }
    for name, body in files.items():
        (store / name).write_text(json.dumps(body))
    out = tmp_path / "report.md"
    assert main(["--store", str(store), "--out", str(out)]) == 0
    text = out.read_text()
    assert "passed 2 candidates; the top 2 were reviewed and 1 kept" in text
    assert "**X tends to move before Y.**" in text and "x leads by 1 Q" in text
    assert "idea (verified): Tracker" in text
    assert "Y is \\| part of X." in text
    assert "| 2013Q1 | slowed | 1.20 | 0.002 | guess: Cooling measures. |" in text


def test_an_archived_dropped_file_renders_as_an_appendix(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    for name, body in {"coverage.json": {"matched_tables": 1, "tables": 1, "series": 1}}.items():
        (store / name).write_text(json.dumps(body))
    archived = tmp_path / "old.json"
    archived.write_text(json.dumps([{**CAND, "drop_reason": "same_quantity", "rationale": "Same index."}]))
    out = tmp_path / "report.md"
    main(["--store", str(store), "--appendix", f"first rules={archived}", "--out", str(out)])
    text = out.read_text()
    assert "## Appendix: first rules" in text and "| same_quantity | Same index. |" in text
