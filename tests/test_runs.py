import json
import os
from unittest.mock import Mock

import pytest

from backend import app as backend


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT", tmp_path)
    runs_file = tmp_path / "runs.json"
    runs_file.write_text(json.dumps([{"run_id": "r1", "status": "running"}]), encoding="utf-8")
    monkeypatch.setattr(backend, "RUNS_FILE", runs_file)
    return tmp_path


def fake_proc(lines, code):
    return Mock(stdout=iter(lines), wait=lambda: code)


def test_failed_run_does_not_inherit_previous_report(run_env):
    stale = run_env / "report.html"
    stale.write_text("previous dataset", encoding="utf-8")
    os.utime(stale, (0, 0))
    backend._drain_process("r1", fake_proc(["boom\n"], 1), run_env / "log.txt")
    assert backend._get_run("r1")["status"] == "failed"
    assert not (run_env / "report_r1.html").exists()


def test_completed_run_claims_fresh_report(run_env):
    def lines():
        (run_env / "report.html").write_text("this run", encoding="utf-8")
        yield "done. best: RandomForest  MAE=1.5 R2=0.73.\n"

    backend._drain_process("r1", fake_proc(lines(), 0), run_env / "log.txt")
    run = backend._get_run("r1")
    assert run["status"] == "completed" and run["r2"] == 0.73
    assert (run_env / "report_r1.html").read_text(encoding="utf-8") == "this run"
