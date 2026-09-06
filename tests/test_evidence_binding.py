"""CPU-only tests for decision-to-report content binding."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_evidence_binding.py"
SPEC = importlib.util.spec_from_file_location("check_evidence_binding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_report_bytes_match_decision(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"value": 1}\n')
    decision = {
        "input_reports": [
            {"sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
        ]
    }
    assert not MODULE.binding_problems(decision, [report])


def test_changed_report_bytes_break_decision_binding(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"value": 1}\n')
    decision = {
        "input_reports": [
            {"sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
        ]
    }
    report.write_text('{"value": 2}\n')
    assert MODULE.binding_problems(decision, [report])


def test_binding_survives_a_safe_publish_rename(tmp_path):
    report = tmp_path / "run" / "report.json"
    report.parent.mkdir()
    report.write_text('{"value": 1}\n')
    decision = {
        "input_reports": [{"sha256": hashlib.sha256(report.read_bytes()).hexdigest()}]
    }
    published = tmp_path / "backward_campaign_20260905T200000Z_report.json"
    published.write_bytes(report.read_bytes())
    assert not MODULE.binding_problems(decision, [published])


def test_git_index_binding_ignores_later_worktree_changes(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    report = tmp_path / "report.json"
    decision = tmp_path / "decision.json"
    report_bytes = b'{"value": 1}\n'
    report.write_bytes(report_bytes)
    decision.write_text(
        json.dumps(
            {"input_reports": [{"sha256": hashlib.sha256(report_bytes).hexdigest()}]}
        )
        + "\n"
    )
    subprocess.run(["git", "-C", tmp_path, "add", "report.json", "decision.json"], check=True)
    report.write_bytes(b'{"value": 2}\n')
    monkeypatch.chdir(tmp_path)
    staged_decision = MODULE._git_blob(":", Path("decision.json"))
    staged_report = MODULE._git_blob(":", Path("report.json"))
    assert not MODULE.binding_problems_from_bytes(staged_decision, [staged_report])
    assert MODULE.binding_problems_from_bytes(staged_decision, [report.read_bytes()])
