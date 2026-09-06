"""CPU-only tests for decision-to-report content binding."""

from __future__ import annotations

import hashlib
import importlib.util
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
