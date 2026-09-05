"""CPU-only failure-path tests for the quick backward smoke gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backward_smoke.py"
SPEC = importlib.util.spec_from_file_location("check_backward_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_empty_smoke_reports_fail_closed():
    decision = MODULE.check_reports([])
    assert decision["status"] == "FAIL"
    assert decision["problems"]


def test_incomplete_implementation_set_fails_closed():
    report = {
        "dtype": "bfloat16",
        "schema_version": 2,
        "shape_set": "gate",
        "selected_implementations": ["current"],
        "correctness_seeds": [0, 1, 2],
        "provenance": {
            "repository_commit": "abc",
            "worktree_dirty": False,
            "argv": ["--quick"],
        },
        "gpu_preflight": {
            "resolved_uuid": "GPU-test",
            "compute_processes_at_start": [],
            "busy_override": False,
        },
        "gpu_postflight": {
            "resolved_uuid": "GPU-test",
            "compute_processes_at_end": [],
        },
        "results": [],
        "correctness_only": [],
    }
    decision = MODULE.check_reports([report, {**report, "dtype": "float16"}])
    assert decision["status"] == "FAIL"
    assert any("implementation set" in problem for problem in decision["problems"])
