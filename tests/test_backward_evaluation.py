"""CPU-only tests for the backward candidate's promotion gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_backward.py"
SPEC = importlib.util.spec_from_file_location("evaluate_backward", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(impl, shape, latency, *, ok=True, cv=0.01, error=1e-3):
    samples = [latency] * 200
    return {
        "impl": impl,
        "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
        "correctness": {
            name: {"ok": ok, "rel_l2": error} for name in ("output", "dv", "dw")
        },
        "backward": {"median_ms": latency, "cv": cv, "trials": [{"samples_ms": samples}]},
        "fwd_bwd": {"median_ms": latency, "cv": cv, "trials": [{"samples_ms": samples}]},
        "correctness_by_seed": [
            {
                "seed": seed,
                "report": {
                    name: {"ok": ok, "rel_l2": error} for name in ("output", "dv", "dw")
                },
            }
            for seed in (0, 1, 2)
        ],
    }


def _report(dtype, *, candidate_ms=0.8, ok=True, dirty=False):
    results = []
    for shape in MODULE.EXPECTED_FULL_SHAPES:
        results.extend(
            [
                _record("current", shape, 1.0),
                _record("source_serial", shape, candidate_ms, ok=ok),
                _record("liger", shape, 0.9),
            ]
        )
    return {
        "dtype": dtype,
        "schema_version": 1,
        "run_id": f"run-{dtype}",
        "shape_set": "full",
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "gpu_preflight": {"compute_processes_at_start": [], "busy_override": False},
        "provenance": {
            "repository_commit": "abc123",
            "tracked_worktree_dirty": dirty,
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": [],
        },
        "results": results,
        "correctness_only": [
            {
                "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
                "implementations": [
                    {
                        "impl": impl,
                        "seed": seed,
                        "correctness": {
                            name: {"ok": True} for name in ("output", "dv", "dw")
                        },
                    }
                    for impl in ("current", "source_serial", "liger")
                    for seed in (0, 1, 2)
                ],
            }
            for shape in MODULE.MASKED_TAIL_SHAPES
        ],
    }


def test_complete_repeatable_wins_are_ready_for_dispatch_review():
    reports = [_report(dtype) for dtype in MODULE.REQUIRED_DTYPES]
    decision = MODULE.evaluate_reports(reports)
    assert decision["status"] == "READY_FOR_DISPATCH_REVIEW"
    assert all(row["classification"] == "WIN" for row in decision["comparisons"])


def test_incomplete_dtype_matrix_requests_more_data():
    decision = MODULE.evaluate_reports([_report("bfloat16")])
    assert decision["status"] == "MORE_DATA"
    assert any("missing dtype" in problem for problem in decision["problems"])


def test_correctness_failure_rejects_candidate():
    reports = [_report(dtype, ok=dtype != "float16") for dtype in MODULE.REQUIRED_DTYPES]
    decision = MODULE.evaluate_reports(reports)
    assert decision["status"] == "REJECT"
    assert decision["correctness_failures"]


def test_candidate_without_material_wins_is_dropped():
    reports = [_report(dtype, candidate_ms=0.96) for dtype in MODULE.REQUIRED_DTYPES]
    decision = MODULE.evaluate_reports(reports)
    assert decision["status"] == "DROP"


def test_dirty_or_unstable_measurements_cannot_promote():
    reports = [_report(dtype, dirty=dtype == "bfloat16") for dtype in MODULE.REQUIRED_DTYPES]
    reports[0]["results"][1]["backward"]["cv"] = 0.2
    decision = MODULE.evaluate_reports(reports)
    assert decision["status"] == "MORE_DATA"
    assert decision["problems"]
    assert decision["unstable"]
