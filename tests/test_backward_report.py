"""CPU-only tests for safe backward report phase reuse."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backward_report.py"
SPEC = importlib.util.spec_from_file_location("check_backward_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

IMPLS = ["current", "cuda_cluster", "liger"]


def _report() -> dict:
    return {
        "schema_version": 2,
        "run_id": "run-test",
        "run_status": "complete",
        "dtype": "bfloat16",
        "shape_set": "full",
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "selected_implementations": IMPLS,
        "gpu_preflight": {
            "resolved_uuid": "GPU-test",
            "foreign_compute_process_count_at_start": 0,
            "busy_override": False,
        },
        "gpu_postflight": {
            "resolved_uuid": "GPU-test",
            "foreign_compute_process_count_at_end": 0,
        },
        "gpu_process_monitor": {
            "device_uuid": "GPU-test",
            "samples": 100,
            "interval_seconds": 0.25,
            "duration_seconds": 25.0,
            "collision_detected": False,
            "collision_events": [],
            "probe_errors": [],
        },
        "provenance": {
            "repository_commit": "abc123",
            "repository_tree": "tree123",
            "repository_branch": "codex/campaign",
            "worktree_dirty": False,
            "tracked_worktree_dirty": False,
            "third_party_commits": {"Liger-Kernel": MODULE.PINNED_LIGER_COMMIT},
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": ["bench/bench_backward.py", "--shape-set", "full"],
        },
        "comparators": {
            "liger": {
                "commit": MODULE.PINNED_LIGER_COMMIT,
                "source_sha256": MODULE.PINNED_LIGER_SOURCE_SHA256,
                "source_path": "src/liger_kernel/ops/attn_res.py",
                "worktree_dirty": False,
                "under_pinned_checkout": True,
            }
        },
    }


def _problems(report: dict) -> list[str]:
    return MODULE.validate_report(
        report,
        dtype="bfloat16",
        shape_set="full",
        implementations=IMPLS,
        quick=False,
        expected_commit="abc123",
        expected_branch="codex/campaign",
        expected_gpu_uuid="GPU-test",
    )


def test_clean_complete_report_can_be_reused():
    assert not _problems(_report())


def test_running_or_contaminated_report_cannot_be_reused():
    report = _report()
    report["run_status"] = "running"
    report["gpu_process_monitor"]["collision_detected"] = True
    assert _problems(report)


def test_report_with_wrong_source_or_zero_monitor_coverage_cannot_be_reused():
    report = _report()
    report["comparators"]["liger"]["source_sha256"] = "wrong"
    report["gpu_process_monitor"]["samples"] = 0
    assert _problems(report)


def test_quick_report_cannot_be_reused_as_full():
    report = _report()
    report["provenance"]["argv"].append("--quick")
    assert _problems(report)
