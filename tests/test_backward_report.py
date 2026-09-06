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


def _timing(order_in_trial: int = 0) -> dict:
    trials = [
        {
            "median_ms": 1.0,
            "p10_ms": 1.0,
            "p90_ms": 1.0,
            "min_ms": 1.0,
            "mean_ms": 1.0,
            "cv": 0.0,
            "reps": 13,
            "warmup": 10,
            "l2_flushed": True,
            "samples_ms": [1.0] * 13,
            "trial": trial,
            "order_in_trial": order_in_trial,
        }
        for trial in range(15)
    ]
    return {
        "median_ms": 1.0,
        "trial_medians_ms": [1.0] * 15,
        "p10_ms": 1.0,
        "p90_ms": 1.0,
        "min_ms": 1.0,
        "mean_ms": 1.0,
        "cv": 0.0,
        "reps": 195,
        "warmup_per_trial": 10,
        "trial_count": 15,
        "l2_flushed": True,
        "trials": trials,
    }


def _correctness() -> dict:
    tolerances = MODULE.EXPECTED_TOLERANCES["bfloat16"]
    return {
        name: {
            "ok": True,
            "rel_l2": 0.001,
            "rel_l2_tol": tolerances[name],
            "dtype_floor_rel_l2": 0.001,
            "err_vs_dtype_floor": 1.0,
            "max_abs_err": 0.001,
            "max_abs_err_over_rms": 0.001,
            "output_rms": 1.0,
            "has_nan": False,
            "has_inf": False,
        }
        for name in ("output", "dv", "dw")
    }


def _kernels() -> dict:
    return {
        "total_kernels": 1,
        "total_cuda_us": 1.0,
        "by_name": {
            "test_kernel": {"launches_per_call": 1.0, "cuda_us_per_call": 1.0}
        },
    }


def _memory() -> dict:
    return {
        "peak_allocated_bytes": 30,
        "workspace_bytes": 10,
        "resident_bytes": 20,
        "incremental_peak_bytes": 20,
        "returned_bytes": 0,
        "accounted_output_bytes": 10,
        "allocation_count": 1,
    }


def _report() -> dict:
    report = {
        "schema_version": 2,
        "run_id": "20260905T200000Z",
        "campaign_attempt": 1,
        "experiment": "backward architecture selection",
        "run_status": "complete",
        "dtype": "bfloat16",
        "shape_set": "full",
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "selected_implementations": IMPLS,
        "notes": [],
        "environment": {
            "torch": "test",
            "torch_cuda": "test",
            "triton": "test",
            "python": "test",
            "platform": "test",
            "device_name": "test",
            "device_cc": "12.0",
            "device_index": 0,
            "timestamp_utc": "test",
        },
        "gpu_preflight": {
            "logical_device": "cuda:0",
            "device_id": "device-0123456789abcdef",
            "device_query": "sanitized",
            "benchmark_process_context_count": 1,
            "foreign_compute_process_count_at_start": 0,
            "exclusive_access_required": True,
            "busy_override": False,
        },
        "gpu_postflight": {
            "device_id": "device-0123456789abcdef",
            "benchmark_process_context_count": 1,
            "foreign_compute_process_count_at_end": 0,
        },
        "gpu_process_monitor": {
            "device_id": "device-0123456789abcdef",
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
            "dirty_paths": [],
            "diff_sha256": "test",
            "third_party_commits": {"Liger-Kernel": MODULE.PINNED_LIGER_COMMIT},
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": ["bench/bench_backward.py", "--shape-set", "full"],
            "input_seed": 0,
            "query_seed": 1,
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
        "tolerances": MODULE.EXPECTED_TOLERANCES["bfloat16"],
        "methodology": {field: "test" for field in MODULE.METHOD_FIELDS},
        "results": [],
        "correctness_only": [],
        "execution_order": [],
    }
    plan = MODULE.get_training_plan("cuda_cluster").as_dict()
    for shape in MODULE.SHAPE_SETS["full"]:
        shape_dict = dict(zip(("n", "b", "t", "d"), shape, strict=True))
        runnable = [
            name
            for name in IMPLS
            if MODULE._plan_support(name, shape, "bfloat16")[0]
        ]
        for name in IMPLS:
            supported, expected_plan = MODULE._plan_support(name, shape, "bfloat16")
            record = {"impl": name, "shape": shape_dict, "status": "test"}
            if expected_plan is not None:
                record["training_plan"] = plan
            if not supported:
                record["skipped"] = "unsupported plan: test"
            else:
                order = runnable.index(name)
                record.update(
                    {
                        "correctness": _correctness(),
                        "correctness_by_seed": [
                            {"seed": seed, "report": _correctness()}
                            for seed in (0, 1, 2)
                        ],
                        "forward": _timing(order),
                        "backward": _timing(order),
                        "fwd_bwd": _timing(order),
                        "forward_kernels": _kernels(),
                        "backward_kernels": _kernels(),
                        "fwd_bwd_kernels": _kernels(),
                        "fwd_bwd_memory": _memory(),
                    }
                )
            report["results"].append(record)
        report["execution_order"].append(
            {
                "shape": shape_dict,
                "metrics": {
                    metric: [
                        {"trial": trial, "implementations": runnable}
                        for trial in range(15)
                    ]
                    for metric in ("forward", "backward", "fwd_bwd")
                },
            }
        )
    for shape in MODULE.CORRECTNESS_ONLY_SHAPES:
        rows = []
        for name in IMPLS:
            supported, expected_plan = MODULE._plan_support(name, shape, "bfloat16")
            for seed in (0, 1, 2):
                row = {"impl": name, "seed": seed}
                if not supported:
                    row["training_plan"] = expected_plan
                    row["skipped"] = "unsupported plan: test"
                else:
                    row["correctness"] = _correctness()
                rows.append(row)
        report["correctness_only"].append(
            {
                "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
                "implementations": rows,
            }
        )
    return report


def _problems(report: dict) -> list[str]:
    return MODULE.validate_report(
        report,
        dtype="bfloat16",
        shape_set="full",
        implementations=IMPLS,
        quick=False,
        expected_commit="abc123",
        expected_branch="codex/campaign",
        expected_tree="tree123",
        expected_device_id="device-0123456789abcdef",
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


def test_unknown_top_level_or_nested_fields_cannot_be_published():
    report = _report()
    report["private_note"] = "must not leave the runner"
    report["provenance"]["prompt"] = "must not leave the runner"
    problems = _problems(report)
    assert any("report contains unexpected fields" in problem for problem in problems)
    assert any("provenance contains unexpected fields" in problem for problem in problems)


def test_truncated_complete_report_cannot_be_reused():
    report = _report()
    report["results"].pop()
    report["execution_order"][0]["metrics"]["forward"].pop()
    problems = _problems(report)
    assert any("result shape and implementation matrix" in problem for problem in problems)
    assert any("execution schedule" in problem for problem in problems)


def test_truncated_numeric_evidence_cannot_be_reused():
    report = _report()
    record = next(item for item in report["results"] if not item.get("skipped"))
    record["correctness"]["output"].pop("rel_l2")
    record["forward"]["trials"][0]["samples_ms"].pop()
    record["backward_kernels"] = {
        "total_kernels": None,
        "total_cuda_us": None,
        "by_name": {"_unavailable": "missing profile"},
    }
    record["fwd_bwd_memory"].pop("workspace_bytes")
    problems = _problems(report)
    assert any("correctness evidence" in problem for problem in problems)
    assert any("forward" in problem and "incomplete trial" in problem for problem in problems)
    assert any("backward_kernels" in problem for problem in problems)
    assert any("fwd_bwd_memory" in problem for problem in problems)


def test_timing_order_must_match_execution_schedule():
    report = _report()
    record = next(item for item in report["results"] if not item.get("skipped"))
    record["forward"]["trials"][0]["order_in_trial"] = 99
    problems = _problems(report)
    assert any("does not match raw timing order" in problem for problem in problems)
