"""CPU-only tests for safe backward report phase reuse."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backward_report.py"
SPEC = importlib.util.spec_from_file_location("check_backward_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

IMPLS = ["current", "cuda_cluster", "liger"]
EXPECTED_COMMIT = "a" * 40
EXPECTED_TREE = "b" * 40
EXPECTED_BRANCH = "codex/campaign"
EXPECTED_DEVICE_ID = "device-0123456789abcdef"


def _timing(order_in_trial: int = 0, *, quick: bool = False) -> dict:
    trial_count, reps, warmup = (2, 10, 8) if quick else (15, 13, 10)
    trials = [
        {
            "median_ms": 1.0,
            "p10_ms": 1.0,
            "p90_ms": 1.0,
            "min_ms": 1.0,
            "mean_ms": 1.0,
            "cv": 0.0,
            "reps": reps,
            "warmup": warmup,
            "l2_flushed": True,
            "samples_ms": [1.0] * reps,
            "trial": trial,
            "order_in_trial": order_in_trial,
        }
        for trial in range(trial_count)
    ]
    return {
        "median_ms": 1.0,
        "trial_medians_ms": [1.0] * trial_count,
        "p10_ms": 1.0,
        "p90_ms": 1.0,
        "min_ms": 1.0,
        "mean_ms": 1.0,
        "cv": 0.0,
        "reps": trial_count * reps,
        "warmup_per_trial": warmup,
        "trial_count": trial_count,
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
        "by_name": {"test_kernel": {"launches_per_call": 1.0, "cuda_us_per_call": 1.0}},
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


def _report(*, quick: bool = False, shape_set: str = "full") -> dict:
    trial_count = 2 if quick else 15
    report = {
        "schema_version": 2,
        "run_id": "20260905T200000Z",
        "campaign_attempt": 1,
        "experiment": "backward architecture selection",
        "run_status": "complete",
        "dtype": "bfloat16",
        "shape_set": shape_set,
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "selected_implementations": IMPLS,
        "notes": [],
        "environment": {
            **MODULE.EXPECTED_ENVIRONMENT,
            "timestamp_utc": "2026-09-05T20:00:00Z",
        },
        "gpu_preflight": {
            "logical_device": "cuda:0",
            "device_id": EXPECTED_DEVICE_ID,
            "device_query": (
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, "
                "580.65.06, 35, 20.50, 300.00, 300, 405"
            ),
            "benchmark_process_context_count": 1,
            "foreign_compute_process_count_at_start": 0,
            "exclusive_access_required": True,
            "busy_override": False,
        },
        "gpu_postflight": {
            "device_id": EXPECTED_DEVICE_ID,
            "benchmark_process_context_count": 1,
            "foreign_compute_process_count_at_end": 0,
        },
        "gpu_process_monitor": {
            "device_id": EXPECTED_DEVICE_ID,
            "samples": 500,
            "interval_seconds": 0.05,
            "duration_seconds": 25.0,
            "collision_detected": False,
            "collision_events": [],
            "probe_errors": [],
        },
        "provenance": {
            "repository_commit": EXPECTED_COMMIT,
            "repository_tree": EXPECTED_TREE,
            "repository_branch": EXPECTED_BRANCH,
            "worktree_dirty": False,
            "tracked_worktree_dirty": False,
            "dirty_paths": [],
            "diff_sha256": MODULE.EMPTY_DIFF_SHA256,
            "third_party_commits": {"Liger-Kernel": MODULE.PINNED_LIGER_COMMIT},
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": MODULE._expected_argv(
                dtype="bfloat16",
                shape_set=shape_set,
                implementations=IMPLS,
                quick=quick,
            ),
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
        "methodology": MODULE._expected_methodology(quick=quick),
        "results": [],
        "correctness_only": [],
        "execution_order": [],
    }
    plan = MODULE.get_training_plan("cuda_cluster").as_dict()
    for shape in MODULE.SHAPE_SETS[shape_set]:
        shape_dict = dict(zip(("n", "b", "t", "d"), shape, strict=True))
        runnable = [name for name in IMPLS if MODULE._plan_support(name, shape, "bfloat16")[0]]
        for name in IMPLS:
            supported, expected_plan = MODULE._plan_support(name, shape, "bfloat16")
            record = {
                "impl": name,
                "shape": shape_dict,
                "status": MODULE._expected_result_status(name),
            }
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
                            {"seed": seed, "report": _correctness()} for seed in (0, 1, 2)
                        ],
                        "forward": _timing(order, quick=quick),
                        "backward": _timing(order, quick=quick),
                        "fwd_bwd": _timing(order, quick=quick),
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
                        for trial in range(trial_count)
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


def _problems(report: object, *, quick: bool = False, shape_set: str = "full") -> list[str]:
    return MODULE.validate_report(
        report,
        dtype="bfloat16",
        shape_set=shape_set,
        implementations=IMPLS,
        quick=quick,
        expected_commit=EXPECTED_COMMIT,
        expected_branch=EXPECTED_BRANCH,
        expected_tree=EXPECTED_TREE,
        expected_device_id=EXPECTED_DEVICE_ID,
    )


def test_clean_complete_report_can_be_reused():
    assert not _problems(_report())


def test_clean_generated_smoke_report_can_be_reused():
    report = _report(quick=True, shape_set="gate")
    assert not _problems(report, quick=True, shape_set="gate")


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


@pytest.mark.parametrize(
    "section",
    [
        "environment",
        "gpu_preflight",
        "gpu_postflight",
        "gpu_process_monitor",
        "provenance",
        "comparators",
    ],
)
def test_runtime_sections_must_be_objects(section: str):
    report = _report()
    report[section] = []
    assert _problems(report)


def test_nonobject_report_and_result_matrix_fail_closed():
    assert _problems([])
    report = _report()
    report["results"] = "not a result list"
    assert _problems(report)


def test_malformed_nested_timing_fails_closed():
    report = _report()
    record = next(item for item in report["results"] if not item.get("skipped"))
    record["forward"]["trials"] = 3
    assert _problems(report)


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


@pytest.mark.parametrize("implementation", IMPLS)
def test_result_status_must_match_the_implementation_class(implementation: str):
    report = _report()
    record = next(item for item in report["results"] if item["impl"] == implementation)
    record["status"] = "private candidate; not reachable from production dispatch"
    assert any("invalid result status" in problem for problem in _problems(report))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch", "2.9.1+cu128"),
        ("torch_cuda", 12.8),
        ("triton", "3.5.1"),
        ("python", "3.13.0"),
        ("platform", "Linux-test"),
        ("device_name", "Different GPU"),
        ("device_cc", 12.0),
        ("device_index", False),
        ("timestamp_utc", "not-a-timestamp"),
    ],
)
def test_environment_values_and_types_are_exact(field: str, value: object):
    report = _report()
    report["environment"][field] = value
    assert _problems(report)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("gpu_preflight", "logical_device", "cuda:1"),
        ("gpu_preflight", "device_query", "sanitized"),
        ("gpu_preflight", "benchmark_process_context_count", True),
        ("gpu_preflight", "benchmark_process_context_count", 0),
        ("gpu_preflight", "benchmark_process_context_count", 2),
        ("gpu_preflight", "foreign_compute_process_count_at_start", False),
        ("gpu_preflight", "exclusive_access_required", 1),
        ("gpu_preflight", "busy_override", 0),
        ("gpu_postflight", "benchmark_process_context_count", False),
        ("gpu_postflight", "benchmark_process_context_count", 0),
        ("gpu_postflight", "benchmark_process_context_count", 2),
        ("gpu_postflight", "foreign_compute_process_count_at_end", False),
        ("gpu_process_monitor", "samples", True),
        ("gpu_process_monitor", "interval_seconds", 0.25),
        ("gpu_process_monitor", "duration_seconds", 25),
        ("gpu_process_monitor", "collision_detected", 0),
        ("gpu_process_monitor", "collision_events", {}),
        ("gpu_process_monitor", "probe_errors", None),
    ],
)
def test_runtime_attestation_values_and_types_are_exact(section: str, field: str, value: object):
    report = _report()
    report[section][field] = value
    assert _problems(report)


def test_environment_timestamp_is_bound_to_the_run_start():
    report = _report()
    report["environment"]["timestamp_utc"] = "2026-09-05T20:01:01Z"
    assert any("not bound" in problem for problem in _problems(report))


def test_methodology_is_exact():
    report = _report()
    report["methodology"]["timing"] = "different"
    assert any("methodology differs" in problem for problem in _problems(report))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dirty_paths", ["src/private.py"]),
        ("diff_sha256", "0" * 64),
        ("input_seed", False),
        ("query_seed", True),
        ("third_party_commits", {"Liger-Kernel": MODULE.PINNED_LIGER_COMMIT, "extra": "x"}),
        ("third_party_dirty", {"Liger-Kernel": 0}),
    ],
)
def test_clean_provenance_values_and_types_are_exact(field: str, value: object):
    report = _report()
    report["provenance"][field] = value
    assert _problems(report)


@pytest.mark.parametrize(
    "argv",
    [
        ["bench/bench_backward.py", "--shape-set", "full"],
        [
            "bench/bench_backward.py",
            "--shape-set",
            "full",
            "--dtype",
            "bfloat16",
            "--impls",
            ",".join(IMPLS),
            "--out",
            "/mnt/private/full_bfloat16.json",
        ],
        [
            "bench/bench_backward.py",
            "--dtype",
            "bfloat16",
            "--shape-set",
            "full",
            "--impls",
            ",".join(IMPLS),
            "--out",
            "<external>/full_bfloat16.json",
        ],
    ],
)
def test_command_provenance_must_have_the_exact_sanitized_shape(argv: list[str]):
    report = _report()
    report["provenance"]["argv"] = argv
    assert any("exact sanitized campaign command" in problem for problem in _problems(report))


def test_comparator_source_path_must_be_exact_and_relative():
    report = _report()
    report["comparators"]["liger"]["source_path"] = "/mnt/Liger-Kernel/attn_res.py"
    assert _problems(report)


@pytest.mark.parametrize(
    "leaked_value",
    [
        "Co-Authored-By: Model <model@example.com>",
        "https://claude.ai/code/session_01Example",
        "/mnt/private/campaign/report.json",
        r"C:\Users\owner\campaign\report.json",
        "hostname=worker-17.internal",
        "worker-17.cluster.internal",
        "owner@example.com",
    ],
)
def test_private_metadata_is_rejected_from_any_publishable_string(leaked_value: str):
    report = _report()
    report["methodology"]["oracle"] = leaked_value
    assert any("private host or task metadata" in problem for problem in _problems(report))


def test_kernel_row_cuda_time_must_sum_to_total():
    report = _report()
    record = next(item for item in report["results"] if not item.get("skipped"))
    record["forward_kernels"]["total_cuda_us"] = 2.0
    assert any("incomplete forward_kernels" in problem for problem in _problems(report))


def test_kernel_row_cuda_time_allows_small_profiler_rounding_error():
    report = _report()
    record = next(item for item in report["results"] if not item.get("skipped"))
    record["forward_kernels"]["total_cuda_us"] = 1.0000005
    assert not _problems(report)
