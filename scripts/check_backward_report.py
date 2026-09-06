#!/usr/bin/env python3
"""Accept only a clean, complete backward report for phase reuse."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PINNED_LIGER_COMMIT = "777799588a89d74c489ed995e3bf006427738e85"
PINNED_LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)

TOP_LEVEL_FIELDS = {
    "schema_version",
    "run_id",
    "experiment",
    "run_status",
    "candidate_reachable_from_production",
    "environment",
    "gpu_preflight",
    "gpu_postflight",
    "gpu_process_monitor",
    "provenance",
    "comparators",
    "dtype",
    "shape_set",
    "selected_implementations",
    "tolerances",
    "correctness_seeds",
    "methodology",
    "notes",
    "results",
    "correctness_only",
    "execution_order",
}
RECORD_FIELDS = {
    "impl",
    "shape",
    "status",
    "training_plan",
    "skipped",
    "correctness",
    "correctness_by_seed",
    "forward",
    "backward",
    "fwd_bwd",
    "forward_kernels",
    "backward_kernels",
    "fwd_bwd_kernels",
    "fwd_bwd_memory",
    "traffic_model",
    "forward_traffic_model",
    "cluster_launch_info",
    "register_launch_info",
    "register_cluster_launch_info",
    "register_cluster_forward_launch_info",
}
ACCURACY_FIELDS = {
    "ok",
    "error",
    "rel_l2",
    "rel_l2_tol",
    "dtype_floor_rel_l2",
    "err_vs_dtype_floor",
    "max_abs_err",
    "max_abs_err_over_rms",
    "output_rms",
    "has_nan",
    "has_inf",
}
TIMING_FIELDS = {
    "median_ms",
    "trial_medians_ms",
    "p10_ms",
    "p90_ms",
    "min_ms",
    "mean_ms",
    "cv",
    "reps",
    "warmup_per_trial",
    "trial_count",
    "l2_flushed",
    "trials",
}
TRIAL_FIELDS = {
    "median_ms",
    "p10_ms",
    "p90_ms",
    "min_ms",
    "mean_ms",
    "cv",
    "reps",
    "warmup",
    "l2_flushed",
    "samples_ms",
    "trial",
    "order_in_trial",
}


def _only_fields(value: object, allowed: set[str], label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        problems.append(f"{label} contains unexpected fields: {unexpected}")


def _check_accuracy_schema(value: object, label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    _only_fields(value, {"output", "dv", "dw"}, label, problems)
    for name, report in value.items():
        _only_fields(report, ACCURACY_FIELDS, f"{label}.{name}", problems)


def _check_plan_schema(value: object, label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    _only_fields(
        value,
        {"name", "forward", "backward", "production", "rationale"},
        label,
        problems,
    )
    _only_fields(
        value.get("forward"), {"family", "saved_state"}, f"{label}.forward", problems
    )
    _only_fields(
        value.get("backward"),
        {"family", "tokens_per_cta", "dw_reduction"},
        f"{label}.backward",
        problems,
    )


def _check_timing_schema(value: object, label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    _only_fields(value, TIMING_FIELDS, label, problems)
    for index, trial in enumerate(value.get("trials", [])):
        _only_fields(trial, TRIAL_FIELDS, f"{label}.trials[{index}]", problems)


def _check_kernel_schema(value: object, label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    _only_fields(value, {"total_kernels", "total_cuda_us", "by_name", "error"}, label, problems)
    by_name = value.get("by_name", {})
    if isinstance(by_name, dict):
        for name, measurement in by_name.items():
            _only_fields(
                measurement,
                {"launches_per_call", "cuda_us_per_call"},
                f"{label}.by_name[{name!r}]",
                problems,
            )


def report_schema_problems(report: dict) -> list[str]:
    """Reject unknown fields at every report level before publication."""
    problems: list[str] = []
    _only_fields(report, TOP_LEVEL_FIELDS, "report", problems)
    _only_fields(
        report.get("environment"),
        {
            "torch",
            "torch_cuda",
            "triton",
            "python",
            "platform",
            "device_name",
            "device_cc",
            "device_index",
            "timestamp_utc",
        },
        "environment",
        problems,
    )
    _only_fields(
        report.get("gpu_preflight"),
        {
            "logical_device",
            "resolved_uuid",
            "resolved_pci_bus_id",
            "device_query",
            "benchmark_process_context_count",
            "foreign_compute_process_count_at_start",
            "exclusive_access_required",
            "busy_override",
        },
        "gpu_preflight",
        problems,
    )
    _only_fields(
        report.get("gpu_postflight"),
        {
            "resolved_uuid",
            "benchmark_process_context_count",
            "foreign_compute_process_count_at_end",
        },
        "gpu_postflight",
        problems,
    )
    monitor = report.get("gpu_process_monitor")
    _only_fields(
        monitor,
        {
            "device_uuid",
            "interval_seconds",
            "samples",
            "duration_seconds",
            "collision_detected",
            "collision_events",
            "probe_errors",
        },
        "gpu_process_monitor",
        problems,
    )
    if isinstance(monitor, dict):
        for index, event in enumerate(monitor.get("collision_events", [])):
            _only_fields(
                event,
                {"time", "foreign_process_count"},
                f"gpu_process_monitor.collision_events[{index}]",
                problems,
            )
    provenance = report.get("provenance")
    _only_fields(
        provenance,
        {
            "argv",
            "repository_commit",
            "repository_tree",
            "repository_branch",
            "worktree_dirty",
            "tracked_worktree_dirty",
            "dirty_paths",
            "diff_sha256",
            "third_party_commits",
            "third_party_dirty",
            "input_seed",
            "query_seed",
        },
        "provenance",
        problems,
    )
    if isinstance(provenance, dict):
        _only_fields(
            provenance.get("third_party_commits"),
            {"Liger-Kernel"},
            "provenance.third_party_commits",
            problems,
        )
        _only_fields(
            provenance.get("third_party_dirty"),
            {"Liger-Kernel"},
            "provenance.third_party_dirty",
            problems,
        )
    comparators = report.get("comparators")
    _only_fields(comparators, {"liger"}, "comparators", problems)
    if isinstance(comparators, dict):
        _only_fields(
            comparators.get("liger"),
            {
                "commit",
                "worktree_dirty",
                "source_path",
                "source_sha256",
                "under_pinned_checkout",
            },
            "comparators.liger",
            problems,
        )
    _only_fields(report.get("tolerances"), {"output", "dv", "dw"}, "tolerances", problems)
    _only_fields(
        report.get("methodology"),
        {"oracle", "timing", "statistics", "cache", "compilation", "backward", "promotion"},
        "methodology",
        problems,
    )

    for record_index, record in enumerate(report.get("results", [])):
        label = f"results[{record_index}]"
        _only_fields(record, RECORD_FIELDS, label, problems)
        _only_fields(record.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        _check_plan_schema(record.get("training_plan"), f"{label}.training_plan", problems)
        _check_accuracy_schema(record.get("correctness"), f"{label}.correctness", problems)
        for seed_index, seed in enumerate(record.get("correctness_by_seed", [])):
            seed_label = f"{label}.correctness_by_seed[{seed_index}]"
            _only_fields(seed, {"seed", "report"}, seed_label, problems)
            _check_accuracy_schema(seed.get("report"), f"{seed_label}.report", problems)
        for metric in ("forward", "backward", "fwd_bwd"):
            _check_timing_schema(record.get(metric), f"{label}.{metric}", problems)
        for profile in ("forward_kernels", "backward_kernels", "fwd_bwd_kernels"):
            _check_kernel_schema(record.get(profile), f"{label}.{profile}", problems)
        _only_fields(
            record.get("fwd_bwd_memory"),
            {
                "peak_allocated_bytes",
                "workspace_bytes",
                "resident_bytes",
                "incremental_peak_bytes",
                "returned_bytes",
                "accounted_output_bytes",
                "allocation_count",
            },
            f"{label}.fwd_bwd_memory",
            problems,
        )
        _only_fields(
            record.get("traffic_model"),
            {
                "strategy",
                "minimum_large_tensor_bytes",
                "logical_large_tensor_bytes",
                "estimated_dram_bytes",
                "statistics_bytes",
                "saved_state_bytes",
                "workspace_bytes",
                "dw_atomic_updates",
                "assumptions",
            },
            f"{label}.traffic_model",
            problems,
        )
        _only_fields(
            record.get("forward_traffic_model"),
            {
                "strategy",
                "minimum_large_tensor_bytes",
                "logical_large_tensor_bytes",
                "estimated_dram_bytes",
                "saved_state_bytes",
                "assumptions",
            },
            f"{label}.forward_traffic_model",
            problems,
        )
        for launch_field in (
            "cluster_launch_info",
            "register_launch_info",
            "register_cluster_launch_info",
            "register_cluster_forward_launch_info",
        ):
            _only_fields(
                record.get(launch_field),
                {
                    "active_clusters",
                    "active_blocks",
                    "cluster_blocks",
                    "dynamic_shared_bytes",
                    "static_shared_bytes",
                    "registers_per_thread",
                    "max_shared_bytes",
                    "multiprocessors",
                    "threads_per_block",
                },
                f"{label}.{launch_field}",
                problems,
            )

    for case_index, case in enumerate(report.get("correctness_only", [])):
        label = f"correctness_only[{case_index}]"
        _only_fields(case, {"shape", "implementations"}, label, problems)
        _only_fields(case.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        for row_index, row in enumerate(case.get("implementations", [])):
            row_label = f"{label}.implementations[{row_index}]"
            _only_fields(
                row,
                {"impl", "seed", "training_plan", "skipped", "correctness"},
                row_label,
                problems,
            )
            _check_plan_schema(row.get("training_plan"), f"{row_label}.training_plan", problems)
            _check_accuracy_schema(row.get("correctness"), f"{row_label}.correctness", problems)

    for schedule_index, schedule in enumerate(report.get("execution_order", [])):
        label = f"execution_order[{schedule_index}]"
        _only_fields(schedule, {"shape", "metrics"}, label, problems)
        _only_fields(schedule.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        metrics = schedule.get("metrics")
        _only_fields(metrics, {"forward", "backward", "fwd_bwd"}, f"{label}.metrics", problems)
        if isinstance(metrics, dict):
            for metric, trials in metrics.items():
                for trial_index, trial in enumerate(trials):
                    _only_fields(
                        trial,
                        {"trial", "implementations"},
                        f"{label}.metrics.{metric}[{trial_index}]",
                        problems,
                    )
    return problems


def validate_report(
    report: dict,
    *,
    dtype: str,
    shape_set: str,
    implementations: list[str],
    quick: bool,
    expected_commit: str,
    expected_branch: str,
    expected_tree: str,
    expected_gpu_uuid: str,
) -> list[str]:
    """Return every reason a report cannot be reused."""
    problems = report_schema_problems(report)
    expected = {
        "schema_version": 2,
        "run_status": "complete",
        "dtype": dtype,
        "shape_set": shape_set,
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            problems.append(f"{field} must equal {value!r}")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", run_id) is None:
        problems.append("run_id must use the benchmark UTC timestamp format")
    if report.get("selected_implementations") != implementations:
        problems.append("selected implementation order differs from the requested phase")
    if report.get("experiment") != "backward architecture selection":
        problems.append("experiment name is not exact")
    if report.get("notes") != []:
        problems.append("publishable reports must not contain free-form notes")

    provenance = report.get("provenance", {})
    if provenance.get("repository_commit") != expected_commit:
        problems.append("repository commit differs from the campaign commit")
    if provenance.get("repository_branch") != expected_branch:
        problems.append("repository branch differs from the campaign branch")
    if provenance.get("repository_tree") != expected_tree:
        problems.append("repository tree differs from the campaign tree")
    if provenance.get("worktree_dirty") is not False:
        problems.append("benchmark worktree was not fully clean")
    if provenance.get("tracked_worktree_dirty") is not False:
        problems.append("tracked benchmark worktree was not clean")
    if provenance.get("third_party_commits", {}).get("Liger-Kernel") != PINNED_LIGER_COMMIT:
        problems.append("pinned Liger commit is missing or wrong")
    if provenance.get("third_party_dirty", {}).get("Liger-Kernel") is not False:
        problems.append("pinned Liger checkout was not clean")
    argv = provenance.get("argv", [])
    if not isinstance(argv, list) or ("--quick" in argv) is not quick:
        problems.append("quick/full command provenance differs from the requested phase")

    liger = report.get("comparators", {}).get("liger", {})
    if (
        liger.get("commit") != PINNED_LIGER_COMMIT
        or liger.get("source_sha256") != PINNED_LIGER_SOURCE_SHA256
        or liger.get("worktree_dirty") is not False
        or liger.get("under_pinned_checkout") is not True
        or not liger.get("source_path")
    ):
        problems.append("imported Liger source provenance is incomplete")

    preflight = report.get("gpu_preflight", {})
    postflight = report.get("gpu_postflight", {})
    uuid = preflight.get("resolved_uuid")
    if uuid != expected_gpu_uuid:
        problems.append("physical GPU differs from the campaign device")
    if (
        not uuid
        or preflight.get("foreign_compute_process_count_at_start") != 0
        or preflight.get("busy_override")
    ):
        problems.append("GPU preflight was not exclusive")
    if (
        postflight.get("resolved_uuid") != uuid
        or postflight.get("foreign_compute_process_count_at_end") != 0
    ):
        problems.append("GPU postflight was not exclusive on the same device")

    monitor = report.get("gpu_process_monitor", {})
    samples = monitor.get("samples")
    interval = monitor.get("interval_seconds")
    duration = monitor.get("duration_seconds")
    coverage_ok = (
        isinstance(samples, int)
        and samples >= 2
        and isinstance(interval, int | float)
        and 0 < interval <= 1
        and isinstance(duration, int | float)
        and duration >= interval
        and samples >= max(2, int(duration / (2 * interval)))
    )
    if (
        monitor.get("device_uuid") != uuid
        or monitor.get("collision_detected") is not False
        or monitor.get("collision_events")
        or monitor.get("probe_errors")
        or not coverage_ok
    ):
        problems.append("sampled GPU monitor was incomplete or contaminated")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--shape-set", required=True)
    parser.add_argument("--impls", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    if not args.report.is_file():
        return 1
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    problems = validate_report(
        report,
        dtype=args.dtype,
        shape_set=args.shape_set,
        implementations=[item for item in args.impls.split(",") if item],
        quick=args.quick,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tree=args.expected_tree,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
