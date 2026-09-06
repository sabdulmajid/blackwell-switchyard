#!/usr/bin/env python3
"""Accept only a clean, complete backward report for phase reuse."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from switchyard.training_plan import get_training_plan, plan_supports  # noqa: E402

PINNED_LIGER_COMMIT = "777799588a89d74c489ed995e3bf006427738e85"
PINNED_LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)

TOP_LEVEL_FIELDS = {
    "schema_version",
    "run_id",
    "campaign_attempt",
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
SUCCESS_ACCURACY_FIELDS = ACCURACY_FIELDS - {"error"}
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
SHAPE_SETS = {
    "gate": {
        (9, 1, 4096, 4096),
        (9, 1, 4096, 8192),
        (32, 1, 4096, 2048),
        (32, 1, 4096, 4096),
    },
    "full": {
        (8, 1, 4096, 4096),
        (9, 1, 128, 4096),
        (9, 1, 512, 4096),
        (9, 1, 4096, 4096),
        (9, 1, 8192, 4096),
        (4, 1, 4096, 8192),
        (5, 1, 4096, 8192),
        (9, 1, 4096, 8192),
        (16, 1, 4096, 2048),
        (17, 1, 4096, 2048),
        (32, 1, 4096, 2048),
        (9, 4, 2048, 4096),
    },
}
CORRECTNESS_ONLY_SHAPES = {(9, 1, 129, 4097), (17, 2, 33, 2049)}
METHOD_FIELDS = {
    "oracle",
    "timing",
    "statistics",
    "cache",
    "compilation",
    "backward",
    "promotion",
}
EXPECTED_TOLERANCES = {
    "bfloat16": {"output": 2e-2, "dv": 3e-2, "dw": 6e-2},
    "float16": {"output": 5e-3, "dv": 1e-2, "dw": 3e-2},
    "float32": {"output": 2e-5, "dv": 1e-4, "dw": 1e-3},
}
MEMORY_FIELDS = {
    "peak_allocated_bytes",
    "workspace_bytes",
    "resident_bytes",
    "incremental_peak_bytes",
    "returned_bytes",
    "accounted_output_bytes",
    "allocation_count",
}
PUBLIC_DEVICE_ID_PATTERN = re.compile(r"device-[0-9a-f]{16}")
PRIVATE_METADATA_PATTERN = re.compile(
    r"co-authored-by|claude|anthropic|wizchem|chatgpt|openai\.com|"
    r"session[-_/][A-Za-z0-9]|file://|/(?:home|tmp|pub[0-9]+)/|GPU-[A-Za-z0-9-]+",
    re.IGNORECASE,
)


def _only_fields(value: object, allowed: set[str], label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        return
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        problems.append(f"{label} contains unexpected fields: {unexpected}")


def _require_fields(value: object, required: set[str], label: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        problems.append(f"{label} must be an object")
        return
    missing = sorted(required - set(value))
    if missing:
        problems.append(f"{label} is missing required fields: {missing}")


def _shape(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    fields = tuple(value.get(name) for name in ("n", "b", "t", "d"))
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in fields
    ):
        return None
    return fields


def _plan_support(
    name: str, shape: tuple[int, int, int, int], dtype: str
) -> tuple[bool, dict | None]:
    if name in {"current", "liger"}:
        return True, None
    try:
        plan = get_training_plan(name)
    except ValueError:
        return False, None
    return plan_supports(plan, *shape, dtype)[0], plan.as_dict()


def _finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= minimum
    )


def _same_number(left: object, right: float) -> bool:
    return _finite_number(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _complete_timing(
    value: object,
    *,
    trial_count: int,
    reps: int,
    warmup: int,
    label: str,
    problems: list[str],
) -> None:
    if not isinstance(value, dict):
        problems.append(f"{label} timing is missing")
        return
    if set(value) != TIMING_FIELDS:
        problems.append(f"{label} timing field set is not exact")
        return
    trials = value.get("trials")
    trial_medians = value.get("trial_medians_ms")
    if (
        value.get("trial_count") != trial_count
        or value.get("reps") != trial_count * reps
        or value.get("warmup_per_trial") != warmup
        or value.get("l2_flushed") is not True
        or not isinstance(trials, list)
        or len(trials) != trial_count
        or not isinstance(trial_medians, list)
        or len(trial_medians) != trial_count
    ):
        problems.append(f"{label} timing matrix is incomplete")
        return
    observed_trials = []
    all_samples: list[float] = []
    calculated_trial_medians: list[float] = []
    for trial in trials:
        if not isinstance(trial, dict):
            problems.append(f"{label} contains a malformed trial")
            return
        if set(trial) != TRIAL_FIELDS:
            problems.append(f"{label} contains a trial with an incomplete field set")
            return
        observed_trials.append(trial.get("trial"))
        samples = trial.get("samples_ms")
        if (
            trial.get("reps") != reps
            or trial.get("warmup") != warmup
            or trial.get("l2_flushed") is not True
            or not isinstance(samples, list)
            or len(samples) != reps
            or any(not _finite_number(sample, minimum=1e-12) for sample in samples)
            or isinstance(trial.get("order_in_trial"), bool)
            or not isinstance(trial.get("order_in_trial"), int)
            or trial["order_in_trial"] < 0
        ):
            problems.append(f"{label} contains an incomplete trial")
            return
        ordered = sorted(samples)
        mean = statistics.fmean(samples)
        expected_trial = {
            "median_ms": ordered[len(ordered) // 2],
            "p10_ms": ordered[int(0.10 * len(ordered))],
            "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
            "min_ms": ordered[0],
            "mean_ms": mean,
            "cv": statistics.pstdev(samples) / mean,
        }
        if any(not _same_number(trial.get(field), expected) for field, expected in expected_trial.items()):
            problems.append(f"{label} contains inconsistent trial statistics")
            return
        all_samples.extend(samples)
        calculated_trial_medians.append(statistics.median(samples))
    if observed_trials != list(range(trial_count)):
        problems.append(f"{label} trial IDs are not exact")
        return
    if any(
        not _same_number(observed, expected)
        for observed, expected in zip(trial_medians, calculated_trial_medians, strict=True)
    ):
        problems.append(f"{label} trial medians do not match the raw samples")
    ordered = sorted(all_samples)
    mean = statistics.fmean(all_samples)
    expected_summary = {
        "median_ms": statistics.median(all_samples),
        "p10_ms": ordered[int(0.10 * len(ordered))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "min_ms": ordered[0],
        "mean_ms": mean,
        "cv": statistics.pstdev(all_samples) / mean,
    }
    if any(not _same_number(value.get(field), expected) for field, expected in expected_summary.items()):
        problems.append(f"{label} summary does not match the raw samples")


def _accuracy_complete(value: object, tolerances: dict[str, float]) -> bool:
    if not isinstance(value, dict) or set(value) != {"output", "dv", "dw"}:
        return False
    for name in ("output", "dv", "dw"):
        report = value[name]
        if not isinstance(report, dict) or set(report) != SUCCESS_ACCURACY_FIELDS:
            return False
        if (
            report.get("ok") is not True
            or report.get("has_nan") is not False
            or report.get("has_inf") is not False
            or report.get("rel_l2_tol") != tolerances[name]
            or not _finite_number(report.get("rel_l2"))
            or report["rel_l2"] > tolerances[name]
            or not _finite_number(report.get("dtype_floor_rel_l2"))
            or not _finite_number(report.get("max_abs_err"))
            or not _finite_number(report.get("output_rms"), minimum=1e-12)
        ):
            return False
        for field in ("err_vs_dtype_floor", "max_abs_err_over_rms"):
            metric = report.get(field)
            if metric is not None and not _finite_number(metric):
                return False
    return True


def _kernel_profile_complete(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"total_kernels", "total_cuda_us", "by_name"}:
        return False
    total = value["total_kernels"]
    by_name = value["by_name"]
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not _finite_number(value["total_cuda_us"], minimum=1e-12)
        or not isinstance(by_name, dict)
        or not by_name
    ):
        return False
    launches = 0.0
    for measurement in by_name.values():
        if not isinstance(measurement, dict) or set(measurement) != {
            "launches_per_call",
            "cuda_us_per_call",
        }:
            return False
        if not _finite_number(measurement["launches_per_call"], minimum=1e-12) or not _finite_number(
            measurement["cuda_us_per_call"], minimum=1e-12
        ):
            return False
        launches += measurement["launches_per_call"]
    return math.isclose(launches, total, rel_tol=1e-9, abs_tol=1e-9)


def _memory_profile_complete(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != MEMORY_FIELDS:
        return False
    if any(
        isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0
        for field in MEMORY_FIELDS
    ):
        return False
    return (
        value["resident_bytes"] > 0
        and value["accounted_output_bytes"] > 0
        and value["peak_allocated_bytes"]
        == value["resident_bytes"] + value["workspace_bytes"]
        and value["workspace_bytes"]
        == max(0, value["incremental_peak_bytes"] - value["accounted_output_bytes"])
    )


def report_completeness_problems(
    report: dict,
    *,
    dtype: str,
    shape_set: str,
    implementations: list[str],
    quick: bool,
) -> list[str]:
    """Require the complete generated phase before a checkpoint can be reused."""
    problems: list[str] = []
    expected_shapes = SHAPE_SETS.get(shape_set)
    if expected_shapes is None:
        return [f"unknown shape set: {shape_set}"]
    trial_count, reps, warmup = (2, 10, 8) if quick else (15, 13, 10)

    methodology = report.get("methodology")
    _require_fields(methodology, METHOD_FIELDS, "methodology", problems)
    if isinstance(methodology, dict) and any(
        not isinstance(methodology.get(field), str) or not methodology[field]
        for field in METHOD_FIELDS
    ):
        problems.append("methodology fields must be nonempty strings")
    expected_tolerances = EXPECTED_TOLERANCES.get(dtype)
    if expected_tolerances is None:
        return [f"unknown dtype: {dtype}"]
    tolerances = report.get("tolerances")
    _require_fields(tolerances, {"output", "dv", "dw"}, "tolerances", problems)
    if isinstance(tolerances, dict) and any(
        isinstance(tolerances.get(field), bool)
        or not isinstance(tolerances.get(field), int | float)
        or tolerances[field] <= 0
        for field in ("output", "dv", "dw")
    ):
        problems.append("tolerances must contain positive numeric values")
    if tolerances != expected_tolerances:
        problems.append("tolerances differ from the fixed dtype contract")

    records = report.get("results")
    if not isinstance(records, list):
        problems.append("results must be a list")
        records = []
    by_pair: dict[tuple[tuple[int, int, int, int], str], dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"results[{index}] must be an object")
            continue
        shape = _shape(record.get("shape"))
        name = record.get("impl")
        if shape not in expected_shapes or name not in implementations:
            problems.append(f"results[{index}] has an unexpected shape or implementation")
            continue
        pair = (shape, name)
        if pair in by_pair:
            problems.append(f"duplicate result record for {name} at {shape}")
            continue
        by_pair[pair] = record
        supported, expected_plan = _plan_support(name, shape, dtype)
        skipped = record.get("skipped")
        if expected_plan is not None and record.get("training_plan") != expected_plan:
            problems.append(f"{name} at {shape} has the wrong training plan")
        if not supported:
            if not isinstance(skipped, str) or not skipped.startswith("unsupported plan:"):
                problems.append(f"{name} at {shape} does not record its unsupported plan")
            continue
        if skipped:
            problems.append(f"supported implementation {name} at {shape} is skipped")
            continue
        correctness = record.get("correctness_by_seed")
        if not isinstance(correctness, list) or [
            item.get("seed") for item in correctness if isinstance(item, dict)
        ] != [0, 1, 2]:
            problems.append(f"{name} at {shape} has an incomplete correctness matrix")
        elif any(
            not _accuracy_complete(item.get("report"), expected_tolerances)
            for item in correctness
        ) or not _accuracy_complete(record.get("correctness"), expected_tolerances):
            problems.append(f"{name} at {shape} has incomplete correctness evidence")
        for metric in ("forward", "backward", "fwd_bwd"):
            _complete_timing(
                record.get(metric),
                trial_count=trial_count,
                reps=reps,
                warmup=warmup,
                label=f"{name} at {shape} {metric}",
                problems=problems,
            )
        for field in ("forward_kernels", "backward_kernels", "fwd_bwd_kernels"):
            if not _kernel_profile_complete(record.get(field)):
                problems.append(f"{name} at {shape} has incomplete {field}")
        if not _memory_profile_complete(record.get("fwd_bwd_memory")):
            problems.append(f"{name} at {shape} has incomplete fwd_bwd_memory")

    expected_pairs = {
        (shape, name) for shape in expected_shapes for name in implementations
    }
    if set(by_pair) != expected_pairs:
        problems.append("result shape and implementation matrix is not exact")

    schedules = report.get("execution_order")
    if not isinstance(schedules, list):
        problems.append("execution_order must be a list")
        schedules = []
    by_shape: dict[tuple[int, int, int, int], dict] = {}
    for schedule in schedules:
        if not isinstance(schedule, dict):
            continue
        shape = _shape(schedule.get("shape"))
        if shape is None or shape in by_shape:
            problems.append("execution schedule contains an invalid or duplicate shape")
            continue
        by_shape[shape] = schedule
    if set(by_shape) != expected_shapes:
        problems.append("execution schedule shape matrix is not exact")
    for shape in expected_shapes & set(by_shape):
        runnable = {
            name
            for name in implementations
            if isinstance(by_pair.get((shape, name)), dict)
            and not by_pair[(shape, name)].get("skipped")
        }
        metrics = by_shape[shape].get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != {
            "forward",
            "backward",
            "fwd_bwd",
        }:
            problems.append(f"execution schedule at {shape} has an incomplete metric set")
            continue
        for metric, trials in metrics.items():
            if not isinstance(trials, list) or len(trials) != trial_count:
                problems.append(f"execution schedule at {shape} {metric} is incomplete")
                continue
            if [
                trial.get("trial") for trial in trials if isinstance(trial, dict)
            ] != list(range(trial_count)):
                problems.append(f"execution schedule at {shape} {metric} has wrong trial IDs")
            for trial in trials:
                names = trial.get("implementations") if isinstance(trial, dict) else None
                if (
                    not isinstance(names, list)
                    or len(names) != len(runnable)
                    or set(names) != runnable
                ):
                    problems.append(f"execution schedule at {shape} {metric} is not exact")
                    break
                trial_id = trial["trial"]
                for order, name in enumerate(names):
                    timing = by_pair[(shape, name)].get(metric)
                    measured_trials = timing.get("trials") if isinstance(timing, dict) else None
                    if (
                        not isinstance(measured_trials, list)
                        or trial_id >= len(measured_trials)
                        or measured_trials[trial_id].get("order_in_trial") != order
                    ):
                        problems.append(
                            f"execution schedule at {shape} {metric} does not match raw timing order"
                        )
                        break

    tails = report.get("correctness_only")
    if not isinstance(tails, list):
        problems.append("correctness_only must be a list")
        tails = []
    by_tail: dict[tuple[int, int, int, int], dict] = {}
    for case in tails:
        if not isinstance(case, dict):
            continue
        shape = _shape(case.get("shape"))
        if shape is None or shape in by_tail:
            problems.append("correctness-only cases contain an invalid or duplicate shape")
            continue
        by_tail[shape] = case
    if set(by_tail) != CORRECTNESS_ONLY_SHAPES:
        problems.append("correctness-only shape matrix is not exact")
    for shape in CORRECTNESS_ONLY_SHAPES & set(by_tail):
        rows = by_tail[shape].get("implementations")
        if not isinstance(rows, list):
            problems.append(f"correctness-only case {shape} has no implementation rows")
            continue
        observed: dict[tuple[str, int], dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = (row.get("impl"), row.get("seed"))
            if key in observed:
                problems.append(f"correctness-only case {shape} has a duplicate row")
            observed[key] = row
        expected = {(name, seed) for name in implementations for seed in (0, 1, 2)}
        if set(observed) != expected:
            problems.append(f"correctness-only case {shape} matrix is not exact")
            continue
        for (name, _seed), row in observed.items():
            supported, expected_plan = _plan_support(name, shape, dtype)
            if (
                expected_plan is not None
                and row.get("training_plan", expected_plan) != expected_plan
            ):
                problems.append(f"correctness-only {name} at {shape} has the wrong plan")
            if not supported:
                if not str(row.get("skipped", "")).startswith("unsupported plan:"):
                    problems.append(f"correctness-only {name} at {shape} must be unsupported")
            elif row.get("skipped") or not _accuracy_complete(
                row.get("correctness"), expected_tolerances
            ):
                problems.append(f"correctness-only {name} at {shape} is incomplete")
    return problems


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
    _require_fields(report, TOP_LEVEL_FIELDS, "report", problems)
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
    _require_fields(
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
            "device_id",
            "device_query",
            "benchmark_process_context_count",
            "foreign_compute_process_count_at_start",
            "exclusive_access_required",
            "busy_override",
        },
        "gpu_preflight",
        problems,
    )
    _require_fields(
        report.get("gpu_preflight"),
        {
            "logical_device",
            "device_id",
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
            "device_id",
            "benchmark_process_context_count",
            "foreign_compute_process_count_at_end",
        },
        "gpu_postflight",
        problems,
    )
    _require_fields(
        report.get("gpu_postflight"),
        {
            "device_id",
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
            "device_id",
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
    _require_fields(
        monitor,
        {
            "device_id",
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
    _require_fields(
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
        METHOD_FIELDS,
        "methodology",
        problems,
    )
    _require_fields(report.get("methodology"), METHOD_FIELDS, "methodology", problems)

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
    expected_device_id: str,
) -> list[str]:
    """Return every reason a report cannot be reused."""
    problems = report_schema_problems(report)
    problems.extend(
        report_completeness_problems(
            report,
            dtype=dtype,
            shape_set=shape_set,
            implementations=implementations,
            quick=quick,
        )
    )
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
    if (
        isinstance(report.get("campaign_attempt"), bool)
        or not isinstance(report.get("campaign_attempt"), int)
        or not 1 <= report["campaign_attempt"] <= 3
    ):
        problems.append("campaign_attempt must be an integer from 1 through 3")

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
    device_id = preflight.get("device_id")
    if device_id != expected_device_id:
        problems.append("device ID differs from the campaign device")
    if not isinstance(device_id, str) or PUBLIC_DEVICE_ID_PATTERN.fullmatch(device_id) is None:
        problems.append("campaign device ID is not an opaque public identifier")
    if (
        not device_id
        or preflight.get("foreign_compute_process_count_at_start") != 0
        or preflight.get("busy_override")
    ):
        problems.append("GPU preflight was not exclusive")
    if (
        postflight.get("device_id") != device_id
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
        monitor.get("device_id") != device_id
        or monitor.get("collision_detected") is not False
        or monitor.get("collision_events")
        or monitor.get("probe_errors")
        or not coverage_ok
    ):
        problems.append("sampled GPU monitor was incomplete or contaminated")
    if PRIVATE_METADATA_PATTERN.search(json.dumps(report, sort_keys=True)):
        problems.append("report contains private host or task metadata")
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
    parser.add_argument("--expected-device-id", required=True)
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
        expected_device_id=args.expected_device_id,
    )
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
