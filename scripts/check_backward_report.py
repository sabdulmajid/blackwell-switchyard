#!/usr/bin/env python3
"""Accept only a clean, complete backward report for phase reuse."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from switchyard.performance import (  # noqa: E402
    backward_traffic_estimate,
    forward_traffic_estimate,
)
from switchyard.training_plan import (  # noqa: E402
    RESIDENT_TILE_MAX,
    get_training_plan,
    plan_supports,
)

PINNED_LIGER_COMMIT = "777799588a89d74c489ed995e3bf006427738e85"
PINNED_LIGER_SOURCE_SHA256 = "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"

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
# A report is publishable only for the repository's documented measurement host.
EXPECTED_ENVIRONMENT = {
    "torch": "2.9.0+cu128",
    "torch_cuda": "12.8",
    "triton": "3.5.0",
    "python": "3.12.3",
    "platform": "Linux-6.8.0-117-generic-x86_64-with-glibc2.39",
    "device_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "device_cc": "12.0",
    "device_index": 0,
}
EXPECTED_DRIVER_VERSION = "580.159.03"
EXPECTED_MULTIPROCESSORS = 188
EXPECTED_MAX_SHARED_BYTES = 101376
ACCEPTED_STATUS = "accepted production dispatch"
EXPERIMENTAL_STATUS = "experimental candidate; not reachable from production dispatch"
LIGER_STATUS = "pinned third-party comparator; RMSNorm gain fixed to one"
EXPERIMENTAL_IMPLEMENTATIONS = {
    "source_serial",
    "serial_recompute_atomic_t4",
    "serial_saved_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
}
PUBLIC_DEVICE_ID_PATTERN = re.compile(r"device-[0-9a-f]{16}")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{3})?Z"
)
PRIVATE_METADATA_PATTERN = re.compile(
    r"co[\s_-]*authored[\s_-]*by|claude(?:\.ai)?|anthropic|wizchem|chatgpt|"
    r"openai(?:\.com)?|session(?:[-_/ ]?(?:id|url))?[-_/:= ]+[A-Za-z0-9]|"
    r"\b(?:host(?:name)?|node(?:name)?|machine(?:name)?)\s*[:=]\s*\S+|"
    r"\b[A-Za-z][A-Za-z0-9+.-]*://|GPU-[A-Za-z0-9-]+|"
    r"\b[A-Z]:[\\/]|\\\\[^\\\s]+\\|(?:^|\s)~/|"
    r"(?:[A-Za-z0-9-]+\.)+(?:internal|local)\b|"
    r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"']+|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _expected_result_status(name: str) -> str | None:
    if name == "current":
        return ACCEPTED_STATUS
    if name == "liger":
        return LIGER_STATUS
    if name in EXPERIMENTAL_IMPLEMENTATIONS:
        return EXPERIMENTAL_STATUS
    return None


def _private_metadata_present(value: object) -> bool:
    if isinstance(value, str):
        if re.fullmatch(r"<external>/[A-Za-z0-9._-]+", value) is not None:
            return False
        return PRIVATE_METADATA_PATTERN.search(value) is not None
    if isinstance(value, dict):
        return any(
            _private_metadata_present(key) or _private_metadata_present(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_private_metadata_present(item) for item in value)
    return False


def _expected_argv(
    *, dtype: str, shape_set: str, implementations: list[str], quick: bool
) -> list[str]:
    phase = "smoke" if quick else "full"
    argv = [
        "bench/bench_backward.py",
        "--shape-set",
        shape_set,
        "--dtype",
        dtype,
    ]
    if quick:
        argv.append("--quick")
    argv.extend(
        [
            "--impls",
            ",".join(implementations),
            "--out",
            f"<external>/{phase}_{dtype}.json",
        ]
    )
    return argv


def _expected_methodology(*, quick: bool) -> dict[str, str]:
    trial_count = 2 if quick else 15
    return {
        "oracle": "float64 forward and first-order gradients on each timed shape",
        "timing": (
            (
                "two interleaved trials of 10 CUDA-event samples"
                if quick
                else "15 interleaved trials of 13 CUDA-event samples"
            )
            + "; all raw samples and trial order stored"
        ),
        "statistics": (
            "paired trial medians are independent observations; individual event "
            "samples are not treated as independent; a dispatch cell must win every "
            f"one of the {trial_count} trials against current and Liger"
        ),
        "cache": "L2 flushed after graph setup and before every timed region",
        "compilation": "excluded by warmup",
        "backward": "measured directly on a retained graph; not median subtraction",
        "promotion": "run scripts/evaluate_backward.py on this raw result",
    }


def _device_query_valid(value: object) -> bool:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
        return False
    fields = [field.strip() for field in value.split(",")]
    if len(fields) != 7 or fields[0] != EXPECTED_ENVIRONMENT["device_name"]:
        return False
    if fields[1] != EXPECTED_DRIVER_VERSION:
        return False
    if re.fullmatch(r"[0-9]+", fields[2]) is None:
        return False
    try:
        power_draw, power_limit = (float(fields[index]) for index in (3, 4))
        sm_clock, memory_clock = (int(fields[index]) for index in (5, 6))
    except ValueError:
        return False
    return (
        math.isfinite(power_draw)
        and power_draw >= 0
        and math.isfinite(power_limit)
        and power_limit > 0
        and sm_clock > 0
        and memory_clock > 0
    )


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _timestamp_valid(value: object, pattern: re.Pattern, time_format: str) -> bool:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, time_format)
    except ValueError:
        return False
    return True


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
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in fields):
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
        if any(
            not _same_number(trial.get(field), expected)
            for field, expected in expected_trial.items()
        ):
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
    if any(
        not _same_number(value.get(field), expected) for field, expected in expected_summary.items()
    ):
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
        floor = report["dtype_floor_rel_l2"]
        expected_floor_ratio = report["rel_l2"] / floor if floor > 0 else None
        if expected_floor_ratio is None:
            if report.get("err_vs_dtype_floor") is not None:
                return False
        elif not _same_number(report.get("err_vs_dtype_floor"), expected_floor_ratio):
            return False
        if not _same_number(
            report.get("max_abs_err_over_rms"),
            report["max_abs_err"] / report["output_rms"],
        ):
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
    cuda_us = []
    for measurement in by_name.values():
        if not isinstance(measurement, dict) or set(measurement) != {
            "launches_per_call",
            "cuda_us_per_call",
        }:
            return False
        if not _finite_number(
            measurement["launches_per_call"], minimum=1e-12
        ) or not _finite_number(measurement["cuda_us_per_call"], minimum=1e-12):
            return False
        launches += measurement["launches_per_call"]
        cuda_us.append(measurement["cuda_us_per_call"])
    return math.isclose(launches, total, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        math.fsum(cuda_us), value["total_cuda_us"], rel_tol=1e-6, abs_tol=1e-6
    )


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
        and value["peak_allocated_bytes"] == value["resident_bytes"] + value["workspace_bytes"]
        and value["workspace_bytes"]
        == max(0, value["incremental_peak_bytes"] - value["accounted_output_bytes"])
    )


def _canonical_json(value: object) -> object:
    return json.loads(json.dumps(value))


def _expected_record_fields(
    name: str, *, supported: bool, has_plan: bool, plan_family: str | None
) -> set[str]:
    fields = {"impl", "shape", "status"}
    if has_plan:
        fields.add("training_plan")
    if not supported:
        fields.add("skipped")
        return fields
    fields.update(
        {
            "correctness",
            "correctness_by_seed",
            "forward",
            "backward",
            "fwd_bwd",
            "forward_kernels",
            "backward_kernels",
            "fwd_bwd_kernels",
            "fwd_bwd_memory",
        }
    )
    if name == "current" or has_plan:
        fields.update({"traffic_model", "forward_traffic_model"})
    if plan_family in {"cuda_cluster", "cuda_cluster4"}:
        fields.add("cluster_launch_info")
    elif plan_family == "cuda_register":
        fields.add("register_launch_info")
    elif plan_family == "cuda_register_cluster":
        fields.add("register_cluster_launch_info")
        if get_training_plan(name).forward.family == "cuda_register_cluster":
            fields.add("register_cluster_forward_launch_info")
    return fields


def _positive_exact_int(value: object) -> bool:
    return type(value) is int and value > 0


def _launch_info_active_workers(
    record: dict,
    *,
    plan_name: str,
    shape: tuple[int, int, int, int],
    dtype: str,
    problems: list[str],
) -> int | None:
    n, _b, _t, d = shape
    plan = get_training_plan(plan_name)
    family = plan.backward.family
    itemsize = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    if family in {"cuda_cluster", "cuda_cluster4"}:
        field = "cluster_launch_info"
        blocks = 2 if family == "cuda_cluster" else 4
        expected_fields = {
            "active_clusters",
            "dynamic_shared_bytes",
            "static_shared_bytes",
            "max_shared_bytes",
            "multiprocessors",
            "threads_per_block",
            "cluster_blocks",
        }
        feature_capacity = (d + blocks - 1) // blocks
        scalar_offset = (n * feature_capacity * itemsize + feature_capacity * itemsize + 15) & ~15
        dynamic_shared = scalar_offset + 4 * (feature_capacity + 13 * n)
        fixed = {
            "dynamic_shared_bytes": dynamic_shared,
            "static_shared_bytes": 1024,
            "max_shared_bytes": EXPECTED_MAX_SHARED_BYTES,
            "multiprocessors": EXPECTED_MULTIPROCESSORS,
            "threads_per_block": 256,
            "cluster_blocks": blocks,
        }
        active_field = "active_clusters"
    elif family == "cuda_register":
        field = "register_launch_info"
        expected_fields = {
            "active_blocks",
            "dynamic_shared_bytes",
            "static_shared_bytes",
            "multiprocessors",
            "threads_per_block",
        }
        fixed = {
            "active_blocks": EXPECTED_MULTIPROCESSORS,
            "dynamic_shared_bytes": 4 * (n * 21 + d),
            "static_shared_bytes": 1024,
            "multiprocessors": EXPECTED_MULTIPROCESSORS,
            "threads_per_block": 512,
        }
        active_field = "active_blocks"
    elif family == "cuda_register_cluster":
        field = "register_cluster_launch_info"
        blocks = 4 if (n, d) == (9, 8192) else 2
        expected_fields = {
            "active_clusters",
            "dynamic_shared_bytes",
            "static_shared_bytes",
            "registers_per_thread",
            "max_shared_bytes",
            "multiprocessors",
            "threads_per_block",
            "cluster_blocks",
        }
        fixed = {
            "dynamic_shared_bytes": 4 * (n * 21 + d // blocks) + itemsize * (d // blocks),
            "static_shared_bytes": 1024,
            "registers_per_thread": 128,
            "max_shared_bytes": EXPECTED_MAX_SHARED_BYTES,
            "multiprocessors": EXPECTED_MULTIPROCESSORS,
            "threads_per_block": 512,
            "cluster_blocks": blocks,
        }
        active_field = "active_clusters"
    else:
        return None

    value = record.get(field)
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or any(not _positive_exact_int(item) for item in value.values())
        or any(value.get(name) != expected for name, expected in fixed.items())
    ):
        problems.append(f"{plan_name} at {shape} has invalid {field}")
        return None
    active = value[active_field]
    if active > EXPECTED_MULTIPROCESSORS:
        problems.append(f"{plan_name} at {shape} has impossible {active_field}")
        return None

    if plan.forward.family == "cuda_register_cluster":
        forward = record.get("register_cluster_forward_launch_info")
        expected_fields = {
            "active_clusters",
            "dynamic_shared_bytes",
            "static_shared_bytes",
            "registers_per_thread",
            "max_shared_bytes",
            "multiprocessors",
            "threads_per_block",
            "cluster_blocks",
        }
        register_sources = 0 if (n, d) == (9, 8192) else 28
        forward_fixed = {
            "dynamic_shared_bytes": 4 * n * 33 + itemsize * (n - register_sources) * (d // blocks),
            "static_shared_bytes": 1024,
            "registers_per_thread": (
                96 if (n, d) == (9, 8192) else 103 if dtype == "bfloat16" else 109
            ),
            "max_shared_bytes": EXPECTED_MAX_SHARED_BYTES,
            "multiprocessors": EXPECTED_MULTIPROCESSORS,
            "threads_per_block": 512,
            "cluster_blocks": blocks,
        }
        if (
            not isinstance(forward, dict)
            or set(forward) != expected_fields
            or any(not _positive_exact_int(item) for item in forward.values())
            or any(forward.get(name) != expected for name, expected in forward_fixed.items())
            or forward["active_clusters"] > EXPECTED_MULTIPROCESSORS
        ):
            problems.append(
                f"{plan_name} at {shape} has invalid register-cluster forward launch info"
            )
    return active


def _expected_traffic_models(
    name: str,
    shape: tuple[int, int, int, int],
    dtype: str,
    record: dict,
    problems: list[str],
) -> tuple[object, object] | None:
    n, b, t, d = shape
    itemsize = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    n_pow2 = 1 << (n - 1).bit_length()
    d_pow2 = 1 << (d - 1).bit_length()
    forward_resident = n_pow2 * d_pow2 <= RESIDENT_TILE_MAX
    if name == "current":
        backward = backward_traffic_estimate(
            "resident" if forward_resident else "tiled",
            n,
            b,
            t,
            d,
            itemsize=itemsize,
        ).as_dict()
        forward = forward_traffic_estimate(
            "resident" if forward_resident else "tiled",
            n,
            b,
            t,
            d,
            itemsize=itemsize,
        ).as_dict()
        return _canonical_json(backward), _canonical_json(forward)
    if name == "liger":
        return None

    plan = get_training_plan(name)
    family = plan.backward.family
    options = {}
    if family in {
        "cuda_cluster",
        "cuda_cluster4",
        "cuda_register",
        "cuda_register_cluster",
    }:
        active = _launch_info_active_workers(
            record,
            plan_name=name,
            shape=shape,
            dtype=dtype,
            problems=problems,
        )
        if active is None:
            return None
        options["persistent_clusters"] = min(b * t, active)
    model_name = {
        "source_serial": ("source_serial_saved" if plan.saves_forward_stats else "source_serial"),
        "cuda_shared": "cuda_shared",
        "cuda_cluster": "cuda_cluster",
        "cuda_cluster4": "cuda_cluster4",
        "cuda_register": "cuda_register",
        "cuda_register_cluster": "cuda_register_cluster",
    }[family]
    backward = backward_traffic_estimate(
        model_name,
        n,
        b,
        t,
        d,
        itemsize=itemsize,
        source_tokens_per_cta=plan.backward.tokens_per_cta,
        source_uses_partials=plan.backward.dw_reduction == "partials",
        **options,
    ).as_dict()
    forward = forward_traffic_estimate(
        (
            "cuda_register_cluster"
            if plan.forward.family == "cuda_register_cluster"
            else "resident"
            if forward_resident
            else "tiled"
        ),
        n,
        b,
        t,
        d,
        itemsize=itemsize,
        saves_backward_coefficients=plan.saves_forward_stats,
    ).as_dict()
    return _canonical_json(backward), _canonical_json(forward)


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
        expected_status = _expected_result_status(name)
        plan_family = expected_plan["backward"]["family"] if expected_plan is not None else None
        expected_fields = _expected_record_fields(
            name,
            supported=supported,
            has_plan=expected_plan is not None,
            plan_family=plan_family,
        )
        if set(record) != expected_fields:
            problems.append(f"{name} at {shape} has a non-exact result field set")
        if expected_status is None or record.get("status") != expected_status:
            problems.append(f"{name} at {shape} has an invalid result status")
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
        if (
            not isinstance(correctness, list)
            or len(correctness) != 3
            or any(
                not isinstance(item, dict) or set(item) != {"seed", "report"}
                for item in correctness
            )
            or [item["seed"] for item in correctness if isinstance(item, dict)] != [0, 1, 2]
            or any(type(item["seed"]) is not int for item in correctness)
        ):
            problems.append(f"{name} at {shape} has an incomplete correctness matrix")
        elif any(
            not _accuracy_complete(item.get("report"), expected_tolerances) for item in correctness
        ) or not _accuracy_complete(record.get("correctness"), expected_tolerances):
            problems.append(f"{name} at {shape} has incomplete correctness evidence")
        elif record.get("correctness") != correctness[0]["report"]:
            problems.append(f"{name} at {shape} aggregate correctness differs from seed zero")
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
        expected_traffic = _expected_traffic_models(name, shape, dtype, record, problems)
        if expected_traffic is not None:
            backward_traffic, forward_traffic = expected_traffic
            if record.get("traffic_model") != backward_traffic:
                problems.append(f"{name} at {shape} has an invalid backward traffic model")
            if record.get("forward_traffic_model") != forward_traffic:
                problems.append(f"{name} at {shape} has an invalid forward traffic model")

    expected_pairs = {(shape, name) for shape in expected_shapes for name in implementations}
    if set(by_pair) != expected_pairs:
        problems.append("result shape and implementation matrix is not exact")

    schedules = report.get("execution_order")
    if not isinstance(schedules, list):
        problems.append("execution_order must be a list")
        schedules = []
    by_shape: dict[tuple[int, int, int, int], dict] = {}
    for index, schedule in enumerate(schedules):
        if not isinstance(schedule, dict):
            problems.append(f"execution_order[{index}] must be an object")
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
            if any(
                not isinstance(trial, dict) or set(trial) != {"trial", "implementations"}
                for trial in trials
            ):
                problems.append(f"execution schedule at {shape} {metric} has malformed trials")
                continue
            if [trial["trial"] for trial in trials] != list(range(trial_count)) or any(
                type(trial["trial"]) is not int for trial in trials
            ):
                problems.append(f"execution schedule at {shape} {metric} has wrong trial IDs")
            for trial in trials:
                names = trial["implementations"]
                if (
                    not isinstance(names, list)
                    or len(names) != len(runnable)
                    or any(not isinstance(name, str) for name in names)
                    or (all(isinstance(name, str) for name in names) and set(names) != runnable)
                ):
                    problems.append(f"execution schedule at {shape} {metric} is not exact")
                    break
                trial_id = trial["trial"]
                if type(trial_id) is not int or not 0 <= trial_id < trial_count:
                    continue
                for order, name in enumerate(names):
                    timing = by_pair.get((shape, name), {}).get(metric)
                    measured_trials = timing.get("trials") if isinstance(timing, dict) else None
                    if (
                        not isinstance(measured_trials, list)
                        or trial_id >= len(measured_trials)
                        or not isinstance(measured_trials[trial_id], dict)
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
    for index, case in enumerate(tails):
        if not isinstance(case, dict):
            problems.append(f"correctness_only[{index}] must be an object")
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
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                problems.append(f"correctness-only case {shape} row {row_index} must be an object")
                continue
            name = row.get("impl")
            seed = row.get("seed")
            if not isinstance(name, str) or type(seed) is not int:
                problems.append(f"correctness-only case {shape} row {row_index} has an invalid key")
                continue
            key = (name, seed)
            if key in observed:
                problems.append(f"correctness-only case {shape} has a duplicate row")
            observed[key] = row
        expected = {(name, seed) for name in implementations for seed in (0, 1, 2)}
        if set(observed) != expected:
            problems.append(f"correctness-only case {shape} matrix is not exact")
            continue
        for (name, _seed), row in observed.items():
            supported, expected_plan = _plan_support(name, shape, dtype)
            expected_fields = {"impl", "seed"}
            if expected_plan is not None:
                expected_fields.add("training_plan")
            expected_fields.add("correctness" if supported else "skipped")
            if set(row) != expected_fields:
                problems.append(f"correctness-only {name} at {shape} has a non-exact field set")
            if expected_plan is not None and row.get("training_plan") != expected_plan:
                problems.append(f"correctness-only {name} at {shape} has the wrong plan")
            if not supported:
                skipped = row.get("skipped")
                if not isinstance(skipped, str) or not skipped.startswith("unsupported plan:"):
                    problems.append(f"correctness-only {name} at {shape} must be unsupported")
            elif not _accuracy_complete(row.get("correctness"), expected_tolerances):
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
    _only_fields(value.get("forward"), {"family", "saved_state"}, f"{label}.forward", problems)
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
    trials = value.get("trials")
    if not isinstance(trials, list):
        return
    for index, trial in enumerate(trials):
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
            "timestamp_utc",
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
            "timestamp_utc",
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
            "timestamp_utc",
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
            "timestamp_utc",
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
            "started_at_utc",
            "ended_at_utc",
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
            "started_at_utc",
            "ended_at_utc",
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
        collision_events = monitor.get("collision_events")
        if not isinstance(collision_events, list):
            collision_events = []
        for index, event in enumerate(collision_events):
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

    records = report.get("results")
    if not isinstance(records, list):
        records = []
    for record_index, record in enumerate(records):
        label = f"results[{record_index}]"
        _only_fields(record, RECORD_FIELDS, label, problems)
        if not isinstance(record, dict):
            continue
        _only_fields(record.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        _check_plan_schema(record.get("training_plan"), f"{label}.training_plan", problems)
        _check_accuracy_schema(record.get("correctness"), f"{label}.correctness", problems)
        correctness_by_seed = record.get("correctness_by_seed")
        if not isinstance(correctness_by_seed, list):
            correctness_by_seed = []
        for seed_index, seed in enumerate(correctness_by_seed):
            seed_label = f"{label}.correctness_by_seed[{seed_index}]"
            _only_fields(seed, {"seed", "report"}, seed_label, problems)
            if isinstance(seed, dict):
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

    correctness_only = report.get("correctness_only")
    if not isinstance(correctness_only, list):
        correctness_only = []
    for case_index, case in enumerate(correctness_only):
        label = f"correctness_only[{case_index}]"
        _only_fields(case, {"shape", "implementations"}, label, problems)
        if not isinstance(case, dict):
            continue
        _only_fields(case.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        rows = case.get("implementations")
        if not isinstance(rows, list):
            rows = []
        for row_index, row in enumerate(rows):
            row_label = f"{label}.implementations[{row_index}]"
            _only_fields(
                row,
                {"impl", "seed", "training_plan", "skipped", "correctness"},
                row_label,
                problems,
            )
            if isinstance(row, dict):
                _check_plan_schema(row.get("training_plan"), f"{row_label}.training_plan", problems)
                _check_accuracy_schema(row.get("correctness"), f"{row_label}.correctness", problems)

    execution_order = report.get("execution_order")
    if not isinstance(execution_order, list):
        execution_order = []
    for schedule_index, schedule in enumerate(execution_order):
        label = f"execution_order[{schedule_index}]"
        _only_fields(schedule, {"shape", "metrics"}, label, problems)
        if not isinstance(schedule, dict):
            continue
        _only_fields(schedule.get("shape"), {"n", "b", "t", "d"}, f"{label}.shape", problems)
        metrics = schedule.get("metrics")
        _only_fields(metrics, {"forward", "backward", "fwd_bwd"}, f"{label}.metrics", problems)
        if isinstance(metrics, dict):
            for metric, trials in metrics.items():
                if not isinstance(trials, list):
                    continue
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
    if not isinstance(report, dict):
        return ["report must be an object"]
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
        "schema_version": 3,
        "run_status": "complete",
        "dtype": dtype,
        "shape_set": shape_set,
    }
    for field, value in expected.items():
        observed = report.get(field)
        if type(observed) is not type(value) or observed != value:
            problems.append(f"{field} must equal {value!r}")
    if report.get("candidate_reachable_from_production") is not False:
        problems.append("candidate_reachable_from_production must equal False")
    correctness_seeds = report.get("correctness_seeds")
    if (
        not isinstance(correctness_seeds, list)
        or correctness_seeds != [0, 1, 2]
        or any(type(seed) is not int for seed in correctness_seeds)
    ):
        problems.append("correctness_seeds must equal [0, 1, 2]")
    run_id = report.get("run_id")
    run_id_valid = _timestamp_valid(run_id, RUN_ID_PATTERN, "%Y%m%dT%H%M%SZ")
    run_time = (
        datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if run_id_valid
        else None
    )
    if not run_id_valid:
        problems.append("run_id must use the benchmark UTC timestamp format")
    selected_implementations = report.get("selected_implementations")
    if (
        not isinstance(selected_implementations, list)
        or selected_implementations != implementations
        or any(not isinstance(name, str) for name in selected_implementations)
    ):
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

    environment = _object(report.get("environment"))
    for field, expected_value in EXPECTED_ENVIRONMENT.items():
        observed = environment.get(field)
        if type(observed) is not type(expected_value) or observed != expected_value:
            problems.append(f"environment.{field} differs from the campaign environment")
    timestamp = environment.get("timestamp_utc")
    timestamp_valid = _timestamp_valid(timestamp, UTC_TIMESTAMP_PATTERN, "%Y-%m-%dT%H:%M:%SZ")
    environment_time = _parse_utc(timestamp)
    if not timestamp_valid:
        problems.append("environment.timestamp_utc must be a UTC timestamp")
    elif run_time is not None and environment_time is not None:
        if not 0 <= (environment_time - run_time).total_seconds() <= 60:
            problems.append("environment timestamp is not bound to the report start")

    if report.get("methodology") != _expected_methodology(quick=quick):
        problems.append("benchmark methodology differs from the campaign contract")

    provenance = _object(report.get("provenance"))
    if provenance.get("repository_commit") != expected_commit:
        problems.append("repository commit differs from the campaign commit")
    if provenance.get("repository_branch") != expected_branch:
        problems.append("repository branch differs from the campaign branch")
    if provenance.get("repository_tree") != expected_tree:
        problems.append("repository tree differs from the campaign tree")
    if (
        not isinstance(expected_commit, str)
        or GIT_OBJECT_PATTERN.fullmatch(expected_commit) is None
        or not isinstance(expected_tree, str)
        or GIT_OBJECT_PATTERN.fullmatch(expected_tree) is None
    ):
        problems.append("expected repository identity is malformed")
    branch = provenance.get("repository_branch")
    if (
        not isinstance(branch, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch) is None
        or ".." in branch
        or "//" in branch
    ):
        problems.append("repository branch provenance is malformed")
    if provenance.get("worktree_dirty") is not False:
        problems.append("benchmark worktree was not fully clean")
    if provenance.get("tracked_worktree_dirty") is not False:
        problems.append("tracked benchmark worktree was not clean")
    if provenance.get("dirty_paths") != []:
        problems.append("clean benchmark provenance must have no dirty paths")
    if provenance.get("diff_sha256") != EMPTY_DIFF_SHA256:
        problems.append("clean benchmark provenance has a nonempty diff hash")
    if provenance.get("third_party_commits") != {"Liger-Kernel": PINNED_LIGER_COMMIT}:
        problems.append("pinned Liger commit is missing or wrong")
    third_party_dirty = provenance.get("third_party_dirty")
    if (
        not isinstance(third_party_dirty, dict)
        or set(third_party_dirty) != {"Liger-Kernel"}
        or third_party_dirty.get("Liger-Kernel") is not False
    ):
        problems.append("pinned Liger checkout was not clean")
    if provenance.get("input_seed") != 0 or type(provenance.get("input_seed")) is not int:
        problems.append("input seed provenance is not exact")
    if provenance.get("query_seed") != 1 or type(provenance.get("query_seed")) is not int:
        problems.append("query seed provenance is not exact")
    if provenance.get("argv") != _expected_argv(
        dtype=dtype,
        shape_set=shape_set,
        implementations=implementations,
        quick=quick,
    ):
        problems.append("command provenance is not the exact sanitized campaign command")

    liger = _object(_object(report.get("comparators")).get("liger"))
    if (
        liger.get("commit") != PINNED_LIGER_COMMIT
        or liger.get("source_sha256") != PINNED_LIGER_SOURCE_SHA256
        or liger.get("worktree_dirty") is not False
        or liger.get("under_pinned_checkout") is not True
        or liger.get("source_path") != "src/liger_kernel/ops/attn_res.py"
    ):
        problems.append("imported Liger source provenance is incomplete")

    preflight = _object(report.get("gpu_preflight"))
    postflight = _object(report.get("gpu_postflight"))
    preflight_time = _parse_utc(preflight.get("timestamp_utc"))
    postflight_time = _parse_utc(postflight.get("timestamp_utc"))
    device_id = preflight.get("device_id")
    if device_id != expected_device_id:
        problems.append("device ID differs from the campaign device")
    if not isinstance(device_id, str) or PUBLIC_DEVICE_ID_PATTERN.fullmatch(device_id) is None:
        problems.append("campaign device ID is not an opaque public identifier")
    preflight_valid = (
        preflight.get("logical_device") == "cuda:0"
        and _device_query_valid(preflight.get("device_query"))
        and preflight_time is not None
        and preflight.get("benchmark_process_context_count") == 1
        and type(preflight.get("benchmark_process_context_count")) is int
        and type(preflight.get("foreign_compute_process_count_at_start")) is int
        and preflight.get("foreign_compute_process_count_at_start") == 0
        and preflight.get("exclusive_access_required") is True
        and preflight.get("busy_override") is False
    )
    if not preflight_valid:
        problems.append("GPU preflight was not exclusive")
    postflight_invalid = (
        postflight.get("device_id") != device_id
        or postflight_time is None
        or postflight.get("benchmark_process_context_count") != 1
        or type(postflight.get("benchmark_process_context_count")) is not int
        or type(postflight.get("foreign_compute_process_count_at_end")) is not int
        or postflight.get("foreign_compute_process_count_at_end") != 0
    )
    if postflight_invalid:
        problems.append("GPU postflight was not exclusive on the same device")

    monitor = _object(report.get("gpu_process_monitor"))
    samples = monitor.get("samples")
    interval = monitor.get("interval_seconds")
    duration = monitor.get("duration_seconds")
    monitor_started = _parse_utc(monitor.get("started_at_utc"))
    monitor_ended = _parse_utc(monitor.get("ended_at_utc"))
    wall_duration = (
        (monitor_ended - monitor_started).total_seconds()
        if monitor_started is not None and monitor_ended is not None
        else None
    )
    timeline_ok = (
        wall_duration is not None
        and wall_duration >= 0
        and type(duration) is float
        and math.isfinite(duration)
        and abs(wall_duration - duration) <= 2.0
        and preflight_time is not None
        and run_time is not None
        and environment_time is not None
        and postflight_time is not None
        and preflight_time <= monitor_started
        and monitor_started <= run_time.replace(microsecond=999999)
        and monitor_started <= environment_time.replace(microsecond=999999)
        and run_time <= monitor_ended
        and environment_time <= monitor_ended
        and postflight_time <= monitor_ended.replace(microsecond=999999)
    )
    coverage_ok = (
        type(samples) is int
        and samples >= 2
        and type(interval) is float
        and math.isfinite(interval)
        and math.isclose(interval, 0.05, rel_tol=0.0, abs_tol=1e-12)
        and type(duration) is float
        and math.isfinite(duration)
        and duration >= interval
        and samples >= max(2, int(duration / (2 * interval)))
    )
    if (
        monitor.get("device_id") != device_id
        or monitor.get("collision_detected") is not False
        or monitor.get("collision_events") != []
        or monitor.get("probe_errors") != []
        or not coverage_ok
        or not timeline_ok
    ):
        problems.append("sampled GPU monitor was incomplete or contaminated")
    if _private_metadata_present(report):
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
        report = json.loads(args.report.read_text(), parse_constant=_reject_json_constant)
    except (OSError, ValueError, json.JSONDecodeError):
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
