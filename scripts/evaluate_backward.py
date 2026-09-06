#!/usr/bin/env python3
"""Apply correctness, provenance, and paired performance gates to one plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from switchyard._benchmark_contracts import (  # noqa: E402
    LIGER_EXACT_WORK_CONTRACT,
    LIGER_UPSTREAM_WORK_CONTRACT,
    PINNED_LIGER_COMMIT,
    PINNED_LIGER_SOURCE_SHA256,
)
from switchyard.performance import (  # noqa: E402
    backward_traffic_estimate,
    forward_traffic_estimate,
)
from switchyard.training_plan import get_training_plan, plan_supports  # noqa: E402

ALL_DTYPES = {"bfloat16", "float16", "float32"}
LIGER_EXACT_SOURCE_SHA256 = hashlib.sha256(
    (REPO / "src" / "switchyard" / "_liger_exact.py").read_bytes()
).hexdigest()
FULL_TRIAL_COUNT = 15
FULL_REPS_PER_TRIAL = 13
FULL_WARMUP_PER_TRIAL = 10
MAX_CAMPAIGN_ATTEMPTS = 3
PRACTICAL_TRAINING_MARGIN = 1.05
MAX_TRAINING_REGRESSION = 0.03
EXPECTED_FULL_SHAPES = {
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
}
ANCHOR_SHAPES = {
    (9, 1, 4096, 4096),
    (9, 1, 4096, 8192),
    (32, 1, 4096, 2048),
}
MASKED_TAIL_SHAPES = {(9, 1, 129, 4097), (17, 2, 33, 2049)}
CAMPAIGN_CANDIDATES = (
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
)
@dataclass(frozen=True)
class Comparison:
    dtype: str
    shape: tuple[int, int, int, int]
    current_ms: float
    candidate_ms: float
    speedup: float
    change: float
    classification: str
    current_fwd_bwd_ms: float
    candidate_fwd_bwd_ms: float
    current_fwd_bwd_speedup: float
    liger_exact_backward_ms: float
    liger_exact_backward_speedup: float
    liger_exact_fwd_bwd_ms: float
    liger_exact_fwd_bwd_speedup: float
    liger_upstream_fwd_bwd_ms: float
    liger_upstream_fwd_bwd_speedup_observational: float


def _shape(record: dict) -> tuple[int, int, int, int]:
    value = record["shape"]
    return value["n"], value["b"], value["t"], value["d"]


def _correct(record: dict) -> bool:
    reports = [record.get("correctness", {})]
    reports.extend(item.get("report", {}) for item in record.get("correctness_by_seed", []))
    return bool(reports) and all(
        all(report.get(name, {}).get("ok", False) for name in ("output", "dv", "dw"))
        for report in reports
    )


def _timing_from_raw(
    record: dict,
    metric: str,
    schedule: list[dict],
    problems: list[str],
    label: str,
    *,
    trial_count: int = FULL_TRIAL_COUNT,
    reps_per_trial: int = FULL_REPS_PER_TRIAL,
    warmup_per_trial: int = FULL_WARMUP_PER_TRIAL,
) -> dict | None:
    """Validate one timing record and rebuild every statistic from raw samples."""
    timing = record.get(metric, {})
    trials = timing.get("trials", [])
    implementation = record.get("impl")
    if len(trials) != trial_count:
        problems.append(f"{label} {implementation}: expected {trial_count} {metric} trials")
        return None
    schedule_by_trial = {item.get("trial"): item for item in schedule}
    if len(schedule) != trial_count or set(schedule_by_trial) != set(range(trial_count)):
        problems.append(f"{label}: {metric} schedule does not contain exact trial IDs")
        return None

    by_trial: dict[int, float] = {}
    all_samples: list[float] = []
    for trial in trials:
        trial_id = trial.get("trial")
        if not isinstance(trial_id, int) or trial_id in by_trial:
            problems.append(f"{label} {implementation}: invalid or duplicate {metric} trial ID")
            return None
        samples = trial.get("samples_ms", [])
        if len(samples) != reps_per_trial or any(
            not isinstance(value, int | float) or not math.isfinite(value) or value <= 0
            for value in samples
        ):
            problems.append(
                f"{label} {implementation}: {metric} trial {trial_id} needs "
                f"{reps_per_trial} finite positive samples"
            )
            return None
        order = schedule_by_trial[trial_id].get("implementations", [])
        if implementation not in order or trial.get("order_in_trial") != order.index(implementation):
            problems.append(
                f"{label} {implementation}: {metric} trial {trial_id} disagrees with schedule"
            )
            return None
        values = [float(value) for value in samples]
        ordered_trial = sorted(values)
        trial_mean = statistics.fmean(values)
        expected_trial_fields = (
            ("median_ms", statistics.median(values)),
            ("p10_ms", ordered_trial[int(0.10 * len(ordered_trial))]),
            ("p90_ms", ordered_trial[min(len(ordered_trial) - 1, int(0.90 * len(ordered_trial)))]),
            ("min_ms", ordered_trial[0]),
            ("mean_ms", trial_mean),
            ("cv", statistics.pstdev(values) / trial_mean),
        )
        if any(
            not isinstance(trial.get(field), int | float)
            or not math.isclose(
                float(trial[field]), rebuilt, rel_tol=1e-9, abs_tol=1e-12
            )
            for field, rebuilt in expected_trial_fields
        ):
            problems.append(
                f"{label} {implementation}: stored {metric} trial {trial_id} summary is wrong"
            )
        if (
            trial.get("reps") != reps_per_trial
            or trial.get("warmup") != warmup_per_trial
            or trial.get("l2_flushed") is not True
        ):
            problems.append(
                f"{label} {implementation}: stored {metric} trial {trial_id} method is wrong"
            )
        by_trial[trial_id] = statistics.median(values)
        all_samples.extend(values)

    if set(by_trial) != set(range(trial_count)):
        problems.append(f"{label} {implementation}: {metric} trial IDs are incomplete")
        return None
    median = statistics.median(all_samples)
    mean = statistics.fmean(all_samples)
    cv = statistics.pstdev(all_samples) / mean
    ordered = sorted(all_samples)
    rebuilt_trial_medians = [by_trial[index] for index in range(trial_count)]
    stored_values = (
        ("median_ms", timing.get("median_ms"), median),
        ("p10_ms", timing.get("p10_ms"), ordered[int(0.10 * len(ordered))]),
        ("p90_ms", timing.get("p90_ms"), ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))]),
        ("min_ms", timing.get("min_ms"), ordered[0]),
        ("mean_ms", timing.get("mean_ms"), mean),
        ("cv", timing.get("cv"), cv),
    )
    for field, stored, rebuilt in stored_values:
        if not isinstance(stored, int | float) or not math.isclose(
            float(stored), rebuilt, rel_tol=1e-9, abs_tol=1e-12
        ):
            problems.append(
                f"{label} {implementation}: stored {metric} {field} disagrees with raw samples"
            )
    stored_trial_medians = timing.get("trial_medians_ms")
    if not isinstance(stored_trial_medians, list) or len(stored_trial_medians) != trial_count or any(
        not isinstance(stored, int | float)
        or not math.isclose(float(stored), rebuilt, rel_tol=1e-9, abs_tol=1e-12)
        for stored, rebuilt in zip(
            stored_trial_medians if isinstance(stored_trial_medians, list) else [],
            rebuilt_trial_medians,
            strict=False,
        )
    ):
        problems.append(
            f"{label} {implementation}: stored {metric} trial medians disagree with raw samples"
        )
    if (
        timing.get("trial_count") != trial_count
        or timing.get("reps") != len(all_samples)
        or timing.get("warmup_per_trial") != warmup_per_trial
    ):
        problems.append(f"{label} {implementation}: stored {metric} sample counts are wrong")
    if timing.get("l2_flushed") is not True:
        problems.append(f"{label} {implementation}: {metric} did not record an L2 flush")
    return {"median_ms": median, "cv": cv, "trial_medians": by_trial}


def _paired_speedups(
    numerator: dict[int, float], denominator: dict[int, float]
) -> list[float] | None:
    """Return paired trial ratios without a formal independence claim."""
    trial_ids = sorted(set(numerator) & set(denominator))
    if len(trial_ids) < FULL_TRIAL_COUNT:
        return None
    return [numerator[index] / denominator[index] for index in trial_ids]


def _required_dtypes(candidate: str) -> set[str]:
    plan = get_training_plan(candidate)
    return {"bfloat16", "float16"} if plan.backward.family.startswith("cuda") else ALL_DTYPES


def _expected_kernel_contract(
    candidate: str, dtype: str
) -> tuple[tuple[str, ...], int, int]:
    """Return required names and the full autograd launch counts.

    Atomic plans zero an FP32 ``dw`` buffer before their main kernel. Low-
    precision plans also cast that buffer to the query dtype before autograd
    returns it. These are part of the measured operator, not free bookkeeping.
    The partials plan does not need the zero-fill kernel.
    """
    plan = get_training_plan(candidate)
    cast_launches = int(dtype != "float32")
    if plan.backward.family in {"cuda_cluster", "cuda_cluster4"}:
        backward_launches = 2 + cast_launches
        return ("feature_cluster_backward_kernel",), backward_launches, backward_launches + 1
    if plan.backward.family == "cuda_register":
        backward_launches = 2 + cast_launches
        return ("register_backward_kernel",), backward_launches, backward_launches + 1
    if plan.backward.family == "cuda_register_cluster":
        backward_launches = 2 + cast_launches
        return (
            ("register_cluster_backward_kernel",),
            backward_launches,
            backward_launches + 1,
        )
    if plan.backward.family == "cuda_shared":
        backward_launches = 2 + cast_launches
        return ("shared_backward_kernel",), backward_launches, backward_launches + 1
    if plan.backward.dw_reduction == "partials":
        backward_launches = 2 + cast_launches
        return (
            ("_bwd_source_serial_grouped", "_reduce_dw_partials"),
            backward_launches,
            backward_launches + 1,
        )
    backward_launches = 2 + cast_launches
    return ("_bwd_source_serial_grouped",), backward_launches, backward_launches + 1


def _check_kernel_contract(
    candidate: str,
    dtype: str,
    record: dict,
    problems: list[str],
    label: str,
) -> None:
    names, backward_launches, step_launches = _expected_kernel_contract(candidate, dtype)
    for field, expected_launches in (
        ("backward_kernels", backward_launches),
        ("fwd_bwd_kernels", step_launches),
    ):
        profile = record.get(field, {})
        if profile.get("total_kernels") != expected_launches:
            problems.append(
                f"{label}: {field} launched {profile.get('total_kernels')}, expected {expected_launches}"
            )
        observed = " ".join(profile.get("by_name", {}))
        for name in names:
            if name not in observed:
                problems.append(f"{label}: {field} did not contain {name}")
    plan = get_training_plan(candidate)
    if plan.forward.family == "cuda_register_cluster":
        forward = record.get("forward_kernels", {})
        if forward.get("total_kernels") != 1:
            problems.append(f"{label}: custom forward must launch exactly one kernel")
        forward_observed = " ".join(forward.get("by_name", {}))
        step_observed = " ".join(record.get("fwd_bwd_kernels", {}).get("by_name", {}))
        required = "register_cluster_forward_kernel"
        if required not in forward_observed:
            problems.append(f"{label}: forward profile did not contain {required}")
        if required not in step_observed:
            problems.append(f"{label}: training profile did not contain {required}")


def _check_current_kernel_contract(
    shape: tuple[int, int, int, int],
    dtype: str,
    record: dict,
    problems: list[str],
    label: str,
) -> None:
    n, _, _, d = shape
    n_pow2 = 1 << (n - 1).bit_length()
    d_pow2 = 1 << (d - 1).bit_length()
    resident = n_pow2 * d_pow2 <= 32768
    names = ("_bwd_resident",) if resident else ("_bwd_stats", "_bwd_apply")
    # Every accepted path zero-fills its FP32 ``dw`` accumulator. bf16 and
    # fp16 paths then cast it once before returning the gradient.
    expected_backward = len(names) + 1 + int(dtype != "float32")
    for field, expected_launches in (
        ("backward_kernels", expected_backward),
        ("fwd_bwd_kernels", expected_backward + 1),
    ):
        profile = record.get(field, {})
        if profile.get("total_kernels") != expected_launches:
            problems.append(
                f"{label}: current {field} launched {profile.get('total_kernels')}, "
                f"expected {expected_launches}"
            )
        observed = " ".join(profile.get("by_name", {}))
        for name in names:
            if name not in observed:
                problems.append(f"{label}: current {field} did not contain {name}")


def _check_register_cluster_launch_info(
    shape: tuple[int, int, int, int],
    launch_info: dict,
    problems: list[str],
    label: str,
) -> None:
    """Require the exact launch-resource contract used by each specialization."""
    if launch_info.get("active_clusters", 0) <= 0:
        problems.append(f"{label}: active register-cluster count is missing")
    expected_blocks = 4 if (shape[0], shape[3]) == (9, 8192) else 2
    if launch_info.get("cluster_blocks") != expected_blocks:
        problems.append(f"{label}: expected a {expected_blocks}-block register cluster")
    local_width = shape[3] // expected_blocks
    expected_dynamic = 4 * (21 * shape[0] + local_width) + 2 * local_width
    if launch_info.get("dynamic_shared_bytes") != expected_dynamic:
        problems.append(f"{label}: register-cluster dynamic memory is wrong")
    if launch_info.get("static_shared_bytes") != 1024:
        problems.append(f"{label}: register-cluster static memory is wrong")
    if launch_info.get("registers_per_thread") != 128:
        problems.append(f"{label}: register-cluster register count is wrong")
    if launch_info.get("threads_per_block") != 512:
        problems.append(f"{label}: register cluster must use 512 threads")
    if launch_info.get("multiprocessors", 0) <= 0:
        problems.append(f"{label}: GPU multiprocessor count is missing")
    if (
        launch_info.get("dynamic_shared_bytes", 0)
        + launch_info.get("static_shared_bytes", 0)
        > launch_info.get("max_shared_bytes", 0)
    ):
        problems.append(f"{label}: register-cluster shared memory exceeds the device limit")


def _check_register_cluster_forward_launch_info(
    shape: tuple[int, int, int, int],
    dtype: str,
    launch_info: dict,
    problems: list[str],
    label: str,
) -> None:
    """Require the exact resource contract of the complete one-read forward."""
    if launch_info.get("active_clusters", 0) <= 0:
        problems.append(f"{label}: active forward register-cluster count is missing")
    n, _, _, d = shape
    expected_blocks = 4 if (n, d) == (9, 8192) else 2
    if launch_info.get("cluster_blocks") != expected_blocks:
        problems.append(
            f"{label}: expected a {expected_blocks}-block forward register cluster"
        )
    register_sources = 0 if n == 9 else 28
    local_width = d // expected_blocks
    expected_dynamic = 4 * n * (2 * 16 + 1) + 2 * (n - register_sources) * local_width
    if launch_info.get("dynamic_shared_bytes") != expected_dynamic:
        problems.append(f"{label}: forward register-cluster dynamic memory is wrong")
    if launch_info.get("static_shared_bytes") != 1024:
        problems.append(f"{label}: forward register-cluster static memory is wrong")
    expected_registers = {
        (9, 8192, "bfloat16"): 96,
        (9, 8192, "float16"): 96,
        (32, 2048, "bfloat16"): 103,
        (32, 2048, "float16"): 109,
    }[(n, d, dtype)]
    if launch_info.get("registers_per_thread") != expected_registers:
        problems.append(f"{label}: forward register-cluster register count is wrong")
    if launch_info.get("threads_per_block") != 512:
        problems.append(f"{label}: forward register cluster must use 512 threads")
    if launch_info.get("multiprocessors", 0) <= 0:
        problems.append(f"{label}: forward GPU multiprocessor count is missing")
    if (
        launch_info.get("dynamic_shared_bytes", 0)
        + launch_info.get("static_shared_bytes", 0)
        > launch_info.get("max_shared_bytes", 0)
    ):
        problems.append(
            f"{label}: forward register-cluster shared memory exceeds the device limit"
        )


def _canonical_model(model) -> dict:
    return json.loads(json.dumps(model.as_dict()))


def _check_memory_contract(
    implementation: str,
    dtype: str,
    shape: tuple[int, int, int, int],
    record: dict,
    problems: list[str],
    label: str,
) -> int | None:
    """Validate the normalized peak-memory record and return workspace bytes."""
    memory = record.get("fwd_bwd_memory")
    fields = {
        "peak_allocated_bytes",
        "workspace_bytes",
        "resident_bytes",
        "incremental_peak_bytes",
        "returned_bytes",
        "accounted_output_bytes",
        "allocation_count",
    }
    if not isinstance(memory, dict) or set(memory) != fields or any(
        not isinstance(memory[field], int) or memory[field] < 0 for field in fields
    ):
        problems.append(f"{label}: complete nonnegative memory record is required")
        return None

    itemsize = 4 if dtype == "float32" else 2
    n, b, t, d = shape
    values_bytes = n * b * t * d * itemsize
    token_bytes = b * t * d * itemsize
    query_bytes = d * itemsize
    expected_accounted = values_bytes + token_bytes + query_bytes
    extra_resident = query_bytes if implementation == "liger_upstream" else 0
    expected_resident = 2 * expected_accounted + extra_resident
    if memory["accounted_output_bytes"] != expected_accounted:
        problems.append(f"{label}: accounted output bytes are wrong")
    if memory["resident_bytes"] != expected_resident:
        problems.append(f"{label}: resident bytes omit an input or mandatory output")
    if memory["returned_bytes"] != 0:
        problems.append(f"{label}: training callable must not retain a returned tensor")
    if memory["peak_allocated_bytes"] != expected_resident + memory["workspace_bytes"]:
        problems.append(f"{label}: normalized peak memory is inconsistent")
    rebuilt_workspace = max(
        0,
        memory["incremental_peak_bytes"] - memory["accounted_output_bytes"],
    )
    if memory["workspace_bytes"] != rebuilt_workspace:
        problems.append(f"{label}: workspace does not reconstruct from allocator evidence")
    return memory["workspace_bytes"]


def _check_traffic_contract(
    candidate: str,
    dtype: str,
    shape: tuple[int, int, int, int],
    record: dict,
    problems: list[str],
    label: str,
) -> None:
    """Reconstruct both traffic models instead of trusting stored summaries."""
    plan = get_training_plan(candidate)
    itemsize = 4 if dtype == "float32" else 2
    family = plan.backward.family
    options = {
        "source_tokens_per_cta": plan.backward.tokens_per_cta,
        "source_uses_partials": plan.backward.dw_reduction == "partials",
    }
    if family in {"cuda_cluster", "cuda_cluster4"}:
        active_workers = record.get("cluster_launch_info", {}).get("active_clusters")
    elif family == "cuda_register":
        active_workers = record.get("register_launch_info", {}).get("active_blocks")
    elif family == "cuda_register_cluster":
        active_workers = record.get("register_cluster_launch_info", {}).get(
            "active_clusters"
        )
    else:
        active_workers = None
    if isinstance(active_workers, int) and active_workers > 0:
        options["persistent_clusters"] = min(shape[1] * shape[2], active_workers)
    model_name = {
        "source_serial": (
            "source_serial_saved" if plan.saves_forward_stats else "source_serial"
        ),
        "cuda_shared": "cuda_shared",
        "cuda_cluster": "cuda_cluster",
        "cuda_cluster4": "cuda_cluster4",
        "cuda_register": "cuda_register",
        "cuda_register_cluster": "cuda_register_cluster",
    }[family]
    expected_backward = _canonical_model(
        backward_traffic_estimate(model_name, *shape, itemsize=itemsize, **options)
    )
    if record.get("traffic_model") != expected_backward:
        problems.append(f"{label}: backward traffic model does not reconstruct exactly")

    n, _, _, d = shape
    resident = (1 << (n - 1).bit_length()) * (1 << (d - 1).bit_length()) <= 32768
    forward_name = (
        "cuda_register_cluster"
        if plan.forward.family == "cuda_register_cluster"
        else "resident" if resident else "tiled"
    )
    expected_forward = _canonical_model(
        forward_traffic_estimate(
            forward_name,
            *shape,
            itemsize=itemsize,
            saves_backward_coefficients=plan.saves_forward_stats,
        )
    )
    if record.get("forward_traffic_model") != expected_forward:
        problems.append(f"{label}: forward traffic model does not reconstruct exactly")


def evaluate_reports(
    reports: list[dict],
    *,
    candidate: str = "cuda_cluster",
    threshold: float = 0.07,
    max_cv: float = 0.05,
    expected_commit: str | None = None,
    expected_branch: str | None = None,
    expected_tree: str | None = None,
) -> dict:
    """Return a deterministic promotion decision for one immutable plan."""
    if candidate not in CAMPAIGN_CANDIDATES:
        raise ValueError(f"{candidate!r} is not a promotion candidate")
    plan = get_training_plan(candidate)
    required_dtypes = _required_dtypes(candidate)
    problems: list[str] = []
    correctness_failures: list[str] = []
    numerical_regressions: list[str] = []
    unstable: list[str] = []
    confidence_failures: list[str] = []
    training_confidence_failures: list[str] = []
    performance_regressions: list[str] = []
    comparisons: list[Comparison] = []
    dispatch_cells: list[dict] = []
    anchor_speedups: list[float] = []
    anchor_current_training_speedups: list[float] = []
    anchor_liger_training_speedups: list[float] = []
    device_ids: set[str] = set()

    dtype_values = [report.get("dtype") for report in reports]
    if len(dtype_values) != len(set(dtype_values)):
        problems.append("each dtype must have exactly one report")
    by_dtype = {report.get("dtype"): report for report in reports if report.get("dtype") in ALL_DTYPES}
    missing_dtypes = required_dtypes - set(by_dtype)
    if missing_dtypes:
        problems.append(f"missing dtype runs: {sorted(missing_dtypes)}")

    commit_values = [report.get("provenance", {}).get("repository_commit") for report in reports]
    tree_values = [report.get("provenance", {}).get("repository_tree") for report in reports]
    commits = {value for value in commit_values if isinstance(value, str) and value}
    trees = {value for value in tree_values if isinstance(value, str) and value}
    if len(commits) != 1 or len(commit_values) != len(reports) or any(
        not isinstance(value, str) or not value for value in commit_values
    ):
        problems.append("every dtype run must record the same nonempty repository commit")
    if len(trees) != 1 or len(tree_values) != len(reports) or any(
        not isinstance(value, str) or not value for value in tree_values
    ):
        problems.append("every dtype run must record the same nonempty repository tree")
    if expected_commit is not None and commits != {expected_commit}:
        problems.append(f"reports do not match expected commit {expected_commit}")
    if expected_tree is not None and trees != {expected_tree}:
        problems.append(f"reports do not match expected tree {expected_tree}")

    for dtype in sorted(required_dtypes & set(by_dtype)):
        report = by_dtype[dtype]
        prefix = dtype
        if (
            report.get("schema_version") != 4
            or not isinstance(report.get("run_id"), str)
            or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", report["run_id"]) is None
        ):
            problems.append(f"{prefix}: schema version 4 or run ID is missing")
        if report.get("run_status") != "complete":
            problems.append(f"{prefix}: report is not a completed run")
        if report.get("shape_set") != "full":
            problems.append(f"{prefix}: production decision requires --shape-set full")
        provenance = report.get("provenance", {})
        if expected_branch is not None and provenance.get("repository_branch") != expected_branch:
            problems.append(f"{prefix}: report does not use expected branch {expected_branch}")
        if provenance.get("worktree_dirty") is not False:
            problems.append(f"{prefix}: benchmark worktree was not recorded as fully clean")
        if provenance.get("tracked_worktree_dirty"):
            problems.append(f"{prefix}: benchmark worktree was dirty")
        if provenance.get("third_party_commits", {}).get("Liger-Kernel") != PINNED_LIGER_COMMIT:
            problems.append(f"{prefix}: pinned Liger commit is missing or wrong")
        if provenance.get("third_party_dirty", {}).get("Liger-Kernel") is not False:
            problems.append(f"{prefix}: pinned Liger worktree cleanliness is missing or false")
        comparators = report.get("comparators", {})
        liger_provenance = comparators.get("liger_upstream", {})
        if (
            liger_provenance.get("commit") != PINNED_LIGER_COMMIT
            or liger_provenance.get("worktree_dirty") is not False
            or liger_provenance.get("under_pinned_checkout") is not True
            or liger_provenance.get("source_sha256") != PINNED_LIGER_SOURCE_SHA256
            or not liger_provenance.get("source_path")
        ):
            problems.append(f"{prefix}: imported Liger source provenance is incomplete")
        liger_exact_provenance = comparators.get("liger_exact", {})
        if (
            liger_exact_provenance.get("source_path")
            != "src/switchyard/_liger_exact.py"
            or liger_exact_provenance.get("derived_from")
            != f"Liger-Kernel@{PINNED_LIGER_COMMIT}"
            or liger_exact_provenance.get("license") != "BSD-2-Clause"
            or liger_exact_provenance.get("under_pinned_checkout") is not False
            or liger_exact_provenance.get("source_sha256") != LIGER_EXACT_SOURCE_SHA256
        ):
            problems.append(f"{prefix}: exact-contract Liger provenance is incomplete")
        if "--quick" in provenance.get("argv", []):
            problems.append(f"{prefix}: quick runs cannot produce a production decision")
        preflight = report.get("gpu_preflight", {})
        if (
            preflight.get("foreign_compute_process_count_at_start") != 0
            or preflight.get("busy_override")
        ):
            problems.append(f"{prefix}: benchmark did not start with exclusive access")
        if not preflight.get("device_id"):
            problems.append(f"{prefix}: campaign device ID was not recorded")
        else:
            device_ids.add(preflight["device_id"])
        postflight = report.get("gpu_postflight", {})
        if postflight.get("device_id") != preflight.get("device_id"):
            problems.append(f"{prefix}: GPU postflight identity is missing or changed")
        if postflight.get("foreign_compute_process_count_at_end") != 0:
            problems.append(f"{prefix}: another compute process appeared during the run")
        monitor = report.get("gpu_process_monitor", {})
        if monitor.get("device_id") != preflight.get("device_id"):
            problems.append(f"{prefix}: sampled GPU process monitor is missing")
        if monitor.get("collision_detected") is not False or monitor.get("collision_events"):
            problems.append(f"{prefix}: sampled monitor observed a competing process")
        samples = monitor.get("samples")
        interval = monitor.get("interval_seconds")
        duration = monitor.get("duration_seconds")
        max_probe_gap = monitor.get("max_probe_gap_seconds")
        monitor_coverage_ok = (
            isinstance(samples, int)
            and samples >= 2
            and isinstance(interval, int | float)
            and 0 < interval <= 1
            and isinstance(duration, int | float)
            and duration >= interval
            and samples >= max(2, int(duration / (2 * interval)))
            and monitor.get("probe_attempts") == samples
            and isinstance(max_probe_gap, int | float)
            and math.isfinite(max_probe_gap)
            and 0 <= max_probe_gap <= 0.25
            and monitor.get("maximum_allowed_probe_gap_seconds") == 0.25
        )
        if monitor.get("probe_errors") or not monitor_coverage_ok:
            problems.append(f"{prefix}: sampled GPU monitor was incomplete")
        if report.get("candidate_reachable_from_production") is not False:
            problems.append(f"{prefix}: candidate isolation flag is missing or true")
        if report.get("correctness_seeds") != [0, 1, 2]:
            problems.append(f"{prefix}: correctness seeds must be [0, 1, 2]")
        if not {"current", candidate, "liger_exact", "liger_upstream"} <= set(
            report.get("selected_implementations", [])
        ):
            problems.append(f"{prefix}: selected implementation set omits a comparator")

        tails = report.get("correctness_only", [])
        observed_tail_shapes = {_shape(case) for case in tails}
        if observed_tail_shapes != MASKED_TAIL_SHAPES:
            problems.append(f"{prefix}: exact masked-tail correctness cases are required")
        for case in tails:
            rows = {
                (item.get("impl"), item.get("seed")): item
                for item in case.get("implementations", [])
            }
            expected = {
                (implementation, seed)
                for implementation in (
                    "current",
                    candidate,
                    "liger_exact",
                    "liger_upstream",
                )
                for seed in (0, 1, 2)
            }
            observed_rows = [
                (item.get("impl"), item.get("seed"))
                for item in case.get("implementations", [])
                if item.get("impl")
                in {"current", candidate, "liger_exact", "liger_upstream"}
            ]
            if len(observed_rows) != len(expected) or set(observed_rows) != expected:
                problems.append(f"{prefix}: masked-tail implementation/seed matrix is incomplete")
            for item in case.get("implementations", []):
                implementation = item.get("impl")
                if implementation not in {
                    "current",
                    candidate,
                    "liger_exact",
                    "liger_upstream",
                }:
                    continue
                skipped = str(item.get("skipped", ""))
                supported, _ = plan_supports(plan, *(_shape(case)), dtype)
                if implementation == candidate and not supported:
                    if not skipped.startswith("unsupported plan:"):
                        problems.append(
                            f"{prefix} masked-tail {candidate} did not record its unsupported plan"
                        )
                    continue
                if skipped or not all(
                    item.get("correctness", {}).get(name, {}).get("ok", False)
                    for name in ("output", "dv", "dw")
                ):
                    failure = f"{prefix} masked-tail {implementation} seed={item.get('seed')}"
                    if implementation == candidate:
                        correctness_failures.append(failure)
                    else:
                        problems.append(f"{failure}: comparator correctness is incomplete")
            for seed in (0, 1, 2):
                accepted = rows.get(("current", seed), {}).get("correctness", {})
                exact = rows.get(("liger_exact", seed), {}).get("correctness", {})
                candidate_correctness = rows.get((candidate, seed), {}).get(
                    "correctness", {}
                )
                candidate_supported, _ = plan_supports(
                    plan, *(_shape(case)), dtype
                )
                for value in ("output", "dv", "dw"):
                    accepted_error = accepted.get(value, {}).get("rel_l2")
                    exact_error = exact.get(value, {}).get("rel_l2")
                    if not isinstance(accepted_error, int | float) or not isinstance(
                        exact_error, int | float
                    ):
                        problems.append(
                            f"{prefix} masked-tail liger_exact seed={seed} {value}: "
                            "numerical floor evidence is missing"
                        )
                    elif exact_error > max(1.05 * accepted_error, 1e-7):
                        problems.append(
                            f"{prefix} masked-tail liger_exact seed={seed} {value}: "
                            f"error {exact_error:.3e} exceeds {accepted_error:.3e}"
                        )
                    if not candidate_supported:
                        continue
                    candidate_error = candidate_correctness.get(value, {}).get(
                        "rel_l2"
                    )
                    if not isinstance(accepted_error, int | float) or not isinstance(
                        candidate_error, int | float
                    ):
                        problems.append(
                            f"{prefix} masked-tail {candidate} seed={seed} {value}: "
                            "numerical floor evidence is missing"
                        )
                    elif candidate_error > max(1.05 * accepted_error, 1e-7):
                        numerical_regressions.append(
                            f"{prefix} masked-tail {candidate} seed={seed} {value}: "
                            f"error {candidate_error:.3e} exceeds {accepted_error:.3e}"
                        )

        execution_order = report.get("execution_order", [])
        schedule_shapes = [_shape(item) for item in execution_order]
        if len(schedule_shapes) != len(set(schedule_shapes)) or set(schedule_shapes) != EXPECTED_FULL_SHAPES:
            problems.append(f"{prefix}: execution schedule shape matrix is not exact")
        schedules = {_shape(item): item for item in execution_order}

        table: dict[tuple[int, int, int, int], dict[str, dict]] = {}
        for record in report.get("results", []):
            shape = _shape(record)
            implementation = record.get("impl", "")
            if implementation in table.setdefault(shape, {}):
                problems.append(f"{prefix} {shape}: duplicate {implementation} record")
            table[shape][implementation] = record
        missing_shapes = EXPECTED_FULL_SHAPES - set(table)
        if missing_shapes:
            problems.append(f"{prefix}: missing {len(missing_shapes)} full-sweep shapes")

        for shape in sorted(EXPECTED_FULL_SHAPES & set(table)):
            records = table[shape]
            label = f"{prefix} {shape}"
            if not {"current", candidate, "liger_exact", "liger_upstream"} <= set(records):
                problems.append(
                    f"{label}: current/{candidate}/liger_exact/liger_upstream set is incomplete"
                )
                continue
            current = records["current"]
            measured = records[candidate]
            liger_exact = records["liger_exact"]
            liger_upstream = records["liger_upstream"]
            supported, reason = plan_supports(plan, *shape, dtype)
            if not supported:
                if not str(measured.get("skipped", "")).startswith("unsupported plan:"):
                    problems.append(f"{label}: unsupported plan was not recorded as skipped ({reason})")
                continue
            if measured.get("skipped"):
                correctness_failures.append(f"{label} {candidate}: {measured['skipped']}")
                continue
            if measured.get("training_plan") != plan.as_dict():
                problems.append(f"{label}: serialized training plan does not match {candidate}")

            shape_schedule = schedules.get(shape, {}).get("metrics", {})
            for metric in ("forward", "backward", "fwd_bwd"):
                trial_schedule = shape_schedule.get(metric, [])
                if len(trial_schedule) != FULL_TRIAL_COUNT:
                    problems.append(
                        f"{label}: {FULL_TRIAL_COUNT} interleaved {metric} "
                        "schedules are required"
                    )
                    continue
                required = {"current", candidate, "liger_exact", "liger_upstream"}
                if any(
                    not required <= set(trial.get("implementations", []))
                    for trial in trial_schedule
                ):
                    problems.append(f"{label}: {metric} schedule is not paired")
                candidate_positions = {
                    trial.get("implementations", []).index(candidate)
                    for trial in trial_schedule
                    if candidate in trial.get("implementations", [])
                }
                if len(candidate_positions) < 2:
                    problems.append(f"{label}: {metric} order was not rotated")
                for comparator in ("current", "liger_exact", "liger_upstream"):
                    candidate_first = sum(
                        trial["implementations"].index(candidate)
                        < trial["implementations"].index(comparator)
                        for trial in trial_schedule
                        if candidate in trial.get("implementations", [])
                        and comparator in trial.get("implementations", [])
                    )
                    balanced_counts = {
                        FULL_TRIAL_COUNT // 2,
                        (FULL_TRIAL_COUNT + 1) // 2,
                    }
                    if candidate_first not in balanced_counts:
                        problems.append(
                            f"{label}: {metric} order does not balance "
                            f"{candidate} against {comparator}"
                        )

            raw_timings: dict[tuple[str, str], dict] = {}
            workspaces: dict[str, int | None] = {}
            for implementation, record in (
                ("current", current),
                (candidate, measured),
                ("liger_exact", liger_exact),
                ("liger_upstream", liger_upstream),
            ):
                if record.get("skipped") or not _correct(record):
                    failure = f"{label} {implementation}"
                    if implementation == candidate:
                        correctness_failures.append(failure)
                    else:
                        problems.append(f"{failure}: comparator correctness is incomplete")
                expected_work_contract = {
                    "liger_exact": LIGER_EXACT_WORK_CONTRACT,
                    "liger_upstream": LIGER_UPSTREAM_WORK_CONTRACT,
                }.get(implementation)
                if (
                    expected_work_contract is not None
                    and record.get("work_contract") != expected_work_contract
                ):
                    problems.append(f"{label} {implementation}: work contract is wrong")
                workspaces[implementation] = _check_memory_contract(
                    implementation,
                    dtype,
                    shape,
                    record,
                    problems,
                    f"{label} {implementation}",
                )
                seed_ids = [
                    item.get("seed") for item in record.get("correctness_by_seed", [])
                ]
                if seed_ids != [0, 1, 2]:
                    problems.append(
                        f"{label} {implementation}: exact ordered correctness seeds are missing"
                    )
                for metric in ("forward", "backward", "fwd_bwd"):
                    evidence = _timing_from_raw(
                        record,
                        metric,
                        shape_schedule.get(metric, []),
                        problems,
                        label,
                    )
                    if evidence is not None:
                        raw_timings[(implementation, metric)] = evidence
                        if evidence["cv"] > max_cv:
                            unstable.append(
                                f"{label} {implementation} {metric}: cv={evidence['cv']}"
                            )

            current_seeds = {
                item["seed"]: item["report"] for item in current.get("correctness_by_seed", [])
            }
            candidate_seeds = {
                item["seed"]: item["report"] for item in measured.get("correctness_by_seed", [])
            }
            liger_exact_seeds = {
                item["seed"]: item["report"]
                for item in liger_exact.get("correctness_by_seed", [])
            }
            if (
                set(current_seeds) != {0, 1, 2}
                or set(candidate_seeds) != {0, 1, 2}
                or set(liger_exact_seeds) != {0, 1, 2}
            ):
                problems.append(f"{label}: exact three-seed correctness set is missing")
            for seed in sorted(set(current_seeds) & set(candidate_seeds)):
                for value in ("output", "dv", "dw"):
                    accepted_error = current_seeds[seed][value].get("rel_l2")
                    candidate_error = candidate_seeds[seed][value].get("rel_l2")
                    if (
                        isinstance(accepted_error, int | float)
                        and isinstance(candidate_error, int | float)
                        and candidate_error > max(1.05 * accepted_error, 1e-7)
                    ):
                        numerical_regressions.append(
                            f"{label} seed={seed} {value}: "
                            f"{candidate_error:.3e} vs {accepted_error:.3e}"
                        )
                    exact_error = liger_exact_seeds.get(seed, {}).get(value, {}).get("rel_l2")
                    if (
                        not isinstance(exact_error, int | float)
                        or not isinstance(accepted_error, int | float)
                        or exact_error > max(1.05 * accepted_error, 1e-7)
                    ):
                        problems.append(
                            f"{label} liger_exact seed={seed} {value}: numerical error "
                            "does not match the accepted error floor"
                        )

            required_timing_keys = {
                (implementation, metric)
                for implementation in (
                    "current",
                    candidate,
                    "liger_exact",
                    "liger_upstream",
                )
                for metric in ("forward", "backward", "fwd_bwd")
            }
            if not required_timing_keys <= set(raw_timings):
                problems.append(f"{label}: validated raw timing matrix is incomplete")
                continue
            current_ms = raw_timings[("current", "backward")]["median_ms"]
            candidate_ms = raw_timings[(candidate, "backward")]["median_ms"]
            liger_exact_backward = raw_timings[("liger_exact", "backward")]["median_ms"]
            current_fwd_bwd = raw_timings[("current", "fwd_bwd")]["median_ms"]
            candidate_fwd_bwd = raw_timings[(candidate, "fwd_bwd")]["median_ms"]
            liger_exact_fwd_bwd = raw_timings[("liger_exact", "fwd_bwd")]["median_ms"]
            liger_upstream_fwd_bwd = raw_timings[("liger_upstream", "fwd_bwd")][
                "median_ms"
            ]
            backward_ratios = _paired_speedups(
                raw_timings[("current", "backward")]["trial_medians"],
                raw_timings[(candidate, "backward")]["trial_medians"],
            )
            liger_exact_backward_ratios = _paired_speedups(
                raw_timings[("liger_exact", "backward")]["trial_medians"],
                raw_timings[(candidate, "backward")]["trial_medians"],
            )
            if backward_ratios is None or liger_exact_backward_ratios is None:
                problems.append(f"{label}: paired backward trial matrix is incomplete")
                continue
            current_training_ratios = _paired_speedups(
                raw_timings[("current", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            liger_exact_training_ratios = _paired_speedups(
                raw_timings[("liger_exact", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            liger_upstream_training_ratios = _paired_speedups(
                raw_timings[("liger_upstream", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            if (
                current_training_ratios is None
                or liger_exact_training_ratios is None
                or liger_upstream_training_ratios is None
            ):
                problems.append(f"{label}: paired training trial matrix is incomplete")
                continue
            speedup = statistics.median(backward_ratios)
            liger_exact_backward_speedup = statistics.median(
                liger_exact_backward_ratios
            )
            change = 1.0 - 1.0 / speedup
            classification = "WIN" if change >= threshold else "LOSS" if change <= -threshold else "NOISE"
            comparisons.append(
                Comparison(
                    dtype,
                    shape,
                    current_ms,
                    candidate_ms,
                    speedup,
                    change,
                    classification,
                    current_fwd_bwd,
                    candidate_fwd_bwd,
                    statistics.median(current_training_ratios),
                    liger_exact_backward,
                    liger_exact_backward_speedup,
                    liger_exact_fwd_bwd,
                    statistics.median(liger_exact_training_ratios),
                    liger_upstream_fwd_bwd,
                    statistics.median(liger_upstream_training_ratios),
                )
            )

            _check_kernel_contract(candidate, dtype, measured, problems, label)
            _check_current_kernel_contract(shape, dtype, current, problems, label)
            _check_traffic_contract(candidate, dtype, shape, measured, problems, label)
            current_forward = raw_timings[("current", "forward")]["median_ms"]
            candidate_forward = raw_timings[(candidate, "forward")]["median_ms"]
            if candidate_forward > 1.15 * current_forward:
                performance_regressions.append(
                    f"{label}: training forward regressed by "
                    f"{candidate_forward / current_forward:.2f}x"
                )
            if candidate_fwd_bwd > (1.0 + MAX_TRAINING_REGRESSION) * current_fwd_bwd:
                performance_regressions.append(
                    f"{label}: complete training regressed by "
                    f"{candidate_fwd_bwd / current_fwd_bwd:.2f}x"
                )
            current_workspace = workspaces["current"]
            candidate_workspace = workspaces[candidate]
            traffic = measured.get("traffic_model", {})
            modeled_extra = traffic.get("saved_state_bytes", 0) + traffic.get(
                "workspace_bytes", 0
            )
            if (
                isinstance(current_workspace, int)
                and isinstance(candidate_workspace, int)
                and candidate_workspace > current_workspace + modeled_extra + 2 * 2**20
            ):
                performance_regressions.append(
                    f"{label}: workspace exceeded the model and 2 MiB allocator allowance"
                )
            if plan.backward.family in {"cuda_cluster", "cuda_cluster4"}:
                launch_info = measured.get("cluster_launch_info", {})
                if launch_info.get("active_clusters", 0) <= 0:
                    problems.append(f"{label}: active cluster count is missing")
                expected_blocks = 2 if plan.backward.family == "cuda_cluster" else 4
                if launch_info.get("cluster_blocks") != expected_blocks:
                    problems.append(
                        f"{label}: expected a {expected_blocks}-block cluster launch"
                    )
                if (
                    launch_info.get("dynamic_shared_bytes", 0)
                    + launch_info.get("static_shared_bytes", 0)
                    > launch_info.get("max_shared_bytes", 0)
                ):
                    problems.append(f"{label}: recorded shared memory exceeds the device limit")
            if plan.backward.family == "cuda_register":
                launch_info = measured.get("register_launch_info", {})
                if launch_info.get("active_blocks", 0) <= 0:
                    problems.append(f"{label}: active register-worker count is missing")
            if plan.backward.family == "cuda_register_cluster":
                _check_register_cluster_launch_info(
                    shape,
                    measured.get("register_cluster_launch_info", {}),
                    problems,
                    label,
                )
                if plan.forward.family == "cuda_register_cluster":
                    _check_register_cluster_forward_launch_info(
                        shape,
                        dtype,
                        measured.get("register_cluster_forward_launch_info", {}),
                        problems,
                        label,
                    )

            current_trial_ratios = _paired_speedups(
                raw_timings[("current", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            liger_exact_trial_ratios = _paired_speedups(
                raw_timings[("liger_exact", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            liger_upstream_trial_ratios = _paired_speedups(
                raw_timings[("liger_upstream", "fwd_bwd")]["trial_medians"],
                raw_timings[(candidate, "fwd_bwd")]["trial_medians"],
            )
            if (
                current_trial_ratios is None
                or liger_exact_trial_ratios is None
                or liger_upstream_trial_ratios is None
            ):
                problems.append(f"{label}: paired training trial matrix is incomplete")
                continue
            current_training_speedup = statistics.median(current_trial_ratios)
            liger_exact_training_speedup = statistics.median(
                liger_exact_trial_ratios
            )
            liger_upstream_training_speedup = statistics.median(
                liger_upstream_trial_ratios
            )
            current_minimum = min(current_trial_ratios)
            liger_exact_minimum = min(liger_exact_trial_ratios)
            liger_exact_backward_minimum = min(liger_exact_backward_ratios)
            cell_reasons = []
            if classification != "WIN" or speedup < 1.10:
                cell_reasons.append(
                    f"backward speedup {speedup:.3f} does not clear the 1.10x cell gate"
                )
            if candidate_forward > 1.15 * current_forward:
                cell_reasons.append(
                    f"forward latency is {candidate_forward / current_forward:.3f}x current"
                )
            if current_training_speedup < PRACTICAL_TRAINING_MARGIN:
                cell_reasons.append(
                    f"current/candidate training speedup is {current_training_speedup:.3f}"
                )
            if liger_exact_backward_speedup < PRACTICAL_TRAINING_MARGIN:
                cell_reasons.append(
                    "liger_exact/candidate backward speedup is "
                    f"{liger_exact_backward_speedup:.3f}"
                )
            if liger_exact_training_speedup < PRACTICAL_TRAINING_MARGIN:
                cell_reasons.append(
                    "liger_exact/candidate training speedup is "
                    f"{liger_exact_training_speedup:.3f}"
                )
            if current_minimum <= 1.0:
                cell_reasons.append(
                    "minimum paired current training speedup does not exceed 1.0"
                )
            if liger_exact_minimum <= 1.0:
                cell_reasons.append(
                    "minimum paired liger_exact training speedup does not exceed 1.0"
                )
            if liger_exact_backward_minimum <= 1.0:
                cell_reasons.append(
                    "minimum paired liger_exact backward speedup does not exceed 1.0"
                )
            if not all(ratio > 1.0 for ratio in current_trial_ratios):
                cell_reasons.append(
                    f"candidate did not beat current in all {FULL_TRIAL_COUNT} trials"
                )
            if not all(ratio > 1.0 for ratio in liger_exact_trial_ratios):
                cell_reasons.append(
                    f"candidate did not beat liger_exact in all {FULL_TRIAL_COUNT} training trials"
                )
            if not all(ratio > 1.0 for ratio in liger_exact_backward_ratios):
                cell_reasons.append(
                    f"candidate did not beat liger_exact in all {FULL_TRIAL_COUNT} backward trials"
                )
            if (
                isinstance(current_workspace, int)
                and isinstance(candidate_workspace, int)
                and candidate_workspace
                > current_workspace + modeled_extra + 2 * 2**20
            ):
                cell_reasons.append("workspace exceeds the modeled allowance")
            dispatch_cells.append(
                {
                    "dtype": dtype,
                    "shape": dict(
                        zip(("n", "b", "t", "d"), shape, strict=True)
                    ),
                    "status": (
                        "READY_FOR_DISPATCH_REVIEW"
                        if not cell_reasons
                        else "FALLBACK"
                    ),
                    "reasons": cell_reasons,
                    "backward_speedup_vs_current": speedup,
                    "backward_speedup_vs_liger_exact": liger_exact_backward_speedup,
                    "training_speedup_vs_current": current_training_speedup,
                    "training_speedup_vs_liger_exact": liger_exact_training_speedup,
                    "training_speedup_vs_liger_upstream_observational": (
                        liger_upstream_training_speedup
                    ),
                    "minimum_paired_current_training_speedup": current_minimum,
                    "minimum_paired_liger_exact_training_speedup": (
                        liger_exact_minimum
                    ),
                    "minimum_paired_liger_exact_backward_speedup": (
                        liger_exact_backward_minimum
                    ),
                    "all_current_trials_win": all(
                        ratio > 1.0 for ratio in current_trial_ratios
                    ),
                    "all_liger_exact_training_trials_win": all(
                        ratio > 1.0 for ratio in liger_exact_trial_ratios
                    ),
                    "all_liger_exact_backward_trials_win": all(
                        ratio > 1.0 for ratio in liger_exact_backward_ratios
                    ),
                }
            )

            if shape in ANCHOR_SHAPES:
                anchor_speedups.append(speedup)
                anchor_current_training_speedups.append(current_training_speedup)
                anchor_liger_training_speedups.append(liger_exact_training_speedup)
                if (
                    current_training_speedup < PRACTICAL_TRAINING_MARGIN
                    or current_minimum <= 1.0
                ):
                    training_confidence_failures.append(
                        f"{label}: current/candidate training speedup="
                        f"{current_training_speedup:.3f}, minimum paired trial="
                        f"{current_minimum:.3f}"
                    )
                if (
                    liger_exact_training_speedup < PRACTICAL_TRAINING_MARGIN
                    or liger_exact_minimum <= 1.0
                    or liger_exact_backward_speedup < PRACTICAL_TRAINING_MARGIN
                    or liger_exact_backward_minimum <= 1.0
                ):
                    confidence_failures.append(
                        f"{label}: liger_exact/candidate training speedup="
                        f"{liger_exact_training_speedup:.3f}, minimum paired training="
                        f"{liger_exact_minimum:.3f}, backward speedup="
                        f"{liger_exact_backward_speedup:.3f}, minimum paired backward="
                        f"{liger_exact_backward_minimum:.3f}"
                    )

    if len(device_ids) > 1:
        problems.append("all dtype runs must use the same campaign device ID")
    expected_anchor_count = sum(
        plan_supports(plan, *shape, dtype)[0]
        for dtype in required_dtypes
        for shape in ANCHOR_SHAPES
    )
    if expected_anchor_count == 0:
        problems.append("candidate supports no required anchor shape")
    if len(anchor_speedups) != expected_anchor_count:
        problems.append("candidate does not cover every supported dtype and anchor shape")

    if correctness_failures or numerical_regressions:
        status = "REJECT"
        rationale = "candidate failed correctness or exceeded 1.05 times accepted numerical error"
    elif problems or unstable:
        status = "MORE_DATA"
        rationale = "the evidence matrix or measurement-quality gate is incomplete"
    else:
        eligible = [
            cell
            for cell in dispatch_cells
            if cell["status"] == "READY_FOR_DISPATCH_REVIEW"
        ]
        if eligible:
            status = "READY_FOR_DISPATCH_REVIEW"
            rationale = (
                "candidate has repeatable cell-level wins; dispatch only the listed "
                "shape and dtype cases"
            )
        else:
            status = "DROP"
            rationale = (
                "candidate has no measured shape and dtype cell that clears every gate"
            )

    if status in {"REJECT", "MORE_DATA"}:
        for cell in dispatch_cells:
            if cell["status"] == "READY_FOR_DISPATCH_REVIEW":
                cell["status"] = "INVALID_REPORT"
                cell["reasons"].append(
                    "global correctness or evidence-quality gate did not pass"
                )
    eligible_dispatches = (
        [
            cell
            for cell in dispatch_cells
            if cell["status"] == "READY_FOR_DISPATCH_REVIEW"
        ]
        if status == "READY_FOR_DISPATCH_REVIEW"
        else []
    )

    return {
        "status": status,
        "rationale": rationale,
        "candidate": candidate,
        "required_dtypes": sorted(required_dtypes),
        "threshold": threshold,
        "max_cv": max_cv,
        "practical_training_margin": PRACTICAL_TRAINING_MARGIN,
        "max_training_regression": MAX_TRAINING_REGRESSION,
        "max_campaign_attempts": MAX_CAMPAIGN_ATTEMPTS,
        "statistical_interpretation": (
            "paired trial ratios are descriptive repeated measurements; no trial "
            "independence or family-wise probability is claimed"
        ),
        "commits": sorted(commits),
        "trees": sorted(trees),
        "problems": problems,
        "correctness_failures": correctness_failures,
        "numerical_regressions": numerical_regressions,
        "unstable": unstable,
        "confidence_failures": confidence_failures,
        "training_confidence_failures": training_confidence_failures,
        "performance_regressions": performance_regressions,
        "dispatch_cells": dispatch_cells,
        "eligible_dispatches": eligible_dispatches,
        "anchor_speedup_floor": min(anchor_speedups) if anchor_speedups else None,
        "anchor_speedup_geomean": (
            math.exp(sum(math.log(value) for value in anchor_speedups) / len(anchor_speedups))
            if anchor_speedups
            else None
        ),
        "anchor_current_training_speedup_floor": (
            min(anchor_current_training_speedups)
            if anchor_current_training_speedups
            else None
        ),
        "anchor_liger_exact_training_speedup_floor": (
            min(anchor_liger_training_speedups)
            if anchor_liger_training_speedups
            else None
        ),
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--candidate", default="cuda_cluster", choices=CAMPAIGN_CANDIDATES
    )
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report_bytes = [path.read_bytes() for path in args.results]
    reports = [json.loads(raw) for raw in report_bytes]
    decision = evaluate_reports(
        reports,
        candidate=args.candidate,
        threshold=args.threshold,
        max_cv=args.max_cv,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        expected_tree=args.expected_tree,
    )
    decision["input_reports"] = [
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for raw in report_bytes
    ]
    if args.json:
        print(json.dumps(decision, indent=2, default=str))
    else:
        for row in decision["comparisons"]:
            shape = "x".join(str(value) for value in row["shape"])
            print(
                f"{row['dtype']:8} {shape:20} {row['current_ms']:8.4f} -> "
                f"{row['candidate_ms']:8.4f} ms  {row['speedup']:5.2f}x  "
                f"{row['classification']}"
            )
        print(f"\n{decision['status']}: {decision['rationale']}")
        for problem in decision["problems"] + decision["unstable"]:
            print(f"  - {problem}")
        for failure in decision["correctness_failures"] + decision["numerical_regressions"]:
            print(f"  - correctness: {failure}")
        for failure in decision["confidence_failures"]:
            print(f"  - comparison: {failure}")
        for failure in decision["training_confidence_failures"]:
            print(f"  - training: {failure}")
        for failure in decision["performance_regressions"]:
            print(f"  - performance: {failure}")

    return {"READY_FOR_DISPATCH_REVIEW": 0, "DROP": 0, "REJECT": 1, "MORE_DATA": 2}[
        decision["status"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())
