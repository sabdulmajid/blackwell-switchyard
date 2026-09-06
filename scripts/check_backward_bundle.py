#!/usr/bin/env python3
"""Reconstruct and validate one publishable backward campaign bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from check_backward_report import validate_report  # noqa: E402
from check_backward_smoke import check_reports  # noqa: E402
from evaluate_backward import evaluate_reports  # noqa: E402

from switchyard.training_plan import get_training_plan  # noqa: E402

ALL_IMPLEMENTATIONS = [
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
    "liger_exact",
    "liger_upstream",
]
PORTABLE_IMPLEMENTATIONS = [
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "liger_exact",
    "liger_upstream",
]
CANDIDATES = [
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
]
EXPECTED_CUDA_INSTANCES = {
    "shared_backward_kernel": 2,
    "feature_cluster_backward_kernel_2block": 2,
    "feature_cluster_backward_kernel_4block": 2,
    "register_backward_kernel": 2,
    "register_cluster_backward_kernel_bfloat16_n9_d8192_c4": 1,
    "register_cluster_backward_kernel_bfloat16_n32_d2048_c2": 1,
    "register_cluster_backward_kernel_float16_n9_d8192_c4": 1,
    "register_cluster_backward_kernel_float16_n32_d2048_c2": 1,
    "register_cluster_forward_kernel_bfloat16_n9_d8192_c4": 1,
    "register_cluster_forward_kernel_bfloat16_n32_d2048_c2": 1,
    "register_cluster_forward_kernel_float16_n9_d8192_c4": 1,
    "register_cluster_forward_kernel_float16_n32_d2048_c2": 1,
}
EXPECTED_CUDA_LIMITS = {
    "shared_backward_kernel": 64,
    "feature_cluster_backward_kernel_2block": 64,
    "feature_cluster_backward_kernel_4block": 80,
    "register_backward_kernel": 128,
    **{
        name: 128
        for name in EXPECTED_CUDA_INSTANCES
        if name.startswith("register_cluster_backward_kernel_")
        or name.startswith("register_cluster_forward_kernel_")
    },
}
EXPECTED_CUDA_GLOBAL_LOADS = {
    "register_cluster_backward_kernel_bfloat16_n9_d8192_c4": 25,
    "register_cluster_backward_kernel_bfloat16_n32_d2048_c2": 37,
    "register_cluster_backward_kernel_float16_n9_d8192_c4": 25,
    "register_cluster_backward_kernel_float16_n32_d2048_c2": 37,
    "register_cluster_forward_kernel_bfloat16_n9_d8192_c4": 20,
    "register_cluster_forward_kernel_bfloat16_n32_d2048_c2": 33,
    "register_cluster_forward_kernel_float16_n9_d8192_c4": 20,
    "register_cluster_forward_kernel_float16_n32_d2048_c2": 33,
}
EXPECTED_CUDA_GLOBAL_LOAD_ROLES = {
    **{
        f"register_cluster_backward_kernel_{dtype}_n9_d8192_c4": {
            "grad_output_pairs": 2,
            "source_pairs": 18,
            "saved_alpha": 1,
            "saved_rstd": 1,
            "saved_norm": 1,
            "query_pairs": 2,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_backward_kernel_{dtype}_n32_d2048_c2": {
            "grad_output_pairs": 1,
            "source_pairs": 32,
            "saved_alpha": 1,
            "saved_rstd": 1,
            "saved_norm": 1,
            "query_pairs": 1,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_forward_kernel_{dtype}_n9_d8192_c4": {
            "query_pairs": 2,
            "source_pairs": 18,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_forward_kernel_{dtype}_n32_d2048_c2": {
            "query_pairs": 1,
            "source_pairs": 32,
        }
        for dtype in ("bfloat16", "float16")
    },
}
EXPECTED_COMPILE_PROVENANCE = {
    "torch": "2.9.0+cu128",
    "triton": "3.5.0",
    "nvcc": "Build cuda_12.8.r12.8/compiler.35583870_0",
}
EXPECTED_CUDA_RESOURCE_ROWS = (
    Counter(
        {
            ("shared_backward_kernel", 48, 1024, 4, None): 2,
            ("feature_cluster_backward_kernel_2block", 64, 1024, 12, None): 2,
            ("feature_cluster_backward_kernel_4block", 72, 1024, 12, None): 2,
            ("register_backward_kernel", 128, 1024, 74, None): 2,
        }
    )
    + Counter(
        {
            (
                name,
                128,
                1024,
                EXPECTED_CUDA_GLOBAL_LOADS[name],
                json.dumps(EXPECTED_CUDA_GLOBAL_LOAD_ROLES[name], sort_keys=True),
            ): 1
            for name in EXPECTED_CUDA_GLOBAL_LOADS
            if "backward" in name
        }
    )
    + Counter(
        {
            (
                "register_cluster_forward_kernel_bfloat16_n9_d8192_c4",
                96,
                1024,
                20,
                json.dumps(
                    EXPECTED_CUDA_GLOBAL_LOAD_ROLES[
                        "register_cluster_forward_kernel_bfloat16_n9_d8192_c4"
                    ],
                    sort_keys=True,
                ),
            ): 1,
            (
                "register_cluster_forward_kernel_bfloat16_n32_d2048_c2",
                103,
                1024,
                33,
                json.dumps(
                    EXPECTED_CUDA_GLOBAL_LOAD_ROLES[
                        "register_cluster_forward_kernel_bfloat16_n32_d2048_c2"
                    ],
                    sort_keys=True,
                ),
            ): 1,
            (
                "register_cluster_forward_kernel_float16_n9_d8192_c4",
                96,
                1024,
                20,
                json.dumps(
                    EXPECTED_CUDA_GLOBAL_LOAD_ROLES[
                        "register_cluster_forward_kernel_float16_n9_d8192_c4"
                    ],
                    sort_keys=True,
                ),
            ): 1,
            (
                "register_cluster_forward_kernel_float16_n32_d2048_c2",
                109,
                1024,
                33,
                json.dumps(
                    EXPECTED_CUDA_GLOBAL_LOAD_ROLES[
                        "register_cluster_forward_kernel_float16_n32_d2048_c2"
                    ],
                    sort_keys=True,
                ),
            ): 1,
        }
    )
)
MIN_IDLE_SECONDS = 1800
MIN_WAIT_POLL_SECONDS = 30
MAX_WAIT_POLL_SECONDS = 300
MAX_WATCHDOG_SECONDS = 0.25
MAX_IDLE_PCIE_KIB_PER_SECOND = 64 * 1024
MIN_FINALIZE_SECONDS = 1800
MAX_CAMPAIGN_ATTEMPTS = 3
PRIVATE_METADATA_PATTERN = re.compile(
    r"co-authored-by|claude|anthropic|wizchem|chatgpt|openai\.com|"
    r"session[-_/][A-Za-z0-9]|file://|"
    r"/(?:home|tmp|pub[0-9]+|mnt|scratch|workspace)/|"
    r"[A-Za-z]:[\\/](?:Users|home|tmp|workspace)[\\/]|"
    r"(?:[A-Za-z0-9-]+\.)+(?:internal|local)\b|GPU-[A-Za-z0-9-]+",
    re.IGNORECASE,
)


def _suffixes() -> list[str]:
    return [
        "offline_compile",
        "smoke_bfloat16",
        "smoke_float16",
        "smoke_decision",
        "full_bfloat16",
        "full_float16",
        "full_float32",
        *(f"decision_{candidate}" for candidate in CANDIDATES),
        "manifest",
    ]


def _canonical(value: dict) -> dict:
    return json.loads(json.dumps(value, default=str))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _inputs(raw_reports: list[bytes]) -> list[dict[str, str]]:
    return [{"sha256": hashlib.sha256(raw).hexdigest()} for raw in raw_reports]


def _unexpected_fields(value: object, allowed: set[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    unexpected = sorted(set(value) - allowed)
    return [f"{label} contains unexpected fields: {unexpected}"] if unexpected else []


def _exact_fields(value: object, expected: set[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    if set(value) != expected:
        return [f"{label} field set is not exact"]
    return []


def _compile_problems(
    report: object,
    expected_commit: str,
    expected_source_sha256: str | None = None,
    expected_contract: dict | None = None,
) -> list[str]:
    problems: list[str] = []
    top_fields = {"target", "gpu_visible", "provenance", "plans", "compilations"}
    problems.extend(_exact_fields(report, top_fields, "offline compile report"))
    if not isinstance(report, dict):
        return problems

    if report.get("target") != {"backend": "cuda", "arch": 120, "warp_size": 32}:
        problems.append("offline compile report does not have the exact sm_120 target")
    if report.get("gpu_visible") is not False:
        problems.append("offline compile report was not produced without a visible GPU")

    provenance = report.get("provenance")
    provenance_fields = {
        "repository_commit",
        "worktree_clean",
        "torch",
        "triton",
        "nvcc",
        "cuda_source_sha256",
    }
    problems.extend(_exact_fields(provenance, provenance_fields, "compile provenance"))
    if isinstance(provenance, dict):
        if provenance.get("repository_commit") != expected_commit:
            problems.append("offline compile report uses the wrong repository commit")
        if provenance.get("worktree_clean") is not True:
            problems.append("offline compile report was not produced from a clean worktree")
        for field, expected in EXPECTED_COMPILE_PROVENANCE.items():
            if provenance.get(field) != expected:
                problems.append(f"offline compile report uses the wrong {field} version")
        source_hash = provenance.get("cuda_source_sha256")
        if (
            not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
        ):
            problems.append("offline compile report has an invalid CUDA source hash")
        if expected_source_sha256 is not None and source_hash != expected_source_sha256:
            problems.append("offline compile report uses different CUDA source bytes")

    expected_plans = [get_training_plan(name).as_dict() for name in ("auto", *CANDIDATES)]
    if report.get("plans") != expected_plans:
        problems.append("offline compile report does not contain the exact campaign plans")

    compilations = report.get("compilations")
    if not isinstance(compilations, list) or not compilations or not all(
        isinstance(item, dict) for item in compilations
    ):
        problems.append("offline compile report has a malformed compilation list")
        return problems
    names = [item.get("name") for item in compilations]
    if (
        not all(isinstance(name, str) for name in names)
        or len(names) != len(set(names))
        or names[-1] != "one_read_cuda"
    ):
        problems.append("offline compile report has an invalid compilation inventory")

    resource_fields = {
        "kernel",
        "registers",
        "stack_bytes",
        "static_shared_bytes",
        "local_bytes",
        "global_load_instructions",
    }
    current_kernel_names = {
        "_bwd_apply",
        "_bwd_resident",
        "_bwd_stats",
        "_fwd_resident",
        "_fwd_tiled",
    }
    for index, compilation in enumerate(compilations[:-1]):
        label = f"offline Triton compilation[{index}]"
        fields = {
            "name",
            "kind",
            "input_dtype",
            "spill_policy",
            "constants",
            "num_warps",
            "num_stages",
            "shared_bytes",
            "resources",
        }
        problems.extend(_exact_fields(compilation, fields, label))
        if compilation.get("kind") != "triton":
            problems.append(f"{label} has the wrong kind")
        if compilation.get("input_dtype") not in {"bfloat16", "float16", "float32"}:
            problems.append(f"{label} has an invalid input dtype")
        policy = compilation.get("spill_policy")
        if policy not in {"forbid", "record"}:
            problems.append(f"{label} has an invalid spill policy")
        if not isinstance(compilation.get("constants"), dict):
            problems.append(f"{label} has malformed constants")
        for field in ("num_warps", "num_stages"):
            if (
                type(compilation.get(field)) is not int
                or compilation[field] <= 0
            ):
                problems.append(f"{label} has an invalid {field}")
        if (
            type(compilation.get("shared_bytes")) is not int
            or compilation["shared_bytes"] < 0
        ):
            problems.append(f"{label} has invalid shared memory")

        resources = compilation.get("resources")
        if (
            not isinstance(resources, list)
            or len(resources) != 1
            or not isinstance(resources[0], dict)
        ):
            problems.append(f"{label} must have one resource record")
            continue
        resource = resources[0]
        problems.extend(_exact_fields(resource, resource_fields, f"{label} resources[0]"))
        scalar_fields = resource_fields - {"kernel"}
        if (
            not isinstance(resource.get("kernel"), str)
            or any(type(resource.get(field)) is not int for field in scalar_fields)
            or any(
                resource[field] < 0
                for field in scalar_fields
                if type(resource.get(field)) is int
            )
        ):
            problems.append(f"{label} resource values are invalid")
            continue
        spills = bool(resource["stack_bytes"] or resource["local_bytes"])
        if policy == "forbid" and spills:
            problems.append(f"{label} violates its spill-free contract")
        if policy == "record" and resource["kernel"] not in current_kernel_names:
            problems.append(f"{label} relaxes spills for a non-baseline kernel")

    cuda = compilations[-1]
    cuda_fields = {
        "name",
        "kind",
        "required_template_instances",
        "register_limits",
        "required_global_load_instruction_counts",
        "required_global_load_role_counts",
        "resources",
    }
    problems.extend(_exact_fields(cuda, cuda_fields, "offline CUDA compilation"))
    if cuda.get("name") != "one_read_cuda" or cuda.get("kind") != "cuda":
        problems.append("offline compile report does not contain the exact CUDA build")
    if (
        cuda.get("required_template_instances") != EXPECTED_CUDA_INSTANCES
        or cuda.get("register_limits") != EXPECTED_CUDA_LIMITS
        or cuda.get("required_global_load_instruction_counts") != EXPECTED_CUDA_GLOBAL_LOADS
        or cuda.get("required_global_load_role_counts")
        != EXPECTED_CUDA_GLOBAL_LOAD_ROLES
    ):
        problems.append("offline CUDA resource contracts differ from the source gate")

    resources = cuda.get("resources")
    if not isinstance(resources, list) or not all(
        isinstance(item, dict) for item in resources
    ):
        problems.append("offline CUDA resource records are malformed")
        return problems
    observed_rows = Counter()
    for index, item in enumerate(resources):
        fields = set(resource_fields)
        roles = item.get("global_load_instruction_roles")
        if roles is not None:
            fields.add("global_load_instruction_roles")
        problems.extend(_exact_fields(item, fields, f"CUDA resources[{index}]"))
        scalar_fields = resource_fields - {"kernel"}
        if (
            not isinstance(item.get("kernel"), str)
            or any(type(item.get(field)) is not int for field in scalar_fields)
            or any(
                item[field] < 0
                for field in scalar_fields
                if type(item.get(field)) is int
            )
            or (roles is not None and not isinstance(roles, dict))
            or (
                isinstance(roles, dict)
                and any(
                    not isinstance(key, str) or type(count) is not int or count <= 0
                    for key, count in roles.items()
                )
            )
        ):
            problems.append(f"CUDA resources[{index}] has invalid value types")
            continue
        observed_rows[
            (
                item.get("kernel"),
                item.get("registers"),
                item.get("static_shared_bytes"),
                item.get("global_load_instructions"),
                json.dumps(roles, sort_keys=True) if roles is not None else None,
            )
        ] += 1
        if item.get("stack_bytes") != 0 or item.get("local_bytes") != 0:
            problems.append(f"offline CUDA resource record spills: {item.get('kernel')}")
    if observed_rows != EXPECTED_CUDA_RESOURCE_ROWS:
        problems.append("offline CUDA resource rows differ from the exact compiler contract")

    if expected_contract is not None:
        expected = _canonical(expected_contract)
        observed = _canonical(report)
        if not isinstance(expected.get("provenance"), dict):
            problems.append("tracked compiler contract has malformed provenance")
        else:
            expected["provenance"]["repository_commit"] = "<campaign-commit>"
            observed["provenance"]["repository_commit"] = "<campaign-commit>"
            if observed != expected:
                problems.append("offline compile report differs from the tracked compiler contract")
    return problems
def validate_bundle(
    prefix: Path,
    *,
    read_bytes: Callable[[Path], bytes],
    changed_paths: set[Path],
    expected_commit: str,
    expected_tree: str,
    expected_branch: str,
    result_branch: str,
    expected_origin: str,
    expected_device_id: str,
    expected_source_sha256: str | None = None,
    expected_compile_contract: dict | None = None,
) -> list[str]:
    problems: list[str] = []
    match = re.fullmatch(r"backward_campaign_([0-9]{8}T[0-9]{6}Z)", prefix.name)
    if match is None:
        return ["bundle prefix does not contain a valid campaign ID"]
    campaign_id = match.group(1)
    paths = {suffix: prefix.with_name(f"{prefix.name}_{suffix}.json") for suffix in _suffixes()}
    expected_paths = set(paths.values())
    if changed_paths != expected_paths:
        problems.append("result commit or index does not contain the exact campaign bundle")

    raw: dict[str, bytes] = {}
    payloads: dict[str, dict] = {}
    for suffix, path in paths.items():
        try:
            raw[suffix] = read_bytes(path)
            payload = json.loads(raw[suffix], parse_constant=_reject_json_constant)
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            payloads[suffix] = payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"cannot read {path.name}: {type(exc).__name__}")
    if len(payloads) != len(paths):
        return problems
    for suffix, contents in raw.items():
        if PRIVATE_METADATA_PATTERN.search(contents.decode("utf-8")):
            problems.append(f"{suffix} contains private host or task metadata")

    problems.extend(
        _compile_problems(
            payloads["offline_compile"],
            expected_commit,
            expected_source_sha256,
            expected_compile_contract,
        )
    )

    phase_specs = {
        "smoke_bfloat16": ("bfloat16", "gate", ALL_IMPLEMENTATIONS, True),
        "smoke_float16": ("float16", "gate", ALL_IMPLEMENTATIONS, True),
        "full_bfloat16": ("bfloat16", "full", ALL_IMPLEMENTATIONS, False),
        "full_float16": ("float16", "full", ALL_IMPLEMENTATIONS, False),
        "full_float32": ("float32", "full", PORTABLE_IMPLEMENTATIONS, False),
    }
    for suffix, (dtype, shape_set, implementations, quick) in phase_specs.items():
        phase_problems = validate_report(
            payloads[suffix],
            dtype=dtype,
            shape_set=shape_set,
            implementations=implementations,
            quick=quick,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            expected_tree=expected_tree,
            expected_device_id=expected_device_id,
        )
        problems.extend(f"{suffix}: {problem}" for problem in phase_problems)
    if payloads["full_bfloat16"].get("run_id") != campaign_id:
        problems.append("bundle campaign ID differs from the bf16 full report")

    smoke_raw = [raw["smoke_bfloat16"], raw["smoke_float16"]]
    smoke = check_reports(
        [payloads["smoke_bfloat16"], payloads["smoke_float16"]],
        expected_commit=expected_commit,
        expected_branch=expected_branch,
        expected_tree=expected_tree,
    )
    smoke["input_reports"] = _inputs(smoke_raw)
    if smoke.get("status") != "PASS" or _canonical(smoke) != payloads["smoke_decision"]:
        problems.append("stored smoke decision does not reconstruct exactly")

    for candidate in CANDIDATES:
        report_suffixes = ["full_bfloat16", "full_float16"]
        if candidate.startswith("serial_"):
            report_suffixes.append("full_float32")
        decision = evaluate_reports(
            [payloads[suffix] for suffix in report_suffixes],
            candidate=candidate,
            expected_commit=expected_commit,
            expected_branch=expected_branch,
            expected_tree=expected_tree,
        )
        decision["input_reports"] = _inputs([raw[suffix] for suffix in report_suffixes])
        if decision.get("status") not in {"DROP", "REJECT", "READY_FOR_DISPATCH_REVIEW"}:
            problems.append(f"{candidate} does not have a terminal reconstructed decision")
        if _canonical(decision) != payloads[f"decision_{candidate}"]:
            problems.append(f"stored {candidate} decision does not reconstruct exactly")

    manifest = payloads["manifest"]
    exact_manifest = {
        "schema_version": 3,
        "repository_commit": expected_commit,
        "repository_tree": expected_tree,
        "benchmark_branch": expected_branch,
        "result_branch": result_branch,
        "origin": expected_origin,
        "device_id": expected_device_id,
        "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
    }
    for field, expected in exact_manifest.items():
        if manifest.get(field) != expected:
            problems.append(f"manifest {field} differs from the campaign contract")
    if (
        isinstance(manifest.get("attempt"), bool)
        or not isinstance(manifest.get("attempt"), int)
        or not 1 <= manifest["attempt"] <= MAX_CAMPAIGN_ATTEMPTS
    ):
        problems.append(f"manifest attempt must be between 1 and {MAX_CAMPAIGN_ATTEMPTS}")
    guards = manifest.get("guard_attestations")
    expected_manifest_fields = set(exact_manifest) | {"attempt", "guard_attestations"}
    if set(manifest) != expected_manifest_fields:
        problems.append("manifest field set is not exact")
    guard_fields = {
        "attempt",
        "idle_activity_scope",
        "process_scope",
        "not_before",
        "idle_started_at",
        "launch_at",
        "idle_seconds",
        "idle_probe_count",
        "max_idle_gpu_utilization_percent",
        "max_idle_memory_mib",
        "max_idle_pcie_rx_kib_per_second",
        "max_idle_pcie_tx_kib_per_second",
        "max_idle_probe_gap_seconds",
        "max_global_compute_process_count",
        "max_non_target_gpu_utilization_percent",
        "max_non_target_memory_mib",
        "max_non_target_pcie_rx_kib_per_second",
        "max_non_target_pcie_tx_kib_per_second",
        "maximum_allowed_pcie_kib_per_second",
        "watchdog_probe_count",
        "max_watchdog_probe_gap_seconds",
        "maximum_allowed_watchdog_probe_gap_seconds",
        "wait_poll_seconds",
        "watchdog_seconds",
        "finalize_seconds",
        "gpu_count",
        "device_id",
    }
    by_attempt: dict[int, tuple[dict, dict[str, datetime]]] = {}
    if not isinstance(guards, list) or not guards:
        problems.append("manifest guard attestations must be a nonempty list")
        guards = []
    for index, guard in enumerate(guards):
        if not isinstance(guard, dict) or set(guard) != guard_fields:
            problems.append(f"manifest guard {index} field set is not exact")
            continue
        moments = {}
        try:
            for field in ("not_before", "idle_started_at", "launch_at"):
                moments[field] = datetime.fromisoformat(guard[field])
                if moments[field].tzinfo is None:
                    raise ValueError("missing UTC offset")
        except (TypeError, ValueError):
            problems.append(f"manifest guard {index} timestamps must include UTC offsets")
        integer_guard_fields = {
            "idle_seconds",
            "idle_probe_count",
            "wait_poll_seconds",
            "finalize_seconds",
            "gpu_count",
            "watchdog_probe_count",
        }
        numeric_guard_valid = True
        for field in integer_guard_fields:
            value = guard[field]
            if type(value) is not int or value <= 0:
                problems.append(f"manifest guard {index} {field} must be a positive integer")
                numeric_guard_valid = False
        watchdog = guard["watchdog_seconds"]
        if type(watchdog) is not float or not math.isfinite(watchdog) or watchdog <= 0:
            problems.append(
                f"manifest guard {index} watchdog_seconds must be a positive finite float"
            )
            numeric_guard_valid = False
        attempt = guard.get("attempt")
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or not 1 <= attempt <= MAX_CAMPAIGN_ATTEMPTS
            or attempt in by_attempt
        ):
            problems.append(f"manifest guard {index} has an invalid or duplicate attempt")
        elif len(moments) == 3:
            by_attempt[attempt] = (guard, moments)
        if guard["device_id"] != expected_device_id:
            problems.append(f"manifest guard {index} has the wrong device ID")
        if guard["idle_activity_scope"] not in {"all_gpus", "target_gpu"}:
            problems.append(f"manifest guard {index} idle_activity_scope is invalid")
        if guard["process_scope"] != "all_gpus":
            problems.append(f"manifest guard {index} process_scope must equal all_gpus")
        if type(guard["idle_seconds"]) is int and guard["idle_seconds"] < MIN_IDLE_SECONDS:
            problems.append(
                f"manifest guard {index} idle_seconds must be at least {MIN_IDLE_SECONDS}"
            )
        idle_utilization = guard["max_idle_gpu_utilization_percent"]
        if isinstance(idle_utilization, bool) or not isinstance(idle_utilization, int):
            problems.append(
                f"manifest guard {index} max_idle_gpu_utilization_percent must be an integer"
            )
        elif idle_utilization != 0:
            problems.append(f"manifest guard {index} max_idle_gpu_utilization_percent must be zero")
        idle_memory = guard["max_idle_memory_mib"]
        if isinstance(idle_memory, bool) or not isinstance(idle_memory, int) or idle_memory < 0:
            problems.append(
                f"manifest guard {index} max_idle_memory_mib must be a nonnegative integer"
            )
        elif idle_memory > 64:
            problems.append(f"manifest guard {index} max_idle_memory_mib exceeds 64 MiB")
        idle_probe_gap = guard["max_idle_probe_gap_seconds"]
        if (
            isinstance(idle_probe_gap, bool)
            or not isinstance(idle_probe_gap, int | float)
            or not math.isfinite(idle_probe_gap)
            or idle_probe_gap < 0
            or idle_probe_gap > guard["wait_poll_seconds"] + 5
        ):
            problems.append(f"manifest guard {index} max_idle_probe_gap_seconds is invalid")
        if guard["max_global_compute_process_count"] != 0:
            problems.append(
                f"manifest guard {index} max_global_compute_process_count must be zero"
            )
        non_target_utilization = guard["max_non_target_gpu_utilization_percent"]
        if (
            isinstance(non_target_utilization, bool)
            or not isinstance(non_target_utilization, int)
            or not 0 <= non_target_utilization <= 100
        ):
            problems.append(
                f"manifest guard {index} max_non_target_gpu_utilization_percent is invalid"
            )
        non_target_memory = guard["max_non_target_memory_mib"]
        if (
            isinstance(non_target_memory, bool)
            or not isinstance(non_target_memory, int)
            or non_target_memory < 0
        ):
            problems.append(f"manifest guard {index} max_non_target_memory_mib is invalid")
        for field in (
            "max_idle_pcie_rx_kib_per_second",
            "max_idle_pcie_tx_kib_per_second",
            "max_non_target_pcie_rx_kib_per_second",
            "max_non_target_pcie_tx_kib_per_second",
        ):
            value = guard[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= MAX_IDLE_PCIE_KIB_PER_SECOND
            ):
                problems.append(f"manifest guard {index} {field} is invalid")
        if (
            guard["maximum_allowed_pcie_kib_per_second"]
            != MAX_IDLE_PCIE_KIB_PER_SECOND
        ):
            problems.append(
                f"manifest guard {index} maximum_allowed_pcie_kib_per_second is invalid"
            )
        watchdog_gap = guard["max_watchdog_probe_gap_seconds"]
        if (
            type(watchdog_gap) is not float
            or not math.isfinite(watchdog_gap)
            or not 0 <= watchdog_gap <= 1.0
        ):
            problems.append(
                f"manifest guard {index} max_watchdog_probe_gap_seconds is invalid"
            )
        if guard["maximum_allowed_watchdog_probe_gap_seconds"] != 1.0:
            problems.append(
                f"manifest guard {index} maximum_allowed_watchdog_probe_gap_seconds is invalid"
            )
        wait_poll = guard["wait_poll_seconds"]
        if type(wait_poll) is int and not (
            MIN_WAIT_POLL_SECONDS <= wait_poll <= MAX_WAIT_POLL_SECONDS
        ):
            problems.append(
                f"manifest guard {index} wait_poll_seconds must be between "
                f"{MIN_WAIT_POLL_SECONDS} and {MAX_WAIT_POLL_SECONDS}"
            )
        if type(watchdog) is float and math.isfinite(watchdog) and watchdog > MAX_WATCHDOG_SECONDS:
            problems.append(
                f"manifest guard {index} watchdog_seconds must be no more than "
                f"{MAX_WATCHDOG_SECONDS}"
            )
        if type(guard["finalize_seconds"]) is int and (
            guard["finalize_seconds"] < MIN_FINALIZE_SECONDS
        ):
            problems.append(
                f"manifest guard {index} finalize_seconds must be at least {MIN_FINALIZE_SECONDS}"
            )
        if type(guard["gpu_count"]) is int and guard["gpu_count"] != 2:
            problems.append(f"manifest guard {index} gpu_count must equal 2")
        if len(moments) == 3 and numeric_guard_valid:
            observed_idle = (moments["launch_at"] - moments["idle_started_at"]).total_seconds()
            if moments["launch_at"] < moments["not_before"]:
                problems.append(f"manifest guard {index} launch precedes not_before")
            if observed_idle < guard["idle_seconds"]:
                problems.append(
                    f"manifest guard {index} does not attest the required idle interval"
                )
            minimum_probes = max(
                2,
                math.floor(guard["idle_seconds"] / guard["wait_poll_seconds"]),
            )
            if guard["idle_probe_count"] < minimum_probes:
                problems.append(f"manifest guard {index} has too few idle probes")

    attempt_order = [guard.get("attempt") for guard in guards if isinstance(guard, dict)]
    valid_attempt_order = [
        value for value in attempt_order if isinstance(value, int) and not isinstance(value, bool)
    ]
    if len(valid_attempt_order) != len(guards) or valid_attempt_order != sorted(
        set(valid_attempt_order)
    ):
        problems.append("manifest guards are not in strictly increasing attempt order")
    if isinstance(manifest.get("attempt"), int) and (
        not by_attempt or max(by_attempt) != manifest["attempt"]
    ):
        problems.append("manifest final attempt has no matching guard attestation")

    ordered_attempts = sorted(by_attempt)
    for suffix in phase_specs:
        report = payloads[suffix]
        attempt = report.get("campaign_attempt")
        if attempt not in by_attempt:
            problems.append(f"{suffix} has no guard for its generating attempt")
            continue
        try:
            phase_time = datetime.strptime(report["run_id"], "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except (KeyError, TypeError, ValueError):
            continue
        _guard, moments = by_attempt[attempt]
        if (phase_time - moments["launch_at"]).total_seconds() < -1.0:
            problems.append(f"{suffix} predates its guarded launch")
        later_launches = [
            by_attempt[value][1]["launch_at"] for value in ordered_attempts if value > attempt
        ]
        if later_launches and phase_time >= min(later_launches):
            problems.append(f"{suffix} is not bound to its recorded attempt")
    return problems


def _git_reader(ref: str) -> Callable[[Path], bytes]:
    def read(path: Path) -> bytes:
        object_name = f":{path.as_posix()}" if ref == ":" else f"{ref}:{path.as_posix()}"
        return subprocess.run(
            ["git", "show", object_name],
            check=True,
            capture_output=True,
        ).stdout

    return read


def _git_changed_paths(ref: str) -> set[Path]:
    command = (
        ["git", "diff", "--cached", "--name-only"]
        if ref == ":"
        else ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", ref]
    )
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return {Path(line) for line in output.splitlines() if line}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--result-branch", required=True)
    parser.add_argument("--expected-origin", required=True)
    parser.add_argument("--expected-device-id", required=True)
    args = parser.parse_args()
    try:
        source = subprocess.run(
            [
                "git",
                "show",
                f"{args.expected_commit}:src/switchyard/csrc/shared_backward.cu",
            ],
            check=True,
            capture_output=True,
        ).stdout
        compile_contract = json.loads(
            subprocess.run(
                [
                    "git",
                    "show",
                    f"{args.expected_commit}:results/backward_candidates_compile_sm120.json",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(compile_contract, dict):
            raise ValueError("tracked compiler contract is not an object")
        problems = validate_bundle(
            args.prefix,
            read_bytes=_git_reader(args.git_ref),
            changed_paths=_git_changed_paths(args.git_ref),
            expected_commit=args.expected_commit,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
            result_branch=args.result_branch,
            expected_origin=args.expected_origin,
            expected_device_id=args.expected_device_id,
            expected_source_sha256=hashlib.sha256(source).hexdigest(),
            expected_compile_contract=compile_contract,
        )
    except (KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        problems = [f"cannot read Git evidence: {type(exc).__name__}"]
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
