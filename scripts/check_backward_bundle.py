#!/usr/bin/env python3
"""Reconstruct and validate one publishable backward campaign bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from check_backward_report import validate_report  # noqa: E402
from check_backward_smoke import check_reports  # noqa: E402
from evaluate_backward import evaluate_reports  # noqa: E402

ALL_IMPLEMENTATIONS = [
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "liger",
]
PORTABLE_IMPLEMENTATIONS = [
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "liger",
]
CANDIDATES = [
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
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
    },
}
EXPECTED_COMPILATIONS = {
    "serial_recompute_atomic_t4_n9_d4096",
    "serial_saved_partials_t16_n9_d4096",
    "serial_saved_partials_t16_n32_d4096",
    "serial_saved_partials_t16_n32_d2048",
    "saved_training_forward_n9_d8192",
    "accepted_tiled_forward_n9_d4096",
    "saved_resident_forward_n8_d1024",
    "dw_partial_reduction",
    "one_read_cuda",
}


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


def _inputs(raw_reports: list[bytes]) -> list[dict[str, str]]:
    return [{"sha256": hashlib.sha256(raw).hexdigest()} for raw in raw_reports]


def _compile_problems(
    report: dict, expected_commit: str, expected_source_sha256: str | None = None
) -> list[str]:
    problems = []
    if report.get("target", {}).get("arch") != 120 or report.get("gpu_visible") is not False:
        problems.append("offline compile report does not target sm_120 without a GPU")
    provenance = report.get("provenance", {})
    if provenance.get("repository_commit") != expected_commit:
        problems.append("offline compile report uses the wrong repository commit")
    if provenance.get("worktree_clean") is not True:
        problems.append("offline compile report was not produced from a clean worktree")
    if (
        expected_source_sha256 is not None
        and provenance.get("cuda_source_sha256") != expected_source_sha256
    ):
        problems.append("offline compile report uses different CUDA source bytes")
    compilation_names = [item.get("name") for item in report.get("compilations", [])]
    if (
        len(compilation_names) != len(EXPECTED_COMPILATIONS)
        or set(compilation_names) != EXPECTED_COMPILATIONS
    ):
        problems.append("offline compile report does not contain the exact compilation set")
    cuda_records = [
        item for item in report.get("compilations", []) if item.get("kind") == "cuda"
    ]
    if len(cuda_records) != 1:
        problems.append("offline compile report must contain one CUDA build")
        return problems
    cuda = cuda_records[0]
    expected_counts = cuda.get("required_template_instances", {})
    limits = cuda.get("register_limits", {})
    resources = cuda.get("resources", [])
    if not isinstance(expected_counts, dict) or not isinstance(limits, dict):
        problems.append("offline CUDA resource contracts are malformed")
        return problems
    if expected_counts != EXPECTED_CUDA_INSTANCES or limits != EXPECTED_CUDA_LIMITS:
        problems.append("offline CUDA resource contracts differ from the source gate")
    if Counter(item.get("kernel") for item in resources) != Counter(expected_counts):
        problems.append("offline CUDA template-instance counts are incomplete")
    for item in resources:
        kernel = item.get("kernel")
        if item.get("stack_bytes") != 0 or item.get("local_bytes") != 0:
            problems.append(f"offline CUDA resource record spills: {kernel}")
        if not isinstance(limits.get(kernel), int) or item.get("registers", 10**9) > limits[kernel]:
            problems.append(f"offline CUDA register budget failed: {kernel}")
    for compilation in report.get("compilations", []):
        if not compilation.get("resources"):
            problems.append(f"offline compilation has no resource record: {compilation.get('name')}")
        for item in compilation.get("resources", []):
            if item.get("stack_bytes") != 0 or item.get("local_bytes") != 0:
                problems.append(f"offline resource record spills: {compilation.get('name')}")
    plan_names = [item.get("name") for item in report.get("plans", [])]
    if len(plan_names) != len(CANDIDATES) or set(plan_names) != set(CANDIDATES):
        problems.append("offline compile report does not contain the exact campaign plans")
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
    expected_gpu_uuid: str,
    expected_source_sha256: str | None = None,
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
            payload = json.loads(raw[suffix])
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            payloads[suffix] = payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            problems.append(f"cannot read {path.name}: {type(exc).__name__}")
    if len(payloads) != len(paths):
        return problems

    problems.extend(
        _compile_problems(
            payloads["offline_compile"], expected_commit, expected_source_sha256
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
            expected_gpu_uuid=expected_gpu_uuid,
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
        "schema_version": 1,
        "repository_commit": expected_commit,
        "repository_tree": expected_tree,
        "benchmark_branch": expected_branch,
        "result_branch": result_branch,
        "origin": expected_origin,
        "gpu_uuid": expected_gpu_uuid,
        "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
    }
    for field, expected in exact_manifest.items():
        if manifest.get(field) != expected:
            problems.append(f"manifest {field} differs from the campaign contract")
    if not isinstance(manifest.get("attempt"), int) or manifest["attempt"] <= 0:
        problems.append("manifest attempt must be a positive integer")
    guard = manifest.get("guard", {})
    expected_manifest_fields = set(exact_manifest) | {"attempt", "guard"}
    if set(manifest) != expected_manifest_fields:
        problems.append("manifest field set is not exact")
    guard_fields = {
        "not_before",
        "idle_seconds",
        "wait_poll_seconds",
        "watchdog_seconds",
        "finalize_seconds",
    }
    if not isinstance(guard, dict) or set(guard) != guard_fields:
        problems.append("manifest guard field set is not exact")
    else:
        try:
            not_before = datetime.fromisoformat(guard["not_before"])
            if not_before.tzinfo is None:
                raise ValueError("missing UTC offset")
        except (TypeError, ValueError):
            problems.append("manifest guard not_before is not an offset timestamp")
        for field in guard_fields - {"not_before"}:
            value = guard[field]
            if not isinstance(value, int | float) or value <= 0:
                problems.append(f"manifest guard {field} must be positive")
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
    parser.add_argument("--expected-gpu-uuid", required=True)
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
        problems = validate_bundle(
            args.prefix,
            read_bytes=_git_reader(args.git_ref),
            changed_paths=_git_changed_paths(args.git_ref),
            expected_commit=args.expected_commit,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
            result_branch=args.result_branch,
            expected_origin=args.expected_origin,
            expected_gpu_uuid=args.expected_gpu_uuid,
            expected_source_sha256=hashlib.sha256(source).hexdigest(),
        )
    except (KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        problems = [f"cannot read Git evidence: {type(exc).__name__}"]
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
