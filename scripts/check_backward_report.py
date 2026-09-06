#!/usr/bin/env python3
"""Accept only a clean, complete backward report for phase reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PINNED_LIGER_COMMIT = "777799588a89d74c489ed995e3bf006427738e85"
PINNED_LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)


def validate_report(
    report: dict,
    *,
    dtype: str,
    shape_set: str,
    implementations: list[str],
    quick: bool,
    expected_commit: str,
    expected_branch: str,
    expected_gpu_uuid: str,
) -> list[str]:
    """Return every reason a report cannot be reused."""
    problems: list[str] = []
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
    if not report.get("run_id"):
        problems.append("run_id is missing")
    if report.get("selected_implementations") != implementations:
        problems.append("selected implementation order differs from the requested phase")

    provenance = report.get("provenance", {})
    if provenance.get("repository_commit") != expected_commit:
        problems.append("repository commit differs from the campaign commit")
    if provenance.get("repository_branch") != expected_branch:
        problems.append("repository branch differs from the campaign branch")
    if not provenance.get("repository_tree"):
        problems.append("repository tree is missing")
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
        and duration >= 0
        and samples >= max(2, int(duration / (2 * interval)))
    )
    if (
        monitor.get("device_uuid") != uuid
        or monitor.get("collision_detected") is not False
        or monitor.get("collision_events")
        or monitor.get("probe_errors")
        or not coverage_ok
    ):
        problems.append("continuous GPU monitor was incomplete or contaminated")
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
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
