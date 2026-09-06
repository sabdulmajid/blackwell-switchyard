#!/usr/bin/env python3
"""Validate quick backward reports before the full GPU campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_backward import (  # noqa: E402
    MASKED_TAIL_SHAPES,
    PINNED_LIGER_COMMIT,
    PINNED_LIGER_SOURCE_SHA256,
    _check_current_kernel_contract,
    _check_kernel_contract,
    _correct,
    _shape,
    _timing_from_raw,
)

from switchyard.training_plan import get_training_plan  # noqa: E402

EXPECTED = {
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
    "liger",
}
CANDIDATES = EXPECTED - {"current", "liger"}
NUMERICAL_FLOOR_CANDIDATES = CANDIDATES - {"cuda_shared"}
GATE_SHAPES = {
    (9, 1, 4096, 4096),
    (9, 1, 4096, 8192),
    (32, 1, 4096, 2048),
    (32, 1, 4096, 4096),
}


def check_reports(
    reports: list[dict],
    *,
    expected_commit: str | None = None,
    expected_branch: str | None = None,
    expected_tree: str | None = None,
) -> dict:
    problems: list[str] = []
    if {report.get("dtype") for report in reports} != {"bfloat16", "float16"}:
        problems.append("smoke gate requires exactly one bf16 and one fp16 report")
    commits = {report.get("provenance", {}).get("repository_commit") for report in reports}
    trees = {report.get("provenance", {}).get("repository_tree") for report in reports}
    device_ids = {report.get("gpu_preflight", {}).get("device_id") for report in reports}
    if len(commits) != 1 or None in commits:
        problems.append("reports do not use one recorded repository commit")
    if len(device_ids) != 1 or None in device_ids:
        problems.append("reports do not use one campaign device ID")
    if len(trees) != 1 or None in trees:
        problems.append("reports do not use one recorded repository tree")
    if expected_commit is not None and commits != {expected_commit}:
        problems.append(f"reports do not match expected commit {expected_commit}")
    if expected_tree is not None and trees != {expected_tree}:
        problems.append(f"reports do not match expected tree {expected_tree}")

    successful = {name: 0 for name in EXPECTED}
    for report in reports:
        dtype = report.get("dtype", "")
        prefix = dtype or "unknown dtype"
        if (
            report.get("schema_version") != 3
            or report.get("shape_set") != "gate"
            or not isinstance(report.get("run_id"), str)
            or re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", report["run_id"]) is None
        ):
            problems.append(f"{prefix}: report is not a schema-2 gate run")
        if report.get("run_status") != "complete":
            problems.append(f"{prefix}: report is not complete")
        if report.get("correctness_seeds") != [0, 1, 2]:
            problems.append(f"{prefix}: correctness seeds must be [0, 1, 2]")
        if set(report.get("selected_implementations", [])) != EXPECTED:
            problems.append(f"{prefix}: selected implementation set is incomplete")
        provenance = report.get("provenance", {})
        if expected_branch is not None and provenance.get("repository_branch") != expected_branch:
            problems.append(f"{prefix}: wrong benchmark branch")
        if provenance.get("worktree_dirty") is not False:
            problems.append(f"{prefix}: worktree was not clean")
        if "--quick" not in provenance.get("argv", []):
            problems.append(f"{prefix}: report is not marked as a quick smoke run")
        if provenance.get("third_party_commits", {}).get("Liger-Kernel") != PINNED_LIGER_COMMIT:
            problems.append(f"{prefix}: pinned Liger commit is missing or wrong")
        if provenance.get("third_party_dirty", {}).get("Liger-Kernel") is not False:
            problems.append(f"{prefix}: pinned Liger checkout is not recorded clean")
        liger = report.get("comparators", {}).get("liger", {})
        if (
            liger.get("commit") != PINNED_LIGER_COMMIT
            or liger.get("under_pinned_checkout") is not True
            or liger.get("worktree_dirty") is not False
            or liger.get("source_sha256") != PINNED_LIGER_SOURCE_SHA256
        ):
            problems.append(f"{prefix}: imported Liger provenance is incomplete")
        preflight = report.get("gpu_preflight", {})
        postflight = report.get("gpu_postflight", {})
        if (
            preflight.get("foreign_compute_process_count_at_start") != 0
            or preflight.get("busy_override")
        ):
            problems.append(f"{prefix}: preflight was not exclusive")
        if postflight.get("foreign_compute_process_count_at_end") != 0:
            problems.append(f"{prefix}: postflight was not exclusive")
        if postflight.get("device_id") != preflight.get("device_id"):
            problems.append(f"{prefix}: preflight and postflight GPU differ")
        monitor = report.get("gpu_process_monitor", {})
        samples = monitor.get("samples")
        interval = monitor.get("interval_seconds")
        duration = monitor.get("duration_seconds")
        monitor_coverage_ok = (
            isinstance(samples, int)
            and samples >= 2
            and isinstance(interval, int | float)
            and 0 < interval <= 1
            and isinstance(duration, int | float)
            and duration >= interval
            and samples >= max(2, int(duration / (2 * interval)))
        )
        if (
            monitor.get("device_id") != preflight.get("device_id")
            or monitor.get("collision_detected") is not False
            or monitor.get("collision_events")
            or monitor.get("probe_errors")
            or not monitor_coverage_ok
        ):
            problems.append(f"{prefix}: sampled GPU monitor is incomplete or contaminated")

        execution_order = report.get("execution_order", [])
        schedule_shapes = [_shape(item) for item in execution_order]
        if len(schedule_shapes) != len(set(schedule_shapes)) or set(schedule_shapes) != GATE_SHAPES:
            problems.append(f"{prefix}: exact gate execution schedule is missing")
        schedules = {_shape(item): item.get("metrics", {}) for item in execution_order}

        by_shape: dict[tuple[int, int, int, int], dict[str, dict]] = {}
        local_success = {name: 0 for name in EXPECTED}
        for record in report.get("results", []):
            shape = _shape(record)
            implementation = record.get("impl", "")
            if implementation in by_shape.setdefault(shape, {}):
                problems.append(f"{prefix} {shape}: duplicate {implementation} record")
            by_shape.setdefault(shape, {})[implementation] = record
            skipped = str(record.get("skipped", ""))
            if skipped:
                if not skipped.startswith("unsupported plan:"):
                    problems.append(f"{prefix} {shape} {implementation}: {skipped}")
                continue
            if implementation not in EXPECTED:
                continue
            successful[implementation] += 1
            local_success[implementation] += 1
            if not _correct(record):
                problems.append(f"{prefix} {shape} {implementation}: correctness failed")
            if [
                item.get("seed") for item in record.get("correctness_by_seed", [])
            ] != [0, 1, 2]:
                problems.append(
                    f"{prefix} {shape} {implementation}: correctness seed matrix is incomplete"
                )
            for metric in ("forward", "backward", "fwd_bwd"):
                _timing_from_raw(
                    record,
                    metric,
                    schedules.get(shape, {}).get(metric, []),
                    problems,
                    f"{prefix} {shape}",
                    trial_count=2,
                    reps_per_trial=10,
                    warmup_per_trial=8,
                )
            kernel_problems: list[str] = []
            label = f"{prefix} {shape}"
            if implementation == "current":
                _check_current_kernel_contract(
                    shape, dtype, record, kernel_problems, label
                )
            elif implementation in CANDIDATES:
                _check_kernel_contract(
                    implementation, dtype, record, kernel_problems, label
                )
                if record.get("training_plan") != get_training_plan(implementation).as_dict():
                    kernel_problems.append(f"{label}: wrong serialized training plan")
            problems.extend(kernel_problems)

        if set(by_shape) != GATE_SHAPES:
            problems.append(f"{prefix}: exact four gate shapes are missing")
        for shape in GATE_SHAPES:
            records = by_shape.get(shape, {})
            if set(records) != EXPECTED:
                problems.append(f"{prefix} {shape}: implementation record matrix is incomplete")
            current = records.get("current", {})
            accepted_by_seed = {
                item.get("seed"): item.get("report", {})
                for item in current.get("correctness_by_seed", [])
            }
            for candidate in NUMERICAL_FLOOR_CANDIDATES:
                measured = records.get(candidate, {})
                if measured.get("skipped"):
                    continue
                candidate_by_seed = {
                    item.get("seed"): item.get("report", {})
                    for item in measured.get("correctness_by_seed", [])
                }
                for seed in set(accepted_by_seed) & set(candidate_by_seed):
                    for value in ("output", "dv", "dw"):
                        accepted = accepted_by_seed[seed].get(value, {}).get("rel_l2")
                        observed = candidate_by_seed[seed].get(value, {}).get("rel_l2")
                        if (
                            isinstance(accepted, int | float)
                            and isinstance(observed, int | float)
                            and observed > max(1.05 * accepted, 1e-7)
                        ):
                            problems.append(
                                f"{prefix} {shape} {candidate} seed={seed} {value}: "
                                f"error {observed:.3e} exceeds {accepted:.3e}"
                            )

        tails = report.get("correctness_only", [])
        if {_shape(case) for case in tails} != MASKED_TAIL_SHAPES:
            problems.append(f"{prefix}: exact masked-tail shapes are missing")
        for case in tails:
            shape = _shape(case)
            observed = {
                (item.get("impl"), item.get("seed"))
                for item in case.get("implementations", [])
            }
            expected = {
                (implementation, seed)
                for implementation in EXPECTED
                for seed in (0, 1, 2)
            }
            if observed != expected:
                problems.append(f"{prefix} masked tail {shape}: implementation matrix is incomplete")
            for item in case.get("implementations", []):
                skipped = str(item.get("skipped", ""))
                if skipped.startswith("unsupported plan:"):
                    continue
                if skipped or not all(
                    item.get("correctness", {}).get(name, {}).get("ok", False)
                    for name in ("output", "dv", "dw")
                ):
                    problems.append(
                        f"{prefix} masked tail {shape} {item.get('impl')} seed={item.get('seed')}"
                    )

        for implementation, count in local_success.items():
            if count == 0:
                problems.append(f"{prefix}: {implementation} had no successful measured shape")

    return {
        "status": "PASS" if not problems else "FAIL",
        "repository_commit": next(iter(commits)) if len(commits) == 1 else None,
        "device_id": next(iter(device_ids)) if len(device_ids) == 1 else None,
        "successful_records": successful,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=2, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-tree", required=True)
    args = parser.parse_args()
    report_bytes = [path.read_bytes() for path in args.reports]
    decision = check_reports(
        [json.loads(raw) for raw in report_bytes],
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
    rendered = json.dumps(decision, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
