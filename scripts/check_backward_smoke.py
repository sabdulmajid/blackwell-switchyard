#!/usr/bin/env python3
"""Validate quick backward reports before the full GPU campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_backward import (  # noqa: E402
    MASKED_TAIL_SHAPES,
    _check_current_kernel_contract,
    _check_kernel_contract,
    _correct,
    _shape,
)

from switchyard.training_plan import get_training_plan  # noqa: E402

EXPECTED = {
    "current",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "liger",
}
CANDIDATES = EXPECTED - {"current", "liger"}


def check_reports(reports: list[dict]) -> dict:
    problems: list[str] = []
    if {report.get("dtype") for report in reports} != {"bfloat16", "float16"}:
        problems.append("smoke gate requires exactly one bf16 and one fp16 report")
    commits = {report.get("provenance", {}).get("repository_commit") for report in reports}
    uuids = {report.get("gpu_preflight", {}).get("resolved_uuid") for report in reports}
    if len(commits) != 1 or None in commits:
        problems.append("reports do not use one recorded repository commit")
    if len(uuids) != 1 or None in uuids:
        problems.append("reports do not use one physical GPU")

    successful = {name: 0 for name in EXPECTED}
    for report in reports:
        dtype = report.get("dtype", "")
        prefix = dtype or "unknown dtype"
        if report.get("schema_version") != 2 or report.get("shape_set") != "gate":
            problems.append(f"{prefix}: report is not a schema-2 gate run")
        if report.get("correctness_seeds") != [0, 1, 2]:
            problems.append(f"{prefix}: correctness seeds must be [0, 1, 2]")
        if set(report.get("selected_implementations", [])) != EXPECTED:
            problems.append(f"{prefix}: selected implementation set is incomplete")
        provenance = report.get("provenance", {})
        if provenance.get("worktree_dirty") is not False:
            problems.append(f"{prefix}: worktree was not clean")
        if "--quick" not in provenance.get("argv", []):
            problems.append(f"{prefix}: report is not marked as a quick smoke run")
        preflight = report.get("gpu_preflight", {})
        postflight = report.get("gpu_postflight", {})
        if preflight.get("compute_processes_at_start") or preflight.get("busy_override"):
            problems.append(f"{prefix}: preflight was not exclusive")
        if postflight.get("compute_processes_at_end"):
            problems.append(f"{prefix}: postflight was not exclusive")
        if postflight.get("resolved_uuid") != preflight.get("resolved_uuid"):
            problems.append(f"{prefix}: preflight and postflight GPU differ")

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
            for metric in ("forward", "backward", "fwd_bwd"):
                timing = record.get(metric, {})
                if timing.get("trial_count") != 2 or len(timing.get("trials", [])) != 2:
                    problems.append(
                        f"{prefix} {shape} {implementation}: incomplete quick {metric} trials"
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

        for shape, records in by_shape.items():
            current = records.get("current", {})
            accepted_by_seed = {
                item.get("seed"): item.get("report", {})
                for item in current.get("correctness_by_seed", [])
            }
            for candidate in CANDIDATES:
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
        "gpu_uuid": next(iter(uuids)) if len(uuids) == 1 else None,
        "successful_records": successful,
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=2, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    decision = check_reports([json.loads(path.read_text()) for path in args.reports])
    rendered = json.dumps(decision, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0 if decision["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
