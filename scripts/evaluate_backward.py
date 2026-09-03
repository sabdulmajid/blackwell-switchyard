#!/usr/bin/env python3
"""Apply correctness, provenance, and paired performance gates to one plan."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from switchyard.training_plan import get_training_plan, plan_supports  # noqa: E402

ALL_DTYPES = {"bfloat16", "float16", "float32"}
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


@dataclass(frozen=True)
class Comparison:
    dtype: str
    shape: tuple[int, int, int, int]
    current_ms: float
    candidate_ms: float
    speedup: float
    change: float
    classification: str


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


def _trial_medians(record: dict, metric: str) -> dict[int, float]:
    result = {}
    for fallback_index, trial in enumerate(record.get(metric, {}).get("trials", [])):
        samples = trial.get("samples_ms", [])
        if samples:
            result[int(trial.get("trial", fallback_index))] = statistics.median(samples)
    return result


def _paired_speedup_lower_bound(numerator: dict[int, float], denominator: dict[int, float]) -> float | None:
    """Bootstrap paired trial ratios, never correlated event-level samples."""
    trial_ids = sorted(set(numerator) & set(denominator))
    if len(trial_ids) < 5:
        return None
    ratios = [numerator[index] / denominator[index] for index in trial_ids]
    rng = random.Random(0)
    estimates = [
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10_000)
    ]
    estimates.sort()
    return estimates[int(0.025 * len(estimates))]


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
    if plan.backward.family == "cuda_cluster":
        backward_launches = 2 + cast_launches
        return ("feature_cluster_backward_kernel",), backward_launches, backward_launches + 1
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


def evaluate_reports(
    reports: list[dict],
    *,
    candidate: str = "cuda_cluster",
    threshold: float = 0.07,
    max_cv: float = 0.05,
) -> dict:
    """Return a deterministic promotion decision for one immutable plan."""
    plan = get_training_plan(candidate)
    required_dtypes = _required_dtypes(candidate)
    problems: list[str] = []
    correctness_failures: list[str] = []
    numerical_regressions: list[str] = []
    unstable: list[str] = []
    confidence_failures: list[str] = []
    performance_regressions: list[str] = []
    comparisons: list[Comparison] = []
    anchor_speedups: list[float] = []
    gpu_uuids: set[str] = set()

    dtype_values = [report.get("dtype") for report in reports]
    if len(dtype_values) != len(set(dtype_values)):
        problems.append("each dtype must have exactly one report")
    by_dtype = {report.get("dtype"): report for report in reports if report.get("dtype") in ALL_DTYPES}
    missing_dtypes = required_dtypes - set(by_dtype)
    if missing_dtypes:
        problems.append(f"missing dtype runs: {sorted(missing_dtypes)}")

    commits = {report.get("provenance", {}).get("repository_commit") for report in reports}
    commits.discard(None)
    if len(commits) != 1:
        problems.append("all dtype runs must use one repository commit")

    for dtype in sorted(required_dtypes & set(by_dtype)):
        report = by_dtype[dtype]
        prefix = dtype
        if report.get("schema_version") != 2 or not report.get("run_id"):
            problems.append(f"{prefix}: schema version 2 or run ID is missing")
        if report.get("shape_set") != "full":
            problems.append(f"{prefix}: production decision requires --shape-set full")
        provenance = report.get("provenance", {})
        if provenance.get("worktree_dirty") is not False:
            problems.append(f"{prefix}: benchmark worktree was not recorded as fully clean")
        if provenance.get("tracked_worktree_dirty"):
            problems.append(f"{prefix}: benchmark worktree was dirty")
        if provenance.get("third_party_dirty", {}).get("Liger-Kernel"):
            problems.append(f"{prefix}: pinned Liger worktree was dirty")
        if "--quick" in provenance.get("argv", []):
            problems.append(f"{prefix}: quick runs cannot produce a production decision")
        preflight = report.get("gpu_preflight", {})
        if preflight.get("compute_processes_at_start") or preflight.get("busy_override"):
            problems.append(f"{prefix}: benchmark did not start with exclusive access")
        if not preflight.get("resolved_uuid"):
            problems.append(f"{prefix}: physical GPU UUID was not recorded")
        else:
            gpu_uuids.add(preflight["resolved_uuid"])
        postflight = report.get("gpu_postflight", {})
        if postflight.get("resolved_uuid") != preflight.get("resolved_uuid"):
            problems.append(f"{prefix}: GPU postflight identity is missing or changed")
        if postflight.get("compute_processes_at_end"):
            problems.append(f"{prefix}: another compute process appeared during the run")
        if report.get("candidate_reachable_from_production") is not False:
            problems.append(f"{prefix}: candidate isolation flag is missing or true")
        if report.get("correctness_seeds") != [0, 1, 2]:
            problems.append(f"{prefix}: correctness seeds must be [0, 1, 2]")

        tails = report.get("correctness_only", [])
        observed_tail_shapes = {_shape(case) for case in tails}
        if observed_tail_shapes != MASKED_TAIL_SHAPES:
            problems.append(f"{prefix}: exact masked-tail correctness cases are required")
        for case in tails:
            expected = {
                (implementation, seed)
                for implementation in ("current", candidate, "liger")
                for seed in (0, 1, 2)
            }
            observed = {
                (item.get("impl"), item.get("seed"))
                for item in case.get("implementations", [])
                if item.get("impl") in {"current", candidate, "liger"}
            }
            if observed != expected:
                problems.append(f"{prefix}: masked-tail implementation/seed matrix is incomplete")
            for item in case.get("implementations", []):
                if item.get("impl") in {"current", candidate, "liger"} and not all(
                    item.get("correctness", {}).get(name, {}).get("ok", False)
                    for name in ("output", "dv", "dw")
                ):
                    correctness_failures.append(
                        f"{prefix} masked-tail {item.get('impl')} seed={item.get('seed')}"
                    )

        schedules = {_shape(item): item for item in report.get("execution_order", [])}

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
            if not {"current", candidate, "liger"} <= set(records):
                problems.append(f"{label}: current/{candidate}/Liger set is incomplete")
                continue
            current, measured, liger = records["current"], records[candidate], records["liger"]
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
                if len(trial_schedule) < 5:
                    problems.append(f"{label}: five interleaved {metric} schedules are required")
                    continue
                required = {"current", candidate, "liger"}
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

            for implementation, record in (
                ("current", current),
                (candidate, measured),
                ("liger", liger),
            ):
                if record.get("skipped") or not _correct(record):
                    failure = f"{label} {implementation}"
                    if implementation == candidate:
                        correctness_failures.append(failure)
                    else:
                        problems.append(f"{failure}: comparator correctness is incomplete")
                for metric in ("forward", "backward", "fwd_bwd"):
                    trial_medians = _trial_medians(record, metric)
                    if len(trial_medians) < 5:
                        problems.append(f"{label} {implementation}: fewer than five {metric} trials")
                    cv = record.get(metric, {}).get("cv")
                    if cv is None or cv > max_cv:
                        unstable.append(f"{label} {implementation} {metric}: cv={cv}")

            current_seeds = {
                item["seed"]: item["report"] for item in current.get("correctness_by_seed", [])
            }
            candidate_seeds = {
                item["seed"]: item["report"] for item in measured.get("correctness_by_seed", [])
            }
            if set(current_seeds) != {0, 1, 2} or set(candidate_seeds) != {0, 1, 2}:
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

            current_ms = current.get("backward", {}).get("median_ms")
            candidate_ms = measured.get("backward", {}).get("median_ms")
            if not isinstance(current_ms, int | float) or not isinstance(
                candidate_ms, int | float
            ):
                problems.append(f"{label}: missing backward median")
                continue
            speedup = current_ms / candidate_ms
            change = 1.0 - candidate_ms / current_ms
            classification = "WIN" if change >= threshold else "LOSS" if change <= -threshold else "NOISE"
            comparisons.append(
                Comparison(dtype, shape, current_ms, candidate_ms, speedup, change, classification)
            )

            _check_kernel_contract(candidate, dtype, measured, problems, label)
            _check_current_kernel_contract(shape, dtype, current, problems, label)
            if "traffic_model" not in measured or "fwd_bwd_memory" not in measured:
                problems.append(f"{label}: traffic or memory record is missing")
            current_forward = current.get("forward", {}).get("median_ms")
            candidate_forward = measured.get("forward", {}).get("median_ms")
            if (
                isinstance(current_forward, int | float)
                and isinstance(candidate_forward, int | float)
                and candidate_forward > 1.15 * current_forward
            ):
                performance_regressions.append(
                    f"{label}: training forward regressed by "
                    f"{candidate_forward / current_forward:.2f}x"
                )
            current_workspace = current.get("fwd_bwd_memory", {}).get("workspace_bytes")
            candidate_workspace = measured.get("fwd_bwd_memory", {}).get("workspace_bytes")
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
            if plan.backward.family == "cuda_cluster":
                launch_info = measured.get("cluster_launch_info", {})
                if launch_info.get("active_clusters", 0) <= 0:
                    problems.append(f"{label}: active cluster count is missing")
                if (
                    launch_info.get("dynamic_shared_bytes", 0)
                    + launch_info.get("static_shared_bytes", 0)
                    > launch_info.get("max_shared_bytes", 0)
                ):
                    problems.append(f"{label}: recorded shared memory exceeds the device limit")

            if shape in ANCHOR_SHAPES:
                anchor_speedups.append(speedup)
                lower = _paired_speedup_lower_bound(
                    _trial_medians(liger, "fwd_bwd"),
                    _trial_medians(measured, "fwd_bwd"),
                )
                if lower is None:
                    problems.append(f"{label}: five paired Liger/candidate trials are required")
                elif lower <= 1.0:
                    confidence_failures.append(
                        f"{label}: paired Liger/candidate 95% lower bound={lower:.3f}"
                    )

    if len(gpu_uuids) > 1:
        problems.append("all dtype runs must use the same physical GPU UUID")

    if correctness_failures or numerical_regressions:
        status = "REJECT"
        rationale = "candidate failed correctness or exceeded 1.05 times accepted numerical error"
    elif problems or unstable:
        status = "MORE_DATA"
        rationale = "the evidence matrix or measurement-quality gate is incomplete"
    else:
        wins = [comparison for comparison in comparisons if comparison.classification == "WIN"]
        anchor_floor = min(anchor_speedups) if anchor_speedups else 0.0
        anchor_geomean = (
            math.exp(sum(math.log(value) for value in anchor_speedups) / len(anchor_speedups))
            if anchor_speedups
            else 0.0
        )
        if (
            len(wins) < 2
            or anchor_floor < 1.10
            or anchor_geomean < 1.15
            or confidence_failures
            or performance_regressions
        ):
            status = "DROP"
            rationale = "candidate did not clear the anchor or paired Liger requirement"
        else:
            status = "READY_FOR_DISPATCH_REVIEW"
            rationale = "candidate has repeatable wins; dispatch only the measured shape and dtype cases"

    return {
        "status": status,
        "rationale": rationale,
        "candidate": candidate,
        "required_dtypes": sorted(required_dtypes),
        "threshold": threshold,
        "max_cv": max_cv,
        "commits": sorted(commits),
        "problems": problems,
        "correctness_failures": correctness_failures,
        "numerical_regressions": numerical_regressions,
        "unstable": unstable,
        "confidence_failures": confidence_failures,
        "performance_regressions": performance_regressions,
        "anchor_speedup_floor": min(anchor_speedups) if anchor_speedups else None,
        "anchor_speedup_geomean": (
            math.exp(sum(math.log(value) for value in anchor_speedups) / len(anchor_speedups))
            if anchor_speedups
            else None
        ),
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--candidate", default="cuda_cluster")
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.results]
    decision = evaluate_reports(
        reports,
        candidate=args.candidate,
        threshold=args.threshold,
        max_cv=args.max_cv,
    )
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
        for failure in decision["performance_regressions"]:
            print(f"  - performance: {failure}")

    return 1 if decision["status"] == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
