#!/usr/bin/env python3
"""Apply the production gates to source-serial backward benchmark results.

The script never changes dispatch. It classifies each measured shape and says
whether the evidence is complete enough to justify a dispatch patch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

REQUIRED_DTYPES = {"bfloat16", "float16", "float32"}
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
    change: float
    classification: str


def _shape(record: dict) -> tuple[int, int, int, int]:
    value = record["shape"]
    return value["n"], value["b"], value["t"], value["d"]


def _correct(record: dict) -> bool:
    correctness = record.get("correctness", {})
    primary = all(
        correctness.get(name, {}).get("ok", False) for name in ("output", "dv", "dw")
    )
    seeded = record.get("correctness_by_seed", [])
    return primary and all(
        all(item.get("report", {}).get(name, {}).get("ok", False) for name in ("output", "dv", "dw"))
        for item in seeded
    )


def _samples(record: dict, metric: str) -> list[float]:
    return [
        sample
        for trial in record.get(metric, {}).get("trials", [])
        for sample in trial.get("samples_ms", [])
    ]


def _speedup_lower_bound(numerator: list[float], denominator: list[float]) -> float | None:
    """Deterministic unpaired bootstrap lower bound for a median speedup."""
    if not numerator or not denominator:
        return None
    rng = random.Random(0)
    estimates = []
    for _ in range(2000):
        top = statistics.median(rng.choices(numerator, k=len(numerator)))
        bottom = statistics.median(rng.choices(denominator, k=len(denominator)))
        estimates.append(top / bottom)
    estimates.sort()
    return estimates[int(0.025 * len(estimates))]


def evaluate_reports(
    reports: list[dict], *, threshold: float = 0.07, max_cv: float = 0.05
) -> dict:
    """Return a deterministic decision from one result file per dtype."""
    problems: list[str] = []
    comparisons: list[Comparison] = []
    correctness_failures: list[str] = []
    numerical_regressions: list[str] = []
    unstable: list[str] = []
    anchor_speedups: list[float] = []
    liger_confidence_failures: list[str] = []
    commits = {report.get("provenance", {}).get("repository_commit") for report in reports}
    commits.discard(None)
    if len(commits) != 1:
        problems.append("all dtype runs must use one repository commit")

    by_dtype: dict[str, dict[tuple[int, int, int, int], dict[str, dict]]] = {}
    for report in reports:
        dtype = report.get("dtype")
        if dtype not in REQUIRED_DTYPES:
            problems.append(f"unknown or missing dtype: {dtype!r}")
            continue
        if report.get("shape_set") != "full":
            problems.append(f"{dtype}: production decision requires --shape-set full")
        if report.get("schema_version") != 1 or not report.get("run_id"):
            problems.append(f"{dtype}: schema version or run ID is missing")
        if report.get("provenance", {}).get("tracked_worktree_dirty"):
            problems.append(f"{dtype}: benchmark provenance says the tracked worktree was dirty")
        if report.get("provenance", {}).get("third_party_dirty", {}).get("Liger-Kernel"):
            problems.append(f"{dtype}: pinned Liger worktree was dirty")
        if "--quick" in report.get("provenance", {}).get("argv", []):
            problems.append(f"{dtype}: quick runs cannot produce a production decision")
        if report.get("gpu_preflight", {}).get("compute_processes_at_start"):
            problems.append(f"{dtype}: benchmark started while the GPU was busy")
        if report.get("gpu_preflight", {}).get("busy_override"):
            problems.append(f"{dtype}: busy-GPU override cannot produce a production result")
        if report.get("candidate_reachable_from_production") is not False:
            problems.append(f"{dtype}: candidate isolation flag is missing or true")
        if report.get("correctness_seeds") != [0, 1, 2]:
            problems.append(f"{dtype}: correctness seeds must be [0, 1, 2]")
        tails = report.get("correctness_only", [])
        tail_shapes = {
            tuple(case.get("shape", {}).get(key, -1) for key in ("n", "b", "t", "d"))
            for case in tails
        }
        if tail_shapes != MASKED_TAIL_SHAPES:
            problems.append(f"{dtype}: exact masked-tail correctness cases are required")
        for case in tails:
            observed = {
                (record.get("impl"), record.get("seed"))
                for record in case.get("implementations", [])
            }
            required = {
                (impl, seed)
                for impl in ("current", "source_serial", "liger")
                for seed in (0, 1, 2)
            }
            if observed != required:
                problems.append(f"{dtype}: masked-tail implementation/seed matrix is incomplete")
            for record in case.get("implementations", []):
                if record.get("impl") in {"current", "source_serial", "liger"} and not all(
                    record.get("correctness", {}).get(name, {}).get("ok", False)
                    for name in ("output", "dv", "dw")
                ):
                    correctness_failures.append(
                        f"{dtype} masked-tail {record.get('impl')} seed={record.get('seed')}"
                    )

        table = by_dtype.setdefault(dtype, {})
        for record in report.get("results", []):
            table.setdefault(_shape(record), {})[record.get("impl", "")] = record

    missing_dtypes = REQUIRED_DTYPES - set(by_dtype)
    if missing_dtypes:
        problems.append(f"missing dtype runs: {sorted(missing_dtypes)}")

    for dtype, table in sorted(by_dtype.items()):
        missing_shapes = EXPECTED_FULL_SHAPES - set(table)
        if missing_shapes:
            problems.append(f"{dtype}: missing {len(missing_shapes)} full-sweep shapes")
        for shape in sorted(EXPECTED_FULL_SHAPES & set(table)):
            pair = table[shape]
            if not {"current", "source_serial", "liger"} <= set(pair):
                problems.append(f"{dtype} {shape}: current/source_serial/Liger set is incomplete")
                continue
            current = pair["current"]
            candidate = pair["source_serial"]
            for name, record in (("current", current), ("source_serial", candidate)):
                if record.get("skipped") or not _correct(record):
                    correctness_failures.append(f"{dtype} {shape} {name}")
                cv = record.get("backward", {}).get("cv")
                if cv is None or cv > max_cv:
                    unstable.append(f"{dtype} {shape} {name}: cv={cv}")
                if len(_samples(record, "backward")) < 200:
                    problems.append(f"{dtype} {shape} {name}: fewer than 200 backward samples")

            current_seeds = {
                item["seed"]: item["report"] for item in current.get("correctness_by_seed", [])
            }
            candidate_seeds = {
                item["seed"]: item["report"]
                for item in candidate.get("correctness_by_seed", [])
            }
            if set(current_seeds) != {0, 1, 2} or set(candidate_seeds) != {0, 1, 2}:
                problems.append(f"{dtype} {shape}: exact three-seed correctness set is missing")
            for seed in sorted(set(current_seeds) & set(candidate_seeds)):
                for value in ("output", "dv", "dw"):
                    current_error = current_seeds[seed][value].get("rel_l2")
                    candidate_error = candidate_seeds[seed][value].get("rel_l2")
                    if isinstance(current_error, (int, float)) and isinstance(
                        candidate_error, (int, float)
                    ) and candidate_error > max(1.05 * current_error, 1e-7):
                        numerical_regressions.append(
                            f"{dtype} {shape} seed={seed} {value}: "
                            f"{candidate_error:.3e} vs {current_error:.3e}"
                        )

            current_ms = current.get("backward", {}).get("median_ms")
            candidate_ms = candidate.get("backward", {}).get("median_ms")
            if not isinstance(current_ms, (int, float)) or not isinstance(
                candidate_ms, (int, float)
            ):
                problems.append(f"{dtype} {shape}: missing backward median")
                continue
            change = (current_ms - candidate_ms) / current_ms
            if change >= threshold:
                classification = "WIN"
            elif change <= -threshold:
                classification = "LOSS"
            else:
                classification = "NOISE"
            comparisons.append(
                Comparison(dtype, shape, current_ms, candidate_ms, change, classification)
            )

            if shape in ANCHOR_SHAPES:
                anchor_speedups.append(current_ms / candidate_ms)
                liger = pair["liger"]
                if liger.get("skipped") or not _correct(liger):
                    correctness_failures.append(f"{dtype} {shape} liger")
                lower = _speedup_lower_bound(
                    _samples(liger, "fwd_bwd"), _samples(candidate, "fwd_bwd")
                )
                if lower is None:
                    problems.append(f"{dtype} {shape}: raw fwd+bwd samples are missing")
                elif lower <= 1.0:
                    liger_confidence_failures.append(
                        f"{dtype} {shape}: Liger/candidate 95% lower bound={lower:.3f}"
                    )

    if correctness_failures or numerical_regressions:
        status = "REJECT"
        rationale = "candidate failed correctness or exceeded 1.05 times current numerical error"
    elif problems or unstable:
        status = "MORE_DATA"
        rationale = "the production evidence matrix or measurement-quality gate is incomplete"
    else:
        wins = [row for row in comparisons if row.classification == "WIN"]
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
            or liger_confidence_failures
        ):
            status = "DROP"
            rationale = (
                "candidate did not clear the anchor speedup or Liger confidence requirement"
            )
        else:
            status = "READY_FOR_DISPATCH_REVIEW"
            rationale = (
                "candidate has repeatable wins; add only measured shape/dtype dispatch cases, "
                "then rerun the complete operator regression suite"
            )

    return {
        "status": status,
        "rationale": rationale,
        "threshold": threshold,
        "max_cv": max_cv,
        "commits": sorted(commits),
        "problems": problems,
        "correctness_failures": correctness_failures,
        "numerical_regressions": numerical_regressions,
        "unstable": unstable,
        "liger_confidence_failures": liger_confidence_failures,
        "anchor_speedup_floor": min(anchor_speedups) if anchor_speedups else None,
        "anchor_speedup_geomean": (
            math.exp(sum(math.log(value) for value in anchor_speedups) / len(anchor_speedups))
            if anchor_speedups
            else None
        ),
        "comparisons": [row.__dict__ for row in comparisons],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.results]
    decision = evaluate_reports(reports, threshold=args.threshold, max_cv=args.max_cv)
    if args.json:
        print(json.dumps(decision, indent=2, default=str))
    else:
        for row in decision["comparisons"]:
            shape = "x".join(str(x) for x in row["shape"])
            print(
                f"{row['dtype']:8} {shape:20} {row['current_ms']:8.4f} -> "
                f"{row['candidate_ms']:8.4f} ms  {100 * row['change']:+6.1f}%  "
                f"{row['classification']}"
            )
        print(f"\n{decision['status']}: {decision['rationale']}")
        for problem in decision["problems"] + decision["unstable"]:
            print(f"  - {problem}")
        for failure in decision["correctness_failures"] + decision["numerical_regressions"]:
            print(f"  - correctness: {failure}")
        for failure in decision["liger_confidence_failures"]:
            print(f"  - comparison: {failure}")

    return 1 if decision["status"] == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
