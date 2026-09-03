"""CPU-only tests for the backward architecture promotion gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_backward.py"
SPEC = importlib.util.spec_from_file_location("evaluate_backward", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CANDIDATE = "cuda_cluster"
CANDIDATE_PLAN = MODULE.get_training_plan(CANDIDATE).as_dict()


def _timing(latency, *, cv=0.01):
    trials = [
        {"trial": trial, "order_in_trial": trial % 3, "samples_ms": [latency] * 40}
        for trial in range(5)
    ]
    return {
        "median_ms": latency,
        "cv": cv,
        "trials": trials,
        "trial_medians_ms": [latency] * 5,
    }


def _record(impl, shape, latency, *, ok=True, cv=0.01, error=1e-3):
    record = {
        "impl": impl,
        "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
        "correctness": {
            name: {"ok": ok, "rel_l2": error} for name in ("output", "dv", "dw")
        },
        "forward": _timing(latency, cv=cv),
        "backward": _timing(latency, cv=cv),
        "fwd_bwd": _timing(latency, cv=cv),
        "correctness_by_seed": [
            {
                "seed": seed,
                "report": {
                    name: {"ok": ok, "rel_l2": error}
                    for name in ("output", "dv", "dw")
                },
            }
            for seed in (0, 1, 2)
        ],
    }
    if impl == CANDIDATE:
        record.update(
            {
                "training_plan": CANDIDATE_PLAN,
                "backward_kernels": {
                    "total_kernels": 3,
                    "by_name": {"feature_cluster_backward_kernel": {}},
                },
                "fwd_bwd_kernels": {
                    "total_kernels": 4,
                    "by_name": {
                        "_fwd_tiled": {},
                        "feature_cluster_backward_kernel": {},
                    },
                },
                "traffic_model": {},
                "fwd_bwd_memory": {"workspace_bytes": 0},
                "cluster_launch_info": {
                    "active_clusters": 94,
                    "dynamic_shared_bytes": 80_000,
                    "static_shared_bytes": 1024,
                    "max_shared_bytes": 101_376,
                },
            }
        )
    if impl == "current":
        n, _, _, d = shape
        resident = (1 << (n - 1).bit_length()) * (1 << (d - 1).bit_length()) <= 32768
        kernel_names = ["_bwd_resident"] if resident else ["_bwd_stats", "_bwd_apply"]
        record.update(
            {
                "backward_kernels": {
                    "total_kernels": len(kernel_names) + 2,
                    "by_name": {name: {} for name in kernel_names},
                },
                "fwd_bwd_kernels": {
                    "total_kernels": len(kernel_names) + 3,
                    "by_name": {"_fwd": {}, **{name: {} for name in kernel_names}},
                },
                "fwd_bwd_memory": {"workspace_bytes": 0},
            }
        )
    return record


def _report(dtype, *, candidate_ms=0.8, ok=True, dirty=False):
    results = []
    for shape in MODULE.EXPECTED_FULL_SHAPES:
        results.extend(
            [
                _record("current", shape, 1.0),
                _record(CANDIDATE, shape, candidate_ms, ok=ok),
                _record("liger", shape, 0.9),
            ]
        )
    return {
        "dtype": dtype,
        "schema_version": 2,
        "run_id": f"run-{dtype}",
        "shape_set": "full",
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "gpu_preflight": {
            "resolved_uuid": "GPU-test",
            "compute_processes_at_start": [],
            "busy_override": False,
        },
        "gpu_postflight": {
            "resolved_uuid": "GPU-test",
            "compute_processes_at_end": [],
        },
        "provenance": {
            "repository_commit": "abc123",
            "worktree_dirty": dirty,
            "tracked_worktree_dirty": dirty,
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": [],
        },
        "results": results,
        "execution_order": [
            {
                "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
                "metrics": {
                    metric: [
                        {
                            "trial": trial,
                            "implementations": (
                                ["current", CANDIDATE, "liger"][trial % 3 :]
                                + ["current", CANDIDATE, "liger"][: trial % 3]
                            ),
                        }
                        for trial in range(5)
                    ]
                    for metric in ("forward", "backward", "fwd_bwd")
                },
            }
            for shape in MODULE.EXPECTED_FULL_SHAPES
        ],
        "correctness_only": [
            {
                "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
                "implementations": [
                    {
                        "impl": impl,
                        "seed": seed,
                        "correctness": {
                            name: {"ok": True} for name in ("output", "dv", "dw")
                        },
                    }
                    for impl in ("current", CANDIDATE, "liger")
                    for seed in (0, 1, 2)
                ],
            }
            for shape in MODULE.MASKED_TAIL_SHAPES
        ],
    }


def _complete_reports(**kwargs):
    return [_report(dtype, **kwargs) for dtype in ("bfloat16", "float16")]


def test_complete_repeatable_wins_are_ready_for_dispatch_review():
    decision = MODULE.evaluate_reports(_complete_reports(), candidate=CANDIDATE)
    assert decision["status"] == "READY_FOR_DISPATCH_REVIEW"
    assert all(row["classification"] == "WIN" for row in decision["comparisons"])


def test_cuda_plan_does_not_require_unsupported_float32_run():
    decision = MODULE.evaluate_reports(_complete_reports(), candidate=CANDIDATE)
    assert decision["required_dtypes"] == ["bfloat16", "float16"]
    assert not any("float32" in problem for problem in decision["problems"])


def test_kernel_contract_counts_support_work_not_only_main_kernels():
    assert MODULE._expected_kernel_contract("cuda_cluster", "bfloat16")[1:] == (3, 4)
    assert MODULE._expected_kernel_contract("serial_saved_partials_t16", "float16")[1:] == (
        3,
        4,
    )
    assert MODULE._expected_kernel_contract("serial_saved_partials_t16", "float32")[1:] == (
        2,
        3,
    )


def test_incomplete_dtype_matrix_requests_more_data():
    decision = MODULE.evaluate_reports([_report("bfloat16")], candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("missing dtype" in problem for problem in decision["problems"])


def test_correctness_failure_rejects_candidate():
    reports = [_report("bfloat16"), _report("float16", ok=False)]
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "REJECT"
    assert decision["correctness_failures"]


def test_comparator_failure_requests_new_data_instead_of_rejecting_candidate():
    reports = _complete_reports()
    for report in reports:
        liger = next(record for record in report["results"] if record["impl"] == "liger")
        liger["correctness"]["dw"]["ok"] = False
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert not decision["correctness_failures"]
    assert any("comparator correctness" in problem for problem in decision["problems"])


def test_candidate_without_material_wins_is_dropped():
    decision = MODULE.evaluate_reports(
        _complete_reports(candidate_ms=0.96), candidate=CANDIDATE
    )
    assert decision["status"] == "DROP"


def test_dirty_or_unstable_measurements_cannot_promote():
    reports = [_report("bfloat16", dirty=True), _report("float16")]
    reports[0]["results"][1]["backward"]["cv"] = 0.2
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert decision["problems"]
    assert decision["unstable"]


def test_duplicate_dtype_or_missing_kernel_identity_cannot_promote():
    reports = _complete_reports()
    reports.append(_report("bfloat16"))
    reports[-1]["results"][1]["backward_kernels"]["by_name"] = {"wrong_kernel": {}}
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("exactly one report" in problem for problem in decision["problems"])
    assert any("did not contain" in problem for problem in decision["problems"])


def test_gpu_identity_and_postflight_are_mandatory():
    reports = _complete_reports()
    reports[1]["gpu_preflight"]["resolved_uuid"] = "GPU-other"
    reports[1]["gpu_postflight"]["resolved_uuid"] = "GPU-other"
    reports[0]["gpu_postflight"]["compute_processes_at_end"] = ["123, other, 1024"]
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("same physical GPU" in problem for problem in decision["problems"])
    assert any("appeared during" in problem for problem in decision["problems"])
