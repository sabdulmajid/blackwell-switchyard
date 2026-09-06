"""CPU-only tests for the backward architecture promotion gate."""

from __future__ import annotations

import importlib.util
import statistics
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


def _trial_order(trial):
    implementations = ["current", CANDIDATE, "liger"]
    offset = trial % len(implementations)
    return implementations[offset:] + implementations[:offset]


def _timing(latency, impl):
    trials = [
        {
            "trial": trial,
            "order_in_trial": _trial_order(trial).index(impl),
            "samples_ms": [latency] * 40,
            "median_ms": latency,
            "p10_ms": latency,
            "p90_ms": latency,
            "min_ms": latency,
            "mean_ms": latency,
            "cv": 0.0,
            "reps": 40,
            "warmup": 25,
            "l2_flushed": True,
        }
        for trial in range(5)
    ]
    return {
        "median_ms": latency,
        "p10_ms": latency,
        "p90_ms": latency,
        "min_ms": latency,
        "mean_ms": latency,
        "cv": 0.0,
        "trials": trials,
        "trial_medians_ms": [latency] * 5,
        "trial_count": 5,
        "reps": 200,
        "warmup_per_trial": 25,
        "l2_flushed": True,
    }


def _record(
    impl,
    shape,
    latency,
    *,
    ok=True,
    error=1e-3,
    forward_ms=None,
    backward_ms=None,
    fwd_bwd_ms=None,
):
    record = {
        "impl": impl,
        "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
        "correctness": {
            name: {"ok": ok, "rel_l2": error} for name in ("output", "dv", "dw")
        },
        "forward": _timing(forward_ms or latency, impl),
        "backward": _timing(backward_ms or latency, impl),
        "fwd_bwd": _timing(fwd_bwd_ms or latency, impl),
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
                    "cluster_blocks": 2,
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


def _report(dtype, *, candidate_ms=0.8, candidate_fwd_bwd=None, ok=True, dirty=False):
    results = []
    for shape in MODULE.EXPECTED_FULL_SHAPES:
        results.extend(
            [
                _record("current", shape, 1.0),
                _record(
                    CANDIDATE,
                    shape,
                    candidate_ms,
                    ok=ok,
                    fwd_bwd_ms=candidate_fwd_bwd,
                ),
                _record("liger", shape, 0.9),
            ]
        )
    return {
        "dtype": dtype,
        "schema_version": 2,
        "run_id": f"run-{dtype}",
        "run_status": "complete",
        "shape_set": "full",
        "selected_implementations": ["current", CANDIDATE, "liger"],
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
        "gpu_process_monitor": {
            "device_uuid": "GPU-test",
            "samples": 100,
            "interval_seconds": 0.25,
            "duration_seconds": 25.0,
            "collision_detected": False,
            "collision_events": [],
            "probe_errors": [],
        },
        "provenance": {
            "repository_commit": "abc123",
            "repository_tree": "tree123",
            "repository_branch": "codex/test",
            "worktree_dirty": dirty,
            "tracked_worktree_dirty": dirty,
            "third_party_commits": {"Liger-Kernel": MODULE.PINNED_LIGER_COMMIT},
            "third_party_dirty": {"Liger-Kernel": False},
            "argv": [],
        },
        "comparators": {
            "liger": {
                "commit": MODULE.PINNED_LIGER_COMMIT,
                "worktree_dirty": False,
                "under_pinned_checkout": True,
                "source_path": "src/liger_kernel/ops/attn_res.py",
                "source_sha256": MODULE.PINNED_LIGER_SOURCE_SHA256,
            }
        },
        "results": results,
        "execution_order": [
            {
                "shape": dict(zip(("n", "b", "t", "d"), shape, strict=True)),
                "metrics": {
                    metric: [
                        {
                            "trial": trial,
                            "implementations": _trial_order(trial),
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
    timing = reports[0]["results"][1]["backward"]
    for trial in timing["trials"]:
        trial["samples_ms"] = [0.4] * 20 + [1.2] * 20
    samples = [value for trial in timing["trials"] for value in trial["samples_ms"]]
    timing["median_ms"] = statistics.median(samples)
    timing["trial_medians_ms"] = [
        statistics.median(trial["samples_ms"]) for trial in timing["trials"]
    ]
    timing["cv"] = statistics.pstdev(samples) / statistics.fmean(samples)
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert decision["problems"]
    assert decision["unstable"]


def test_stored_summary_tampering_cannot_promote():
    reports = _complete_reports()
    reports[0]["results"][1]["backward"]["median_ms"] = 0.01
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("disagrees with raw samples" in problem for problem in decision["problems"])


def test_per_trial_summary_and_method_tampering_cannot_promote():
    reports = _complete_reports()
    trial = reports[0]["results"][1]["backward"]["trials"][0]
    trial["p90_ms"] = 0.01
    trial["warmup"] = 0
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("trial 0 summary is wrong" in problem for problem in decision["problems"])
    assert any("trial 0 method is wrong" in problem for problem in decision["problems"])


def test_faster_backward_cannot_hide_slower_complete_training():
    decision = MODULE.evaluate_reports(
        _complete_reports(candidate_ms=0.8, candidate_fwd_bwd=1.14),
        candidate=CANDIDATE,
    )
    assert decision["status"] == "DROP"
    assert decision["performance_regressions"]


def test_missing_commit_or_liger_provenance_cannot_promote():
    reports = _complete_reports()
    reports[1]["provenance"].pop("repository_commit")
    reports[0]["comparators"].pop("liger")
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("nonempty repository commit" in problem for problem in decision["problems"])
    assert any("Liger source provenance" in problem for problem in decision["problems"])


def test_wrong_liger_source_hash_or_empty_monitor_cannot_promote():
    reports = _complete_reports()
    reports[0]["comparators"]["liger"]["source_sha256"] = "wrong"
    reports[1]["gpu_process_monitor"]["samples"] = 0
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("Liger source provenance" in problem for problem in decision["problems"])
    assert any("monitor was incomplete" in problem for problem in decision["problems"])


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
