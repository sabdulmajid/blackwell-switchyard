"""CPU-only tests for the backward architecture promotion gate."""

from __future__ import annotations

import importlib.util
import json
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
    offset = (trial // 2) % len(implementations)
    order = implementations[offset:] + implementations[:offset]
    return list(reversed(order)) if trial % 2 else order


def _timing(latency, impl):
    trials = [
        {
            "trial": trial,
            "order_in_trial": _trial_order(trial).index(impl),
            "samples_ms": [latency] * MODULE.FULL_REPS_PER_TRIAL,
            "median_ms": latency,
            "p10_ms": latency,
            "p90_ms": latency,
            "min_ms": latency,
            "mean_ms": latency,
            "cv": 0.0,
            "reps": MODULE.FULL_REPS_PER_TRIAL,
            "warmup": MODULE.FULL_WARMUP_PER_TRIAL,
            "l2_flushed": True,
        }
        for trial in range(MODULE.FULL_TRIAL_COUNT)
    ]
    return {
        "median_ms": latency,
        "p10_ms": latency,
        "p90_ms": latency,
        "min_ms": latency,
        "mean_ms": latency,
        "cv": 0.0,
        "trials": trials,
        "trial_medians_ms": [latency] * MODULE.FULL_TRIAL_COUNT,
        "trial_count": MODULE.FULL_TRIAL_COUNT,
        "reps": MODULE.FULL_TRIAL_COUNT * MODULE.FULL_REPS_PER_TRIAL,
        "warmup_per_trial": MODULE.FULL_WARMUP_PER_TRIAL,
        "l2_flushed": True,
    }


def _set_trial_latency(timing, trial_id, latency):
    trial = timing["trials"][trial_id]
    samples = [latency] * MODULE.FULL_REPS_PER_TRIAL
    trial.update(
        {
            "samples_ms": samples,
            "median_ms": latency,
            "p10_ms": latency,
            "p90_ms": latency,
            "min_ms": latency,
            "mean_ms": latency,
            "cv": 0.0,
        }
    )
    all_samples = [
        value for item in timing["trials"] for value in item["samples_ms"]
    ]
    ordered = sorted(all_samples)
    timing.update(
        {
            "median_ms": statistics.median(all_samples),
            "p10_ms": ordered[int(0.10 * len(ordered))],
            "p90_ms": ordered[int(0.90 * len(ordered))],
            "min_ms": ordered[0],
            "mean_ms": statistics.fmean(all_samples),
            "cv": statistics.pstdev(all_samples) / statistics.fmean(all_samples),
            "trial_medians_ms": [
                statistics.median(item["samples_ms"])
                for item in timing["trials"]
            ],
        }
    )


def _memory(shape, dtype, workspace=0):
    itemsize = 4 if dtype == "float32" else 2
    n, b, t, d = shape
    accounted = (n * b * t * d + b * t * d + d) * itemsize
    return {
        "peak_allocated_bytes": 2 * accounted + workspace,
        "workspace_bytes": workspace,
        "resident_bytes": 2 * accounted,
        "incremental_peak_bytes": accounted + workspace,
        "returned_bytes": 0,
        "accounted_output_bytes": accounted,
        "allocation_count": 3,
    }


def _record(
    impl,
    shape,
    latency,
    *,
    dtype="bfloat16",
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
        "fwd_bwd_memory": _memory(shape, dtype),
    }
    if impl == CANDIDATE:
        itemsize = 4 if dtype == "float32" else 2
        n, _, _, d = shape
        resident = (1 << (n - 1).bit_length()) * (1 << (d - 1).bit_length()) <= 32768
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
                "traffic_model": json.loads(
                    json.dumps(
                        MODULE.backward_traffic_estimate(
                            "cuda_cluster",
                            *shape,
                            itemsize=itemsize,
                            persistent_clusters=min(shape[1] * shape[2], 94),
                        ).as_dict()
                    )
                ),
                "forward_traffic_model": json.loads(
                    json.dumps(
                        MODULE.forward_traffic_estimate(
                            "resident" if resident else "tiled",
                            *shape,
                            itemsize=itemsize,
                            saves_backward_coefficients=True,
                        ).as_dict()
                    )
                ),
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
            }
        )
    return record


def _report(dtype, *, candidate_ms=0.8, candidate_fwd_bwd=None, ok=True, dirty=False):
    results = []
    for shape in MODULE.EXPECTED_FULL_SHAPES:
        results.extend(
            [
                _record("current", shape, 1.0, dtype=dtype),
                _record(
                    CANDIDATE,
                    shape,
                    candidate_ms,
                    dtype=dtype,
                    ok=ok,
                    fwd_bwd_ms=candidate_fwd_bwd,
                ),
                _record("liger", shape, 0.9, dtype=dtype),
            ]
        )
    return {
        "dtype": dtype,
        "schema_version": 2,
        "run_id": "20260905T200000Z" if dtype == "bfloat16" else "20260905T200001Z",
        "run_status": "complete",
        "shape_set": "full",
        "selected_implementations": ["current", CANDIDATE, "liger"],
        "candidate_reachable_from_production": False,
        "correctness_seeds": [0, 1, 2],
        "gpu_preflight": {
            "device_id": "device-0123456789abcdef",
            "foreign_compute_process_count_at_start": 0,
            "busy_override": False,
        },
        "gpu_postflight": {
            "device_id": "device-0123456789abcdef",
            "foreign_compute_process_count_at_end": 0,
        },
        "gpu_process_monitor": {
            "device_id": "device-0123456789abcdef",
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
                        for trial in range(MODULE.FULL_TRIAL_COUNT)
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


def _complete_register_cluster_full_reports():
    candidate = "cuda_register_cluster_full"
    plan = MODULE.get_training_plan(candidate)
    reports = _complete_reports()
    for report in reports:
        dtype = report["dtype"]
        report["selected_implementations"] = ["current", candidate, "liger"]
        for schedule in report["execution_order"]:
            for trials in schedule["metrics"].values():
                for trial in trials:
                    trial["implementations"] = [
                        candidate if name == CANDIDATE else name
                        for name in trial["implementations"]
                    ]
        for case in report["correctness_only"]:
            for item in case["implementations"]:
                if item["impl"] == CANDIDATE:
                    seed = item["seed"]
                    item.clear()
                    item.update(
                        {
                            "impl": candidate,
                            "seed": seed,
                            "training_plan": plan.as_dict(),
                            "skipped": "unsupported plan: fixed specialization",
                        }
                    )
        for index, record in enumerate(report["results"]):
            if record["impl"] != CANDIDATE:
                continue
            shape = MODULE._shape(record)
            supported, reason = MODULE.plan_supports(plan, *shape, dtype)
            if not supported:
                report["results"][index] = {
                    "impl": candidate,
                    "shape": record["shape"],
                    "training_plan": plan.as_dict(),
                    "skipped": f"unsupported plan: {reason}",
                }
                continue
            n, b, t, d = shape
            blocks = 4 if n == 9 else 2
            local_width = d // blocks
            register_sources = 0 if n == 9 else 28
            active_clusters = 48
            backward_dynamic = 4 * (21 * n + local_width) + 2 * local_width
            forward_dynamic = (
                4 * n * 33 + 2 * (n - register_sources) * local_width
            )
            record.update(
                {
                    "impl": candidate,
                    "training_plan": plan.as_dict(),
                    "forward_kernels": {
                        "total_kernels": 1,
                        "by_name": {"register_cluster_forward_kernel": {}},
                    },
                    "backward_kernels": {
                        "total_kernels": 3,
                        "by_name": {"register_cluster_backward_kernel": {}},
                    },
                    "fwd_bwd_kernels": {
                        "total_kernels": 4,
                        "by_name": {
                            "register_cluster_forward_kernel": {},
                            "register_cluster_backward_kernel": {},
                        },
                    },
                    "traffic_model": json.loads(
                        json.dumps(
                            MODULE.backward_traffic_estimate(
                                "cuda_register_cluster",
                                *shape,
                                itemsize=2,
                                persistent_clusters=min(b * t, active_clusters),
                            ).as_dict()
                        )
                    ),
                    "forward_traffic_model": json.loads(
                        json.dumps(
                            MODULE.forward_traffic_estimate(
                                "cuda_register_cluster",
                                *shape,
                                itemsize=2,
                                saves_backward_coefficients=True,
                            ).as_dict()
                        )
                    ),
                    "register_cluster_launch_info": {
                        "active_clusters": active_clusters,
                        "cluster_blocks": blocks,
                        "dynamic_shared_bytes": backward_dynamic,
                        "static_shared_bytes": 1024,
                        "registers_per_thread": 128,
                        "threads_per_block": 512,
                        "multiprocessors": 192,
                        "max_shared_bytes": 101_376,
                    },
                    "register_cluster_forward_launch_info": {
                        "active_clusters": active_clusters,
                        "cluster_blocks": blocks,
                        "dynamic_shared_bytes": forward_dynamic,
                        "static_shared_bytes": 1024,
                        "registers_per_thread": {
                            (9, "bfloat16"): 96,
                            (9, "float16"): 96,
                            (32, "bfloat16"): 103,
                            (32, "float16"): 109,
                        }[(n, dtype)],
                        "threads_per_block": 512,
                        "multiprocessors": 192,
                        "max_shared_bytes": 101_376,
                    },
                }
            )
    return reports


def test_complete_repeatable_wins_are_ready_for_dispatch_review():
    decision = MODULE.evaluate_reports(_complete_reports(), candidate=CANDIDATE)
    assert decision["status"] == "READY_FOR_DISPATCH_REVIEW"
    assert all(row["classification"] == "WIN" for row in decision["comparisons"])


def test_complete_one_read_plan_reconstructs_and_selects_only_supported_cells():
    decision = MODULE.evaluate_reports(
        _complete_register_cluster_full_reports(),
        candidate="cuda_register_cluster_full",
    )
    assert decision["status"] == "READY_FOR_DISPATCH_REVIEW"
    assert len(decision["eligible_dispatches"]) == 4
    assert {
        (cell["shape"]["n"], cell["shape"]["d"])
        for cell in decision["eligible_dispatches"]
    } == {(9, 8192), (32, 2048)}


def test_cuda_plan_does_not_require_unsupported_float32_run():
    decision = MODULE.evaluate_reports(_complete_reports(), candidate=CANDIDATE)
    assert decision["required_dtypes"] == ["bfloat16", "float16"]
    assert not any("float32" in problem for problem in decision["problems"])


def test_kernel_contract_counts_support_work_not_only_main_kernels():
    assert MODULE._expected_kernel_contract("cuda_cluster", "bfloat16")[1:] == (3, 4)
    assert MODULE._expected_kernel_contract("cuda_register_cluster", "float16") == (
        ("register_cluster_backward_kernel",),
        3,
        4,
    )
    assert MODULE._expected_kernel_contract("serial_saved_partials_t16", "float16")[1:] == (
        3,
        4,
    )
    assert MODULE._expected_kernel_contract("serial_saved_partials_t16", "float32")[1:] == (
        2,
        3,
    )


def test_register_cluster_launch_contract_rejects_each_resource_mismatch():
    shape = (9, 1, 4096, 8192)
    blocks = 4
    local_width = shape[3] // blocks
    dynamic = 4 * (21 * shape[0] + local_width) + 2 * local_width
    valid = {
        "active_clusters": 1,
        "cluster_blocks": blocks,
        "dynamic_shared_bytes": dynamic,
        "static_shared_bytes": 1024,
        "registers_per_thread": 128,
        "threads_per_block": 512,
        "multiprocessors": 192,
        "max_shared_bytes": dynamic + 1024,
    }
    problems = []
    MODULE._check_register_cluster_launch_info(shape, valid, problems, "probe")
    assert problems == []

    invalid_values = {
        "active_clusters": 0,
        "cluster_blocks": 2,
        "dynamic_shared_bytes": dynamic - 4,
        "static_shared_bytes": 0,
        "registers_per_thread": 109,
        "threads_per_block": 256,
        "multiprocessors": 0,
        "max_shared_bytes": dynamic,
    }
    for field, value in invalid_values.items():
        launch_info = valid | {field: value}
        field_problems = []
        MODULE._check_register_cluster_launch_info(
            shape, launch_info, field_problems, "probe"
        )
        assert field_problems, field


def test_register_cluster_forward_launch_contract_is_exact_per_dtype():
    shape = (32, 1, 4096, 2048)
    valid = {
        "active_clusters": 1,
        "cluster_blocks": 2,
        "dynamic_shared_bytes": 12_416,
        "static_shared_bytes": 1024,
        "registers_per_thread": 109,
        "threads_per_block": 512,
        "multiprocessors": 192,
        "max_shared_bytes": 101_376,
    }
    problems = []
    MODULE._check_register_cluster_forward_launch_info(
        shape, "float16", valid, problems, "probe"
    )
    assert problems == []
    invalid = valid | {"registers_per_thread": 108}
    MODULE._check_register_cluster_forward_launch_info(
        shape, "float16", invalid, problems, "probe"
    )
    assert any("register count" in problem for problem in problems)


def test_incomplete_dtype_matrix_requests_more_data():
    decision = MODULE.evaluate_reports([_report("bfloat16")], candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("missing dtype" in problem for problem in decision["problems"])


def test_duplicate_masked_tail_row_requests_more_data():
    reports = _complete_reports()
    reports[0]["correctness_only"][0]["implementations"].append(
        reports[0]["correctness_only"][0]["implementations"][0].copy()
    )
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any(
        "masked-tail implementation/seed matrix" in problem
        for problem in decision["problems"]
    )


def test_correctness_failure_rejects_candidate():
    reports = [_report("bfloat16"), _report("float16", ok=False)]
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "REJECT"
    assert decision["correctness_failures"]
    assert decision["eligible_dispatches"] == []


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


def test_one_material_cell_win_is_preserved_for_exact_dispatch():
    reports = _complete_reports(candidate_ms=0.96)
    winning_shape = (9, 1, 4096, 8192)
    measured = next(
        record
        for record in reports[0]["results"]
        if record["impl"] == CANDIDATE and MODULE._shape(record) == winning_shape
    )
    for metric in ("forward", "backward", "fwd_bwd"):
        measured[metric] = _timing(0.8, CANDIDATE)

    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)

    assert decision["status"] == "READY_FOR_DISPATCH_REVIEW"
    assert [
        (cell["dtype"], tuple(cell["shape"].values()))
        for cell in decision["eligible_dispatches"]
    ] == [("bfloat16", winning_shape)]


def test_dirty_or_unstable_measurements_cannot_promote():
    reports = [_report("bfloat16", dirty=True), _report("float16")]
    timing = reports[0]["results"][1]["backward"]
    for trial in timing["trials"]:
        trial["samples_ms"] = [0.4] * 6 + [1.2] * 7
        ordered = sorted(trial["samples_ms"])
        trial.update(
            {
                "median_ms": statistics.median(ordered),
                "p10_ms": ordered[int(0.10 * len(ordered))],
                "p90_ms": ordered[int(0.90 * len(ordered))],
                "min_ms": ordered[0],
                "mean_ms": statistics.fmean(ordered),
                "cv": statistics.pstdev(ordered) / statistics.fmean(ordered),
            }
        )
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
    assert decision["eligible_dispatches"] == []


def test_campaign_wide_sign_gate_controls_familywise_error():
    assert MODULE.CAMPAIGN_DISPATCH_HYPOTHESES == 264
    assert MODULE.MAX_CAMPAIGN_ATTEMPTS == 3
    assert MODULE.FAMILYWISE_SIGN_ERROR_BOUND < 0.05


def test_unbalanced_trial_order_cannot_promote():
    reports = _complete_reports()
    fixed_order = ["current", CANDIDATE, "liger"]
    for report in reports:
        for shape_schedule in report["execution_order"]:
            for trials in shape_schedule["metrics"].values():
                for trial in trials:
                    trial["implementations"] = fixed_order
        for record in report["results"]:
            for metric in ("forward", "backward", "fwd_bwd"):
                for trial in record[metric]["trials"]:
                    trial["order_in_trial"] = fixed_order.index(record["impl"])
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert decision["eligible_dispatches"] == []
    assert any("order does not balance" in problem for problem in decision["problems"])


def test_one_losing_training_trial_blocks_that_dispatch_cell():
    reports = _complete_reports()
    shape = (9, 1, 4096, 8192)
    candidate = next(
        record
        for record in reports[0]["results"]
        if record["impl"] == CANDIDATE and MODULE._shape(record) == shape
    )
    _set_trial_latency(candidate["fwd_bwd"], trial_id=0, latency=0.95)

    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    cell = next(
        item
        for item in decision["dispatch_cells"]
        if item["dtype"] == "bfloat16"
        and tuple(item["shape"].values()) == shape
    )
    assert cell["status"] == "FALLBACK"
    assert cell["all_liger_trials_win"] is False
    assert any("all 15 trials" in reason for reason in cell["reasons"])


def test_incomplete_memory_evidence_cannot_promote():
    reports = _complete_reports()
    for report in reports:
        for record in report["results"]:
            if record["impl"] in {"current", CANDIDATE}:
                record["fwd_bwd_memory"] = {}
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert decision["eligible_dispatches"] == []
    assert any("memory record" in problem for problem in decision["problems"])


def test_stored_summary_tampering_cannot_promote():
    reports = _complete_reports()
    reports[0]["results"][1]["backward"]["median_ms"] = 0.01
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("disagrees with raw samples" in problem for problem in decision["problems"])


def test_traffic_summary_tampering_cannot_promote():
    reports = _complete_reports()
    reports[0]["results"][1]["traffic_model"]["logical_large_tensor_bytes"] += 1
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("traffic model does not reconstruct" in problem for problem in decision["problems"])


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
    reports[1]["gpu_preflight"]["device_id"] = "device-fedcba9876543210"
    reports[1]["gpu_postflight"]["device_id"] = "device-fedcba9876543210"
    reports[0]["gpu_postflight"]["foreign_compute_process_count_at_end"] = 1
    decision = MODULE.evaluate_reports(reports, candidate=CANDIDATE)
    assert decision["status"] == "MORE_DATA"
    assert any("same campaign device ID" in problem for problem in decision["problems"])
    assert any("appeared during" in problem for problem in decision["problems"])
