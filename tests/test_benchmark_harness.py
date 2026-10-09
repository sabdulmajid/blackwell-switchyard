"""CPU-only tests for benchmark timing, profiling, and process monitoring."""

from __future__ import annotations

import importlib.util
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from bench import bench_backward

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "harness.py"
SPEC = importlib.util.spec_from_file_location("benchmark_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "harness_evaluator", SCRIPT.parents[1] / "scripts" / "evaluate_backward.py"
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = EVALUATOR
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


@pytest.fixture
def cuda_elapsed_time(monkeypatch):
    elapsed_time = Mock()
    event = SimpleNamespace(record=lambda: None, elapsed_time=elapsed_time)
    monkeypatch.setattr(MODULE.torch.cuda, "Event", lambda **_kwargs: event)
    monkeypatch.setattr(MODULE.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(MODULE, "_flush_l2", lambda _device: None)
    monkeypatch.setattr(bench_backward, "measure_latency", MODULE.measure_latency)
    return elapsed_time


@pytest.mark.parametrize("record_samples", [False, True])
@pytest.mark.parametrize(
    ("samples", "median"),
    [([3.0], 3.0), ([7.0, 1.0, 3.0], 3.0), ([7.0, 1.0, 9.0, 3.0], 5.0)],
    ids=["single", "odd", "even"],
)
def test_measure_latency_statistics(cuda_elapsed_time, samples, median, record_samples):
    cuda_elapsed_time.side_effect = samples
    fn = Mock()
    timing = MODULE.measure_latency(
        fn, device=object(), warmup=2, reps=len(samples), record_samples=record_samples
    )

    assert timing.median_ms == median
    assert timing.p10_ms == min(samples)
    assert timing.p90_ms == max(samples)
    assert timing.min_ms == min(samples)
    assert timing.mean_ms == statistics.fmean(samples)
    assert timing.cv == pytest.approx(statistics.pstdev(samples) / statistics.fmean(samples))
    assert timing.reps == len(samples)
    assert timing.warmup == 2
    assert timing.l2_flushed is True
    assert fn.call_count == 2 + len(samples)
    assert cuda_elapsed_time.call_count == len(samples)
    if record_samples:
        assert timing.samples_ms == samples
        assert timing.as_dict()["samples_ms"] == samples
    else:
        assert timing.samples_ms is None
        assert "samples_ms" not in timing.as_dict()


@pytest.mark.parametrize("quick", [True, False], ids=["even-quick", "odd-full"])
def test_interleaved_harness_statistics_match_raw_validator(cuda_elapsed_time, quick):
    trial_count, reps, warmup = (2, 10, 8) if quick else (15, 13, 10)
    count = 2 * trial_count * reps
    samples = [float(1 + (index * 7) % count) for index in range(count)]
    cuda_elapsed_time.side_effect = samples
    summaries, schedule = bench_backward._measure_paired_trials(
        {"current": lambda: None, "candidate": lambda: None},
        device=object(), quick=quick, rotation=0,
    )

    for name, summary in summaries.items():
        for trial in summary["trials"]:
            start = (2 * trial["trial"] + trial["order_in_trial"]) * reps
            assert trial["samples_ms"] == samples[start : start + reps]
        problems = []
        rebuilt = EVALUATOR._timing_from_raw(
            {"impl": name, "backward": summary},
            "backward", schedule, problems, name,
            trial_count=trial_count, reps_per_trial=reps, warmup_per_trial=warmup,
        )
        assert not problems
        assert rebuilt is not None
        assert rebuilt["median_ms"] == summary["median_ms"]
        assert list(rebuilt["trial_medians"].values()) == summary["trial_medians_ms"]
    assert cuda_elapsed_time.call_count == count


def test_gpu_process_monitor_records_a_transient_competitor(monkeypatch):
    samples = iter(
        [
            [SimpleNamespace(pid=os.getpid())],
            [SimpleNamespace(pid=os.getpid()), SimpleNamespace(pid=987654)],
        ]
    )
    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda _uuid: object(),
        nvmlDeviceGetComputeRunningProcesses=lambda _handle: next(samples),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    monitor = MODULE.GPUProcessMonitor(
        "GPU-test",
        report_device_id="device-0123456789abcdef",
        interval_seconds=60.0,
    )
    monitor.start()
    report = monitor.stop()

    assert report["samples"] == 2
    assert report["device_id"] == "device-0123456789abcdef"
    assert report["started_at_utc"].endswith("Z")
    assert report["ended_at_utc"].endswith("Z")
    assert report["first_probe_at_utc"].endswith("Z")
    assert report["last_probe_at_utc"].endswith("Z")
    assert report["probe_attempts"] == 2
    assert 0 <= report["max_probe_gap_seconds"] <= report["maximum_allowed_probe_gap_seconds"]
    assert report["collision_detected"]
    assert report["collision_events"][0]["foreign_process_count"] == 1
    assert not report["probe_errors"]


def test_gpu_process_monitor_requests_immediate_exit_on_collision(monkeypatch):
    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda _uuid: object(),
        nvmlDeviceGetComputeRunningProcesses=lambda _handle: [
            SimpleNamespace(pid=os.getpid()),
            SimpleNamespace(pid=987654),
        ],
    )
    exits = []
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(os, "_exit", exits.append)

    monitor = MODULE.GPUProcessMonitor("GPU-test", interval_seconds=60.0, abort_on_collision=True)
    monitor.start()
    monitor.stop()

    assert exits == [75]


def test_repository_provenance_redacts_external_argv_path(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    inside = repo / "result.json"
    outside = tmp_path / "private" / "result.json"
    monkeypatch.setattr(sys, "argv", ["bench.py", "--out", str(outside), str(inside)])

    provenance = MODULE.repository_provenance(repo)

    assert "input_seed" not in provenance
    assert "query_seed" not in provenance
    assert provenance["argv"] == [
        "bench.py",
        "--out",
        "<external>/result.json",
        "result.json",
    ]


def test_repository_provenance_fails_closed_when_git_fails(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(subprocess, "run", fail)

    try:
        MODULE.repository_provenance(tmp_path)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("repository provenance accepted a failed Git query")


def test_kernel_profiler_separates_compute_and_auxiliary_device_operations(monkeypatch):
    events = [
        SimpleNamespace(
            device_type=MODULE.torch.autograd.DeviceType.CUDA,
            key="test_compute_kernel",
            count=3,
            self_device_time_total=30.0,
        ),
        SimpleNamespace(
            device_type=MODULE.torch.autograd.DeviceType.CUDA,
            key="Memcpy DtoD (Device -> Device)",
            count=3,
            self_device_time_total=6.0,
        ),
        SimpleNamespace(
            device_type=MODULE.torch.autograd.DeviceType.CUDA,
            key="Memset (Device)",
            count=6,
            self_device_time_total=3.0,
        ),
    ]

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def key_averages(self):
            return events

    monkeypatch.setattr(MODULE.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(MODULE.torch.profiler, "profile", lambda **_kwargs: FakeProfile())

    report = MODULE.count_kernels(lambda: None, device=object(), iters=3).as_dict()

    assert report["total_kernels"] == 1
    assert report["total_cuda_us"] == 10.0
    assert set(report["by_name"]) == {"test_compute_kernel"}
    assert report["total_auxiliary_operations"] == 3
    assert report["total_auxiliary_cuda_us"] == 3.0
    assert {
        name: measurement["kind"]
        for name, measurement in report["auxiliary_by_name"].items()
    } == {
        "Memcpy DtoD (Device -> Device)": "memcpy",
        "Memset (Device)": "memset",
    }
