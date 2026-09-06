"""CPU-only tests for benchmark process-contamination monitoring."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "harness.py"
SPEC = importlib.util.spec_from_file_location("benchmark_harness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
