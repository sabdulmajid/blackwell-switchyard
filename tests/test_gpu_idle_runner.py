"""CPU-only tests for the guarded GPU idle runner."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_when_gpu_idle.py"
SPEC = importlib.util.spec_from_file_location("run_when_gpu_idle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _recovery_context(attempt: int = 3) -> dict:
    return {
        "target_uuid": "GPU-test",
        "attempt": attempt,
        "not_before": "2026-01-01T00:00:00+00:00",
        "idle_seconds": 1800,
        "wait_poll_seconds": 60,
        "watchdog_seconds": 0.25,
        "finalize_seconds": 1800,
        "idle_started_at": "2026-01-01T00:00:00+00:00",
        "launch_at": "2026-01-01T00:30:00+00:00",
        "idle_probe_count": 31,
        "max_idle_gpu_utilization_percent": 0,
        "max_idle_memory_mib": 8,
        "gpu_count": 2,
    }


def test_nvidia_smi_parsers_keep_physical_identity():
    assert MODULE._parse_inventory("0, GPU-a\n1, GPU-b\n") == {
        0: "GPU-a",
        1: "GPU-b",
    }
    assert MODULE._parse_compute_apps("GPU-b, 42\n") == [
        {"gpu_uuid": "GPU-b", "pid": 42}
    ]
    activity = MODULE._parse_activity("0, 0, 2\n1, 0, 3\n")
    assert activity[1] == {"utilization_percent": 0, "memory_used_mib": 3}
    assert MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    activity[0]["utilization_percent"] = 1
    assert not MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    activity[0] = {"utilization_percent": 0, "memory_used_mib": 65}
    assert not MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    assert not MODULE._is_idle_activity({0: activity[0]}, {0: "GPU-a", 1: "GPU-b"})


def test_descendant_check_walks_process_tree_and_stops_cycles():
    parents = {30: 20, 20: 10, 10: 1, 40: 40}
    lookup = parents.get
    assert MODULE._is_descendant(30, 10, lookup)
    assert not MODULE._is_descendant(30, 11, lookup)
    assert not MODULE._is_descendant(40, 10, lookup)


@pytest.mark.parametrize(
    "text",
    ["", "index, uuid\n", "0\n", "x, GPU-a\n", "0, not-a-uuid\n", "0, GPU-a\n0, GPU-b\n"],
)
def test_inventory_parser_fails_closed(text):
    with pytest.raises(ValueError):
        MODULE._parse_inventory(text)


@pytest.mark.parametrize(
    "text",
    ["GPU-a, not-a-pid\n", "GPU-a, 42, python\n", "not-a-uuid, 42\n"],
)
def test_compute_process_parser_fails_closed(text):
    with pytest.raises(ValueError):
        MODULE._parse_compute_apps(text)


@pytest.mark.parametrize(
    "text",
    ["", "0, 0\n", "0, busy, 2\n", "0, 101, 2\n", "0, 0, 2\n0, 0, 2\n"],
)
def test_activity_parser_fails_closed(text):
    with pytest.raises(ValueError):
        MODULE._parse_activity(text)


def test_state_recorder_resumes_attempt_count_without_process_data(tmp_path):
    state = tmp_path / "state.json"
    recorder = MODULE.StateRecorder(state, "campaign-a")
    recorder.write("launching_workload", attempt=2, target_uuid="GPU-test")
    recorder.set_recovery_context(_recovery_context(attempt=2))
    resumed = MODULE.StateRecorder(state, "campaign-a")
    assert resumed.attempts == 2
    assert resumed.last_phase == "launching_workload"
    assert resumed.public_device_id == recorder.public_device_id
    assert MODULE.PUBLIC_DEVICE_ID_PATTERN.fullmatch(resumed.public_device_id)
    assert resumed.recovery_context == _recovery_context(attempt=2)
    assert "pid" not in state.read_text().lower()
    with pytest.raises(ValueError, match="different campaign"):
        MODULE.StateRecorder(state, "campaign-b")


def test_state_recorder_rejects_malformed_history(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"campaign_identity": "campaign-a", "events": "not-a-list"})
    )
    with pytest.raises(ValueError, match="event history"):
        MODULE.StateRecorder(state, "campaign-a")


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.pop("gpu_count"),
        lambda value: value.update(attempt=True),
        lambda value: value.update(target_uuid="not-a-gpu"),
        lambda value: value.update(launch_at="missing-offset"),
        lambda value: value.update(max_idle_memory_mib=65),
    ],
)
def test_recovery_context_fails_closed(change):
    context = _recovery_context()
    change(context)
    with pytest.raises(ValueError, match="recovery"):
        MODULE._validate_recovery_context(context)


def test_recovery_environment_contains_only_reconstructable_guard_values():
    context = _recovery_context()
    environment = MODULE._recovery_environment(
        {"PATH": "/bin", "UNRELATED": "kept"}, context, "device-0123456789abcdef"
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert environment["SWITCHYARD_RUN_ATTEMPT"] == "3"
    assert environment["SWITCHYARD_GUARD_IDLE_PROBE_COUNT"] == "31"
    assert environment["SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB"] == "8"
    assert environment["SWITCHYARD_PUBLIC_DEVICE_ID"] == "device-0123456789abcdef"
    assert environment["UNRELATED"] == "kept"


def test_main_recovers_completed_gpu_phase_before_inventory_or_attempt_limit(
    tmp_path, monkeypatch
):
    command = ["/bin/true"]
    state = tmp_path / "state.json"
    log = tmp_path / "runner.log"
    lock = tmp_path / "runner.lock"
    marker = tmp_path / "gpu_complete"
    cwd = tmp_path / "worktree"
    cwd.mkdir()
    parsed = Namespace(
        not_before=0.0,
        gpu_index=1,
        idle_seconds=1800,
        wait_poll_seconds=60,
        watchdog_seconds=0.25,
        finalize_seconds=1800,
        deadline_hours=36.0,
        max_attempts=3,
        state=state,
        log=log,
        lock=lock,
        gpu_lock_dir=tmp_path,
        gpu_complete_marker=marker,
        cwd=cwd,
    )
    recorder = MODULE.StateRecorder(state, MODULE._campaign_identity(parsed, command))
    recorder.write("gpu_phase_complete", attempt=3)
    recorder.set_recovery_context(_recovery_context())
    marker.touch()

    observed = {}

    def recover(_command, **kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(MODULE, "_run_cpu_recovery", recover)
    monkeypatch.setattr(
        MODULE, "_inventory", lambda: pytest.fail("GPU inventory must not be queried")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--not-before",
            "1970-01-01T00:00:00+00:00",
            "--gpu-index",
            "1",
            "--idle-seconds",
            "1800",
            "--wait-poll-seconds",
            "60",
            "--watchdog-seconds",
            "0.25",
            "--finalize-seconds",
            "1800",
            "--deadline-hours",
            "36",
            "--max-attempts",
            "3",
            "--state",
            str(state),
            "--log",
            str(log),
            "--lock",
            str(lock),
            "--gpu-lock-dir",
            str(tmp_path),
            "--gpu-complete-marker",
            str(marker),
            "--cwd",
            str(cwd),
            "--",
            *command,
        ],
    )
    assert MODULE.main() == 0
    assert observed["attempt"] == 3
    assert observed["environment"]["SWITCHYARD_RUN_ATTEMPT"] == "3"
    assert observed["inherited_fds"]


def test_runner_source_exports_idle_attestation_without_process_ids():
    source = SCRIPT.read_text()
    assert "SWITCHYARD_GUARD_IDLE_STARTED_AT" in source
    assert "SWITCHYARD_GUARD_LAUNCH_AT" in source
    assert "SWITCHYARD_GUARD_IDLE_PROBE_COUNT" in source
    assert "SWITCHYARD_GUARD_MAX_IDLE_GPU_UTILIZATION_PERCENT" in source
    assert "SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB" in source
    assert "SWITCHYARD_GUARD_GPU_COUNT" in source
    assert "SWITCHYARD_PUBLIC_DEVICE_ID" in source
    assert "pass_fds=(lock_handle.fileno(), gpu_lock_handle.fileno())" in source


def test_cpu_recovery_disables_gpu_visibility_and_preserves_attempt(tmp_path):
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    result = MODULE._run_cpu_recovery(
        ["/bin/sh", "-c", 'test -z "$CUDA_VISIBLE_DEVICES"'],
        cwd=tmp_path,
        environment={**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-private"},
        log_path=tmp_path / "runner.log",
        recorder=recorder,
        attempt=3,
        deadline=time.time() + 10,
        finalize_seconds=5,
        retry_seconds=1,
    )
    assert result == 0
    assert recorder.attempts == 3
    assert recorder.last_phase == "complete"
