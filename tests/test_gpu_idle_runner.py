"""CPU-only tests for the guarded GPU idle runner."""

from __future__ import annotations

import importlib.util
import json
import os
import select
import signal
import stat
import subprocess
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
        "idle_activity_scope": "target_gpu",
        "process_scope": "all_gpus",
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
        "max_idle_pcie_rx_kib_per_second": 2048,
        "max_idle_pcie_tx_kib_per_second": 4096,
        "max_idle_probe_gap_seconds": 60.1,
        "max_global_compute_process_count": 0,
        "max_non_target_gpu_utilization_percent": 100,
        "max_non_target_memory_mib": 2,
        "max_non_target_pcie_rx_kib_per_second": 1024,
        "max_non_target_pcie_tx_kib_per_second": 2048,
        "maximum_allowed_pcie_kib_per_second": 65536,
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
    activity = MODULE._merge_pcie_throughput(
        MODULE._parse_activity("0, 0, 2\n1, 0, 3\n"),
        {0: (0, 0), 1: (0, 0)},
    )
    assert activity[1] == {
        "utilization_percent": 0,
        "memory_used_mib": 3,
        "pcie_rx_kib_per_second": 0,
        "pcie_tx_kib_per_second": 0,
    }
    assert MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    activity[0]["utilization_percent"] = 1
    assert not MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    activity[0] = {"utilization_percent": 0, "memory_used_mib": 65}
    assert not MODULE._is_idle_activity(activity, {0: "GPU-a", 1: "GPU-b"})
    assert not MODULE._is_idle_activity({0: activity[0]}, {0: "GPU-a", 1: "GPU-b"})

    activity = MODULE._merge_pcie_throughput(
        MODULE._parse_activity("0, 100, 2\n1, 0, 3\n"),
        {0: (0, 0), 1: (0, 0)},
    )
    assert MODULE._is_idle_activity(
        activity,
        {0: "GPU-a", 1: "GPU-b"},
        activity_scope="target_gpu",
        target_index=1,
    )
    activity[1]["utilization_percent"] = 1
    assert not MODULE._is_idle_activity(
        activity,
        {0: "GPU-a", 1: "GPU-b"},
        activity_scope="target_gpu",
        target_index=1,
    )
    activity[1]["utilization_percent"] = 0
    activity[0]["pcie_rx_kib_per_second"] = (
        MODULE.MAX_IDLE_PCIE_KIB_PER_SECOND + 1
    )
    assert not MODULE._is_idle_activity(
        activity,
        {0: "GPU-a", 1: "GPU-b"},
        activity_scope="target_gpu",
        target_index=1,
    )
    assert not MODULE._is_idle_activity(
        {1: activity[1]},
        {0: "GPU-a", 1: "GPU-b"},
        activity_scope="target_gpu",
        target_index=1,
    )


def test_descendant_check_walks_process_tree_and_stops_cycles():
    parents = {30: 20, 20: 10, 10: 1, 40: 40}
    lookup = parents.get
    assert MODULE._is_descendant(30, 10, lookup)
    assert not MODULE._is_descendant(30, 11, lookup)
    assert not MODULE._is_descendant(40, 10, lookup)


def test_watchdog_gap_includes_launch_to_first_probe_interval():
    probe_at, maximum = MODULE._record_probe_gap(100.0, 0.0, now=130.0)
    assert probe_at == 130.0
    assert maximum == 30.0


def test_supervisor_cleanup_terminates_the_active_process_group(tmp_path):
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    MODULE._set_active_process(process, recorder)
    MODULE._cleanup_active_process()
    assert process.poll() is not None
    assert recorder.last_phase == "stopping_workload"


def test_terminate_group_rejects_a_workload_without_its_own_session(tmp_path):
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    process = subprocess.Popen(["sleep", "60"])
    try:
        with pytest.raises(RuntimeError, match="must own the session"):
            MODULE._terminate_group(process, recorder, "test cleanup")
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


def test_terminate_group_never_signals_a_reaped_workload(tmp_path, monkeypatch):
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait(timeout=5)
    monkeypatch.setattr(
        MODULE.os, "killpg", lambda *_: pytest.fail("must not signal a recycled group")
    )
    MODULE._terminate_group(process, recorder, "test cleanup")


@pytest.mark.parametrize("launch", ["direct", "exited_parent", "timeout"])
def test_cleanup_kills_stubborn_child_after_parent_exits(tmp_path, launch):
    # Adopt only this test's orphan so it can be reaped even without a helpful init.
    libc = MODULE.ctypes.CDLL(None, use_errno=True)
    previous = MODULE.ctypes.c_int()
    assert libc.prctl(37, MODULE.ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    child_code = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(os.getpid(), flush=True)
time.sleep(60)
"""
    parent_code = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
subprocess.Popen([sys.executable, "-c", sys.argv[1]])
time.sleep(60)
"""
    command = [sys.executable, "-c", parent_code, child_code]
    if launch == "timeout":
        command = [
            "env", "-u", "SWITCHYARD_TARGET_GPU_UUID", "CUDA_VISIBLE_DEVICES=",
            "timeout", "--kill-after=1s", "30s", *command,
        ]
    process = None
    unrelated = None
    child_pid = None
    try:
        unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
        process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE)
        assert select.select([process.stdout], [], [], 5)[0], "child did not become ready"
        child_pid = int(process.stdout.readline())
        assert os.getpgid(child_pid) == process.pid
        assert MODULE._is_descendant(child_pid, process.pid)
        if launch == "exited_parent":
            process.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 5
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                assert time.monotonic() < deadline, "parent did not exit"
                time.sleep(0.01)

        recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
        MODULE._set_active_process(process, recorder)
        MODULE._cleanup_active_process()
        assert process.returncode is not None
        if launch != "timeout":
            assert process.returncode == 0
        assert unrelated.poll() is None
        deadline = time.monotonic() + 5
        while True:
            reaped, status = os.waitpid(child_pid, os.WNOHANG)
            if reaped:
                child_pid = None
                assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
                break
            assert time.monotonic() < deadline, "stubborn child survived cleanup"
            time.sleep(0.01)
    finally:
        if process is not None:
            MODULE._clear_active_process(process)
            if process.returncode is None or child_pid is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
            process.stdout.close()
        if child_pid is not None:
            os.waitpid(child_pid, 0)
        if unrelated is not None:
            unrelated.kill()
            unrelated.wait(timeout=5)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


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
    [
        "",
        "0, 0\n",
        "0, busy, 2\n",
        "0, 101, 2\n",
        "0, 0, 2\n0, 0, 2\n",
    ],
)
def test_activity_parser_fails_closed(text):
    with pytest.raises(ValueError):
        MODULE._parse_activity(text)


@pytest.mark.parametrize(
    "pcie",
    [
        {1: (0, 0)},
        {0: (-1, 0)},
        {0: (0, True)},
    ],
)
def test_pcie_merge_fails_closed(pcie):
    with pytest.raises(ValueError):
        MODULE._merge_pcie_throughput(MODULE._parse_activity("0, 0, 2\n"), pcie)


def test_nvml_pcie_units_are_interpreted_conservatively():
    assert MODULE._nvml_kb_as_conservative_kib_per_second(0) == 0
    assert MODULE._nvml_kb_as_conservative_kib_per_second(1024) == 1024
    assert MODULE._nvml_kb_as_conservative_kib_per_second(1) == 1
    with pytest.raises(ValueError):
        MODULE._nvml_kb_as_conservative_kib_per_second(True)


def test_parent_death_signal_requires_the_exact_supervisor(monkeypatch):
    exits = []
    monkeypatch.setattr(MODULE.os, "getppid", lambda: 41)
    monkeypatch.setattr(MODULE.os, "_exit", lambda code: exits.append(code))
    MODULE._set_parent_death_signal(42)
    assert exits == [127]


def test_parent_death_signal_closes_the_post_prctl_race(monkeypatch):
    class Libc:
        @staticmethod
        def prctl(*_args):
            return 0

    parents = iter((42, 41))
    exits = []
    monkeypatch.setattr(MODULE.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(MODULE.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(MODULE.ctypes, "CDLL", lambda *_args, **_kwargs: Libc())
    MODULE._set_parent_death_signal(42)
    assert exits == [127]


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
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="different campaign"):
        MODULE.StateRecorder(state, "campaign-b")


def test_state_recorder_keeps_gpu_uuid_out_of_terminal_output(tmp_path, capsys):
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    recorder.write("waiting", target_uuid="GPU-private", target_index=1)
    assert "GPU-private" not in capsys.readouterr().out
    assert recorder.events[-1]["target_uuid"] == "GPU-private"


def test_public_probe_errors_never_include_physical_identity():
    error = ValueError("malformed row contains GPU-01234567-secret")
    assert MODULE._public_probe_error(error) == "ValueError"
    assert "GPU-" not in MODULE._public_probe_error(error)


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
        lambda value: value.update(idle_activity_scope="some_gpus"),
        lambda value: value.update(process_scope="target_gpu"),
        lambda value: value.update(max_idle_probe_gap_seconds=66),
        lambda value: value.update(max_global_compute_process_count=1),
        lambda value: value.update(
            max_non_target_pcie_rx_kib_per_second=(
                MODULE.MAX_IDLE_PCIE_KIB_PER_SECOND + 1
            )
        ),
        lambda value: value.update(maximum_allowed_pcie_kib_per_second=1),
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
        {
            "PATH": "/bin",
            "UNRELATED": "kept",
            "SWITCHYARD_TARGET_GPU_UUID": "GPU-inherited",
            "SWITCHYARD_GUARD_TARGET_GPU_UUID": "GPU-inherited",
        },
        context,
        "device-0123456789abcdef",
        "campaign-hash",
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert environment["SWITCHYARD_TARGET_GPU_UUID"] == "GPU-test"
    assert environment["SWITCHYARD_GUARD_TARGET_GPU_UUID"] == "GPU-test"
    assert environment["SWITCHYARD_RUN_ATTEMPT"] == "3"
    assert environment["SWITCHYARD_GUARD_IDLE_PROBE_COUNT"] == "31"
    assert environment["SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB"] == "8"
    assert environment["SWITCHYARD_GUARD_MAX_IDLE_PCIE_RX_KIB_PER_SECOND"] == "2048"
    assert environment["SWITCHYARD_GUARD_MAXIMUM_ALLOWED_PCIE_KIB_PER_SECOND"] == "65536"
    assert environment["SWITCHYARD_GUARD_IDLE_ACTIVITY_SCOPE"] == "target_gpu"
    assert environment["SWITCHYARD_GUARD_PROCESS_SCOPE"] == "all_gpus"
    assert environment["SWITCHYARD_GUARD_MAX_IDLE_PROBE_GAP_SECONDS"] == "60.1"
    assert environment["SWITCHYARD_PUBLIC_DEVICE_ID"] == "device-0123456789abcdef"
    assert environment["SWITCHYARD_RUNNER_CAMPAIGN_IDENTITY"] == "campaign-hash"
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
        idle_activity_scope="target_gpu",
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
            "--idle-activity-scope",
            "target_gpu",
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
    assert "SWITCHYARD_GUARD_IDLE_ACTIVITY_SCOPE" in source
    assert "SWITCHYARD_GUARD_PROCESS_SCOPE" in source
    assert "SWITCHYARD_GUARD_MAX_IDLE_PROBE_GAP_SECONDS" in source
    assert "SWITCHYARD_PUBLIC_DEVICE_ID" in source
    assert "pass_fds=(lock_handle.fileno(), gpu_lock_handle.fileno())" in source
    assert "observed_apps = _compute_apps()" in source
    assert "for item in observed_apps" in source
    assert "preexec_fn=_parent_death_preexec()" in source
    assert "os.getppid() != expected_parent_pid" in source
    assert "atexit.register(_cleanup_active_process)" in source
    assert 'if key != "target_uuid"' in source
    assert '"gpu_completion_requires_new_attempt"' in source
    assert 'f"blackwell-switchyard-{target_uuid}.lock"' not in source


@pytest.mark.parametrize("inherited_target", [None, "GPU-inherited"])
def test_cpu_recovery_disables_gpu_visibility_and_preserves_attempt(
    tmp_path, monkeypatch, inherited_target
):
    if inherited_target is None:
        monkeypatch.delenv("SWITCHYARD_TARGET_GPU_UUID", raising=False)
    else:
        monkeypatch.setenv("SWITCHYARD_TARGET_GPU_UUID", inherited_target)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "GPU-private",
        "SWITCHYARD_GUARD_TARGET_GPU_UUID": "GPU-test",
    }
    conftest = SCRIPT.parents[1] / "tests" / "conftest.py"
    child = """
import importlib.util
import os
import runpy
import sys
from types import SimpleNamespace

assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert "SWITCHYARD_TARGET_GPU_UUID" not in os.environ
assert os.environ["SWITCHYARD_GUARD_TARGET_GPU_UUID"] == "GPU-test"
assert os.environ["SWITCHYARD_CPU_RECOVERY"] == "1"
hooks = runpy.run_path(sys.argv[1])
def forbid_monitor(*args, **kwargs):
    raise AssertionError("CPU recovery must not load the GPU monitor")
importlib.util.spec_from_file_location = forbid_monitor
config = SimpleNamespace()
hooks["pytest_configure"](config)
hooks["pytest_unconfigure"](config)
"""
    recorder = MODULE.StateRecorder(tmp_path / "state.json", "campaign-a")
    result = MODULE._run_cpu_recovery(
        [sys.executable, "-c", child, str(conftest)],
        cwd=tmp_path,
        environment=environment,
        log_path=tmp_path / "runner.log",
        recorder=recorder,
        attempt=3,
        deadline=time.time() + 10,
        finalize_seconds=5,
        retry_seconds=1,
    )
    assert result == 0
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-private"
    assert environment.get("SWITCHYARD_TARGET_GPU_UUID") == inherited_target
    assert recorder.attempts == 3
    assert recorder.last_phase == "complete"
    assert stat.S_IMODE((tmp_path / "runner.log").stat().st_mode) == 0o600
