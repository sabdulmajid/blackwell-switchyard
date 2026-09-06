"""CPU-only tests for the guarded GPU idle runner."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_when_gpu_idle.py"
SPEC = importlib.util.spec_from_file_location("run_when_gpu_idle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_nvidia_smi_parsers_keep_physical_identity():
    assert MODULE._parse_inventory("0, GPU-a\n1, GPU-b\n") == {
        0: "GPU-a",
        1: "GPU-b",
    }
    assert MODULE._parse_compute_apps("GPU-b, 42\n") == [
        {"gpu_uuid": "GPU-b", "pid": 42}
    ]


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


def test_state_recorder_resumes_attempt_count_without_process_data(tmp_path):
    state = tmp_path / "state.json"
    recorder = MODULE.StateRecorder(state, "campaign-a")
    recorder.write("launching_workload", attempt=2, target_uuid="GPU-test")
    resumed = MODULE.StateRecorder(state, "campaign-a")
    assert resumed.attempts == 2
    assert resumed.last_phase == "launching_workload"
    assert resumed.public_device_id == recorder.public_device_id
    assert MODULE.PUBLIC_DEVICE_ID_PATTERN.fullmatch(resumed.public_device_id)
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


def test_runner_source_exports_idle_attestation_without_process_ids():
    source = SCRIPT.read_text()
    assert "SWITCHYARD_GUARD_IDLE_STARTED_AT" in source
    assert "SWITCHYARD_GUARD_LAUNCH_AT" in source
    assert "SWITCHYARD_GUARD_IDLE_PROBE_COUNT" in source
    assert "SWITCHYARD_GUARD_GPU_COUNT" in source
    assert "SWITCHYARD_PUBLIC_DEVICE_ID" in source


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
