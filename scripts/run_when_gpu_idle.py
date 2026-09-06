#!/usr/bin/env python3
"""Run one command after the required GPU activity has stayed idle.

The runner uses no GPU while it waits. At a low frequency, it checks NVIDIA's
global process table and the configured activity scope. The activity scope can
cover every GPU or only the selected GPU. During the GPU phase, the runner
continues to check the global process table. If an unrelated GPU process appears
anywhere, it stops its own process group and waits for a new bounded attempt.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

CPU_RECOVERY_EXIT = 74
MAX_IDLE_MEMORY_MIB = 64
MAX_IDLE_PCIE_KIB_PER_SECOND = 64 * 1024
MAX_IDLE_PROBE_DELAY_SECONDS = 5.0
MAX_WATCHDOG_PROBE_GAP_SECONDS = 1.0
IDLE_ACTIVITY_SCOPES = frozenset({"all_gpus", "target_gpu"})
PROCESS_SCOPE = "all_gpus"
PUBLIC_DEVICE_ID_PATTERN = re.compile(r"device-[0-9a-f]{16}")
GPU_UUID_PATTERN = re.compile(r"GPU-[A-Za-z0-9-]+")
RECOVERY_CONTEXT_FIELDS = {
    "target_uuid",
    "idle_activity_scope",
    "process_scope",
    "attempt",
    "not_before",
    "idle_seconds",
    "wait_poll_seconds",
    "watchdog_seconds",
    "finalize_seconds",
    "idle_started_at",
    "launch_at",
    "idle_probe_count",
    "max_idle_gpu_utilization_percent",
    "max_idle_memory_mib",
    "max_idle_pcie_rx_kib_per_second",
    "max_idle_pcie_tx_kib_per_second",
    "max_idle_probe_gap_seconds",
    "max_global_compute_process_count",
    "max_non_target_gpu_utilization_percent",
    "max_non_target_memory_mib",
    "max_non_target_pcie_rx_kib_per_second",
    "max_non_target_pcie_tx_kib_per_second",
    "maximum_allowed_pcie_kib_per_second",
    "gpu_count",
}


def _parse_inventory(text: str) -> dict[int, str]:
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if not rows:
        raise ValueError("nvidia-smi returned an empty GPU inventory")
    result: dict[int, str] = {}
    for row in rows:
        if len(row) != 2 or not row[0].strip().isdigit() or not row[1].strip().startswith("GPU-"):
            raise ValueError(f"malformed GPU inventory row: {row!r}")
        index = int(row[0].strip())
        if index in result:
            raise ValueError(f"duplicate GPU index {index}")
        result[index] = row[1].strip()
    return result


def _parse_compute_apps(text: str) -> list[dict[str, str | int]]:
    result = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if (
            len(row) != 2
            or not row[0].strip().startswith("GPU-")
            or not row[1].strip().isdigit()
        ):
            raise ValueError(f"malformed compute-process row: {row!r}")
        result.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
            }
        )
    return result


def _parse_activity(text: str) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 3 or any(not field.strip().isdigit() for field in row):
            raise ValueError(f"malformed GPU activity row: {row!r}")
        index, utilization, memory_used = (int(field.strip()) for field in row)
        if index in result or not 0 <= utilization <= 100:
            raise ValueError(f"invalid GPU activity row: {row!r}")
        result[index] = {
            "utilization_percent": utilization,
            "memory_used_mib": memory_used,
        }
    if not result:
        raise ValueError("nvidia-smi returned an empty GPU activity table")
    return result


def _merge_pcie_throughput(
    activity: dict[int, dict[str, int]], pcie: dict[int, tuple[int, int]]
) -> dict[int, dict[str, int]]:
    if set(activity) != set(pcie):
        raise ValueError("NVML and nvidia-smi returned different GPU indices")
    merged = {index: values.copy() for index, values in activity.items()}
    for index, (rx_kib, tx_kib) in pcie.items():
        if (
            isinstance(rx_kib, bool)
            or isinstance(tx_kib, bool)
            or not isinstance(rx_kib, int)
            or not isinstance(tx_kib, int)
            or rx_kib < 0
            or tx_kib < 0
        ):
            raise ValueError(f"invalid NVML PCIe throughput for GPU {index}")
        merged[index]["pcie_rx_kib_per_second"] = rx_kib
        merged[index]["pcie_tx_kib_per_second"] = tx_kib
    return merged


def _parent_pid(pid: int) -> int | None:
    try:
        # Everything after the final ')' starts with state and parent PID.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1])
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return None


def _record_probe_gap(
    previous_probe_at: float, maximum_gap_seconds: float, *, now: float | None = None
) -> tuple[float, float]:
    """Include the complete interval since the preceding watchdog boundary."""
    probe_at = time.monotonic() if now is None else now
    return probe_at, max(maximum_gap_seconds, probe_at - previous_probe_at)


def _is_descendant(
    pid: int,
    root_pid: int,
    parent: Callable[[int], int | None] = _parent_pid,
) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        next_pid = parent(pid)
        if next_pid is None:
            return False
        pid = next_pid
    return False


def _smi(*arguments: str) -> str:
    return subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout


def _inventory() -> dict[int, str]:
    return _parse_inventory(
        _smi("--query-gpu=index,uuid", "--format=csv,noheader,nounits")
    )


def _compute_apps() -> list[dict[str, str | int]]:
    return _parse_compute_apps(
        _smi(
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        )
    )


def _activity() -> dict[int, dict[str, int]]:
    activity = _parse_activity(
        _smi(
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            pcie = {}
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                pcie[index] = (
                    int(
                        pynvml.nvmlDeviceGetPcieThroughput(
                            handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                        )
                    ),
                    int(
                        pynvml.nvmlDeviceGetPcieThroughput(
                            handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                        )
                    ),
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        raise OSError(f"NVML PCIe telemetry failed: {type(exc).__name__}: {exc}") from exc
    return _merge_pcie_throughput(activity, pcie)


def _is_idle_activity(
    activity: dict[int, dict[str, int]],
    inventory: dict[int, str],
    *,
    activity_scope: str = "all_gpus",
    target_index: int | None = None,
) -> bool:
    if set(activity) != set(inventory):
        return False
    if activity_scope == "all_gpus":
        guarded_indices = set(inventory)
    elif activity_scope == "target_gpu" and target_index in inventory:
        guarded_indices = {target_index}
    else:
        return False
    return all(
        values["utilization_percent"] == 0
        and values["memory_used_mib"] <= MAX_IDLE_MEMORY_MIB
        for index, values in activity.items()
        if index in guarded_indices
    ) and all(
        values["pcie_rx_kib_per_second"] <= MAX_IDLE_PCIE_KIB_PER_SECOND
        and values["pcie_tx_kib_per_second"] <= MAX_IDLE_PCIE_KIB_PER_SECOND
        for values in activity.values()
    )


def _guarded_activity_values(
    activity: dict[int, dict[str, int]],
    *,
    activity_scope: str,
    target_index: int,
) -> list[dict[str, int]]:
    indices = activity if activity_scope == "all_gpus" else {target_index}
    return [activity[index] for index in indices]


class StateRecorder:
    def __init__(self, path: Path, campaign_identity: str):
        self.path = path
        self.campaign_identity = campaign_identity
        self.events: list[dict] = []
        self._attempts = 0
        self.recovery_context: dict | None = None
        self.public_device_id = f"device-{secrets.token_hex(8)}"
        if path.exists():
            os.chmod(path, 0o600)
            payload = json.loads(path.read_text())
            if payload.get("campaign_identity") != campaign_identity:
                raise ValueError("runner state belongs to a different campaign")
            events = payload.get("events")
            if not isinstance(events, list) or not all(
                isinstance(event, dict) for event in events
            ):
                raise ValueError("runner state has an invalid event history")
            self.events = events[-200:]
            public_device_id = payload.get("public_device_id")
            if not isinstance(public_device_id, str) or PUBLIC_DEVICE_ID_PATTERN.fullmatch(
                public_device_id
            ) is None:
                raise ValueError("runner state has an invalid public device ID")
            self.public_device_id = public_device_id
            recorded_attempts = payload.get("attempts", 0)
            if not isinstance(recorded_attempts, int) or recorded_attempts < 0:
                raise ValueError("runner state has an invalid attempt count")
            event_attempts = [
                event.get("attempt")
                for event in self.events
                if isinstance(event.get("attempt"), int)
            ]
            self._attempts = max(recorded_attempts, *event_attempts, 0)
            recovery_context = payload.get("recovery_context")
            if recovery_context is not None:
                self.recovery_context = _validate_recovery_context(recovery_context)
                if self.recovery_context["attempt"] != self._attempts:
                    raise ValueError(
                        "runner state recovery context does not match its attempt count"
                    )

    @property
    def last_phase(self) -> str | None:
        return self.events[-1].get("phase") if self.events else None

    @property
    def attempts(self) -> int:
        return self._attempts

    def _save(self, current: dict | None) -> None:
        payload = {
            "campaign_identity": self.campaign_identity,
            "public_device_id": self.public_device_id,
            "attempts": self._attempts,
            "current": current,
            "events": self.events[-200:],
            "recovery_context": self.recovery_context,
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
        os.chmod(self.path, 0o600)

    def set_recovery_context(self, context: dict | None) -> None:
        self.recovery_context = (
            None if context is None else _validate_recovery_context(context)
        )
        self._save(self.events[-1] if self.events else None)

    def write(self, phase: str, **fields) -> None:
        event = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            **fields,
        }
        attempt = event.get("attempt")
        if isinstance(attempt, int):
            self._attempts = max(self._attempts, attempt)
        self.events.append(event)
        self._save(event)
        print(json.dumps(event, sort_keys=True), flush=True)


def _offset_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return False
    return moment.tzinfo is not None


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _positive_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and value > 0
    )


def _validate_recovery_context(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != RECOVERY_CONTEXT_FIELDS:
        raise ValueError("runner state has an invalid recovery context field set")
    if (
        not isinstance(value["target_uuid"], str)
        or GPU_UUID_PATTERN.fullmatch(value["target_uuid"]) is None
        or value["idle_activity_scope"] not in IDLE_ACTIVITY_SCOPES
        or value["process_scope"] != PROCESS_SCOPE
        or not _positive_int(value["attempt"])
        or not _offset_timestamp(value["not_before"])
        or not _positive_int(value["idle_seconds"])
        or not _positive_int(value["wait_poll_seconds"])
        or not _positive_number(value["watchdog_seconds"])
        or not _positive_int(value["finalize_seconds"])
        or not _offset_timestamp(value["idle_started_at"])
        or not _offset_timestamp(value["launch_at"])
        or not _positive_int(value["idle_probe_count"])
        or value["max_idle_gpu_utilization_percent"] != 0
        or isinstance(value["max_idle_memory_mib"], bool)
        or not isinstance(value["max_idle_memory_mib"], int)
        or not 0 <= value["max_idle_memory_mib"] <= MAX_IDLE_MEMORY_MIB
        or isinstance(value["max_idle_pcie_rx_kib_per_second"], bool)
        or not isinstance(value["max_idle_pcie_rx_kib_per_second"], int)
        or not 0
        <= value["max_idle_pcie_rx_kib_per_second"]
        <= MAX_IDLE_PCIE_KIB_PER_SECOND
        or isinstance(value["max_idle_pcie_tx_kib_per_second"], bool)
        or not isinstance(value["max_idle_pcie_tx_kib_per_second"], int)
        or not 0
        <= value["max_idle_pcie_tx_kib_per_second"]
        <= MAX_IDLE_PCIE_KIB_PER_SECOND
        or isinstance(value["max_idle_probe_gap_seconds"], bool)
        or not isinstance(value["max_idle_probe_gap_seconds"], int | float)
        or not 0
        <= value["max_idle_probe_gap_seconds"]
        <= value["wait_poll_seconds"] + MAX_IDLE_PROBE_DELAY_SECONDS
        or value["max_global_compute_process_count"] != 0
        or isinstance(value["max_non_target_gpu_utilization_percent"], bool)
        or not isinstance(value["max_non_target_gpu_utilization_percent"], int)
        or not 0 <= value["max_non_target_gpu_utilization_percent"] <= 100
        or isinstance(value["max_non_target_memory_mib"], bool)
        or not isinstance(value["max_non_target_memory_mib"], int)
        or value["max_non_target_memory_mib"] < 0
        or isinstance(value["max_non_target_pcie_rx_kib_per_second"], bool)
        or not isinstance(value["max_non_target_pcie_rx_kib_per_second"], int)
        or not 0
        <= value["max_non_target_pcie_rx_kib_per_second"]
        <= MAX_IDLE_PCIE_KIB_PER_SECOND
        or isinstance(value["max_non_target_pcie_tx_kib_per_second"], bool)
        or not isinstance(value["max_non_target_pcie_tx_kib_per_second"], int)
        or not 0
        <= value["max_non_target_pcie_tx_kib_per_second"]
        <= MAX_IDLE_PCIE_KIB_PER_SECOND
        or value["maximum_allowed_pcie_kib_per_second"]
        != MAX_IDLE_PCIE_KIB_PER_SECOND
        or not _positive_int(value["gpu_count"])
    ):
        raise ValueError("runner state has invalid recovery context values")
    if value["attempt"] > 10**6:
        raise ValueError("runner state has an invalid recovery attempt")
    return value.copy()


def _recovery_environment(
    base_environment: dict[str, str], context: dict, public_device_id: str
) -> dict[str, str]:
    context = _validate_recovery_context(context)
    environment = base_environment.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": context["target_uuid"],
            "SWITCHYARD_TARGET_GPU_UUID": context["target_uuid"],
            "SWITCHYARD_GUARD_IDLE_ACTIVITY_SCOPE": context["idle_activity_scope"],
            "SWITCHYARD_GUARD_PROCESS_SCOPE": context["process_scope"],
            "SWITCHYARD_PUBLIC_DEVICE_ID": public_device_id,
            "SWITCHYARD_RUN_ATTEMPT": str(context["attempt"]),
            "SWITCHYARD_GUARD_NOT_BEFORE": context["not_before"],
            "SWITCHYARD_GUARD_IDLE_SECONDS": str(context["idle_seconds"]),
            "SWITCHYARD_GUARD_WAIT_POLL_SECONDS": str(
                context["wait_poll_seconds"]
            ),
            "SWITCHYARD_GUARD_WATCHDOG_SECONDS": str(context["watchdog_seconds"]),
            "SWITCHYARD_GUARD_FINALIZE_SECONDS": str(context["finalize_seconds"]),
            "SWITCHYARD_GUARD_IDLE_STARTED_AT": context["idle_started_at"],
            "SWITCHYARD_GUARD_LAUNCH_AT": context["launch_at"],
            "SWITCHYARD_GUARD_IDLE_PROBE_COUNT": str(context["idle_probe_count"]),
            "SWITCHYARD_GUARD_MAX_IDLE_GPU_UTILIZATION_PERCENT": str(
                context["max_idle_gpu_utilization_percent"]
            ),
            "SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB": str(
                context["max_idle_memory_mib"]
            ),
            "SWITCHYARD_GUARD_MAX_IDLE_PCIE_RX_KIB_PER_SECOND": str(
                context["max_idle_pcie_rx_kib_per_second"]
            ),
            "SWITCHYARD_GUARD_MAX_IDLE_PCIE_TX_KIB_PER_SECOND": str(
                context["max_idle_pcie_tx_kib_per_second"]
            ),
            "SWITCHYARD_GUARD_MAX_IDLE_PROBE_GAP_SECONDS": str(
                context["max_idle_probe_gap_seconds"]
            ),
            "SWITCHYARD_GUARD_MAX_GLOBAL_COMPUTE_PROCESS_COUNT": str(
                context["max_global_compute_process_count"]
            ),
            "SWITCHYARD_GUARD_MAX_NON_TARGET_GPU_UTILIZATION_PERCENT": str(
                context["max_non_target_gpu_utilization_percent"]
            ),
            "SWITCHYARD_GUARD_MAX_NON_TARGET_MEMORY_MIB": str(
                context["max_non_target_memory_mib"]
            ),
            "SWITCHYARD_GUARD_MAX_NON_TARGET_PCIE_RX_KIB_PER_SECOND": str(
                context["max_non_target_pcie_rx_kib_per_second"]
            ),
            "SWITCHYARD_GUARD_MAX_NON_TARGET_PCIE_TX_KIB_PER_SECOND": str(
                context["max_non_target_pcie_tx_kib_per_second"]
            ),
            "SWITCHYARD_GUARD_MAXIMUM_ALLOWED_PCIE_KIB_PER_SECOND": str(
                context["maximum_allowed_pcie_kib_per_second"]
            ),
            "SWITCHYARD_GUARD_GPU_COUNT": str(context["gpu_count"]),
        }
    )
    return environment


def _campaign_identity(args: argparse.Namespace, command: list[str]) -> str:
    environment_keys = (
        "SWITCHYARD_EXPECTED_HEAD",
        "SWITCHYARD_CAMPAIGN_BRANCH",
        "SWITCHYARD_CAMPAIGN_DIR",
        "SWITCHYARD_RESULT_BRANCH",
        "THIRD_PARTY_DIR",
        "SWITCHYARD_TOOLCHAIN_DIR",
    )
    payload = {
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "deadline_hours": args.deadline_hours,
        "finalize_seconds": args.finalize_seconds,
        "gpu_complete_marker": (
            str(args.gpu_complete_marker.resolve())
            if args.gpu_complete_marker is not None
            else None
        ),
        "gpu_index": args.gpu_index,
        "idle_activity_scope": args.idle_activity_scope,
        "gpu_lock_dir": str(args.gpu_lock_dir.resolve()),
        "idle_seconds": args.idle_seconds,
        "max_attempts": args.max_attempts,
        "not_before": args.not_before,
        "wait_poll_seconds": args.wait_poll_seconds,
        "watchdog_seconds": args.watchdog_seconds,
        "environment": {key: os.environ.get(key) for key in environment_keys},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _terminate_group(process: subprocess.Popen, recorder: StateRecorder, reason: str) -> None:
    recorder.write("stopping_workload", reason=reason)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def _run_cpu_recovery(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    recorder: StateRecorder,
    attempt: int,
    deadline: float,
    finalize_seconds: int,
    retry_seconds: int,
    inherited_fds: tuple[int, ...] = (),
) -> int:
    """Retry post-GPU publication without querying a GPU or spending an attempt."""
    cpu_environment = environment.copy()
    cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
    cpu_environment["SWITCHYARD_CPU_RECOVERY"] = "1"
    while time.time() < deadline:
        recorder.write("cpu_recovery_started", attempt=attempt)
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with log_path.open("a") as log:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=cpu_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                pass_fds=inherited_fds,
            )
            try:
                return_code = process.wait(
                    timeout=min(finalize_seconds, max(1.0, deadline - time.time()))
                )
            except subprocess.TimeoutExpired:
                _terminate_group(
                    process,
                    recorder,
                    f"CPU recovery exceeded {finalize_seconds} seconds",
                )
                return_code = CPU_RECOVERY_EXIT
        if return_code == 0:
            recorder.write("complete", attempt=attempt, return_code=0)
            return 0
        if return_code == 75:
            return 75
        if return_code != CPU_RECOVERY_EXIT:
            recorder.write("failed", attempt=attempt, return_code=return_code)
            return return_code or 1
        recorder.write("cpu_recovery_waiting", attempt=attempt)
        time.sleep(min(retry_seconds, max(0.0, deadline - time.time())))
    recorder.write("expired", attempts=attempt)
    return CPU_RECOVERY_EXIT


def _parse_not_before(value: str) -> float:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError("--not-before must include a UTC offset")
    return moment.timestamp()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--not-before", required=True, type=_parse_not_before)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--idle-activity-scope",
        choices=sorted(IDLE_ACTIVITY_SCOPES),
        default="all_gpus",
        help="require idle telemetry from every GPU or only the selected GPU",
    )
    parser.add_argument("--idle-seconds", type=int, default=1800)
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    parser.add_argument("--watchdog-seconds", type=float, default=0.25)
    parser.add_argument("--finalize-seconds", type=int, default=1800)
    parser.add_argument("--deadline-hours", type=float, default=36.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--gpu-lock-dir", default=Path("/tmp"), type=Path)
    parser.add_argument("--gpu-complete-marker", type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if min(
        args.idle_seconds,
        args.wait_poll_seconds,
        args.watchdog_seconds,
        args.finalize_seconds,
        args.max_attempts,
    ) <= 0:
        parser.error("all intervals and --max-attempts must be positive")
    if args.deadline_hours <= 0:
        parser.error("--deadline-hours must be positive")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")

    args.lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_handle = args.lock.open("w")
    os.chmod(args.lock, 0o600)
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another runner owns {args.lock}") from None
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    try:
        recorder = StateRecorder(args.state, _campaign_identity(args, command))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot resume runner state: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if recorder.last_phase == "complete":
        return 0
    attempt = recorder.attempts
    deadline = time.time() + args.deadline_hours * 3600
    if (
        args.gpu_complete_marker is not None
        and args.gpu_complete_marker.exists()
        and recorder.recovery_context is not None
    ):
        context = recorder.recovery_context
        recovery_code = _run_cpu_recovery(
            command,
            cwd=args.cwd,
            environment=_recovery_environment(
                {
                    **os.environ,
                    "SWITCHYARD_RUNNER_STATE": str(args.state.resolve()),
                },
                context,
                recorder.public_device_id,
            ),
            log_path=args.log,
            recorder=recorder,
            attempt=context["attempt"],
            deadline=deadline,
            finalize_seconds=args.finalize_seconds,
            retry_seconds=args.wait_poll_seconds,
            inherited_fds=(lock_handle.fileno(),),
        )
        if recovery_code != 75:
            return recovery_code
        args.gpu_complete_marker.unlink(missing_ok=True)
        recorder.set_recovery_context(None)
        recorder.write("cpu_recovery_requires_gpu", attempt=attempt)
    if attempt >= args.max_attempts:
        recorder.write("expired", attempts=attempt)
        return 75
    if time.time() < args.not_before:
        recorder.write(
            "sleeping_until_not_before",
            not_before=datetime.fromtimestamp(args.not_before).astimezone().isoformat(),
        )
        time.sleep(max(0.0, args.not_before - time.time()))
    inventory = _inventory()
    if args.gpu_index not in inventory:
        recorder.write("failed", reason=f"GPU index {args.gpu_index} does not exist")
        return 2
    target_uuid = inventory[args.gpu_index]
    args.gpu_lock_dir.mkdir(parents=True, exist_ok=True)
    target_lock_id = hashlib.sha256(target_uuid.encode()).hexdigest()[:16]
    gpu_lock_path = args.gpu_lock_dir / f"blackwell-switchyard-device-{target_lock_id}.lock"
    gpu_lock_handle = gpu_lock_path.open("w")
    os.chmod(gpu_lock_path, 0o600)
    try:
        fcntl.flock(gpu_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        recorder.write("failed", reason=f"another switchyard runner owns {gpu_lock_path}")
        return 75
    gpu_lock_handle.write(f"{os.getpid()}\n")
    gpu_lock_handle.flush()
    recorder.write(
        (
            "waiting_for_all_gpus_idle"
            if args.idle_activity_scope == "all_gpus"
            else "waiting_for_target_gpu_idle"
        ),
        target_index=args.gpu_index,
        target_uuid=target_uuid,
        idle_activity_scope=args.idle_activity_scope,
        process_scope=PROCESS_SCOPE,
        required_idle_seconds=args.idle_seconds,
        wait_poll_seconds=args.wait_poll_seconds,
    )

    idle_since: float | None = None
    idle_since_wall: float | None = None
    idle_probe_count = 0
    max_idle_memory_mib = 0
    max_idle_pcie_rx_kib_per_second = 0
    max_idle_pcie_tx_kib_per_second = 0
    max_idle_probe_gap_seconds = 0.0
    last_idle_probe_at: float | None = None
    max_non_target_gpu_utilization_percent = 0
    max_non_target_memory_mib = 0
    max_non_target_pcie_rx_kib_per_second = 0
    max_non_target_pcie_tx_kib_per_second = 0
    while time.time() < deadline and attempt < args.max_attempts:
        try:
            apps = _compute_apps()
            activity = _activity()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            recorder.write("wait_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue

        if apps or not _is_idle_activity(
            activity,
            inventory,
            activity_scope=args.idle_activity_scope,
            target_index=args.gpu_index,
        ):
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            time.sleep(args.wait_poll_seconds)
            continue
        if idle_since is None:
            idle_since = time.monotonic()
            idle_since_wall = time.time()
            recorder.write("idle_grace_started")
        probe_at = time.monotonic()
        if last_idle_probe_at is not None:
            probe_gap = probe_at - last_idle_probe_at
            if probe_gap > args.wait_poll_seconds + MAX_IDLE_PROBE_DELAY_SECONDS:
                idle_since = probe_at
                idle_since_wall = time.time()
                idle_probe_count = 0
                max_idle_memory_mib = 0
                max_idle_pcie_rx_kib_per_second = 0
                max_idle_pcie_tx_kib_per_second = 0
                max_idle_probe_gap_seconds = 0.0
                max_non_target_gpu_utilization_percent = 0
                max_non_target_memory_mib = 0
                max_non_target_pcie_rx_kib_per_second = 0
                max_non_target_pcie_tx_kib_per_second = 0
                recorder.write("idle_grace_restarted_after_probe_gap")
            else:
                max_idle_probe_gap_seconds = max(
                    max_idle_probe_gap_seconds, probe_gap
                )
        last_idle_probe_at = probe_at
        idle_probe_count += 1
        guarded_activity = _guarded_activity_values(
            activity,
            activity_scope=args.idle_activity_scope,
            target_index=args.gpu_index,
        )
        max_idle_memory_mib = max(
            max_idle_memory_mib,
            *(values["memory_used_mib"] for values in guarded_activity),
        )
        max_idle_pcie_rx_kib_per_second = max(
            max_idle_pcie_rx_kib_per_second,
            *(values["pcie_rx_kib_per_second"] for values in guarded_activity),
        )
        max_idle_pcie_tx_kib_per_second = max(
            max_idle_pcie_tx_kib_per_second,
            *(values["pcie_tx_kib_per_second"] for values in guarded_activity),
        )
        non_target = [
            values for index, values in activity.items() if index != args.gpu_index
        ]
        if non_target:
            max_non_target_gpu_utilization_percent = max(
                max_non_target_gpu_utilization_percent,
                *(values["utilization_percent"] for values in non_target),
            )
            max_non_target_memory_mib = max(
                max_non_target_memory_mib,
                *(values["memory_used_mib"] for values in non_target),
            )
            max_non_target_pcie_rx_kib_per_second = max(
                max_non_target_pcie_rx_kib_per_second,
                *(values["pcie_rx_kib_per_second"] for values in non_target),
            )
            max_non_target_pcie_tx_kib_per_second = max(
                max_non_target_pcie_tx_kib_per_second,
                *(values["pcie_tx_kib_per_second"] for values in non_target),
            )
        elapsed = time.monotonic() - idle_since
        if elapsed < args.idle_seconds:
            time.sleep(min(args.wait_poll_seconds, args.idle_seconds - elapsed))
            continue

        # Close the final race before launch. The workload has its own GPU
        # preflight as a second independent check.
        try:
            final_apps = _compute_apps()
            final_activity = _activity()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            recorder.write("final_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue
        if final_apps or not _is_idle_activity(
            final_activity,
            inventory,
            activity_scope=args.idle_activity_scope,
            target_index=args.gpu_index,
        ):
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            continue

        final_probe_at = time.monotonic()
        if (
            last_idle_probe_at is None
            or final_probe_at - last_idle_probe_at
            > args.wait_poll_seconds + MAX_IDLE_PROBE_DELAY_SECONDS
        ):
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            recorder.write("final_probe_gap_exceeded")
            continue
        max_idle_probe_gap_seconds = max(
            max_idle_probe_gap_seconds, final_probe_at - last_idle_probe_at
        )
        last_idle_probe_at = final_probe_at
        attempt += 1
        idle_probe_count += 1
        final_guarded_activity = _guarded_activity_values(
            final_activity,
            activity_scope=args.idle_activity_scope,
            target_index=args.gpu_index,
        )
        max_idle_memory_mib = max(
            max_idle_memory_mib,
            *(values["memory_used_mib"] for values in final_guarded_activity),
        )
        max_idle_pcie_rx_kib_per_second = max(
            max_idle_pcie_rx_kib_per_second,
            *(values["pcie_rx_kib_per_second"] for values in final_guarded_activity),
        )
        max_idle_pcie_tx_kib_per_second = max(
            max_idle_pcie_tx_kib_per_second,
            *(values["pcie_tx_kib_per_second"] for values in final_guarded_activity),
        )
        final_non_target = [
            values for index, values in final_activity.items() if index != args.gpu_index
        ]
        if final_non_target:
            max_non_target_gpu_utilization_percent = max(
                max_non_target_gpu_utilization_percent,
                *(values["utilization_percent"] for values in final_non_target),
            )
            max_non_target_memory_mib = max(
                max_non_target_memory_mib,
                *(values["memory_used_mib"] for values in final_non_target),
            )
            max_non_target_pcie_rx_kib_per_second = max(
                max_non_target_pcie_rx_kib_per_second,
                *(values["pcie_rx_kib_per_second"] for values in final_non_target),
            )
            max_non_target_pcie_tx_kib_per_second = max(
                max_non_target_pcie_tx_kib_per_second,
                *(values["pcie_tx_kib_per_second"] for values in final_non_target),
            )
        launch_time = time.time()
        if idle_since_wall is None:
            recorder.write("failed", reason="idle wall-clock evidence is missing")
            return 2
        context = {
            "target_uuid": target_uuid,
            "idle_activity_scope": args.idle_activity_scope,
            "process_scope": PROCESS_SCOPE,
            "attempt": attempt,
            "not_before": datetime.fromtimestamp(args.not_before)
            .astimezone()
            .isoformat(),
            "idle_seconds": args.idle_seconds,
            "wait_poll_seconds": args.wait_poll_seconds,
            "watchdog_seconds": args.watchdog_seconds,
            "finalize_seconds": args.finalize_seconds,
            "idle_started_at": datetime.fromtimestamp(idle_since_wall)
            .astimezone()
            .isoformat(),
            "launch_at": datetime.fromtimestamp(launch_time).astimezone().isoformat(),
            "idle_probe_count": idle_probe_count,
            "max_idle_gpu_utilization_percent": 0,
            "max_idle_memory_mib": max_idle_memory_mib,
            "max_idle_pcie_rx_kib_per_second": (
                max_idle_pcie_rx_kib_per_second
            ),
            "max_idle_pcie_tx_kib_per_second": (
                max_idle_pcie_tx_kib_per_second
            ),
            "max_idle_probe_gap_seconds": max_idle_probe_gap_seconds,
            "max_global_compute_process_count": 0,
            "max_non_target_gpu_utilization_percent": (
                max_non_target_gpu_utilization_percent
            ),
            "max_non_target_memory_mib": max_non_target_memory_mib,
            "max_non_target_pcie_rx_kib_per_second": (
                max_non_target_pcie_rx_kib_per_second
            ),
            "max_non_target_pcie_tx_kib_per_second": (
                max_non_target_pcie_tx_kib_per_second
            ),
            "maximum_allowed_pcie_kib_per_second": (
                MAX_IDLE_PCIE_KIB_PER_SECOND
            ),
            "gpu_count": len(inventory),
        }
        environment = _recovery_environment(
            {
                **os.environ,
                "SWITCHYARD_RUNNER_STATE": str(args.state.resolve()),
            },
            context,
            recorder.public_device_id,
        )
        if args.gpu_complete_marker is not None:
            args.gpu_complete_marker.unlink(missing_ok=True)
        recorder.set_recovery_context(context)
        recorder.write("launching_workload", attempt=attempt, target_uuid=target_uuid)
        args.log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with args.log.open("a") as log:
            os.chmod(args.log, 0o600)
            process = subprocess.Popen(
                command,
                cwd=args.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                pass_fds=(lock_handle.fileno(), gpu_lock_handle.fileno()),
            )
            recorder.write(
                "workload_started",
                attempt=attempt,
                target_uuid=target_uuid,
            )
            blind_probes = 0
            foreign_apps: list[dict[str, str | int]] = []
            gpu_complete_since: float | None = None
            deadline_reached = False
            finalization_timed_out = False
            watchdog_probe_count = 0
            watchdog_max_probe_gap_seconds = 0.0
            # Measure from process launch. This includes the first query's
            # runtime and prevents a stalled first query from reporting a zero gap.
            last_watchdog_probe_at = time.monotonic()
            while process.poll() is None:
                time.sleep(args.watchdog_seconds)
                if process.poll() is not None:
                    break
                if time.time() >= deadline:
                    _terminate_group(process, recorder, "campaign deadline reached")
                    deadline_reached = True
                    break
                if gpu_complete_since is not None:
                    if time.monotonic() - gpu_complete_since > args.finalize_seconds:
                        _terminate_group(
                            process,
                            recorder,
                            f"post-GPU finalization exceeded {args.finalize_seconds} seconds",
                        )
                        finalization_timed_out = True
                        break
                    continue
                try:
                    observed_apps = _compute_apps()
                    blind_probes = 0
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    blind_probes += 1
                    if blind_probes >= 3:
                        _terminate_group(
                            process,
                            recorder,
                            f"three watchdog failures: {type(exc).__name__}: {exc}"[:300],
                        )
                        foreign_apps = [{"watchdog": "unavailable"}]
                        break
                    continue
                last_watchdog_probe_at, watchdog_max_probe_gap_seconds = (
                    _record_probe_gap(
                        last_watchdog_probe_at,
                        watchdog_max_probe_gap_seconds,
                    )
                )
                watchdog_probe_count += 1
                if watchdog_max_probe_gap_seconds > MAX_WATCHDOG_PROBE_GAP_SECONDS:
                    _terminate_group(
                        process,
                        recorder,
                        "global GPU watchdog probe gap exceeded one second",
                    )
                    foreign_apps = [{"watchdog": "probe_gap"}]
                    break
                foreign_apps = [
                    item
                    for item in observed_apps
                    if not _is_descendant(int(item["pid"]), process.pid)
                ]
                if foreign_apps:
                    _terminate_group(process, recorder, "foreign GPU process appeared")
                    break
                if (
                    args.gpu_complete_marker is not None
                    and args.gpu_complete_marker.exists()
                ):
                    gpu_complete_since = time.monotonic()
                    recorder.write(
                        "gpu_phase_complete",
                        attempt=attempt,
                        watchdog_probe_count=watchdog_probe_count,
                        max_watchdog_probe_gap_seconds=(
                            watchdog_max_probe_gap_seconds
                        ),
                    )

            return_code = process.poll()

        if foreign_apps:
            recorder.write(
                "workload_requeued",
                attempt=attempt,
                foreign_process_count=len(foreign_apps),
                return_code=return_code,
            )
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            continue
        if deadline_reached:
            recorder.write("expired", attempts=attempt)
            return 75
        if finalization_timed_out:
            recovery_code = _run_cpu_recovery(
                command,
                cwd=args.cwd,
                environment=environment,
                log_path=args.log,
                recorder=recorder,
                attempt=attempt,
                deadline=deadline,
                finalize_seconds=args.finalize_seconds,
                retry_seconds=args.wait_poll_seconds,
                inherited_fds=(lock_handle.fileno(), gpu_lock_handle.fileno()),
            )
            if recovery_code == 75:
                recorder.write("cpu_recovery_requires_gpu", attempt=attempt)
                idle_since = None
                idle_since_wall = None
                idle_probe_count = 0
                max_idle_memory_mib = 0
                max_idle_pcie_rx_kib_per_second = 0
                max_idle_pcie_tx_kib_per_second = 0
                max_idle_probe_gap_seconds = 0.0
                last_idle_probe_at = None
                max_non_target_gpu_utilization_percent = 0
                max_non_target_memory_mib = 0
                max_non_target_pcie_rx_kib_per_second = 0
                max_non_target_pcie_tx_kib_per_second = 0
                continue
            return recovery_code
        if return_code == 0:
            recorder.write("complete", attempt=attempt, return_code=return_code)
            return 0
        if return_code == CPU_RECOVERY_EXIT:
            if args.gpu_complete_marker is None or not args.gpu_complete_marker.exists():
                recorder.write(
                    "failed",
                    attempt=attempt,
                    reason="CPU recovery was requested before GPU completion",
                )
                return 2
            recovery_code = _run_cpu_recovery(
                command,
                cwd=args.cwd,
                environment=environment,
                log_path=args.log,
                recorder=recorder,
                attempt=attempt,
                deadline=deadline,
                finalize_seconds=args.finalize_seconds,
                retry_seconds=args.wait_poll_seconds,
                inherited_fds=(lock_handle.fileno(), gpu_lock_handle.fileno()),
            )
            if recovery_code == 75:
                recorder.write("cpu_recovery_requires_gpu", attempt=attempt)
                idle_since = None
                idle_since_wall = None
                idle_probe_count = 0
                max_idle_memory_mib = 0
                max_idle_pcie_rx_kib_per_second = 0
                max_idle_pcie_tx_kib_per_second = 0
                max_idle_probe_gap_seconds = 0.0
                last_idle_probe_at = None
                max_non_target_gpu_utilization_percent = 0
                max_non_target_memory_mib = 0
                max_non_target_pcie_rx_kib_per_second = 0
                max_non_target_pcie_tx_kib_per_second = 0
                continue
            return recovery_code
        if return_code == 75:
            recorder.write("workload_requested_requeue", attempt=attempt)
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            max_idle_pcie_rx_kib_per_second = 0
            max_idle_pcie_tx_kib_per_second = 0
            max_idle_probe_gap_seconds = 0.0
            last_idle_probe_at = None
            max_non_target_gpu_utilization_percent = 0
            max_non_target_memory_mib = 0
            max_non_target_pcie_rx_kib_per_second = 0
            max_non_target_pcie_tx_kib_per_second = 0
            continue
        recorder.write("failed", attempt=attempt, return_code=return_code)
        return return_code or 1

    recorder.write("expired", attempts=attempt)
    return 75


if __name__ == "__main__":
    sys.exit(main())
