#!/usr/bin/env python3
"""Run one command after all GPUs have been idle for a sustained interval.

The runner uses no GPU while it waits. At a low frequency, it checks NVIDIA's
process table, utilization, and allocated memory. During the command, it
monitors only the selected GPU. If an unrelated process appears there, it stops
its own process group and waits for a new idle interval before the next bounded
attempt.
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
PUBLIC_DEVICE_ID_PATTERN = re.compile(r"device-[0-9a-f]{16}")


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


def _parent_pid(pid: int) -> int | None:
    try:
        # Everything after the final ')' starts with state and parent PID.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1])
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return None


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
    return _parse_activity(
        _smi(
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        )
    )


def _is_idle_activity(
    activity: dict[int, dict[str, int]], inventory: dict[int, str]
) -> bool:
    return set(activity) == set(inventory) and all(
        values["utilization_percent"] == 0
        and values["memory_used_mib"] <= MAX_IDLE_MEMORY_MIB
        for values in activity.values()
    )


class StateRecorder:
    def __init__(self, path: Path, campaign_identity: str):
        self.path = path
        self.campaign_identity = campaign_identity
        self.events: list[dict] = []
        self._attempts = 0
        self.public_device_id = f"device-{secrets.token_hex(8)}"
        if path.exists():
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

    @property
    def last_phase(self) -> str | None:
        return self.events[-1].get("phase") if self.events else None

    @property
    def attempts(self) -> int:
        return self._attempts

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
        payload = {
            "campaign_identity": self.campaign_identity,
            "public_device_id": self.public_device_id,
            "attempts": self._attempts,
            "current": event,
            "events": self.events[-200:],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.path)
        print(json.dumps(event, sort_keys=True), flush=True)


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
) -> int:
    """Retry post-GPU publication without querying a GPU or spending an attempt."""
    cpu_environment = environment.copy()
    cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
    cpu_environment["SWITCHYARD_CPU_RECOVERY"] = "1"
    while time.time() < deadline:
        recorder.write("cpu_recovery_started", attempt=attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=cpu_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
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

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock.open("w")
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
    if attempt >= args.max_attempts:
        recorder.write("expired", attempts=attempt)
        return 75
    if time.time() < args.not_before:
        recorder.write(
            "sleeping_until_not_before",
            not_before=datetime.fromtimestamp(args.not_before).astimezone().isoformat(),
        )
        time.sleep(max(0.0, args.not_before - time.time()))
    deadline = time.time() + args.deadline_hours * 3600

    inventory = _inventory()
    if args.gpu_index not in inventory:
        recorder.write("failed", reason=f"GPU index {args.gpu_index} does not exist")
        return 2
    target_uuid = inventory[args.gpu_index]
    args.gpu_lock_dir.mkdir(parents=True, exist_ok=True)
    gpu_lock_path = args.gpu_lock_dir / f"blackwell-switchyard-{target_uuid}.lock"
    gpu_lock_handle = gpu_lock_path.open("w")
    try:
        fcntl.flock(gpu_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        recorder.write("failed", reason=f"another switchyard runner owns {gpu_lock_path}")
        return 75
    gpu_lock_handle.write(f"{os.getpid()}\n")
    gpu_lock_handle.flush()
    recorder.write(
        "waiting_for_all_gpus_idle",
        target_index=args.gpu_index,
        target_uuid=target_uuid,
        required_idle_seconds=args.idle_seconds,
        wait_poll_seconds=args.wait_poll_seconds,
    )

    idle_since: float | None = None
    idle_since_wall: float | None = None
    idle_probe_count = 0
    max_idle_memory_mib = 0
    while time.time() < deadline and attempt < args.max_attempts:
        try:
            apps = _compute_apps()
            activity = _activity()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            recorder.write("wait_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue

        if apps or not _is_idle_activity(activity, inventory):
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            time.sleep(args.wait_poll_seconds)
            continue
        if idle_since is None:
            idle_since = time.monotonic()
            idle_since_wall = time.time()
            recorder.write("idle_grace_started")
        idle_probe_count += 1
        max_idle_memory_mib = max(
            max_idle_memory_mib,
            *(values["memory_used_mib"] for values in activity.values()),
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
            recorder.write("final_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue
        if final_apps or not _is_idle_activity(final_activity, inventory):
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            continue

        attempt += 1
        idle_probe_count += 1
        max_idle_memory_mib = max(
            max_idle_memory_mib,
            *(values["memory_used_mib"] for values in final_activity.values()),
        )
        launch_time = time.time()
        if idle_since_wall is None:
            recorder.write("failed", reason="idle wall-clock evidence is missing")
            return 2
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = target_uuid
        environment["SWITCHYARD_TARGET_GPU_UUID"] = target_uuid
        environment["SWITCHYARD_PUBLIC_DEVICE_ID"] = recorder.public_device_id
        environment["SWITCHYARD_RUN_ATTEMPT"] = str(attempt)
        environment["SWITCHYARD_GUARD_NOT_BEFORE"] = datetime.fromtimestamp(
            args.not_before
        ).astimezone().isoformat()
        environment["SWITCHYARD_GUARD_IDLE_SECONDS"] = str(args.idle_seconds)
        environment["SWITCHYARD_GUARD_WAIT_POLL_SECONDS"] = str(args.wait_poll_seconds)
        environment["SWITCHYARD_GUARD_WATCHDOG_SECONDS"] = str(args.watchdog_seconds)
        environment["SWITCHYARD_GUARD_FINALIZE_SECONDS"] = str(args.finalize_seconds)
        environment["SWITCHYARD_GUARD_IDLE_STARTED_AT"] = datetime.fromtimestamp(
            idle_since_wall
        ).astimezone().isoformat()
        environment["SWITCHYARD_GUARD_LAUNCH_AT"] = datetime.fromtimestamp(
            launch_time
        ).astimezone().isoformat()
        environment["SWITCHYARD_GUARD_IDLE_PROBE_COUNT"] = str(idle_probe_count)
        environment["SWITCHYARD_GUARD_MAX_IDLE_GPU_UTILIZATION_PERCENT"] = "0"
        environment["SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB"] = str(max_idle_memory_mib)
        environment["SWITCHYARD_GUARD_GPU_COUNT"] = str(len(inventory))
        if args.gpu_complete_marker is not None:
            args.gpu_complete_marker.unlink(missing_ok=True)
        recorder.write("launching_workload", attempt=attempt, target_uuid=target_uuid)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=args.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
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
            while process.poll() is None:
                time.sleep(args.watchdog_seconds)
                if process.poll() is not None:
                    break
                if time.time() >= deadline:
                    _terminate_group(process, recorder, "campaign deadline reached")
                    deadline_reached = True
                    break
                if (
                    args.gpu_complete_marker is not None
                    and args.gpu_complete_marker.exists()
                ):
                    if gpu_complete_since is None:
                        gpu_complete_since = time.monotonic()
                        recorder.write("gpu_phase_complete", attempt=attempt)
                    elif time.monotonic() - gpu_complete_since > args.finalize_seconds:
                        _terminate_group(
                            process,
                            recorder,
                            f"post-GPU finalization exceeded {args.finalize_seconds} seconds",
                        )
                        finalization_timed_out = True
                        break
                    continue
                try:
                    target_apps = [
                        item for item in _compute_apps() if item["gpu_uuid"] == target_uuid
                    ]
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
                foreign_apps = [
                    item
                    for item in target_apps
                    if not _is_descendant(int(item["pid"]), process.pid)
                ]
                if foreign_apps:
                    _terminate_group(process, recorder, "foreign GPU process appeared")
                    break

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
            )
            if recovery_code == 75:
                recorder.write("cpu_recovery_requires_gpu", attempt=attempt)
                idle_since = None
                idle_since_wall = None
                idle_probe_count = 0
                max_idle_memory_mib = 0
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
            )
            if recovery_code == 75:
                recorder.write("cpu_recovery_requires_gpu", attempt=attempt)
                idle_since = None
                idle_since_wall = None
                idle_probe_count = 0
                max_idle_memory_mib = 0
                continue
            return recovery_code
        if return_code == 75:
            recorder.write("workload_requested_requeue", attempt=attempt)
            idle_since = None
            idle_since_wall = None
            idle_probe_count = 0
            max_idle_memory_mib = 0
            continue
        recorder.write("failed", attempt=attempt, return_code=return_code)
        return return_code or 1

    recorder.write("expired", attempts=attempt)
    return 75


if __name__ == "__main__":
    sys.exit(main())
