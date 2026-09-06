#!/usr/bin/env python3
"""Run one command after all GPUs have been idle for a sustained interval.

The runner uses no GPU while it waits. It checks NVIDIA's process table at a
low frequency. During the command, it monitors only the selected GPU. If an
unrelated process appears there, it stops its own process group and waits for a
new idle interval before one bounded retry.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path


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


class StateRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.events: list[dict] = []

    def write(self, phase: str, **fields) -> None:
        event = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "phase": phase,
            **fields,
        }
        self.events.append(event)
        payload = {"current": event, "events": self.events[-200:]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.path)
        print(json.dumps(event, sort_keys=True), flush=True)


def _terminate_group(process: subprocess.Popen, recorder: StateRecorder, reason: str) -> None:
    recorder.write("stopping_workload", pid=process.pid, reason=reason)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


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
    parser.add_argument("--watchdog-seconds", type=int, default=2)
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

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another runner owns {args.lock}") from None
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    recorder = StateRecorder(args.state)
    deadline = time.time() + args.deadline_hours * 3600
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
    attempt = 0
    while time.time() < deadline and attempt < args.max_attempts:
        try:
            apps = _compute_apps()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            recorder.write("wait_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue

        if apps:
            idle_since = None
            time.sleep(args.wait_poll_seconds)
            continue
        if idle_since is None:
            idle_since = time.monotonic()
            recorder.write("idle_grace_started")
        elapsed = time.monotonic() - idle_since
        if elapsed < args.idle_seconds:
            time.sleep(min(args.wait_poll_seconds, args.idle_seconds - elapsed))
            continue

        # Close the final race before launch. The workload has its own GPU
        # preflight as a second independent check.
        try:
            final_apps = _compute_apps()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            recorder.write("final_probe_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            time.sleep(args.wait_poll_seconds)
            continue
        if final_apps:
            idle_since = None
            continue

        attempt += 1
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = target_uuid
        environment["SWITCHYARD_TARGET_GPU_UUID"] = target_uuid
        environment["SWITCHYARD_RUN_ATTEMPT"] = str(attempt)
        environment["SWITCHYARD_GUARD_NOT_BEFORE"] = datetime.fromtimestamp(
            args.not_before
        ).astimezone().isoformat()
        environment["SWITCHYARD_GUARD_IDLE_SECONDS"] = str(args.idle_seconds)
        environment["SWITCHYARD_GUARD_WAIT_POLL_SECONDS"] = str(args.wait_poll_seconds)
        environment["SWITCHYARD_GUARD_WATCHDOG_SECONDS"] = str(args.watchdog_seconds)
        environment["SWITCHYARD_GUARD_FINALIZE_SECONDS"] = str(args.finalize_seconds)
        if args.gpu_complete_marker is not None:
            args.gpu_complete_marker.unlink(missing_ok=True)
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
                pid=process.pid,
                target_uuid=target_uuid,
            )
            blind_probes = 0
            foreign_apps: list[dict[str, str | int]] = []
            gpu_complete_since: float | None = None
            while process.poll() is None:
                time.sleep(args.watchdog_seconds)
                if process.poll() is not None:
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
                foreign_apps=foreign_apps,
                return_code=return_code,
            )
            idle_since = None
            continue
        if return_code == 0:
            recorder.write("complete", attempt=attempt, return_code=return_code)
            return 0
        if return_code == 75:
            recorder.write("workload_requested_requeue", attempt=attempt)
            idle_since = None
            continue
        recorder.write("failed", attempt=attempt, return_code=return_code)
        return return_code or 1

    recorder.write("expired", attempts=attempt)
    return 75


if __name__ == "__main__":
    sys.exit(main())
