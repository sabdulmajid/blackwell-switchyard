#!/usr/bin/env python3
"""Verify that a decision is bound to the exact report bytes it evaluated."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def binding_problems_from_bytes(decision_bytes: bytes, report_bytes: list[bytes]) -> list[str]:
    try:
        decision = json.loads(decision_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ["decision is not valid UTF-8 JSON"]
    expected = [{"sha256": hashlib.sha256(raw).hexdigest()} for raw in report_bytes]
    if decision.get("input_reports") != expected:
        return ["decision input hashes do not match the current report bytes"]
    return []


def binding_problems(decision: dict, reports: list[Path]) -> list[str]:
    expected = [{"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in reports]
    return [] if decision.get("input_reports") == expected else [
        "decision input hashes do not match the current report bytes"
    ]


def _git_blob(ref: str, path: Path) -> bytes:
    object_name = f":{path.as_posix()}" if ref == ":" else f"{ref}:{path.as_posix()}"
    return subprocess.run(
        ["git", "show", object_name],
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--git-ref",
        help="read every path from this Git tree (use ':' for the index)",
    )
    args = parser.parse_args()
    try:
        if args.git_ref:
            decision_bytes = _git_blob(args.git_ref, args.decision)
            report_bytes = [_git_blob(args.git_ref, path) for path in args.reports]
        else:
            decision_bytes = args.decision.read_bytes()
            report_bytes = [path.read_bytes() for path in args.reports]
        problems = binding_problems_from_bytes(decision_bytes, report_bytes)
    except (OSError, subprocess.SubprocessError) as exc:
        problems = [f"cannot read evidence: {type(exc).__name__}"]
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
