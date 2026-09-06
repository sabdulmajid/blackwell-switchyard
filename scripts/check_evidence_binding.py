#!/usr/bin/env python3
"""Verify that a decision is bound to the exact report bytes it evaluated."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def binding_problems(decision: dict, reports: list[Path]) -> list[str]:
    expected = [
        {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in reports
    ]
    if decision.get("input_reports") != expected:
        return ["decision input hashes do not match the current report bytes"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        decision = json.loads(args.decision.read_text())
        problems = binding_problems(decision, args.reports)
    except (OSError, json.JSONDecodeError) as exc:
        problems = [f"cannot read evidence: {type(exc).__name__}"]
    for problem in problems:
        print(problem, file=sys.stderr)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
