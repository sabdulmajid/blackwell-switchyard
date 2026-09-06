"""CPU-only tests for release performance-baseline selection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_regression.py"
SPEC = importlib.util.spec_from_file_location("check_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_report(
    path: Path,
    *,
    commit: str = "a" * 40,
    dirty: bool = False,
    quick: bool = False,
    skip_profile: bool = False,
) -> None:
    argv = ["bench/bench_operator.py", "--set", "default"]
    if skip_profile:
        argv.append("--skip-kernel-profile")
    path.write_text(
        json.dumps(
            {
                "environment": {
                    "device_name": "GPU",
                    "device_cc": "12.0",
                    "torch": "2.9",
                    "triton": "3.5",
                },
                "provenance": {
                    "argv": argv,
                    "repository_commit": commit,
                    "tracked_worktree_dirty": dirty,
                },
                "dtype": "bfloat16",
                "quick": quick,
                "results": [
                    {
                        "impl": "switchyard_triton",
                        "shape": {"n": 9, "b": 1, "t": 4096, "d": 2048},
                        "forward": {"median_ms": 0.1},
                        "fwd_bwd": {"median_ms": 0.3},
                        "forward_memory": {"workspace_bytes": 0},
                    }
                ],
            }
        )
    )


def test_release_report_selection_is_explicit_and_path_private(tmp_path):
    report = tmp_path / "operator_default_bfloat16.json"
    _write_report(report)
    records, provenance = MODULE.load_current([report])
    assert len(records) == 1
    assert provenance["repository_commit"] == "a" * 40
    assert provenance["reports"][0]["name"] == report.name
    assert str(tmp_path) not in json.dumps(provenance)


@pytest.mark.parametrize(
    "change",
    [
        {"dirty": True},
        {"quick": True},
        {"skip_profile": True},
    ],
)
def test_release_report_selection_rejects_weakened_runs(tmp_path, change):
    report = tmp_path / "operator_default_bfloat16.json"
    _write_report(report, **change)
    with pytest.raises(ValueError, match="clean, full-profile release run"):
        MODULE.load_current([report])


def test_release_report_selection_rejects_duplicates_and_mixed_commits(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first)
    _write_report(second)
    with pytest.raises(ValueError, match="duplicate current measurement"):
        MODULE.load_current([first, second])

    _write_report(second, commit="b" * 40)
    with pytest.raises(ValueError, match="one repository commit"):
        MODULE.load_current([first, second])
