"""CPU-only tests for release performance-baseline selection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_regression.py"
SPEC = importlib.util.spec_from_file_location("check_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BENCHMARK = SCRIPT.parents[1] / "bench" / "bench_operator.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "regression_gate_bench_operator", BENCHMARK
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK_MODULE = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK_MODULE
BENCHMARK_SPEC.loader.exec_module(BENCHMARK_MODULE)

REPO = SCRIPT.parents[1]
TEST_COMMIT = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
TEST_TREE = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD^{tree}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()


def _write_report(
    path: Path,
    *,
    commit: str = TEST_COMMIT,
    tree: str = TEST_TREE,
    dirty: bool = False,
    untracked_dirty: bool = False,
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
                    "repository_tree": tree,
                    "repository_branch": "codex/test",
                    "worktree_dirty": dirty or untracked_dirty,
                    "tracked_worktree_dirty": dirty,
                    "dirty_paths": ["private.tmp"] if dirty or untracked_dirty else [],
                    "diff_sha256": (
                        "c" * 64 if dirty else MODULE.EMPTY_DIFF_SHA256
                    ),
                },
                "dtype": "bfloat16",
                "shape_set": "default",
                "quick": quick,
                "results": [
                    {
                        "impl": "switchyard_triton",
                        "shape": {"n": n, "b": b, "t": t, "d": d},
                        "forward": {"median_ms": 0.1},
                        "fwd_bwd": {"median_ms": 0.3},
                        "forward_memory": {"workspace_bytes": 0},
                    }
                    for n, b, t, d in sorted(MODULE.EXPECTED_DEFAULT_SHAPES)
                ],
            }
        )
    )


def test_release_report_selection_is_explicit_and_path_private(tmp_path):
    report = tmp_path / "operator_default_bfloat16.json"
    _write_report(report)
    records, provenance = MODULE.load_current([report])
    assert len(records) == len(MODULE.EXPECTED_DEFAULT_SHAPES)
    assert provenance["repository_commit"] == TEST_COMMIT
    assert provenance["reports"][0]["name"] == report.name
    assert str(tmp_path) not in json.dumps(provenance)


def test_release_shape_contract_matches_the_benchmark_default():
    actual = {
        (shape.n, shape.b, shape.t, shape.d)
        for shape in BENCHMARK_MODULE.SHAPE_SETS["default"]
    }
    assert actual == MODULE.EXPECTED_DEFAULT_SHAPES


@pytest.mark.parametrize(
    "change",
    [
        {"dirty": True},
        {"untracked_dirty": True},
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
    first = tmp_path / "first" / "operator_default_bfloat16.json"
    second = tmp_path / "second" / "operator_default_bfloat16.json"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_report(first)
    _write_report(second)
    with pytest.raises(ValueError, match="duplicate current measurement"):
        MODULE.load_current([first, second])

    parent_commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent_tree = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD^^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write_report(second, commit=parent_commit, tree=parent_tree)
    with pytest.raises(ValueError, match="one repository commit"):
        MODULE.load_current([first, second])


def test_release_report_rejects_nonhex_commit_zero_latency_and_private_name(tmp_path):
    report = tmp_path / "operator_default_bfloat16.json"
    _write_report(report, commit="z" * 40)
    with pytest.raises(ValueError, match="clean, full-profile release run"):
        MODULE.load_current([report])

    _write_report(report)
    data = json.loads(report.read_text())
    data["results"][0]["forward"]["median_ms"] = 0.0
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid forward ms"):
        MODULE.load_current([report])

    private = tmp_path / "operator_default_bfloat16_session-secret.json"
    _write_report(private)
    with pytest.raises(ValueError, match="non-release filename"):
        MODULE.load_current([private])


def _baseline(report: Path, *, key: str | None = None, torch: str = "2.9") -> dict:
    records, provenance = MODULE.load_current([report])
    metrics = MODULE.compact(records)
    if key is not None:
        metrics = {key: next(iter(metrics.values()))}
    environment = dict(provenance["environment"])
    environment["torch"] = torch
    return {
        "schema_version": 2,
        "environment": environment,
        "provenance": provenance,
        "metrics": metrics,
    }


def test_regression_gate_fails_when_a_required_row_is_missing(tmp_path, monkeypatch):
    report = tmp_path / "operator_default_bfloat16.json"
    baseline = tmp_path / "operator_baseline.json"
    _write_report(report)
    baseline.write_text(
        json.dumps(
            _baseline(
                report,
                key="switchyard_triton|N99|B1|T4096|D2048|bfloat16",
            )
        )
    )
    monkeypatch.setattr(MODULE, "BASELINE_FILE", baseline)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--current", str(report)])
    assert MODULE.main() == 1


def test_regression_gate_rejects_environment_mismatch(tmp_path, monkeypatch):
    report = tmp_path / "operator_default_bfloat16.json"
    baseline = tmp_path / "operator_baseline.json"
    _write_report(report)
    baseline.write_text(json.dumps(_baseline(report, torch="2.8")))
    monkeypatch.setattr(MODULE, "BASELINE_FILE", baseline)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--current", str(report)])
    assert MODULE.main() == 2


def test_baseline_acceptance_whitelists_environment_metadata(tmp_path, monkeypatch):
    report = tmp_path / "operator_default_bfloat16.json"
    baseline = tmp_path / "operator_baseline.json"
    _write_report(report)
    data = json.loads(report.read_text())
    data["environment"]["private_path"] = "/private/worktree"
    report.write_text(json.dumps(data))
    monkeypatch.setattr(MODULE, "BASELINE_FILE", baseline)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--accept", "--current", str(report)])
    assert MODULE.main() == 0
    stored = json.loads(baseline.read_text())
    assert set(stored["environment"]) == set(MODULE.RELEASE_ENV_FIELDS)
    assert "private_path" not in baseline.read_text()
