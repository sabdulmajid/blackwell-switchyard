"""CPU-only checks for complete result-bundle reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backward_bundle.py"
COMPILE_REPORT = (
    Path(__file__).resolve().parents[1] / "results" / "backward_candidates_compile_sm120.json"
)
SPEC = importlib.util.spec_from_file_location("check_backward_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _compile_report(*, stack_bytes=0):
    report = json.loads(COMPILE_REPORT.read_text())
    if stack_bytes:
        report["compilations"][-1]["resources"][0]["stack_bytes"] = stack_bytes
    return report


def _compile_arguments(report):
    provenance = report["provenance"]
    contract = json.loads(COMPILE_REPORT.read_text())
    return provenance["repository_commit"], provenance["cuda_source_sha256"], contract


def test_bundle_has_one_exact_file_for_every_phase_and_candidate():
    suffixes = MODULE._suffixes()
    assert len(suffixes) == 8 + len(MODULE.CANDIDATES)
    assert len(suffixes) == len(set(suffixes))


def test_compile_gate_accepts_only_clean_spill_free_resources():
    report = _compile_report()
    assert not MODULE._compile_problems(report, *_compile_arguments(report))
    report = _compile_report(stack_bytes=8)
    problems = MODULE._compile_problems(report, *_compile_arguments(report))
    assert any("spills" in problem for problem in problems)


def test_compile_gate_rejects_unknown_nested_fields():
    report = _compile_report()
    report["provenance"]["private_note"] = "must not be published"
    report["compilations"][-1]["resources"][0]["prompt"] = "must not be published"
    problems = MODULE._compile_problems(report, *_compile_arguments(report))
    assert any("compile provenance field set is not exact" in item for item in problems)
    assert any("CUDA resources[0] field set is not exact" in item for item in problems)


def test_compile_gate_binds_plan_and_triton_compiler_contracts():
    mutations = [
        lambda report: report["plans"][0].update(rationale="different"),
        lambda report: report["compilations"][0]["constants"].update(BLOCK_D=2048),
        lambda report: report["compilations"][0]["resources"][0].update(registers=1),
        lambda report: report["compilations"][0].update(spill_policy="record"),
        lambda report: report["target"].update(backend="not-cuda"),
        lambda report: report["provenance"].update(triton="different"),
    ]
    for mutate in mutations:
        report = _compile_report()
        mutate(report)
        assert MODULE._compile_problems(report, *_compile_arguments(report))


def test_compile_gate_rejects_malformed_lists_without_crashing():
    report = _compile_report()
    report["compilations"][0] = "not-an-object"
    assert MODULE._compile_problems(report, *_compile_arguments(report))

    report = _compile_report()
    report["compilations"][-1]["resources"][0]["registers"] = []
    assert MODULE._compile_problems(report, *_compile_arguments(report))


def test_private_metadata_filter_covers_common_host_paths_and_names():
    rejected = [
        "/mnt/team/run.json",
        "/scratch/user/run.json",
        "/workspace/project/run.json",
        r"C:\Users\person\run.json",
        "worker.cluster.internal",
        "buildbox.lab.local",
    ]
    for value in rejected:
        assert MODULE.PRIVATE_METADATA_PATTERN.search(value), value


def _valid_bundle(monkeypatch):
    campaign_id = "20260906T034000Z"
    prefix = Path("results") / f"backward_campaign_{campaign_id}"
    paths = {
        suffix: prefix.with_name(f"{prefix.name}_{suffix}.json") for suffix in MODULE._suffixes()
    }
    payloads = {
        "offline_compile": {},
        "smoke_bfloat16": {"run_id": "20260906T033100Z", "campaign_attempt": 1},
        "smoke_float16": {"run_id": "20260906T033200Z", "campaign_attempt": 1},
        "full_bfloat16": {"run_id": campaign_id, "campaign_attempt": 1},
        "full_float16": {"run_id": "20260906T034100Z", "campaign_attempt": 1},
        "full_float32": {"run_id": "20260906T034200Z", "campaign_attempt": 1},
    }
    monkeypatch.setattr(MODULE, "_compile_problems", lambda *_args: [])
    monkeypatch.setattr(MODULE, "validate_report", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(MODULE, "check_reports", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        MODULE,
        "evaluate_reports",
        lambda _reports, *, candidate, **_kwargs: {
            "status": "DROP",
            "candidate": candidate,
        },
    )

    raw = {suffix: json.dumps(payload).encode() for suffix, payload in payloads.items()}
    smoke_raw = [raw["smoke_bfloat16"], raw["smoke_float16"]]
    payloads["smoke_decision"] = {
        "status": "PASS",
        "input_reports": [{"sha256": hashlib.sha256(item).hexdigest()} for item in smoke_raw],
    }
    for candidate in MODULE.CANDIDATES:
        report_suffixes = ["full_bfloat16", "full_float16"]
        if candidate.startswith("serial_"):
            report_suffixes.append("full_float32")
        payloads[f"decision_{candidate}"] = {
            "status": "DROP",
            "candidate": candidate,
            "input_reports": [
                {"sha256": hashlib.sha256(raw[suffix]).hexdigest()} for suffix in report_suffixes
            ],
        }
    payloads["manifest"] = {
        "schema_version": 1,
        "repository_commit": "abc",
        "repository_tree": "tree",
        "benchmark_branch": "codex/campaign",
        "result_branch": "codex/results",
        "origin": "https://github.com/sabdulmajid/blackwell-switchyard.git",
        "device_id": "device-0123456789abcdef",
        "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
        "attempt": 1,
        "guard_attestations": [
            {
                "attempt": 1,
                "not_before": "2026-09-05T23:00:00-04:00",
                "idle_started_at": "2026-09-05T23:00:00-04:00",
                "launch_at": "2026-09-05T23:30:00-04:00",
                "idle_seconds": 1800,
                "idle_probe_count": 31,
                "max_idle_gpu_utilization_percent": 0,
                "max_idle_memory_mib": 2,
                "wait_poll_seconds": 60,
                "watchdog_seconds": 0.25,
                "finalize_seconds": 1800,
                "gpu_count": 2,
                "device_id": "device-0123456789abcdef",
            }
        ],
    }
    raw = {suffix: json.dumps(payload).encode() for suffix, payload in payloads.items()}
    by_path = {paths[suffix]: value for suffix, value in raw.items()}
    arguments = {
        "read_bytes": by_path.__getitem__,
        "changed_paths": set(paths.values()),
        "expected_commit": "abc",
        "expected_tree": "tree",
        "expected_branch": "codex/campaign",
        "result_branch": "codex/results",
        "expected_origin": "https://github.com/sabdulmajid/blackwell-switchyard.git",
        "expected_device_id": "device-0123456789abcdef",
    }
    return prefix, by_path, arguments


def test_complete_bundle_reconstructs_every_stored_decision(monkeypatch):
    prefix, _by_path, arguments = _valid_bundle(monkeypatch)
    assert not MODULE.validate_bundle(prefix, **arguments)


def test_bundle_rejects_report_bytes_changed_after_decision(monkeypatch):
    prefix, by_path, arguments = _valid_bundle(monkeypatch)
    full_bf16 = prefix.with_name(f"{prefix.name}_full_bfloat16.json")
    by_path[full_bf16] += b" "
    problems = MODULE.validate_bundle(prefix, **arguments)
    assert any("does not reconstruct exactly" in problem for problem in problems)


def test_bundle_rejects_weakened_collision_guards(monkeypatch):
    unsafe_values = {
        "idle_seconds": MODULE.MIN_IDLE_SECONDS - 1,
        "wait_poll_seconds": MODULE.MIN_WAIT_POLL_SECONDS - 1,
        "watchdog_seconds": MODULE.MAX_WATCHDOG_SECONDS + 0.01,
        "finalize_seconds": MODULE.MIN_FINALIZE_SECONDS - 1,
        "max_idle_gpu_utilization_percent": 1,
        "max_idle_memory_mib": 65,
        "gpu_count": 1,
    }
    for field, value in unsafe_values.items():
        prefix, by_path, arguments = _valid_bundle(monkeypatch)
        manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
        manifest = json.loads(by_path[manifest_path])
        manifest["guard_attestations"][0][field] = value
        by_path[manifest_path] = json.dumps(manifest).encode()
        problems = MODULE.validate_bundle(prefix, **arguments)
        assert any("guard" in problem and field in problem for problem in problems), field


def test_bundle_rejects_nonstandard_nonfinite_json_numbers(monkeypatch):
    for field in (
        "watchdog_seconds",
        "finalize_seconds",
        "gpu_count",
        "idle_probe_count",
    ):
        prefix, by_path, arguments = _valid_bundle(monkeypatch)
        manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
        manifest = json.loads(by_path[manifest_path])
        manifest["guard_attestations"][0][field] = float("nan")
        by_path[manifest_path] = json.dumps(manifest).encode()
        problems = MODULE.validate_bundle(prefix, **arguments)
        assert any("cannot read" in problem for problem in problems), field


def test_bundle_binds_guard_times_and_target_gpu(monkeypatch):
    unsafe_values = {
        "idle_started_at": "2026-09-05T23:29:00-04:00",
        "launch_at": "2026-09-05T22:59:00-04:00",
        "idle_probe_count": 1,
        "device_id": "device-fedcba9876543210",
    }
    for field, value in unsafe_values.items():
        prefix, by_path, arguments = _valid_bundle(monkeypatch)
        manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
        manifest = json.loads(by_path[manifest_path])
        manifest["guard_attestations"][0][field] = value
        by_path[manifest_path] = json.dumps(manifest).encode()
        assert MODULE.validate_bundle(prefix, **arguments), field


def test_bundle_binds_each_phase_to_its_generating_attempt(monkeypatch):
    prefix, by_path, arguments = _valid_bundle(monkeypatch)
    manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
    manifest = json.loads(by_path[manifest_path])
    manifest["attempt"] = 2
    manifest["guard_attestations"].append(
        {
            **manifest["guard_attestations"][0],
            "attempt": 2,
            "idle_started_at": "2026-09-06T04:00:00+00:00",
            "launch_at": "2026-09-06T04:30:00+00:00",
        }
    )
    by_path[manifest_path] = json.dumps(manifest).encode()
    full_bf16 = prefix.with_name(f"{prefix.name}_full_bfloat16.json")
    report = json.loads(by_path[full_bf16])
    report["campaign_attempt"] = 2
    by_path[full_bf16] = json.dumps(report).encode()
    problems = MODULE.validate_bundle(prefix, **arguments)
    assert any("predates its guarded launch" in problem for problem in problems)
