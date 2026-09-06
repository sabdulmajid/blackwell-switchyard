"""CPU-only checks for complete result-bundle reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backward_bundle.py"
SPEC = importlib.util.spec_from_file_location("check_backward_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _compile_report(*, stack_bytes=0):
    resources = []
    for kernel, count in MODULE.EXPECTED_CUDA_INSTANCES.items():
        resources.extend(
            {
                "kernel": kernel,
                "registers": MODULE.EXPECTED_CUDA_LIMITS[kernel],
                "stack_bytes": stack_bytes,
                "local_bytes": 0,
            }
            for _ in range(count)
        )
    compilations = [
        {
            "name": "one_read_cuda",
            "kind": "cuda",
            "required_template_instances": MODULE.EXPECTED_CUDA_INSTANCES,
            "register_limits": MODULE.EXPECTED_CUDA_LIMITS,
            "resources": resources,
        }
    ]
    compilations.extend(
        {
            "name": name,
            "kind": "triton",
            "resources": [
                {
                    "kernel": name,
                    "registers": 1,
                    "stack_bytes": 0,
                    "local_bytes": 0,
                }
            ],
        }
        for name in MODULE.EXPECTED_COMPILATIONS - {"one_read_cuda"}
    )
    return {
        "target": {"arch": 120},
        "gpu_visible": False,
        "provenance": {"repository_commit": "abc", "worktree_clean": True},
        "plans": [{"name": name} for name in MODULE.CANDIDATES],
        "compilations": compilations,
    }


def test_bundle_has_one_exact_file_for_every_phase_and_candidate():
    suffixes = MODULE._suffixes()
    assert len(suffixes) == 8 + len(MODULE.CANDIDATES)
    assert len(suffixes) == len(set(suffixes))


def test_compile_gate_accepts_only_clean_spill_free_resources():
    assert not MODULE._compile_problems(_compile_report(), "abc")
    problems = MODULE._compile_problems(_compile_report(stack_bytes=8), "abc")
    assert any("spills" in problem for problem in problems)


def _valid_bundle(monkeypatch):
    campaign_id = "20260905T210000Z"
    prefix = Path("results") / f"backward_campaign_{campaign_id}"
    paths = {
        suffix: prefix.with_name(f"{prefix.name}_{suffix}.json")
        for suffix in MODULE._suffixes()
    }
    payloads = {
        "offline_compile": {},
        "smoke_bfloat16": {"run_id": "20260905T205900Z"},
        "smoke_float16": {"run_id": "20260905T205901Z"},
        "full_bfloat16": {"run_id": campaign_id},
        "full_float16": {"run_id": "20260905T210001Z"},
        "full_float32": {"run_id": "20260905T210002Z"},
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
        "input_reports": [
            {"sha256": hashlib.sha256(item).hexdigest()} for item in smoke_raw
        ],
    }
    for candidate in MODULE.CANDIDATES:
        report_suffixes = ["full_bfloat16", "full_float16"]
        if candidate.startswith("serial_"):
            report_suffixes.append("full_float32")
        payloads[f"decision_{candidate}"] = {
            "status": "DROP",
            "candidate": candidate,
            "input_reports": [
                {"sha256": hashlib.sha256(raw[suffix]).hexdigest()}
                for suffix in report_suffixes
            ],
        }
    payloads["manifest"] = {
        "schema_version": 1,
        "repository_commit": "abc",
        "repository_tree": "tree",
        "benchmark_branch": "codex/campaign",
        "result_branch": "codex/results",
        "origin": "https://github.com/sabdulmajid/blackwell-switchyard.git",
        "gpu_uuid": "GPU-test",
        "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
        "attempt": 1,
        "guard": {
            "not_before": "2026-09-05T23:00:00-04:00",
            "idle_seconds": 1800,
            "wait_poll_seconds": 60,
            "watchdog_seconds": 0.25,
            "finalize_seconds": 1800,
        },
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
        "expected_gpu_uuid": "GPU-test",
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
