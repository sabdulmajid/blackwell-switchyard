"""CPU-only contract tests for the private backward benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "bench_backward.py"
SPEC = importlib.util.spec_from_file_location("bench_backward", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_benchmark_registers_complete_private_plans():
    values = torch.empty(9, 1, 1, 16)
    implementations, _ = MODULE.build_implementations(values)
    expected = {
        "current",
        "source_serial",
        "serial_recompute_atomic_t4",
        "serial_saved_atomic_t4",
        "serial_saved_partials_t16",
        "cuda_shared",
        "cuda_cluster",
    }
    assert expected <= set(implementations)
    for name in expected - {"current"}:
        assert implementations[name]["plan"].production is False


def test_trial_summary_preserves_independent_trial_medians():
    trials = [
        {"samples_ms": [float(index), float(index + 2)]}
        for index in range(1, 6)
    ]
    summary = MODULE._summarize_trials(trials, warmup=25)
    assert summary["trial_medians_ms"] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert summary["trial_count"] == 5
    assert summary["reps"] == 10
