"""CPU-only checks for the DDP reduction acceptance rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "bench_ddp.py"
SPEC = importlib.util.spec_from_file_location("bench_ddp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_ddp_gradient_average_passes():
    result = MODULE._gradient_check(
        0.0,
        2.0,
        rank_count=2,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    assert result["passed"] is True
    assert result["effective_tolerance"] == 0.0


def test_ddp_gradient_mismatch_fails_the_default_rule():
    result = MODULE._gradient_check(
        1e-5,
        2.0,
        rank_count=2,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    assert result["passed"] is False


def test_ddp_gradient_tolerance_is_explicit_and_recorded():
    result = MODULE._gradient_check(
        2e-3,
        2.0,
        rank_count=2,
        absolute_tolerance=1e-3,
        relative_tolerance=1e-3,
    )
    assert result["passed"] is True
    assert result["effective_tolerance"] == 3e-3
