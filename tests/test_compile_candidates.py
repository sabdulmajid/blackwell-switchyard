"""CPU-only checks for the offline campaign compilation matrix."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compile_candidates.py"
SPEC = importlib.util.spec_from_file_location("compile_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    SPEC.loader.exec_module(MODULE)
finally:
    if _VISIBLE_DEVICES is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _VISIBLE_DEVICES


def _config(spec):
    return (*spec.constants.values(), spec.num_warps, spec.num_stages)


def _inventory():
    inventory = defaultdict(set)
    for spec in MODULE.triton_compilation_specs():
        inventory[(spec.kernel_name, spec.input_dtype)].add(_config(spec))
    return dict(inventory)


STANDARD_FORWARD = {
    "_fwd_resident": {
        (4, 8192, 8, 1),
        (8, 4096, 8, 1),
        (16, 2048, 8, 1),
    },
    "_fwd_tiled": {
        (8, 2048, 8, 3),
        (16, 2048, 8, 1),
        (16, 2048, 8, 3),
        (32, 2048, 8, 1),
    },
}
SOURCE_SERIAL = {
    (8, 4096, 4, False, False, 8, 1),
    (8, 4096, 16, True, True, 8, 1),
    (16, 2048, 4, False, False, 8, 1),
    (16, 2048, 16, True, True, 8, 1),
    (16, 4096, 4, False, False, 8, 1),
    (16, 4096, 16, True, True, 8, 1),
    (32, 2048, 16, True, True, 8, 1),
    (32, 4096, 16, True, True, 8, 1),
}


def test_matrix_covers_every_campaign_runtime_specialization():
    expected = {}
    for dtype in MODULE.INPUT_POINTER_TYPES:
        for kernel, configs in STANDARD_FORWARD.items():
            expected[(kernel, dtype)] = configs
        expected[("_bwd_resident", dtype)] = {
            (4, 8192, 4, 8, 1),
            (8, 4096, 4, 8, 1),
            (16, 2048, 4, 8, 1),
        }
        expected[("_bwd_stats", dtype)] = {
            (8, 1024, 8, 1),
            (16, 1024, 8, 1),
            (32, 1024, 8, 1),
        }
        expected[("_bwd_apply", dtype)] = {
            (8, 1024, 32, 8, 1),
            (16, 1024, 32, 8, 1),
            (32, 1024, 32, 8, 1),
        }
    for dtype in ("bfloat16", "float16"):
        expected[("_bwd_source_serial_grouped", dtype)] = SOURCE_SERIAL
    expected[("_bwd_source_serial_grouped", "float32")] = SOURCE_SERIAL - {
        (32, 4096, 16, True, True, 8, 1)
    }

    for dtype in ("bfloat16", "float16"):
        expected[("_fwd_resident_saved", dtype)] = STANDARD_FORWARD["_fwd_resident"]
        expected[("_fwd_tiled_saved", dtype)] = {
            (8, 2048, 8, 3),
            (16, 2048, 8, 1),
            (16, 2048, 8, 3),
            (32, 1024, 16, 1),
        }
    expected[("_fwd_resident_saved", "float32")] = {
        (8, 4096, 8, 1),
        (16, 2048, 8, 1),
    }
    expected[("_fwd_tiled_saved", "float32")] = {
        (16, 2048, 8, 1),
        (32, 1024, 16, 1),
    }
    expected[("_reduce_dw_partials", "float32")] = {(8, 128, 4, 1)}

    assert _inventory() == expected
    assert sum(map(len, expected.values())) == 90


def test_matrix_names_are_unique_sorted_and_repeatable():
    first = MODULE.triton_compilation_specs()
    second = MODULE.triton_compilation_specs()
    first_names = [spec.name for spec in first]
    assert first_names == sorted(first_names)
    assert len(first_names) == len(set(first_names)) == 90
    assert first_names == [spec.name for spec in second]
    assert "bwd_source_serial_grouped_bfloat16_bn8_bd4096_t4_saved0_partial0_w8_s1" in first_names
    assert "bwd_source_serial_grouped_float32_bn16_bd2048_t16_saved1_partial1_w8_s1" in first_names


def test_every_input_pointer_uses_the_recorded_dtype():
    for spec in MODULE.triton_compilation_specs():
        pointer = MODULE.INPUT_POINTER_TYPES[spec.input_dtype]
        for field in ("v_ptr", "w_ptr", "g_ptr", "out_ptr", "dv_ptr"):
            if field in spec.signature:
                assert spec.signature[field] == pointer, (spec.name, field)


def test_compilation_record_retains_resources_and_rejects_spills(monkeypatch):
    spec = next(
        item for item in MODULE.triton_compilation_specs() if item.require_spill_free
    )
    compiled = SimpleNamespace(
        asm={"cubin": b"cubin"}, metadata=SimpleNamespace(shared=4096)
    )
    monkeypatch.setattr(MODULE, "triton_compile", lambda *args, **kwargs: compiled)
    resource = {
        "kernel": spec.kernel_name,
        "registers": 64,
        "stack_bytes": 0,
        "static_shared_bytes": 1024,
        "local_bytes": 0,
        "global_load_instructions": 12,
    }
    monkeypatch.setattr(MODULE, "_resource_usage", lambda path: [resource])

    record = MODULE._compile_triton(spec)
    assert record["input_dtype"] == spec.input_dtype
    assert record["spill_policy"] == "forbid"
    assert record["resources"] == [resource]

    spilled = {**resource, "local_bytes": 8}
    monkeypatch.setattr(MODULE, "_resource_usage", lambda path: [spilled])
    with pytest.raises(RuntimeError, match="local storage"):
        MODULE._compile_triton(spec)


def test_baseline_specializations_record_existing_spills(monkeypatch):
    spec = next(
        item for item in MODULE.triton_compilation_specs() if not item.require_spill_free
    )
    compiled = SimpleNamespace(
        asm={"cubin": b"cubin"}, metadata=SimpleNamespace(shared=4096)
    )
    monkeypatch.setattr(MODULE, "triton_compile", lambda *args, **kwargs: compiled)
    resource = {
        "kernel": spec.kernel_name,
        "registers": 255,
        "stack_bytes": 64,
        "static_shared_bytes": 1024,
        "local_bytes": 0,
        "global_load_instructions": 12,
    }
    monkeypatch.setattr(MODULE, "_resource_usage", lambda path: [resource])

    record = MODULE._compile_triton(spec)
    assert record["spill_policy"] == "record"
    assert record["resources"] == [resource]
