#!/usr/bin/env python3
"""Compile the guarded backward campaign matrix for sm_120 without a GPU.

This is the first experiment gate. It catches unsupported Triton IR, CUDA
compile failures, register explosions, and local-memory spills before scarce
GPU time is used. Run it after ``source scripts/env.sh``::

    CUDA_VISIBLE_DEVICES="" python scripts/compile_candidates.py

The command builds only temporary files unless ``--out`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
    raise SystemExit('refusing to compile with a visible GPU; set CUDA_VISIBLE_DEVICES=""')
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler import compile as triton_compile  # noqa: E402

from bench_backward import CORRECTNESS_ONLY_SHAPES, SHAPE_SETS  # noqa: E402
from switchyard._backward_candidates import (  # noqa: E402
    _bwd_source_serial_grouped,
    _fwd_resident_saved,
    _fwd_tiled_saved,
    _reduce_dw_partials,
    _source_serial_launch,
    saved_forward_launch_config,
)
from switchyard._liger_exact import (  # noqa: E402
    _liger_exact_bwd_kernel,
    _liger_exact_fwd_kernel,
)
from switchyard.cuda_op import _load_extension  # noqa: E402
from switchyard.training_plan import get_training_plan, plan_supports  # noqa: E402
from switchyard.triton_op import (  # noqa: E402
    _bwd_apply,
    _bwd_launch,
    _bwd_resident,
    _bwd_stats,
    _fwd_resident,
    _fwd_tiled,
    _launch_config,
)

TARGET = GPUTarget("cuda", 120, 32)
CUOBJDUMP = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")) / "bin/cuobjdump"
NVDISASM = CUOBJDUMP.with_name("nvdisasm")
CUDA_INSTANCE_COUNTS = {
    "shared_backward_kernel": 2,
    "feature_cluster_backward_kernel_2block": 2,
    "feature_cluster_backward_kernel_4block": 2,
    "register_backward_kernel": 2,
    "register_cluster_backward_kernel_bfloat16_n9_d8192_c4": 1,
    "register_cluster_backward_kernel_bfloat16_n32_d2048_c2": 1,
    "register_cluster_backward_kernel_float16_n9_d8192_c4": 1,
    "register_cluster_backward_kernel_float16_n32_d2048_c2": 1,
    "register_cluster_forward_kernel_bfloat16_n9_d8192_c4": 1,
    "register_cluster_forward_kernel_bfloat16_n32_d2048_c2": 1,
    "register_cluster_forward_kernel_float16_n9_d8192_c4": 1,
    "register_cluster_forward_kernel_float16_n32_d2048_c2": 1,
}
CUDA_REGISTER_LIMITS = {
    "shared_backward_kernel": 64,
    "feature_cluster_backward_kernel_2block": 64,
    # The four-block tile uses at most two blocks per SM because of shared
    # memory. 80 registers still clears that occupancy limit on the target's
    # 65,536-register SM while leaving eight registers of compiler headroom.
    "feature_cluster_backward_kernel_4block": 80,
    "register_backward_kernel": 128,
    "register_cluster_backward_kernel_bfloat16_n9_d8192_c4": 128,
    "register_cluster_backward_kernel_bfloat16_n32_d2048_c2": 128,
    "register_cluster_backward_kernel_float16_n9_d8192_c4": 128,
    "register_cluster_backward_kernel_float16_n32_d2048_c2": 128,
    "register_cluster_forward_kernel_bfloat16_n9_d8192_c4": 128,
    "register_cluster_forward_kernel_bfloat16_n32_d2048_c2": 128,
    "register_cluster_forward_kernel_float16_n9_d8192_c4": 128,
    "register_cluster_forward_kernel_float16_n32_d2048_c2": 128,
}
CUDA_GLOBAL_LOAD_INSTRUCTION_COUNTS = {
    "register_cluster_backward_kernel_bfloat16_n9_d8192_c4": 25,
    "register_cluster_backward_kernel_bfloat16_n32_d2048_c2": 37,
    "register_cluster_backward_kernel_float16_n9_d8192_c4": 25,
    "register_cluster_backward_kernel_float16_n32_d2048_c2": 37,
    "register_cluster_forward_kernel_bfloat16_n9_d8192_c4": 20,
    "register_cluster_forward_kernel_bfloat16_n32_d2048_c2": 33,
    "register_cluster_forward_kernel_float16_n9_d8192_c4": 20,
    "register_cluster_forward_kernel_float16_n32_d2048_c2": 33,
}
CUDA_GLOBAL_LOAD_ROLE_COUNTS = {
    **{
        f"register_cluster_backward_kernel_{dtype}_n9_d8192_c4": {
            "grad_output_pairs": 2,
            "source_pairs": 18,
            "saved_alpha": 1,
            "saved_rstd": 1,
            "saved_norm": 1,
            "query_pairs": 2,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_backward_kernel_{dtype}_n32_d2048_c2": {
            "grad_output_pairs": 1,
            "source_pairs": 32,
            "saved_alpha": 1,
            "saved_rstd": 1,
            "saved_norm": 1,
            "query_pairs": 1,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_forward_kernel_{dtype}_n9_d8192_c4": {
            "query_pairs": 2,
            "source_pairs": 18,
        }
        for dtype in ("bfloat16", "float16")
    },
    **{
        f"register_cluster_forward_kernel_{dtype}_n32_d2048_c2": {
            "query_pairs": 1,
            "source_pairs": 32,
        }
        for dtype in ("bfloat16", "float16")
    },
}

INPUT_POINTER_TYPES = {
    "bfloat16": "*bf16",
    "float16": "*fp16",
    "float32": "*fp32",
}

# These phases mirror the guarded campaign. Comparator kernels use a recorded
# spill policy because they are baselines, not promotion candidates.
CAMPAIGN_CANDIDATE_PLAN_NAMES = (
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
    "cuda_register_cluster",
    "cuda_register_cluster_full",
)
CAMPAIGN_PLAN_NAMES = ("auto", *CAMPAIGN_CANDIDATE_PLAN_NAMES)
CAMPAIGN_PORTABLE_PLAN_NAMES = (
    "auto",
    "serial_recompute_atomic_t4",
    "serial_saved_partials_t16",
)
CAMPAIGN_PHASES = (
    ("gate", "bfloat16", CAMPAIGN_PLAN_NAMES),
    ("gate", "float16", CAMPAIGN_PLAN_NAMES),
    ("full", "bfloat16", CAMPAIGN_PLAN_NAMES),
    ("full", "float16", CAMPAIGN_PLAN_NAMES),
    ("full", "float32", CAMPAIGN_PORTABLE_PLAN_NAMES),
)


@dataclass(frozen=True)
class TritonCompilationSpec:
    """One unique specialization that the guarded campaign can launch."""

    name: str
    kernel_name: str
    function: object
    input_dtype: str
    signature: dict[str, str]
    constants: dict[str, int | bool | float]
    num_warps: int
    num_stages: int
    require_spill_free: bool


def _input_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "v_ptr": pointer,
        "w_ptr": pointer,
        "g_ptr": pointer,
        "saved_alpha_ptr": "*fp32",
        "saved_rstd_ptr": "*fp32",
        "saved_norm_ptr": "*fp32",
        "dv_ptr": pointer,
        "dw_output_ptr": "*fp32",
        "n_src": "i32",
        "D": "i32",
        "eps": "fp32",
        "n_tokens": "i32",
        "stride_vn": "i64",
        "stride_vt": "i64",
        "stride_vd": "i64",
        "stride_gt": "i64",
        "stride_gd": "i64",
        "stride_sn": "i64",
        "stride_partial": "i64",
    }


def _forward_signature(dtype: str, *, saved: bool) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    common = {
        "n_src": "i32",
        "D": "i32",
        "eps": "fp32",
        "stride_vn": "i64",
        "stride_vt": "i64",
        "stride_vd": "i64",
        "stride_ot": "i64",
        "stride_od": "i64",
    }
    if not saved:
        return {
            "v_ptr": pointer,
            "w_ptr": pointer,
            "out_ptr": pointer,
            **common,
        }
    return {
        "v_ptr": pointer,
        "w_ptr": pointer,
        "out_ptr": pointer,
        "saved_alpha_ptr": "*fp32",
        "saved_rstd_ptr": "*fp32",
        "saved_norm_ptr": "*fp32",
        **common,
        "stride_sn": "i64",
    }


REDUCTION_SIGNATURE = {
    "partial_ptr": "*fp32",
    "dw_ptr": "*fp32",
    "n_partials": "i32",
    "D": "i32",
    "stride_partial": "i64",
}


def _current_resident_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "v_ptr": pointer,
        "w_ptr": pointer,
        "g_ptr": pointer,
        "dv_ptr": pointer,
        "dw_ptr": "*fp32",
        "n_src": "i32",
        "D": "i32",
        "eps": "fp32",
        "stride_vn": "i64",
        "stride_vt": "i64",
        "stride_vd": "i64",
        "stride_gt": "i64",
        "stride_gd": "i64",
        "n_tokens": "i32",
    }


def _current_stats_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "v_ptr": pointer,
        "w_ptr": pointer,
        "g_ptr": pointer,
        "alpha_ptr": "*fp32",
        "da_ptr": "*fp32",
        "dssq_ptr": "*fp32",
        "n_src": "i32",
        "D": "i32",
        "eps": "fp32",
        "stride_vn": "i64",
        "stride_vt": "i64",
        "stride_vd": "i64",
        "stride_gt": "i64",
        "stride_gd": "i64",
        "stride_sn": "i64",
    }


def _current_apply_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "v_ptr": pointer,
        "w_ptr": pointer,
        "g_ptr": pointer,
        "alpha_ptr": "*fp32",
        "da_ptr": "*fp32",
        "dssq_ptr": "*fp32",
        "dv_ptr": pointer,
        "dw_ptr": "*fp32",
        "n_src": "i32",
        "D": "i32",
        "n_tokens": "i32",
        "stride_vn": "i64",
        "stride_vt": "i64",
        "stride_vd": "i64",
        "stride_gt": "i64",
        "stride_gd": "i64",
        "stride_sn": "i64",
    }


def _liger_exact_forward_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "values_ptr": pointer,
        "query_ptr": pointer,
        "output_ptr": pointer,
        "alpha_ptr": "*fp32",
        "rstd_ptr": "*fp32",
        "n_src": "i32",
        "n_tokens": "i32",
        "D": "i32",
        "eps": "fp32",
    }


def _liger_exact_backward_signature(dtype: str) -> dict[str, str]:
    pointer = INPUT_POINTER_TYPES[dtype]
    return {
        "grad_output_ptr": pointer,
        "values_ptr": pointer,
        "query_ptr": pointer,
        "alpha_ptr": "*fp32",
        "rstd_ptr": "*fp32",
        "grad_values_ptr": pointer,
        "grad_query_ptr": "*fp32",
        "n_src": "i32",
        "n_tokens": "i32",
        "D": "i32",
    }


def _spec_name(
    kernel_name: str,
    dtype: str,
    constants: dict[str, int | bool | float],
    warps: int,
    stages: int,
) -> str:
    aliases = {
        "BLOCK_N": "bn",
        "BLOCK_D": "bd",
        "BLOCK_P": "bp",
        "TOKENS": "t",
        "USE_SAVED": "saved",
        "WRITE_PARTIAL": "partial",
    }
    parts = [kernel_name.removeprefix("_"), dtype]
    for key, value in constants.items():
        rendered = int(value) if isinstance(value, bool) else value
        parts.append(f"{aliases[key]}{rendered}")
    parts.extend((f"w{warps}", f"s{stages}"))
    return "_".join(parts)


def _make_spec(
    kernel_name: str,
    function: object,
    dtype: str,
    signature: dict[str, str],
    constants: dict[str, int | bool | float],
    *,
    warps: int,
    stages: int = 1,
    require_spill_free: bool = True,
) -> TritonCompilationSpec:
    return TritonCompilationSpec(
        name=_spec_name(kernel_name, dtype, constants, warps, stages),
        kernel_name=kernel_name,
        function=function,
        input_dtype=dtype,
        signature=signature,
        constants=constants,
        num_warps=warps,
        num_stages=stages,
        require_spill_free=require_spill_free,
    )


def _forward_spec(plan, shape, dtype: str) -> TritonCompilationSpec | None:
    if plan.forward.family == "cuda_register_cluster":
        return None
    block_n = triton.next_power_of_2(shape.n)
    resident, block_d, warps, stages = _launch_config(block_n, shape.d)
    saved = plan.saves_forward_stats
    if saved:
        resident, block_d, warps, stages = saved_forward_launch_config(
            resident=resident,
            block_n=block_n,
            block_d=block_d,
            warps=warps,
            stages=stages,
        )
    if saved:
        function = _fwd_resident_saved if resident else _fwd_tiled_saved
    else:
        function = _fwd_resident if resident else _fwd_tiled
    return _make_spec(
        function.fn.__name__,
        function,
        dtype,
        _forward_signature(dtype, saved=saved),
        {"BLOCK_N": block_n, "BLOCK_D": block_d},
        warps=warps,
        stages=stages,
        require_spill_free=plan.name != "auto",
    )


def _backward_specs(plan, shape, dtype: str) -> list[TritonCompilationSpec]:
    block_n = triton.next_power_of_2(shape.n)
    if plan.backward.family == "auto":
        resident, tokens, warps, stages = _bwd_launch(block_n, shape.d)
        if resident:
            return [
                _make_spec(
                    "_bwd_resident",
                    _bwd_resident,
                    dtype,
                    _current_resident_signature(dtype),
                    {
                        "BLOCK_N": block_n,
                        "BLOCK_D": triton.next_power_of_2(shape.d),
                        "TOKENS": tokens,
                    },
                    warps=warps,
                    stages=stages,
                    require_spill_free=False,
                )
            ]
        block_d = min(1024, triton.next_power_of_2(shape.d))
        return [
            _make_spec(
                "_bwd_stats",
                _bwd_stats,
                dtype,
                _current_stats_signature(dtype),
                {"BLOCK_N": block_n, "BLOCK_D": block_d},
                warps=8,
                require_spill_free=False,
            ),
            _make_spec(
                "_bwd_apply",
                _bwd_apply,
                dtype,
                _current_apply_signature(dtype),
                {"BLOCK_N": block_n, "BLOCK_D": block_d, "TOKENS": 32},
                warps=8,
                require_spill_free=False,
            ),
        ]
    if plan.backward.family != "source_serial":
        return []

    block_d, warps = _source_serial_launch(shape.d)
    write_partial = plan.backward.dw_reduction == "partials"
    specs = [
        _make_spec(
            "_bwd_source_serial_grouped",
            _bwd_source_serial_grouped,
            dtype,
            _input_signature(dtype),
            {
                "BLOCK_N": block_n,
                "BLOCK_D": block_d,
                "TOKENS": plan.backward.tokens_per_cta,
                "USE_SAVED": plan.saves_forward_stats,
                "WRITE_PARTIAL": write_partial,
            },
            warps=warps,
        )
    ]
    if write_partial:
        specs.append(
            _make_spec(
                "_reduce_dw_partials",
                _reduce_dw_partials,
                "float32",
                REDUCTION_SIGNATURE,
                {"BLOCK_P": 8, "BLOCK_D": 128},
                warps=4,
            )
        )
    return specs


def triton_compilation_specs() -> list[TritonCompilationSpec]:
    """Derive the unique Triton matrix from every guarded campaign phase."""
    unique: dict[tuple, TritonCompilationSpec] = {}
    for shape_set, dtype, plan_names in CAMPAIGN_PHASES:
        shapes = (*SHAPE_SETS[shape_set], *CORRECTNESS_ONLY_SHAPES)
        for shape in shapes:
            block_n = next(bucket for bucket in (4, 8, 16, 32) if shape.n <= bucket)
            block_d = triton.next_power_of_2(shape.d)
            warps = 16 if block_d >= 8192 else 8 if block_d >= 2048 else 4
            comparator_specs = (
                _make_spec(
                    "_liger_exact_fwd_kernel",
                    _liger_exact_fwd_kernel,
                    dtype,
                    _liger_exact_forward_signature(dtype),
                    {"BLOCK_D": block_d, "BLOCK_N": block_n},
                    warps=warps,
                ),
                _make_spec(
                    "_liger_exact_bwd_kernel",
                    _liger_exact_bwd_kernel,
                    dtype,
                    _liger_exact_backward_signature(dtype),
                    {"BLOCK_D": block_d, "BLOCK_N": block_n},
                    warps=warps,
                ),
            )
            for spec in comparator_specs:
                key = (
                    spec.kernel_name,
                    spec.input_dtype,
                    tuple(sorted(spec.constants.items())),
                    spec.num_warps,
                    spec.num_stages,
                )
                unique.setdefault(key, spec)
            for plan_name in plan_names:
                plan = get_training_plan(plan_name)
                supported, _ = plan_supports(
                    plan, shape.n, shape.b, shape.t, shape.d, dtype
                )
                if not supported:
                    continue
                specs = _backward_specs(plan, shape, dtype)
                forward = _forward_spec(plan, shape, dtype)
                if forward is not None:
                    specs.append(forward)
                for spec in specs:
                    key = (
                        spec.kernel_name,
                        spec.input_dtype,
                        tuple(sorted(spec.constants.items())),
                        spec.num_warps,
                        spec.num_stages,
                    )
                    previous = unique.get(key)
                    if previous is None or (
                        spec.require_spill_free and not previous.require_spill_free
                    ):
                        unique[key] = spec
    return sorted(unique.values(), key=lambda spec: spec.name)


def _normalize_kernel_name(name: str) -> str:
    normalized_name = name
    for known_name in (
        "feature_cluster_backward_kernel",
        "register_cluster_backward_kernel",
        "register_cluster_forward_kernel",
        "register_backward_kernel",
        "shared_backward_kernel",
        "_bwd_source_serial_grouped",
        "_reduce_dw_partials",
        "_fwd_resident_saved",
        "_fwd_tiled_saved",
        "_fwd_resident",
        "_fwd_tiled",
    ):
        if known_name not in name:
            continue
        normalized_name = known_name
        if known_name == "feature_cluster_backward_kernel":
            cluster_blocks = 4 if "Li4E" in name else 2
            normalized_name = f"{known_name}_{cluster_blocks}block"
        elif known_name in {
            "register_cluster_backward_kernel",
            "register_cluster_forward_kernel",
        }:
            shape = re.search(r"Li(\d+)ELi(\d+)ELi(\d+)E", name)
            dtype = (
                "bfloat16"
                if "__nv_bfloat16" in name
                else "float16" if "__half" in name else "unknown"
            )
            if shape is not None:
                n, d, cluster_blocks = shape.groups()
                normalized_name = (
                    f"{known_name}_{dtype}_n{n}_d{d}_c{cluster_blocks}"
                )
        break
    return normalized_name


def _sass_global_load_counts(binary: Path) -> dict[str, int]:
    output = subprocess.run(
        [str(CUOBJDUMP), "--dump-sass", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    counts: dict[str, int] = {}
    for section in re.split(r"(?=\n\s*Function : )", output):
        match = re.search(r"Function : (\S+)", section)
        if match is None:
            continue
        counts[match.group(1)] = len(re.findall(r"\bLDG(?:\.[A-Z0-9_]+)*\s", section))
    return counts


def _unique_source_line(
    lines: list[str], start: int, end: int, needle: str
) -> int:
    matches = [
        index + 1
        for index in range(start, end)
        if needle in lines[index]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one CUDA source anchor for {needle!r}, got {matches}")
    return matches[0]


def _cuda_load_role_lines(source: Path) -> dict[int, str]:
    """Map audited source statements to stable logical load roles."""
    lines = source.read_text().splitlines()
    backward_start = next(
        index
        for index, line in enumerate(lines)
        if "void register_cluster_backward_kernel(" in line
    )
    forward_start = next(
        index
        for index, line in enumerate(lines)
        if "void register_cluster_forward_kernel(" in line
    )
    forward_end = next(
        index
        for index in range(forward_start + 1, len(lines))
        if lines[index].startswith("size_t shared_bytes(")
    )
    anchors = {
        "grad_output_pairs": (
            backward_start,
            forward_start,
            "grad_out + static_cast<int64_t>(token) * Width + feature)[0]",
        ),
        "source_pairs_backward": (
            backward_start,
            forward_start,
            "const uint32_t raw = reinterpret_cast<const uint32_t*>(values + value_offset)[0]",
        ),
        "saved_alpha": (backward_start, forward_start, "alpha = saved_alpha[saved_offset]"),
        "saved_rstd": (
            backward_start,
            forward_start,
            "const float rstd = saved_rstd[saved_offset]",
        ),
        "saved_norm": (
            backward_start,
            forward_start,
            "const float norm = saved_norm[saved_offset]",
        ),
        "query_pairs_backward": (
            backward_start,
            forward_start,
            "reinterpret_cast<const uint32_t*>(query + feature)[0]",
        ),
        "query_pairs_forward": (
            forward_start,
            forward_end,
            "held_query[slot] = reinterpret_cast<const uint32_t*>(query + feature)[0]",
        ),
        "source_pairs_forward": (
            forward_start,
            forward_end,
            "reinterpret_cast<const uint32_t*>(values + value_offset)[0]",
        ),
    }
    role_lines = {}
    for anchor, (start, end, needle) in anchors.items():
        role = anchor.removesuffix("_backward").removesuffix("_forward")
        line = _unique_source_line(lines, start, end, needle)
        if line in role_lines:
            raise RuntimeError(f"CUDA load roles share source line {line}")
        role_lines[line] = role
    return role_lines


def _sass_global_load_roles(binary: Path, source: Path) -> dict[str, dict[str, int]]:
    """Attribute each static LDG to an audited source statement via line info."""
    role_lines = _cuda_load_role_lines(source)
    with tempfile.TemporaryDirectory(prefix="switchyard-cubin-") as directory:
        extracted = Path(directory)
        subprocess.run(
            [str(CUOBJDUMP), "--extract-elf", "all", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            cwd=extracted,
        )
        cubins = sorted(extracted.glob("*.cubin"))
        if len(cubins) != 1:
            raise RuntimeError(f"expected one embedded cubin, got {len(cubins)}")
        output = subprocess.run(
            [str(NVDISASM), "--print-line-info", str(cubins[0])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    result: dict[str, dict[str, int]] = {}
    for section in re.split(r"(?=//-+ \.text\.)", output):
        symbol = re.search(r"\.text\.(\S+)", section)
        if symbol is None:
            continue
        kernel = _normalize_kernel_name(symbol.group(1))
        if kernel not in CUDA_GLOBAL_LOAD_ROLE_COUNTS:
            continue
        if kernel in result:
            raise RuntimeError(f"duplicate line-info SASS section for {kernel}")
        current_line: int | None = None
        counts: Counter[str] = Counter()
        for line in section.splitlines():
            location = re.search(r'File "([^"]+)", line (\d+)', line)
            if location is not None:
                current_line = (
                    int(location.group(2))
                    if Path(location.group(1)).name == source.name
                    else None
                )
            if re.search(r"\bLDG(?:\.[A-Z0-9_]+)*\s", line):
                counts[role_lines.get(current_line, "unmapped")] += 1
        result[kernel] = dict(counts)
    return result


def _resource_usage(binary: Path) -> list[dict[str, int | str]]:
    output = subprocess.run(
        [str(CUOBJDUMP), "--dump-resource-usage", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sass_global_loads = _sass_global_load_counts(binary)
    records = []
    for name, registers, stack, shared, local in re.findall(
        r"Function ([^:]+):\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        output,
    ):
        records.append(
            {
                "kernel": _normalize_kernel_name(name),
                "registers": int(registers),
                "stack_bytes": int(stack),
                "static_shared_bytes": int(shared),
                "local_bytes": int(local),
                "global_load_instructions": sass_global_loads.get(name),
            }
        )
    if not records:
        raise RuntimeError(f"could not parse resource usage from {binary}")
    return records


def _compile_triton(spec: TritonCompilationSpec) -> dict:
    compiled = triton_compile(
        ASTSource(spec.function, spec.signature, spec.constants),
        target=TARGET,
        options={"num_warps": spec.num_warps, "num_stages": spec.num_stages},
    )
    with tempfile.NamedTemporaryFile(suffix=".cubin") as binary:
        binary.write(compiled.asm["cubin"])
        binary.flush()
        resources = _resource_usage(Path(binary.name))
    if spec.require_spill_free and any(
        item["local_bytes"] or item["stack_bytes"] for item in resources
    ):
        raise RuntimeError(
            f"{spec.name} has compiler-reported local storage: {resources}"
        )
    return {
        "name": spec.name,
        "kind": "triton",
        "input_dtype": spec.input_dtype,
        "spill_policy": "forbid" if spec.require_spill_free else "record",
        "constants": spec.constants,
        "num_warps": spec.num_warps,
        "num_stages": spec.num_stages,
        "shared_bytes": compiled.metadata.shared,
        "resources": resources,
    }


def compile_all() -> dict:
    records = [_compile_triton(spec) for spec in triton_compilation_specs()]

    cuda_source = REPO / "src/switchyard/csrc/shared_backward.cu"
    with tempfile.TemporaryDirectory(prefix="switchyard-cuda-build-") as build_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
        extension = _load_extension()
        binary = Path(extension.__file__)
        cuda_resources = _resource_usage(binary)
        cuda_load_roles = _sass_global_load_roles(binary, cuda_source)
    if any(item["local_bytes"] or item["stack_bytes"] for item in cuda_resources):
        raise RuntimeError(f"CUDA candidate spills to local memory: {cuda_resources}")
    instance_counts = Counter(item["kernel"] for item in cuda_resources)
    if instance_counts != Counter(CUDA_INSTANCE_COUNTS):
        raise RuntimeError(
            "CUDA build does not contain the exact required template instances: "
            f"{dict(instance_counts)}"
        )
    for item in cuda_resources:
        limit = CUDA_REGISTER_LIMITS[item["kernel"]]
        if item["registers"] > limit:
            raise RuntimeError(
                f"{item['kernel']} uses {item['registers']} registers; budget is {limit}: {item}"
            )
    observed_loads = {
        item["kernel"]: item["global_load_instructions"]
        for item in cuda_resources
        if item["kernel"] in CUDA_GLOBAL_LOAD_INSTRUCTION_COUNTS
    }
    if observed_loads != CUDA_GLOBAL_LOAD_INSTRUCTION_COUNTS:
        raise RuntimeError(
            "CUDA generated code does not preserve the one-read load contract: "
            f"{observed_loads}"
        )
    if cuda_load_roles != CUDA_GLOBAL_LOAD_ROLE_COUNTS:
        raise RuntimeError(
            "CUDA generated code does not preserve the role-specific one-read contract: "
            f"{cuda_load_roles}"
        )
    for item in cuda_resources:
        if item["kernel"] in cuda_load_roles:
            item["global_load_instruction_roles"] = cuda_load_roles[item["kernel"]]
    records.append(
        {
            "name": "one_read_cuda",
            "kind": "cuda",
            "required_template_instances": CUDA_INSTANCE_COUNTS,
            "register_limits": CUDA_REGISTER_LIMITS,
            "required_global_load_instruction_counts": (
                CUDA_GLOBAL_LOAD_INSTRUCTION_COUNTS
            ),
            "required_global_load_role_counts": CUDA_GLOBAL_LOAD_ROLE_COUNTS,
            "resources": cuda_resources,
        }
    )

    git_commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    nvcc_version = subprocess.run(
        [str(CUOBJDUMP.with_name("nvcc")), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    return {
        "target": asdict(TARGET),
        "gpu_visible": False,
        "provenance": {
            "repository_commit": git_commit,
            "worktree_clean": not git_status,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "nvcc": nvcc_version,
            "cuda_source_sha256": hashlib.sha256(cuda_source.read_bytes()).hexdigest(),
        },
        "plans": [
            get_training_plan(name).as_dict()
            for name in CAMPAIGN_PLAN_NAMES
        ],
        "compilations": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compile_all()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
