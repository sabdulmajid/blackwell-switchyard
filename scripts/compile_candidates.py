#!/usr/bin/env python3
"""Compile every backward candidate for sm_120 without opening a GPU.

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
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
    raise SystemExit('refusing to compile with a visible GPU; set CUDA_VISIBLE_DEVICES=""')
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[1]

import torch  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402
from triton.compiler import compile as triton_compile  # noqa: E402

from switchyard._backward_candidates import (  # noqa: E402
    _bwd_source_serial_grouped,
    _fwd_resident_saved,
    _fwd_tiled_saved,
    _reduce_dw_partials,
)
from switchyard.cuda_op import _load_extension  # noqa: E402
from switchyard.training_plan import get_training_plan  # noqa: E402
from switchyard.triton_op import _fwd_tiled  # noqa: E402

TARGET = GPUTarget("cuda", 120, 32)
CUOBJDUMP = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")) / "bin/cuobjdump"
CUDA_INSTANCE_COUNTS = {
    "shared_backward_kernel": 2,
    "feature_cluster_backward_kernel": 2,
}
CUDA_REGISTER_LIMITS = {
    "shared_backward_kernel": 64,
    "feature_cluster_backward_kernel": 64,
}

POINTER_SIGNATURE = {
    "v_ptr": "*bf16",
    "w_ptr": "*bf16",
    "g_ptr": "*bf16",
    "saved_alpha_ptr": "*fp32",
    "saved_rstd_ptr": "*fp32",
    "saved_norm_ptr": "*fp32",
    "dv_ptr": "*bf16",
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

FORWARD_SIGNATURE = {
    "v_ptr": "*bf16",
    "w_ptr": "*bf16",
    "out_ptr": "*bf16",
    "saved_alpha_ptr": "*fp32",
    "saved_rstd_ptr": "*fp32",
    "saved_norm_ptr": "*fp32",
    "n_src": "i32",
    "D": "i32",
    "eps": "fp32",
    "stride_vn": "i64",
    "stride_vt": "i64",
    "stride_vd": "i64",
    "stride_ot": "i64",
    "stride_od": "i64",
    "stride_sn": "i64",
}

ACCEPTED_FORWARD_SIGNATURE = {
    key: value
    for key, value in FORWARD_SIGNATURE.items()
    if key not in {"saved_alpha_ptr", "saved_rstd_ptr", "saved_norm_ptr", "stride_sn"}
}


def _resource_usage(binary: Path) -> list[dict[str, int | str]]:
    output = subprocess.run(
        [str(CUOBJDUMP), "--dump-resource-usage", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    records = []
    for name, registers, stack, shared, local in re.findall(
        r"Function ([^:]+):\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        output,
    ):
        normalized_name = name
        for known_name in (
            "feature_cluster_backward_kernel",
            "shared_backward_kernel",
            "_bwd_source_serial_grouped",
            "_reduce_dw_partials",
            "_fwd_resident_saved",
            "_fwd_tiled_saved",
            "_fwd_resident",
            "_fwd_tiled",
        ):
            if known_name in name:
                normalized_name = known_name
                break
        records.append(
            {
                "kernel": normalized_name,
                "registers": int(registers),
                "stack_bytes": int(stack),
                "static_shared_bytes": int(shared),
                "local_bytes": int(local),
            }
        )
    if not records:
        raise RuntimeError(f"could not parse resource usage from {binary}")
    return records


def _compile_triton(
    name: str,
    function,
    signature: dict[str, str],
    constants: dict,
    *,
    warps: int,
    stages: int = 1,
) -> dict:
    compiled = triton_compile(
        ASTSource(function, signature, constants),
        target=TARGET,
        options={"num_warps": warps, "num_stages": stages},
    )
    with tempfile.NamedTemporaryFile(suffix=".cubin") as binary:
        binary.write(compiled.asm["cubin"])
        binary.flush()
        resources = _resource_usage(Path(binary.name))
    if any(item["local_bytes"] for item in resources):
        raise RuntimeError(f"{name} has compiler-reported local storage: {resources}")
    return {
        "name": name,
        "kind": "triton",
        "constants": constants,
        "num_warps": warps,
        "num_stages": stages,
        "shared_bytes": compiled.metadata.shared,
        "resources": resources,
    }


def compile_all() -> dict:
    records = []
    for name, n, d, tokens, saved, partial, warps in (
        ("serial_recompute_atomic_t4_n9_d4096", 16, 4096, 4, False, False, 8),
        ("serial_saved_partials_t16_n9_d8192", 16, 8192, 16, True, True, 16),
        ("serial_saved_partials_t16_n32_d2048", 32, 2048, 16, True, True, 8),
    ):
        records.append(
            _compile_triton(
                name,
                _bwd_source_serial_grouped,
                POINTER_SIGNATURE,
                {
                    "BLOCK_N": n,
                    "BLOCK_D": d,
                    "TOKENS": tokens,
                    "USE_SAVED": saved,
                    "WRITE_PARTIAL": partial,
                },
                warps=warps,
            )
        )

    records.append(
        _compile_triton(
            "saved_training_forward_n9_d8192",
            _fwd_tiled_saved,
            FORWARD_SIGNATURE,
            {"BLOCK_N": 16, "BLOCK_D": 2048},
            warps=8,
            stages=3,
        )
    )
    records.append(
        _compile_triton(
            "accepted_tiled_forward_n9_d4096",
            _fwd_tiled,
            ACCEPTED_FORWARD_SIGNATURE,
            {"BLOCK_N": 16, "BLOCK_D": 2048},
            warps=8,
        )
    )
    records.append(
        _compile_triton(
            "saved_resident_forward_n8_d1024",
            _fwd_resident_saved,
            FORWARD_SIGNATURE,
            {"BLOCK_N": 8, "BLOCK_D": 1024},
            warps=4,
        )
    )
    records.append(
        _compile_triton(
            "dw_partial_reduction",
            _reduce_dw_partials,
            {
                "partial_ptr": "*fp32",
                "dw_ptr": "*fp32",
                "n_partials": "i32",
                "D": "i32",
                "stride_partial": "i64",
            },
            {"BLOCK_P": 8, "BLOCK_D": 128},
            warps=4,
        )
    )

    with tempfile.TemporaryDirectory(prefix="switchyard-cuda-build-") as build_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
        extension = _load_extension()
        cuda_resources = _resource_usage(Path(extension.__file__))
    if any(item["local_bytes"] or item["stack_bytes"] for item in cuda_resources):
        raise RuntimeError(f"CUDA candidate spills to local memory: {cuda_resources}")
    instance_counts = Counter(item["kernel"] for item in cuda_resources)
    if instance_counts != Counter(CUDA_INSTANCE_COUNTS):
        raise RuntimeError(
            "CUDA build must contain bf16 and fp16 instances of both candidates: "
            f"{dict(instance_counts)}"
        )
    for item in cuda_resources:
        limit = CUDA_REGISTER_LIMITS[item["kernel"]]
        if item["registers"] > limit:
            raise RuntimeError(
                f"{item['kernel']} uses {item['registers']} registers; budget is {limit}"
            )
    records.append(
        {
            "name": "one_read_cuda",
            "kind": "cuda",
            "required_template_instances": CUDA_INSTANCE_COUNTS,
            "register_limits": CUDA_REGISTER_LIMITS,
            "resources": cuda_resources,
        }
    )

    cuda_source = REPO / "src/switchyard/csrc/shared_backward.cu"
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
            for name in (
                "serial_recompute_atomic_t4",
                "serial_saved_partials_t16",
                "cuda_shared",
                "cuda_cluster",
            )
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
