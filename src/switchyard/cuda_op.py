"""Optional CUDA experiments that target the one-read backward lower bound.

The extension is compiled lazily. Importing :mod:`switchyard` never builds it.
No public or production dispatch path selects these kernels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import torch

_EXTENSION: ModuleType | None = None


def _source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "shared_backward.cu"


def _load_extension() -> ModuleType:
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        source = _source_path()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _EXTENSION = load(
            name=f"switchyard_shared_backward_{digest}",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_120,code=sm_120"],
            with_cuda=True,
            verbose=False,
        )
    return _EXTENSION


def cuda_shared_backward(
    values: torch.Tensor,
    query: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
    *,
    clustered: bool,
    saved_state: tuple[torch.Tensor, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dv, dw_fp32)`` from a one-read shared-memory candidate.

    The persistent feature-sharded cluster consumes the three exact FP32
    coefficient tensors saved by its forward plan. The one-block diagnostic
    recomputes those coefficients while its source values remain in shared
    memory.
    """
    if clustered and len(saved_state) != 3:
        raise ValueError("clustered backward requires alpha, rstd, and norm coefficient")
    if not clustered and saved_state:
        raise ValueError("one-block shared backward does not consume saved state")
    saved_alpha, saved_rstd, saved_norm = (
        saved_state if clustered else (query, query, query)
    )
    extension = _load_extension()
    dv, dw = extension.shared_backward(
        values,
        query,
        grad_out,
        saved_alpha,
        saved_rstd,
        saved_norm,
        eps,
        clustered,
    )
    return dv, dw


def cuda_cluster_launch_info(values: torch.Tensor) -> dict[str, int]:
    """Return the runtime occupancy inputs for a feature-sharded cluster."""
    fields = (
        "active_clusters",
        "dynamic_shared_bytes",
        "static_shared_bytes",
        "max_shared_bytes",
        "multiprocessors",
        "threads_per_block",
    )
    values = values.contiguous()
    return dict(zip(fields, _load_extension().cluster_launch_info(values), strict=True))
