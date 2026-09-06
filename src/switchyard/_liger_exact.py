"""Exact-contract Liger-style comparator for Block AttnRes benchmarks.

This file is adapted from Liger-Kernel's Attention Residuals operator at
commit 777799588a89d74c489ed995e3bf006427738e85. The original code is
Copyright 2024 LinkedIn Corporation and licensed under the BSD 2-Clause
License. Switchyard removes the affine RMSNorm gain and its gradient so this
local comparator performs exactly the same functional work as Switchyard's
two-input operator. See ``licenses/BSD-2-Clause-Liger-Kernel.txt``.

This comparator preserves Liger's token-owned, two-source-pass architecture.
It is benchmark-only and is never eligible for production dispatch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .reference import DEFAULT_EPS
from .triton_op import _validate_inputs


@triton.jit
def _liger_exact_fwd_kernel(
    values_ptr,
    query_ptr,
    output_ptr,
    alpha_ptr,
    rstd_ptr,
    n_src,
    n_tokens,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Use one program per token and preserve Liger's two source passes."""
    token = tl.program_id(0)
    columns = tl.arange(0, BLOCK_D)
    feature_mask = columns < D
    source_lanes = tl.arange(0, BLOCK_N)
    query = tl.load(query_ptr + columns, mask=feature_mask, other=0.0).to(tl.float32)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    for source in tl.static_range(0, BLOCK_N):
        if source < n_src:
            offset = source * n_tokens * D + token * D
            values = tl.load(
                values_ptr + offset + columns, mask=feature_mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(values * values, axis=0) / D + eps)
            tl.store(rstd_ptr + token * n_src + source, rstd)
            logit = tl.sum(query * (values * rstd), axis=0)
            logits = tl.where(source_lanes == source, logit, logits)

    logit_max = tl.max(logits, axis=0)
    unnormalized = tl.where(
        source_lanes < n_src,
        tl.exp(logits - logit_max),
        0.0,
    )
    alpha = unnormalized / tl.sum(unnormalized, axis=0)
    tl.store(
        alpha_ptr + token * n_src + source_lanes,
        alpha,
        mask=source_lanes < n_src,
    )

    output = tl.zeros((BLOCK_D,), tl.float32)
    for source in tl.static_range(0, BLOCK_N):
        if source < n_src:
            offset = source * n_tokens * D + token * D
            values = tl.load(
                values_ptr + offset + columns, mask=feature_mask, other=0.0
            ).to(tl.float32)
            coefficient = tl.sum(tl.where(source_lanes == source, alpha, 0.0))
            output += coefficient * values
    tl.store(output_ptr + token * D + columns, output, mask=feature_mask)


@triton.jit
def _liger_exact_bwd_kernel(
    grad_output_ptr,
    values_ptr,
    query_ptr,
    alpha_ptr,
    rstd_ptr,
    grad_values_ptr,
    grad_query_ptr,
    n_src,
    n_tokens,
    D,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute only the two gradients in Switchyard's functional contract."""
    token = tl.program_id(0)
    columns = tl.arange(0, BLOCK_D)
    feature_mask = columns < D
    source_lanes = tl.arange(0, BLOCK_N)
    grad_output = tl.load(
        grad_output_ptr + token * D + columns, mask=feature_mask, other=0.0
    ).to(tl.float32)
    query = tl.load(query_ptr + columns, mask=feature_mask, other=0.0).to(tl.float32)

    output_paths = tl.zeros((BLOCK_N,), tl.float32)
    alpha = tl.zeros((BLOCK_N,), tl.float32)
    for source in tl.static_range(0, BLOCK_N):
        if source < n_src:
            offset = source * n_tokens * D + token * D
            values = tl.load(
                values_ptr + offset + columns, mask=feature_mask, other=0.0
            ).to(tl.float32)
            output_path = tl.sum(grad_output * values, axis=0)
            coefficient = tl.load(alpha_ptr + token * n_src + source)
            output_paths = tl.where(source_lanes == source, output_path, output_paths)
            alpha = tl.where(source_lanes == source, coefficient, alpha)

    centered = tl.sum(alpha * output_paths, axis=0)
    grad_logits = alpha * (output_paths - centered)
    grad_query = tl.zeros((BLOCK_D,), tl.float32)
    for source in tl.static_range(0, BLOCK_N):
        if source < n_src:
            offset = source * n_tokens * D + token * D
            values = tl.load(
                values_ptr + offset + columns, mask=feature_mask, other=0.0
            ).to(tl.float32)
            coefficient = tl.sum(tl.where(source_lanes == source, alpha, 0.0))
            grad_logit = tl.sum(
                tl.where(source_lanes == source, grad_logits, 0.0)
            )
            rstd = tl.load(rstd_ptr + token * n_src + source)
            query_dot = tl.sum(query * values, axis=0)
            norm_coefficient = grad_logit * query_dot * rstd * rstd * rstd / D
            grad_values = (
                coefficient * grad_output
                + grad_logit * rstd * query
                - norm_coefficient * values
            )
            tl.store(
                grad_values_ptr + offset + columns,
                grad_values,
                mask=feature_mask,
            )
            grad_query += grad_logit * rstd * values

    tl.atomic_add(grad_query_ptr + columns, grad_query, mask=feature_mask)


def _launch_configuration(n: int, d: int) -> tuple[int, int, int]:
    if n > 32:
        raise ValueError("the exact-contract Liger-style comparator supports at most 32 sources")
    block_n = next(bucket for bucket in (4, 8, 16, 32) if n <= bucket)
    block_d = triton.next_power_of_2(d)
    warps = 16 if block_d >= 8192 else 8 if block_d >= 2048 else 4
    return block_n, block_d, warps


def liger_exact_forward(
    values: torch.Tensor, query: torch.Tensor, eps: float = DEFAULT_EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output and the exact FP32 state needed by the comparator backward."""
    n, b, t, d = _validate_inputs(values, query, eps)
    values = values.contiguous()
    query = query.contiguous()
    n_tokens = b * t
    block_n, block_d, warps = _launch_configuration(n, d)
    output = torch.empty(b, t, d, device=values.device, dtype=values.dtype)
    alpha = torch.empty(n_tokens, n, device=values.device, dtype=torch.float32)
    rstd = torch.empty_like(alpha)
    _liger_exact_fwd_kernel[(n_tokens,)](
        values,
        query,
        output,
        alpha,
        rstd,
        n,
        n_tokens,
        d,
        eps,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=1,
    )
    return output, values, alpha, rstd


def liger_exact_backward(
    grad_output: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    alpha: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source and query gradients without an affine-gain gradient."""
    grad_output = grad_output.contiguous()
    n, b, t, d = values.shape
    n_tokens = b * t
    block_n, block_d, warps = _launch_configuration(n, d)
    grad_values = torch.empty_like(values)
    grad_query = torch.zeros(d, device=query.device, dtype=torch.float32)
    _liger_exact_bwd_kernel[(n_tokens,)](
        grad_output,
        values,
        query,
        alpha,
        rstd,
        grad_values,
        grad_query,
        n,
        n_tokens,
        d,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=warps,
        num_stages=1,
    )
    return grad_values, grad_query.to(query.dtype)


class _LigerExactAttnResFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, query: torch.Tensor, eps: float):
        output, values, alpha, rstd = liger_exact_forward(values, query, eps)
        ctx.save_for_backward(values, query.contiguous(), alpha, rstd)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        values, query, alpha, rstd = ctx.saved_tensors
        grad_values, grad_query = liger_exact_backward(
            grad_output, values, query, alpha, rstd
        )
        return grad_values, grad_query, None


def block_attn_res_liger_exact(
    values: torch.Tensor, query: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Run the local exact-work Liger-style benchmark comparator."""
    return _LigerExactAttnResFunction.apply(values, query, eps)
