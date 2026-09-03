"""Private Triton backward architectures and their complete launch path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .training_plan import TrainingPlan


@triton.jit
def _fwd_resident_saved(
    v_ptr,
    w_ptr,
    out_ptr,
    saved_alpha_ptr,
    saved_rstd_ptr,
    saved_norm_ptr,
    n_src,
    D,
    eps,
    stride_vn,
    stride_vt,
    stride_vd,
    stride_ot,
    stride_od,
    stride_sn,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Resident training forward with exact FP32 backward coefficients."""
    token = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D
    pointers = (
        v_ptr
        + token * stride_vt
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )
    values = tl.load(
        pointers, mask=n_mask[:, None] & d_mask[None, :], other=0.0
    ).to(tl.float32)
    query = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    sum_of_squares = tl.sum(values * values, axis=1)
    query_dot = tl.sum(values * query[None, :], axis=1)
    rstd = tl.rsqrt(sum_of_squares / D + eps)
    logits = tl.where(n_mask, query_dot * rstd, float("-inf"))
    unnormalized = tl.exp(logits - tl.max(logits, axis=0))
    alpha = unnormalized / tl.sum(unnormalized, axis=0)

    saved_offset = offs_n * stride_sn + token
    tl.store(saved_alpha_ptr + saved_offset, alpha, mask=n_mask)
    tl.store(saved_rstd_ptr + saved_offset, rstd, mask=n_mask)
    tl.store(
        saved_norm_ptr + saved_offset,
        query_dot * rstd * rstd * rstd / D,
        mask=n_mask,
    )
    output = tl.sum(alpha[:, None] * values, axis=0)
    tl.store(out_ptr + token * stride_ot + offs_d * stride_od, output, mask=d_mask)


@triton.jit
def _fwd_tiled_saved(
    v_ptr,
    w_ptr,
    out_ptr,
    saved_alpha_ptr,
    saved_rstd_ptr,
    saved_norm_ptr,
    n_src,
    D,
    eps,
    stride_vn,
    stride_vt,
    stride_vd,
    stride_ot,
    stride_od,
    stride_sn,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Tiled training forward with exact FP32 backward coefficients."""
    token = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_src
    sum_of_squares = tl.zeros([BLOCK_N], dtype=tl.float32)
    query_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
    for feature_start in range(0, D, BLOCK_D):
        offs_d = feature_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        pointers = (
            v_ptr
            + token * stride_vt
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        values = tl.load(
            pointers, mask=n_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.float32)
        query = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        sum_of_squares += tl.sum(values * values, axis=1)
        query_dot += tl.sum(values * query[None, :], axis=1)

    rstd = tl.rsqrt(sum_of_squares / D + eps)
    logits = tl.where(n_mask, query_dot * rstd, float("-inf"))
    unnormalized = tl.exp(logits - tl.max(logits, axis=0))
    alpha = unnormalized / tl.sum(unnormalized, axis=0)
    saved_offset = offs_n * stride_sn + token
    tl.store(saved_alpha_ptr + saved_offset, alpha, mask=n_mask)
    tl.store(saved_rstd_ptr + saved_offset, rstd, mask=n_mask)
    tl.store(
        saved_norm_ptr + saved_offset,
        query_dot * rstd * rstd * rstd / D,
        mask=n_mask,
    )

    for feature_start in range(0, D, BLOCK_D):
        offs_d = feature_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        pointers = (
            v_ptr
            + token * stride_vt
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        values = tl.load(
            pointers, mask=n_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.float32)
        output = tl.sum(alpha[:, None] * values, axis=0)
        tl.store(out_ptr + token * stride_ot + offs_d * stride_od, output, mask=d_mask)


def launch_saved_training_forward(
    values: torch.Tensor,
    query: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    *,
    resident: bool,
    block_n: int,
    block_d: int,
    warps: int,
    stages: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch an isolated training forward and return its FP32 coefficients."""
    n, b, t, _ = values.shape
    n_tokens = b * t
    saved_alpha = torch.empty(n, n_tokens, device=values.device, dtype=torch.float32)
    saved_rstd = torch.empty_like(saved_alpha)
    saved_norm = torch.empty_like(saved_alpha)
    kernel = _fwd_resident_saved if resident else _fwd_tiled_saved
    kernel[(n_tokens,)](
        values,
        query,
        output,
        saved_alpha,
        saved_rstd,
        saved_norm,
        n,
        values.shape[-1],
        eps,
        values.stride(0),
        values.stride(2),
        values.stride(3),
        output.stride(1),
        output.stride(2),
        saved_alpha.stride(0),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=stages,
    )
    return saved_alpha, saved_rstd, saved_norm


@triton.jit
def _bwd_source_serial_grouped(
    v_ptr,
    w_ptr,
    g_ptr,
    saved_alpha_ptr,
    saved_rstd_ptr,
    saved_norm_ptr,
    dv_ptr,
    dw_output_ptr,
    n_src,
    D,
    eps,
    n_tokens,
    stride_vn,
    stride_vt,
    stride_vd,
    stride_gt,
    stride_gd,
    stride_sn,
    stride_partial,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKENS: tl.constexpr,
    USE_SAVED: tl.constexpr,
    WRITE_PARTIAL: tl.constexpr,
):
    """Keep adjacent source passes inside one grouped token program.

    This is the portable two-read candidate. It fixes the accepted path's
    grid-wide locality loss. It cannot cross the one-read traffic floor that
    the feature-sharded CUDA cluster targets.
    """
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D

    w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for token_index in tl.range(0, TOKENS, loop_unroll_factor=1):
        tok = pid * TOKENS + token_index
        if tok < n_tokens:
            g = tl.load(
                g_ptr + tok * stride_gt + offs_d * stride_gd,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            rstd = tl.zeros([BLOCK_N], dtype=tl.float32)
            query_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
            norm_coefficient = tl.zeros([BLOCK_N], dtype=tl.float32)
            grad_dot = tl.zeros([BLOCK_N], dtype=tl.float32)

            if USE_SAVED:
                saved_offset = offs_n * stride_sn + tok
                alpha = tl.load(saved_alpha_ptr + saved_offset, mask=n_mask, other=0.0)
                rstd = tl.load(saved_rstd_ptr + saved_offset, mask=n_mask, other=0.0)
                norm_coefficient = tl.load(
                    saved_norm_ptr + saved_offset, mask=n_mask, other=0.0
                )

            for source in tl.static_range(0, BLOCK_N):
                if source < n_src:
                    vp = (
                        v_ptr
                        + tok * stride_vt
                        + source * stride_vn
                        + offs_d * stride_vd
                    )
                    value = tl.load(vp, mask=d_mask, other=0.0).to(tl.float32)
                    select_source = offs_n == source
                    grad_dot_source = tl.sum(value * g, axis=0)
                    grad_dot = tl.where(select_source, grad_dot_source, grad_dot)
                    if not USE_SAVED:
                        sum_of_squares = tl.sum(value * value, axis=0)
                        query_dot_source = tl.sum(value * w, axis=0)
                        rstd_source = tl.rsqrt(sum_of_squares / D + eps)
                        rstd = tl.where(select_source, rstd_source, rstd)
                        query_dot = tl.where(select_source, query_dot_source, query_dot)

            if not USE_SAVED:
                logits = tl.where(n_mask, query_dot * rstd, float("-inf"))
                unnormalized = tl.where(
                    n_mask, tl.exp(logits - tl.max(logits, axis=0)), 0.0
                )
                alpha = unnormalized / tl.sum(unnormalized, axis=0)
                norm_coefficient = query_dot * rstd * rstd * rstd / D

            dlogit = alpha * (grad_dot - tl.sum(alpha * grad_dot, axis=0))

            for source in tl.static_range(0, BLOCK_N):
                if source < n_src:
                    select_source = offs_n == source
                    alpha_source = tl.sum(tl.where(select_source, alpha, 0.0), axis=0)
                    dlogit_source = tl.sum(tl.where(select_source, dlogit, 0.0), axis=0)
                    rstd_source = tl.sum(tl.where(select_source, rstd, 0.0), axis=0)
                    norm_source = tl.sum(
                        tl.where(select_source, norm_coefficient, 0.0), axis=0
                    )

                    vp = (
                        v_ptr
                        + tok * stride_vt
                        + source * stride_vn
                        + offs_d * stride_vd
                    )
                    value = tl.load(vp, mask=d_mask, other=0.0).to(tl.float32)
                    dv = (
                        alpha_source * g
                        + dlogit_source * rstd_source * w
                        - dlogit_source * norm_source * value
                    )
                    tl.store(
                        dv_ptr
                        + tok * stride_vt
                        + source * stride_vn
                        + offs_d * stride_vd,
                        dv,
                        mask=d_mask,
                    )
                    dw_acc += dlogit_source * rstd_source * value

    if WRITE_PARTIAL:
        tl.store(
            dw_output_ptr + pid * stride_partial + offs_d,
            dw_acc,
            mask=d_mask,
        )
    else:
        tl.atomic_add(dw_output_ptr + offs_d, dw_acc, mask=d_mask)


@triton.jit
def _reduce_dw_partials(
    partial_ptr,
    dw_ptr,
    n_partials,
    D,
    stride_partial,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Deterministically reduce private FP32 query-gradient rows."""
    offs_d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    total = tl.zeros([BLOCK_D], dtype=tl.float32)
    for partial_start in range(0, n_partials, BLOCK_P):
        offs_p = partial_start + tl.arange(0, BLOCK_P)
        values = tl.load(
            partial_ptr + offs_p[:, None] * stride_partial + offs_d[None, :],
            mask=(offs_p[:, None] < n_partials) & d_mask[None, :],
            other=0.0,
        )
        total += tl.sum(values, axis=0)
    tl.store(dw_ptr + offs_d, total, mask=d_mask)


def _source_serial_launch(d: int) -> tuple[int, int]:
    block_d = triton.next_power_of_2(d)
    warps = 4
    if block_d >= 2048:
        warps = 8
    if block_d >= 8192:
        warps = 16
    return block_d, warps


def launch_source_serial_backward(
    values: torch.Tensor,
    query: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
    plan: TrainingPlan,
    saved_state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one complete grouped source-serial plan."""
    n, b, t, d = values.shape
    n_tokens = b * t
    use_saved = plan.saves_forward_stats
    if use_saved:
        if len(saved_state) != 3:
            raise ValueError("saved source-serial plan requires three forward tensors")
        saved_alpha, saved_rstd, saved_norm = saved_state
        stride_sn = saved_alpha.stride(0)
    else:
        if saved_state:
            raise ValueError("recompute source-serial plan received unexpected saved state")
        saved_alpha = saved_rstd = saved_norm = values
        stride_sn = 0

    groups = triton.cdiv(n_tokens, plan.backward.tokens_per_cta)
    write_partial = plan.backward.dw_reduction == "partials"
    if write_partial:
        partial = torch.empty(groups, d, device=values.device, dtype=torch.float32)
        dw_output = partial
        stride_partial = partial.stride(0)
        dw = torch.empty(d, device=values.device, dtype=torch.float32)
    else:
        dw = torch.zeros(d, device=values.device, dtype=torch.float32)
        dw_output = dw
        stride_partial = 0

    dv = torch.empty_like(values)
    block_d, warps = _source_serial_launch(d)
    _bwd_source_serial_grouped[(groups,)](
        values,
        query,
        grad_out,
        saved_alpha,
        saved_rstd,
        saved_norm,
        dv,
        dw_output,
        n,
        d,
        eps,
        n_tokens,
        values.stride(0),
        values.stride(2),
        values.stride(3),
        grad_out.stride(1),
        grad_out.stride(2),
        stride_sn,
        stride_partial,
        BLOCK_N=triton.next_power_of_2(n),
        BLOCK_D=block_d,
        TOKENS=plan.backward.tokens_per_cta,
        USE_SAVED=use_saved,
        WRITE_PARTIAL=write_partial,
        num_warps=warps,
        num_stages=1,
    )
    if write_partial:
        _reduce_dw_partials[(triton.cdiv(d, 128),)](
            partial,
            dw,
            groups,
            d,
            partial.stride(0),
            BLOCK_P=8,
            BLOCK_D=128,
            num_warps=4,
            num_stages=1,
        )
    return dv, dw
