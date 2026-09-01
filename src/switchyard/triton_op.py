"""Fused Block AttnRes in Triton.

Design, and the measurements behind it
--------------------------------------
The operator moves ``(N+1)*B*T*D`` elements at minimum and does 6 FLOPs per
source element, giving an arithmetic intensity of about 2.7 FLOP/byte in bf16.
This machine's ridge point is roughly 200 FLOP/byte (299 TFLOP/s over
1462 GB/s), so the operator sits about 75x below it. There is no compute
problem here at all: the only thing worth optimizing is bytes moved, and after
that, kernels launched.

The minimum traffic is one read of every source plus one write of the output.
A framework implementation cannot reach that, because the softmax over the
source axis has to finish before the weighted sum can start, so ``v`` is read
twice. Two strategies close that gap, and which one wins depends on ``D``:

``_fwd_resident``
    When ``N * D`` values fit in registers, load ``v`` once into a
    ``[BLOCK_N, BLOCK_D]`` tile, compute the norms, logits, softmax and the
    weighted sum without touching it again, and write the output. Exactly the
    minimum traffic, one kernel.

``_fwd_tiled``
    Otherwise, loop over ``D``: accumulate the sum of squares and the query dot
    product, then loop again to apply the weights. This reads ``v`` twice from
    the memory system, but one program handles one token, so the sources touched
    by all concurrently-resident programs are far smaller than this GPU's
    128 MiB L2. The second read is therefore served at L2 bandwidth, measured at
    4.15x DRAM, making its true cost about a quarter of a DRAM read rather than
    a full one.

Both fold the RMS normalization into the score rather than materializing the
normalized tensor, using ``dot(w, v/rms(v)) == dot(w, v)/rms(v)``. That is what
removes the ``[N, B, T, D]`` intermediate entirely.

All reductions accumulate in fp32 regardless of input dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .reference import DEFAULT_EPS

__all__ = ["block_attn_res_triton", "block_attn_res_batched", "BlockAttnResTriton"]


@triton.jit
def _fwd_resident(
    v_ptr, w_ptr, out_ptr,
    n_src, D, eps,
    stride_vn, stride_vt, stride_vd,
    stride_ot, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per token. ``v`` is read once and stays in registers."""
    tok = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D

    ptrs = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    v = tl.load(ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    ssq = tl.sum(v * v, axis=1)
    dot = tl.sum(v * w[None, :], axis=1)

    logit = dot * tl.rsqrt(ssq / D + eps)
    logit = tl.where(n_mask, logit, float("-inf"))
    p = tl.exp(logit - tl.max(logit, axis=0))
    alpha = p / tl.sum(p, axis=0)

    out = tl.sum(alpha[:, None] * v, axis=0)
    tl.store(out_ptr + tok * stride_ot + offs_d * stride_od, out, mask=d_mask)


@triton.jit
def _fwd_tiled(
    v_ptr, w_ptr, out_ptr,
    n_src, D, eps,
    stride_vn, stride_vt, stride_vd,
    stride_ot, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per token, looping over ``D``. Second pass is served by L2."""
    tok = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_src

    ssq = tl.zeros([BLOCK_N], dtype=tl.float32)
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        ptrs = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        ssq += tl.sum(v * v, axis=1)
        dot += tl.sum(v * w[None, :], axis=1)

    logit = dot * tl.rsqrt(ssq / D + eps)
    logit = tl.where(n_mask, logit, float("-inf"))
    p = tl.exp(logit - tl.max(logit, axis=0))
    alpha = p / tl.sum(p, axis=0)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        ptrs = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        out = tl.sum(alpha[:, None] * v, axis=0)
        tl.store(out_ptr + tok * stride_ot + offs_d * stride_od, out, mask=d_mask)


#: Largest resident tile, in fp32 values, that still beats the tiled kernel.
#: Chosen by measurement, not by reasoning about registers -- see
#: ``docs/tuning.md``. The value matters: at N=9, D=2048 (tile 32768, and the
#: single most representative shape for this operator, since the paper uses
#: about 8 blocks plus the embedding) the tiled path reaches 616 GB/s and the
#: resident path 1365 GB/s. An earlier threshold of 16384 sent exactly that
#: shape down the slow path.
_RESIDENT_TILE_MAX = 32768


def _launch_config(n_pow2: int, d: int) -> tuple[bool, int, int, int]:
    """Pick the kernel and its launch parameters.

    Returns ``(resident, block_d, num_warps, num_stages)``.

    Every constant here came from sweeping the space on the target GPU. Two
    results were counterintuitive enough to be worth recording:

    * ``num_stages=1`` wins almost everywhere. The kernel is bandwidth-bound
      with a short dependency chain, so extra pipeline stages buy no latency
      hiding and cost occupancy.
    * The tiled kernel wants ``BLOCK_D=2048``, not the small tile that would be
      natural for a reduction. Larger tiles mean fewer, longer, more perfectly
      coalesced bursts, and this operator has bandwidth to saturate rather than
      latency to hide.
    """
    tile = n_pow2 * triton.next_power_of_2(d)
    if tile <= _RESIDENT_TILE_MAX:
        return True, triton.next_power_of_2(d), (4 if tile <= 16384 else 8), 1
    block_d = min(2048, triton.next_power_of_2(d))
    return False, block_d, 8, (3 if d >= 8192 else 1)


@triton.jit
def _speed_of_light(
    v_ptr, out_ptr, n_src, D,
    stride_vn, stride_vt, stride_vd, stride_ot, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Touch exactly the operator's bytes and do almost no arithmetic.

    This is the honest ceiling for the fused kernel: any implementation must
    read every source element and write every output element, so whatever this
    achieves is the fastest the real kernel could possibly run. Comparing
    against it is more defensible than comparing against a generic copy
    benchmark, because it has the same access pattern, the same tile shape and
    the same launch geometry -- it simply skips the softmax.
    """
    tok = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_src
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        ptrs = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + tok * stride_ot + offs_d * stride_od, tl.sum(v, axis=0), mask=d_mask)


def speed_of_light(v: torch.Tensor) -> torch.Tensor:
    """Run :func:`_speed_of_light` over ``v``; used only by the benchmark."""
    n, b, t, d = v.shape
    v = v.contiguous()
    out = torch.empty(b, t, d, device=v.device, dtype=v.dtype)
    n_pow2 = triton.next_power_of_2(n)
    _, block_d, warps, stages = _launch_config(n_pow2, d)
    _speed_of_light[(b * t,)](
        v, out, n, d,
        v.stride(0), v.stride(2), v.stride(3), out.stride(1), out.stride(2),
        BLOCK_N=n_pow2, BLOCK_D=min(block_d, 2048), num_warps=warps, num_stages=stages,
    )
    return out


@triton.jit
def _bwd_resident(
    v_ptr, w_ptr, g_ptr, dv_ptr, dw_ptr,
    n_src, D, eps,
    stride_vn, stride_vt, stride_vd,
    stride_gt, stride_gd,
    n_tokens,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, TOKENS: tl.constexpr,
):
    """Fused backward. One program per ``TOKENS`` tokens; ``v`` read once each.

    Derivation, with ``r_n = rsqrt(ssq_n/D + eps)``, ``a_n = dot(w, v_n)``,
    ``logit_n = a_n * r_n`` and ``alpha = softmax(logit)``::

        G_n      = dot(g, v_n)
        S        = sum_n alpha_n G_n
        dlogit_n = alpha_n * (G_n - S)              # softmax backward
        da_n     = dlogit_n * r_n
        dssq_n   = -dlogit_n * a_n * r_n**3 / (2D)  # through the rsqrt
        dv_n     = alpha_n * g + da_n * w + 2 * dssq_n * v_n
        dw      += sum_n da_n * v_n

    The forward statistics are recomputed here rather than saved. They cost one
    pass over data that has to be read anyway to produce ``dv``, whereas saving
    them would cost a write in forward and a read here -- and this operator has
    no arithmetic to spare bytes for.

    ``dw`` reduces over every token, so each program accumulates it privately
    and contributes once with an atomic, which is why ``TOKENS`` exists: it
    trades parallelism for a proportional cut in atomic traffic.
    """
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D

    w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in tl.static_range(TOKENS):
        tok = pid * TOKENS + i
        if tok < n_tokens:
            vp = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(vp, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            g = tl.load(g_ptr + tok * stride_gt + offs_d * stride_gd,
                        mask=d_mask, other=0.0).to(tl.float32)

            ssq = tl.sum(v * v, axis=1)
            a = tl.sum(v * w[None, :], axis=1)
            r = tl.rsqrt(ssq / D + eps)

            logit = tl.where(n_mask, a * r, float("-inf"))
            p = tl.exp(logit - tl.max(logit, axis=0))
            alpha = p / tl.sum(p, axis=0)

            gg = tl.sum(v * g[None, :], axis=1)
            s = tl.sum(alpha * gg, axis=0)
            dlogit = alpha * (gg - s)

            da = dlogit * r
            dssq = -dlogit * a * r * r * r / (2.0 * D)

            dv = alpha[:, None] * g[None, :] + da[:, None] * w[None, :] + (2.0 * dssq)[:, None] * v
            tl.store(dv_ptr + tok * stride_vt + offs_n[:, None] * stride_vn
                     + offs_d[None, :] * stride_vd,
                     dv, mask=n_mask[:, None] & d_mask[None, :])

            dw_acc += tl.sum(da[:, None] * v, axis=0)

    tl.atomic_add(dw_ptr + offs_d, dw_acc, mask=d_mask)


def _bwd_launch(n_pow2: int, d: int) -> tuple[bool, int, int, int]:
    """Whether the resident backward applies, and its launch parameters.

    Returns ``(resident, tokens, warps, stages)``. When ``resident`` is false the
    two-kernel tiled backward below is used instead.
    """
    tile = n_pow2 * triton.next_power_of_2(d)
    if tile > _RESIDENT_TILE_MAX:
        return False, 0, 0, 0
    return True, 4, (4 if tile <= 16384 else 8), 1


@triton.jit
def _bwd_stats(
    v_ptr, w_ptr, g_ptr, alpha_ptr, da_ptr, dssq_ptr,
    n_src, D, eps,
    stride_vn, stride_vt, stride_vd,
    stride_gt, stride_gd, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """First half of the tiled backward: per-token softmax statistics.

    Produces the three ``[N, B*T]`` quantities the apply pass needs. They are
    tiny -- three fp32 scalars per (source, token), against the ``[N,B,T,D]``
    tensors either side -- so materializing them is far cheaper than the
    alternative, which would be recomputing the full-``D`` reductions inside
    every ``D`` tile of the apply pass.
    """
    tok = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < n_src

    ssq = tl.zeros([BLOCK_N], dtype=tl.float32)
    a = tl.zeros([BLOCK_N], dtype=tl.float32)
    gg = tl.zeros([BLOCK_N], dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        vp = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(vp, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + tok * stride_gt + offs_d * stride_gd,
                    mask=d_mask, other=0.0).to(tl.float32)
        ssq += tl.sum(v * v, axis=1)
        a += tl.sum(v * w[None, :], axis=1)
        gg += tl.sum(v * g[None, :], axis=1)

    r = tl.rsqrt(ssq / D + eps)
    logit = tl.where(n_mask, a * r, float("-inf"))
    p = tl.exp(logit - tl.max(logit, axis=0))
    alpha = p / tl.sum(p, axis=0)

    dlogit = alpha * (gg - tl.sum(alpha * gg, axis=0))
    da = dlogit * r
    dssq = -dlogit * a * r * r * r / (2.0 * D)

    sp = offs_n * stride_sn + tok
    tl.store(alpha_ptr + sp, alpha, mask=n_mask)
    tl.store(da_ptr + sp, da, mask=n_mask)
    tl.store(dssq_ptr + sp, dssq, mask=n_mask)


@triton.jit
def _bwd_apply(
    v_ptr, w_ptr, g_ptr, alpha_ptr, da_ptr, dssq_ptr, dv_ptr, dw_ptr,
    n_src, D, n_tokens,
    stride_vn, stride_vt, stride_vd,
    stride_gt, stride_gd, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, TOKENS: tl.constexpr,
):
    """Second half: write ``dv`` and reduce ``dw``.

    The grid is (token chunks) x (D tiles). Chunking tokens is what makes ``dw``
    affordable: it reduces over every token, so a program that owned a single
    token would need one atomic per output element per token -- tens of millions
    of them, all contending on the same ``D`` addresses. Accumulating over
    ``TOKENS`` tokens first cuts that by the same factor.
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D

    w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(TOKENS):
        tok = pid_t * TOKENS + i
        if tok < n_tokens:
            vp = v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(vp, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            g = tl.load(g_ptr + tok * stride_gt + offs_d * stride_gd,
                        mask=d_mask, other=0.0).to(tl.float32)
            sp = offs_n * stride_sn + tok
            alpha = tl.load(alpha_ptr + sp, mask=n_mask, other=0.0)
            da = tl.load(da_ptr + sp, mask=n_mask, other=0.0)
            dssq = tl.load(dssq_ptr + sp, mask=n_mask, other=0.0)

            dv = alpha[:, None] * g[None, :] + da[:, None] * w[None, :] + (2.0 * dssq)[:, None] * v
            dvp = (dv_ptr + tok * stride_vt + offs_n[:, None] * stride_vn
                   + offs_d[None, :] * stride_vd)
            tl.store(dvp, dv, mask=n_mask[:, None] & d_mask[None, :])
            dw_acc += tl.sum(da[:, None] * v, axis=0)

    tl.atomic_add(dw_ptr + offs_d, dw_acc, mask=d_mask)


@triton.jit
def _fwd_batched_resident(
    v_ptr, q_ptr, out_ptr,
    n_src: tl.constexpr, n_q: tl.constexpr, D: tl.constexpr, eps,
    stride_vn, stride_vt, stride_vd,
    stride_qs, stride_qd,
    stride_os, stride_ot, stride_od,
    BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Answer several queries while keeping the source tile resident."""
    tok = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < n_src
    d_mask = offs_d < D

    v = tl.load(
        v_ptr + tok * stride_vt + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=n_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)

    for s in tl.static_range(0, BLOCK_S):
        q = tl.load(
            q_ptr + s * stride_qs + offs_d * stride_qd,
            mask=(s < n_q) & d_mask, other=0.0,
        ).to(tl.float32)
        logit = tl.sum(v * q[None, :], axis=1) * inv_rms
        logit = tl.where(n_mask, logit, float("-inf"))
        p = tl.exp(logit - tl.max(logit, axis=0))
        alpha = p / tl.sum(p, axis=0)
        out = tl.sum(alpha[:, None] * v, axis=0)
        tl.store(
            out_ptr + s * stride_os + tok * stride_ot + offs_d * stride_od,
            out,
            mask=(s < n_q) & d_mask,
        )


_BATCHED_QUERY_MAX = 16


def _batched_launch(n: int, d: int, s: int) -> tuple[bool, int, int, int, int]:
    """Return ``(resident, block_n, block_s, block_d, warps)`` for batched forward."""
    block_n = triton.next_power_of_2(n)
    block_s = triton.next_power_of_2(s)
    block_d = triton.next_power_of_2(d)
    if block_n * block_d > _RESIDENT_TILE_MAX or block_s > _BATCHED_QUERY_MAX:
        return False, block_n, block_s, block_d, 0

    # Measured on the target Blackwell part. The full-D reduction needs more
    # warps as D, the number of live sources, or the unrolled query count grows.
    warps = max(2, block_d // 1024)
    if d >= 2048 and n >= 8:
        warps *= 2
    if n >= 15 or block_s >= 16:
        warps *= 2
    return True, block_n, block_s, block_d, min(8, warps)


def block_attn_res_batched(
    v: torch.Tensor, queries: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Answer several pseudo-queries against one shared source tensor.

    The optimized path keeps the source tile resident and reuses it across all
    queries, reducing ``S`` calls that move ``S * (N + 1)`` slabs to one call
    that moves ``N + S``. It applies when the resident tile is within the same
    measured 32768-value budget as the single-query kernel and ``S <= 16``;
    other shapes fall back to accurate per-query calls.

    This output-only API does not return the log-sum-exp statistics needed to
    merge later intra-block sources, so it is not by itself the paper's complete
    two-phase schedule. It is forward-only; training should use the single-query
    autograd operator until a batched backward exists.

    Args:
        v: ``[N, B, T, D]`` sources.
        queries: ``[S, D]`` pseudo-queries.
        eps: RMSNorm epsilon.

    Returns:
        ``[S, B, T, D]``; entry ``s`` equals
        ``block_attn_res_triton(v, queries[s], eps)``.
    """
    if v.ndim != 4:
        raise ValueError(f"v must be [N, B, T, D], got {tuple(v.shape)}")
    if queries.ndim != 2 or queries.shape[1] != v.shape[-1]:
        raise ValueError(f"queries must be [S, {v.shape[-1]}], got {tuple(queries.shape)}")
    if not v.is_cuda or not queries.is_cuda:
        raise ValueError("v and queries must be CUDA tensors")
    if v.device != queries.device:
        raise ValueError("v and queries must be on the same CUDA device")
    if v.dtype != queries.dtype:
        raise ValueError("v and queries must have the same dtype")
    if v.shape[0] == 0 or queries.shape[0] == 0 or v.shape[-1] == 0:
        raise ValueError("N, S, and D must be positive")
    if v.requires_grad or queries.requires_grad:
        raise RuntimeError(
            "block_attn_res_batched has no backward; use block_attn_res_triton per query "
            "for training, or run this under torch.no_grad() for inference"
        )

    n, b, t, d = v.shape
    s = queries.shape[0]
    v = v.contiguous()
    queries = queries.contiguous()
    resident, block_n, block_s, block_d, warps = _batched_launch(n, d, s)
    if not resident:
        return torch.stack([block_attn_res_triton(v, queries[i], eps) for i in range(s)])

    out = torch.empty(s, b, t, d, device=v.device, dtype=v.dtype)
    if b * t == 0:
        return out
    _fwd_batched_resident[(b * t,)](
        v, queries, out,
        n, s, d, eps,
        v.stride(0), v.stride(2), v.stride(3),
        queries.stride(0), queries.stride(1),
        out.stride(0), out.stride(2), out.stride(3),
        BLOCK_N=block_n, BLOCK_S=block_s, BLOCK_D=block_d,
        num_warps=warps, num_stages=1,
    )
    return out


class _BlockAttnResTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, w: torch.Tensor, eps: float):
        if v.ndim != 4:
            raise ValueError(f"v must be [N, B, T, D], got {tuple(v.shape)}")
        n, b, t, d = v.shape
        if w.shape != (d,):
            raise ValueError(f"w must be [{d}], got {tuple(w.shape)}")
        if not v.is_cuda:
            raise ValueError("v must be on a CUDA device")

        # The kernel walks tokens with a single stride, so B and T are flattened.
        # Requiring contiguity here keeps the indexing honest rather than
        # silently producing wrong results on a transposed input.
        v = v.contiguous()
        w = w.contiguous()
        out = torch.empty(b, t, d, device=v.device, dtype=v.dtype)

        n_pow2 = triton.next_power_of_2(n)
        resident, block_d, warps, stages = _launch_config(n_pow2, d)
        kernel = _fwd_resident if resident else _fwd_tiled
        kernel[(b * t,)](
            v, w, out,
            n, d, eps,
            v.stride(0), v.stride(2), v.stride(3),
            out.stride(1), out.stride(2),
            BLOCK_N=n_pow2, BLOCK_D=block_d,
            num_warps=warps, num_stages=stages,
        )
        ctx.save_for_backward(v, w)
        ctx.eps = eps
        ctx.strategy = "resident" if resident else "tiled"
        return out

    @staticmethod
    def backward(ctx, grad_out):
        v, w = ctx.saved_tensors
        n, b, t, d = v.shape
        n_pow2 = triton.next_power_of_2(n)
        usable, tokens, warps, stages = _bwd_launch(n_pow2, d)

        grad_out = grad_out.contiguous()
        dv = torch.empty_like(v)
        # dw is reduced across every token by atomics, so it must be fp32
        # regardless of the working dtype: bf16 atomics would lose most of the
        # contributions to swamping once the token count is large.
        dw = torch.zeros(d, device=v.device, dtype=torch.float32)
        n_tokens = b * t

        if usable:
            _bwd_resident[(triton.cdiv(n_tokens, tokens),)](
                v, w, grad_out, dv, dw,
                n, d, ctx.eps,
                v.stride(0), v.stride(2), v.stride(3),
                grad_out.stride(1), grad_out.stride(2),
                n_tokens,
                BLOCK_N=n_pow2, BLOCK_D=triton.next_power_of_2(d),
                TOKENS=tokens, num_warps=warps, num_stages=stages,
            )
            return dv, dw.to(w.dtype), None

        # Tiled backward: two kernels, because dw reduces over every token while
        # the softmax statistics need a full-D reduction per token, and no single
        # loop nesting satisfies both.
        block_d = min(1024, triton.next_power_of_2(d))
        alpha = torch.empty(n, n_tokens, device=v.device, dtype=torch.float32)
        da = torch.empty_like(alpha)
        dssq = torch.empty_like(alpha)

        _bwd_stats[(n_tokens,)](
            v, w, grad_out, alpha, da, dssq,
            n, d, ctx.eps,
            v.stride(0), v.stride(2), v.stride(3),
            grad_out.stride(1), grad_out.stride(2), alpha.stride(0),
            BLOCK_N=n_pow2, BLOCK_D=block_d, num_warps=8, num_stages=1,
        )
        tokens_per_cta = 32
        _bwd_apply[(triton.cdiv(n_tokens, tokens_per_cta), triton.cdiv(d, block_d))](
            v, w, grad_out, alpha, da, dssq, dv, dw,
            n, d, n_tokens,
            v.stride(0), v.stride(2), v.stride(3),
            grad_out.stride(1), grad_out.stride(2), alpha.stride(0),
            BLOCK_N=n_pow2, BLOCK_D=block_d, TOKENS=tokens_per_cta,
            num_warps=8, num_stages=1,
        )
        return dv, dw.to(w.dtype), None


def block_attn_res_triton(v: torch.Tensor, w: torch.Tensor, eps: float = DEFAULT_EPS):
    """Fused Block AttnRes. Same contract as ``block_attn_res_reference``."""
    return _BlockAttnResTriton.apply(v, w, eps)


class BlockAttnResTriton(torch.nn.Module):
    """Drop-in replacement for :class:`switchyard.reference.BlockAttnRes`."""

    def __init__(self, d_model: int, norm_affine: bool = True, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w = torch.nn.Parameter(torch.zeros(d_model))
        self.g = torch.nn.Parameter(torch.ones(d_model)) if norm_affine else None

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        q = self.w if self.g is None else self.w * self.g
        return block_attn_res_triton(v, q, self.eps)
