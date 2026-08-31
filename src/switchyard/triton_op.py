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

__all__ = ["block_attn_res_triton", "BlockAttnResTriton"]


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
    """Backward writes ``dv`` as well as reading ``v``, so its tile budget is
    tighter than the forward's. Returns ``(usable, tokens, warps, stages)``."""
    tile = n_pow2 * triton.next_power_of_2(d)
    if tile > _RESIDENT_TILE_MAX:
        return False, 0, 0, 0
    return True, 4, (4 if tile <= 16384 else 8), 1


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

        if not usable:
            # Shapes whose tile will not stay in registers fall back to autograd
            # over the folded formulation. Correct, and still avoids
            # materializing the normalized tensor, but not fused.
            with torch.enable_grad():
                vv = v.detach().requires_grad_(True)
                ww = w.detach().requires_grad_(True)
                dots = torch.einsum("d,nbtd->nbt", ww, vv)
                inv_rms = torch.rsqrt(vv.pow(2).mean(-1) + ctx.eps)
                out = torch.einsum("nbt,nbtd->btd", (dots * inv_rms).softmax(0), vv)
                gv, gw = torch.autograd.grad(out, (vv, ww), grad_out)
            return gv, gw, None

        grad_out = grad_out.contiguous()
        dv = torch.empty_like(v)
        # dw is reduced across every token by atomics, so it must be fp32
        # regardless of the working dtype: bf16 atomics would lose most of the
        # contributions to swamping.
        dw = torch.zeros(d, device=v.device, dtype=torch.float32)

        n_tokens = b * t
        grid = (triton.cdiv(n_tokens, tokens),)
        _bwd_resident[grid](
            v, w, grad_out, dv, dw,
            n, d, ctx.eps,
            v.stride(0), v.stride(2), v.stride(3),
            grad_out.stride(1), grad_out.stride(2),
            n_tokens,
            BLOCK_N=n_pow2, BLOCK_D=triton.next_power_of_2(d),
            TOKENS=tokens, num_warps=warps, num_stages=stages,
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
