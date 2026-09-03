"""Canonical, paper-faithful Block Attention Residuals.

This module is the project's correctness oracle. It is written to be read next to
the paper, not to be fast. Every other execution path in this repository is
validated against it.

Source
------
Attention Residuals, Kimi Team (Moonshot AI), arXiv:2603.15031, March 2026.
The operator below implements Eq. 2-4 and Eq. 6 and matches the PyTorch-style
pseudocode in Figure 2 of that paper. The upstream repository
(github.com/MoonshotAI/Attention-Residuals) is documentation-only and ships no
source, so this is an independent implementation from the published mathematics.

Shape notation used throughout this repository
----------------------------------------------
``N``
    Length of the *source axis*: the number of previous states being attended
    over. This is the ``n`` subscript in the paper's own einsum
    ``'n b t d -> n b t'``. Concretely a source is either a completed block
    representation ``b_j``, the token embedding ``b_0``, or the current
    intra-block partial sum ``b_n^i``.

    Beware the collision in the paper's prose: there, ``N`` is the *architectural
    block count* and a layer attends over ``N``-or-``N+1`` sources depending on
    its position in its block. In this codebase ``N`` always means the actual
    stacked source count, so a model with ``n_blocks=8`` produces operator calls
    with ``N`` between 1 and 9.
``B``
    Batch size.
``T``
    Sequence length. Depth attention is applied independently per ``(b, t)``.
``D``
    Hidden dimension.

The operator's value tensor is therefore ``v: [N, B, T, D]`` and its output is
``[B, T, D]``. The pseudo-query is ``w: [D]`` -- one vector per layer, shared
across every batch element and token position.

The mathematics
---------------
For a single token, with sources ``v_0 .. v_{N-1}`` and pseudo-query ``w``::

    k_i     = v_i / sqrt(mean(v_i ** 2) + eps)      # RMSNorm over D
    logit_i = dot(w, k_i)                           # no 1/sqrt(D) scaling
    alpha   = softmax(logit)                        # over the N source axis
    out     = sum_i alpha_i * v_i                   # weighted sum of RAW sources

Three details are easy to get wrong and are worth stating explicitly, because
each one silently changes the result rather than raising an error:

1. The weighted sum is over the **raw** sources ``v``, not the normalized ``k``.
   Normalization exists only to keep a large-magnitude source from dominating
   the attention weights (paper Sec. 3.1); it is not applied to the values.
2. The softmax runs over the **source/depth axis**, independently for every
   ``(b, t)`` position. This is depth attention, not sequence attention -- there
   is no interaction between token positions anywhere in this operator.
3. There is **no** ``1/sqrt(D)`` scaling on the logits. The paper defines
   ``phi(q, k) = exp(q^T RMSNorm(k))`` with no temperature term.

A learnable RMSNorm gain is deliberately not part of the functional operator.
If the norm carried a gain ``g``, then

    dot(w, g * v_hat) == dot(w * g, v_hat)

so ``g`` is exactly redundant with the pseudo-query in the forward pass. The
:class:`BlockAttnRes` module below keeps ``g`` as a real parameter for
faithfulness to the paper's "one RMSNorm and one pseudo-query per layer", and
folds ``w * g`` before calling the operator; autograd then distributes gradient
to both. The kernels only ever see the folded vector.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "rms_norm",
    "block_attn_res_reference",
    "block_attn_res_oracle",
    "BlockAttnRes",
    "attn_with_stats",
    "merge_online_softmax",
    "attn_res_flops",
    "attn_res_min_bytes",
    "arithmetic_intensity",
]

# The paper does not state an epsilon. 1e-6 matches the RMSNorm convention used
# by Llama and by Kimi Linear, the architecture AttnRes was integrated into.
DEFAULT_EPS = 1e-6


def rms_norm(x: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Root-mean-square normalization over the last dimension, without gain.

    Written out rather than delegated to :class:`torch.nn.RMSNorm` so the
    reference stays readable next to the paper. ``tests/`` checks the two agree.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def block_attn_res_reference(v: Tensor, w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Block Attention Residuals, exactly as written in the paper.

    This is a direct transcription of the mathematics. It materializes every
    intermediate and makes no attempt to be efficient -- that is the point.

    Args:
        v: ``[N, B, T, D]`` stacked source states. Source ``0`` is conventionally
            the token embedding ``b_0``; the last source is the current
            intra-block partial sum when one exists.
        w: ``[D]`` pseudo-query for this layer.
        eps: RMSNorm epsilon.

    Returns:
        ``[B, T, D]`` the input to the layer.
    """
    if v.ndim != 4:
        raise ValueError(f"v must be [N, B, T, D], got shape {tuple(v.shape)}")
    if w.ndim != 1:
        raise ValueError(f"w must be [D], got shape {tuple(w.shape)}")
    if w.shape[0] != v.shape[-1]:
        raise ValueError(f"w has D={w.shape[0]} but v has D={v.shape[-1]}")
    if min(v.shape) <= 0:
        raise ValueError(f"N, B, T, and D must be positive, got shape {tuple(v.shape)}")
    if v.device != w.device:
        raise ValueError("v and w must be on the same device")
    if not v.is_floating_point() or not w.is_floating_point():
        raise TypeError("v and w must be floating-point tensors")
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a finite positive number, got {eps!r}")

    k = rms_norm(v, eps)                                # [N, B, T, D]
    logits = torch.einsum("d,nbtd->nbt", w, k)          # [N, B, T]
    alpha = logits.softmax(0)                           # over the source axis
    return torch.einsum("nbt,nbtd->btd", alpha, v)      # [B, T, D]


def block_attn_res_oracle(
    v: Tensor, w: Tensor, eps: float = DEFAULT_EPS, dtype: torch.dtype = torch.float64
) -> Tensor:
    """High-precision oracle, returned in ``dtype``.

    Used by the test suite to separate "the fast path is wrong" from "the
    reference itself lost precision at this shape". Compare a low-precision
    result against this rather than against :func:`block_attn_res_reference`
    when the question is how much error is acceptable.
    """
    return block_attn_res_reference(v.to(dtype), w.to(dtype), eps)


class BlockAttnRes(nn.Module):
    """One layer's Block AttnRes residual, as an ``nn.Module``.

    Holds the two parameters the paper adds per layer: the pseudo-query ``w``
    and the RMSNorm gain ``g``.

    ``w`` is initialized to **zeros**, which the paper requires (Sec. 5): it
    makes the initial attention weights uniform over sources, so the layer
    starts as an equal-weight average and training does not destabilize.

    Args:
        d_model: hidden dimension ``D``.
        norm_affine: keep a learnable RMSNorm gain. Faithful to the paper's
            per-layer RMSNorm, but note the gain is mathematically redundant
            with ``w`` -- see the module docstring for why.
        eps: RMSNorm epsilon.
    """

    def __init__(self, d_model: int, norm_affine: bool = True, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w = nn.Parameter(torch.zeros(d_model))
        self.g = nn.Parameter(torch.ones(d_model)) if norm_affine else None

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, norm_affine={self.g is not None}, eps={self.eps}"

    def effective_query(self) -> Tensor:
        """``w * g``, the single vector the kernels actually consume."""
        return self.w if self.g is None else self.w * self.g

    def forward(self, v: Tensor) -> Tensor:
        """``v: [N, B, T, D]`` -> ``[B, T, D]``."""
        return block_attn_res_reference(v, self.effective_query(), self.eps)


def uniform_attention_check(n: int) -> float:
    """The attention weight every source receives at initialization.

    Trivial, but it is the invariant that the zero-init requirement buys, and
    the training smoke test asserts against it.
    """
    return 1.0 / n


def attn_with_stats(
    v: Tensor, w: Tensor, eps: float = DEFAULT_EPS
) -> tuple[Tensor, Tensor, Tensor]:
    """``ATTNWITHSTATS`` from the paper's Algorithm 1.

    The two-phase schedule splits the sources into a batched inter-block group
    and a sequential intra-block group, then merges the partial results with an
    online softmax. For that merge to be exact each phase must return the
    running max and the sum of exponentials alongside its unnormalized output.

    Returns:
        ``(o, m, ell)`` with ``o: [B, T, D]``, ``m: [B, T]``, ``ell: [B, T]``,
        where ``m`` is the max logit, ``ell = sum_i exp(logit_i - m)``, and
        ``o = sum_i exp(logit_i - m) * v_i``. Then ``o / ell`` equals
        :func:`block_attn_res_reference`.
    """
    k = rms_norm(v, eps)
    logits = torch.einsum("d,nbtd->nbt", w, k)      # [N, B, T]
    m = logits.amax(0)                              # [B, T]
    p = (logits - m).exp()                          # [N, B, T]
    o = torch.einsum("nbt,nbtd->btd", p, v)
    return o, m, p.sum(0)


def merge_online_softmax(
    o1: Tensor, m1: Tensor, l1: Tensor, o2: Tensor, m2: Tensor, l2: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Combine two partial attention results into one, exactly.

    Implements Algorithm 1 line 12. Given ``(o, m, ell)`` triples for two
    disjoint source groups, returns the triple for their union. Merging and then
    dividing agrees with attending over the concatenated sources in a single
    pass, up to floating-point rounding only.
    """
    m = torch.maximum(m1, m2)
    s1 = (m1 - m).exp()
    s2 = (m2 - m).exp()
    o = o1 * s1.unsqueeze(-1) + o2 * s2.unsqueeze(-1)
    return o, m, l1 * s1 + l2 * s2


def attn_res_flops(n: int, b: int, t: int, d: int) -> int:
    """Arithmetic for one operator call, counting a multiply-add as 2 FLOPs.

    Used by the benchmark harness to report achieved FLOP/s. The three passes
    over ``[N, B, T, D]`` are: the sum of squares for RMSNorm, the query dot
    product, and the weighted accumulation.
    """
    elems = n * b * t * d
    return 2 * elems * 3


def attn_res_min_bytes(n: int, b: int, t: int, d: int, itemsize: int = 2) -> int:
    """Lower bound on DRAM traffic for one operator call.

    Every source element must be read at least once and the output written
    once. Anything a real implementation moves beyond this is overhead --
    re-reads, materialized intermediates, or spills. This is the denominator
    for the project's bandwidth-efficiency numbers.
    """
    return (n * b * t * d + b * t * d) * itemsize


def arithmetic_intensity(n: int, b: int, t: int, d: int, itemsize: int = 2) -> float:
    """FLOPs per byte at the theoretical-minimum traffic.

    Independent of ``B``, ``T`` and ``D``; it depends only on ``N``. That is
    the whole story of this operator: it is hopelessly memory-bound, and no
    amount of tiling changes the ratio. Compare against the machine's
    ridge point (peak FLOP/s divided by achievable bandwidth) to confirm.
    """
    return attn_res_flops(n, b, t, d) / attn_res_min_bytes(n, b, t, d, itemsize)


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """Machine balance in FLOPs per byte. Below this, a kernel is memory-bound."""
    return peak_flops / peak_bandwidth


def _sanity() -> None:
    """Cheap self-check of the invariants the rest of the project relies on."""
    torch.manual_seed(0)
    n, b, t, d = 5, 2, 7, 64
    v = torch.randn(n, b, t, d, dtype=torch.float64)
    w = torch.randn(d, dtype=torch.float64)

    # Zero query gives a uniform average.
    out0 = block_attn_res_reference(v, torch.zeros(d, dtype=torch.float64))
    assert torch.allclose(out0, v.mean(0)), "zero query must average the sources"

    # The two-phase split reproduces the single-pass result.
    o1, m1, l1 = attn_with_stats(v[:3], w)
    o2, m2, l2 = attn_with_stats(v[3:], w)
    o, _, ell = merge_online_softmax(o1, m1, l1, o2, m2, l2)
    merged = o / ell.unsqueeze(-1)
    assert torch.allclose(merged, block_attn_res_reference(v, w), atol=1e-12), "merge must be exact"

    # A single source is the identity.
    assert torch.allclose(block_attn_res_reference(v[:1], w), v[0]), "N=1 must be identity"

    assert math.isclose(arithmetic_intensity(9, 1, 1, 1), 6 * 9 / (10 * 2))
    print("reference self-check passed")


if __name__ == "__main__":
    _sanity()
