"""Framework-native Block AttnRes formulations.

These are the baselines the optimized paths must beat. They are written to give
PyTorch and Inductor the best possible shot at the problem, because a custom
kernel that only beats a deliberately clumsy baseline proves nothing.

Every function here takes the same ``(v, w, eps)`` and returns the same
``[B, T, D]`` as :func:`switchyard.reference.block_attn_res_reference`. They
differ only in how the arithmetic is arranged.

The formulations
----------------
``paper_form``
    A transcription of the paper's Figure 2 pseudocode. Materializes the
    normalized tensor ``k`` in full. This is what a careful reader of the paper
    writes on the first try, and it is the weakest of the three.

``folded_form``
    Uses ``dot(w, v / rms(v)) == dot(w, v) / rms(v)`` to avoid ever
    materializing ``k``. Both reductions over ``D`` -- the sum of squares and
    the query dot product -- are then computable in a single pass over ``v``,
    and the ``[N, B, T, D]`` intermediate disappears entirely. This is the
    strongest framework-native formulation found, and it is the primary
    baseline.

``chunked_form``
    ``folded_form`` with the token axis tiled, so that a tile's slice of ``v``
    stays resident in L2 between the scoring pass and the weighted-sum pass.
    On this GPU L2 is 128 MiB, which is large enough to hold a meaningful tile,
    so this is a real strategy rather than a curiosity.

Numerics
--------
``folded_form`` is not bit-identical to ``paper_form``: dividing the dot product
once is a different rounding sequence than dividing every element of ``v`` and
then reducing. It is in fact slightly *more* accurate, because ``v`` is rounded
one time fewer. The test suite quantifies the difference against the float64
oracle rather than asserting the two agree with each other.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .reference import DEFAULT_EPS

__all__ = [
    "paper_form",
    "folded_form",
    "batched_folded_form",
    "chunked_form",
    "FORMULATIONS",
]


def paper_form(v: Tensor, w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Direct transcription of the paper's pseudocode. Materializes ``k``."""
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    logits = torch.einsum("d,nbtd->nbt", w, k)
    return torch.einsum("nbt,nbtd->btd", logits.softmax(0), v)


def folded_form(v: Tensor, w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Fold the RMS scale into the score. Never materializes the normalized tensor.

    The two reductions over ``D`` are written adjacently and over the same
    tensor so that Inductor can fuse them into one pass over ``v``.
    """
    dots = torch.einsum("d,nbtd->nbt", w, v)                 # [N, B, T]
    inv_rms = torch.rsqrt(v.pow(2).mean(-1) + eps)           # [N, B, T]
    alpha = (dots * inv_rms).softmax(0)
    return torch.einsum("nbt,nbtd->btd", alpha, v)


def batched_folded_form(v: Tensor, queries: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Folded framework baseline for ``S`` queries sharing the same sources."""
    dots = torch.einsum("sd,nbtd->snbt", queries, v)
    inv_rms = torch.rsqrt(v.pow(2).mean(-1) + eps)
    alpha = (dots * inv_rms.unsqueeze(0)).softmax(1)
    return torch.einsum("snbt,nbtd->sbtd", alpha, v)


def chunked_form(v: Tensor, w: Tensor, eps: float = DEFAULT_EPS, chunk: int = 2048) -> Tensor:
    """``folded_form`` tiled over the token axis to keep a tile resident in L2.

    ``chunk`` is a number of ``T`` positions. The default is deliberately
    conservative; the benchmark suite sweeps it. Setting it larger than ``T``
    degenerates to :func:`folded_form`.
    """
    _, _, t, _ = v.shape
    if chunk >= t:
        return folded_form(v, w, eps)
    return torch.cat(
        [folded_form(v[:, :, i : i + chunk], w, eps) for i in range(0, t, chunk)], dim=1
    )


#: Name -> callable, for the benchmark harness to iterate over.
FORMULATIONS = {
    "paper_form": paper_form,
    "folded_form": folded_form,
    "chunked_form": chunked_form,
}
