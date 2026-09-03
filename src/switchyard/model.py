"""A decoder-only Transformer with a switchable residual mechanism.

This module exists to answer one question the operator microbenchmark cannot:
*how much of a real training step does Block AttnRes actually cost?* Everything
here is therefore built so that the three residual mechanisms differ in exactly
one place and are otherwise bit-for-bit the same model.

The three modes
---------------
``"standard"``
    Ordinary PreNorm residual, ``x = x + sublayer(norm(x))``. The control.
``"attnres"``
    Block AttnRes evaluated with :func:`switchyard.baselines.folded_form`, the
    strongest framework-native formulation.
``"switchyard"``
    Block AttnRes evaluated with our fused Triton kernel.

The two AttnRes modes share every module, every parameter and every code path;
only the callable that evaluates the operator differs. ``"standard"`` shares
everything except the per-site pseudo-query and RMSNorm gain, which it does not
have. Parameter initialization draws from the global RNG only in
``nn.Linear``/``nn.Embedding``, and the AttnRes parameters are constant-valued
(zeros and ones), so seeding once before construction gives all three modes
identical shared weights. ``tests/test_model.py`` asserts this.

The paper's structure
---------------------
A "layer" in arXiv:2603.15031 is a **sublayer**, so a Transformer layer holds
two AttnRes sites: one in front of attention and one in front of the MLP, each
with its own pseudo-query and RMSNorm. ``n_blocks`` partitions the ``2 *
n_layers`` sublayers into contiguous groups of ``block_size`` sublayers.

Two structures are maintained:

``blocks``
    Append-only. ``blocks[0]`` is the token embedding; ``blocks[j]`` is the
    representation of completed block ``j``.
``partial``
    A running **unweighted** sum of raw sublayer outputs inside the current
    block, reset at each block boundary. The partial a block ends with *is* that
    block's representation, which is why the reset needs no extra work.

For sublayer ``i`` in block ``n`` the sources are ``blocks[0..n]`` when ``i`` is
the first sublayer of its block and ``blocks[0..n] + [partial]`` otherwise. The
operator's output *replaces* the residual stream input; nothing is added to it::

    h       = attnres(sources)
    out     = sublayer(norm(h))
    partial = partial + out

A final readout site attends over ``blocks[0..n_blocks]`` to produce the hidden
state the LM head consumes. The paper's Eq. 6 covers the sublayers; it does not
say how the final hidden state is formed, and the obvious alternative -- feeding
the last block's ``partial`` straight to the head -- severs the embedding from
the output entirely. Giving the head its own AttnRes site is the reading
consistent with "every consumer of the residual stream gets one". It is the same
in both AttnRes modes, so it cannot affect the comparison between them.

Assembling the sources: the stacking cost
-----------------------------------------
The paper's pseudocode calls ``torch.stack(blocks + [partial])`` at every site.
That copies ``N * B * T * D`` elements per site and keeps every one of those
stacks alive for backward. Summed over a 24-layer model that is 265 slabs of
``B * T * D`` copied and held -- far more than the operator itself moves.

``sources="arena"`` instead preallocates one ``[n_blocks + 1, B, T, D]`` buffer
and writes into slices. The layout is forced by the requirement that each site's
sources be one contiguous run: slot ``0`` holds the embedding, slot ``j`` holds
``blocks[j]``, and during block ``n`` slot ``n + 1`` holds the running partial --
whose final value is exactly ``blocks[n + 1]``, so no separate write happens at a
block boundary. Total traffic is one slab per sublayer plus one for the
embedding: 49 slabs instead of 265, and 9 slabs of live memory instead of 265.

``sources="stack"`` keeps the naive version so the difference can be measured
rather than asserted. ``bench/bench_model.py`` reports both.

Why the arena needs a custom autograd Function
----------------------------------------------
Writing into a slice of a shared buffer bumps that buffer's autograd version
counter, and the counter is shared by every view of it. Site 3 saves
``arena[:4]`` for backward, site 4 writes ``arena[4]``, and backward dies with
"a variable needed for gradient computation has been modified by an inplace
operation" -- even though the bytes site 3 saved were never touched. The check is
storage-granular; the writes are slot-granular.

:class:`_ArenaAttnRes` resolves this without copying:

* the operator is handed an alias built with :meth:`torch.Tensor.set_`, which
  shares the arena's storage but carries its **own** version counter, so no
  spurious error is raised;
* the block-representation slots really are written once, so they are still
  valid at backward time -- the aliasing is safe, not merely silenced;
* the one slot that *is* rewritten is the partial's, so backward restores it
  from the saved ``partial`` before running the operator's backward. The graph
  guarantees the ordering this needs: ``h_{i+1}`` depends on ``partial_i``
  depends on ``h_i``, so site ``i + 1``'s backward always completes before site
  ``i``'s begins, and each restores its own value first;
* the arena is allocated per forward call, so two forwards before a backward
  (gradient accumulation) cannot interfere.

The inner graph is built during forward under ``enable_grad`` and replayed in
backward, so the operator's forward is never recomputed. ``tests/test_model.py``
checks arena and stack agree on both outputs and gradients, which is what makes
the reasoning above testable rather than merely plausible.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .baselines import folded_form
from .reference import DEFAULT_EPS, BlockAttnRes

__all__ = [
    "ModelConfig",
    "Transformer",
    "RESIDUAL_MODES",
    "SOURCE_MODES",
    "source_count_schedule",
    "slab_copies",
]

#: Residual mechanisms the model can be built with.
RESIDUAL_MODES = ("standard", "attnres", "switchyard")

#: How the ``[N, B, T, D]`` source tensor is assembled at each AttnRes site.
SOURCE_MODES = ("arena", "stack")

#: Residual mode -> the callable that evaluates the operator. Both take
#: ``(v, w, eps)`` and return ``[B, T, D]``.
def _block_attn_res_triton(v: Tensor, w: Tensor, eps: float) -> Tensor:
    """Import the optional Triton dependency only when the GPU path is used."""
    from .triton_op import block_attn_res_triton

    return block_attn_res_triton(v, w, eps)


_IMPLS: dict[str, Callable[[Tensor, Tensor, float], Tensor]] = {
    "attnres": folded_form,
    "switchyard": _block_attn_res_triton,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Shape and structure of the model.

    Defaults are the benchmark scale: 24 layers at ``d_model=2048``, about
    1.3 B parameters. See ``bench/bench_model.py`` for the memory arithmetic
    behind the choice.
    """

    vocab_size: int = 32768
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 5632  # 8/3 * d_model rounded to a multiple of 256, the SwiGLU convention
    n_blocks: int = 8
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    norm_eps: float = DEFAULT_EPS
    tie_embeddings: bool = True
    residual: str = "standard"
    sources: str = "arena"

    def __post_init__(self) -> None:
        if self.residual not in RESIDUAL_MODES:
            raise ValueError(f"residual must be one of {RESIDUAL_MODES}, got {self.residual!r}")
        if self.sources not in SOURCE_MODES:
            raise ValueError(f"sources must be one of {SOURCE_MODES}, got {self.sources!r}")
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "n_blocks", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.norm_eps <= 0 or not math.isfinite(self.norm_eps):
            raise ValueError(f"norm_eps must be finite and positive, got {self.norm_eps}")
        if self.rope_theta <= 0 or not math.isfinite(self.rope_theta):
            raise ValueError(f"rope_theta must be finite and positive, got {self.rope_theta}")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if self.n_sublayers % self.n_blocks:
            raise ValueError(
                f"n_blocks {self.n_blocks} must divide the sublayer count {self.n_sublayers} "
                f"(= 2 * n_layers); a block is a whole number of sublayers"
            )

    @property
    def n_sublayers(self) -> int:
        """A layer in the paper is a sublayer, and each Transformer layer has two."""
        return 2 * self.n_layers

    @property
    def block_size(self) -> int:
        """Sublayers per AttnRes block."""
        return self.n_sublayers // self.n_blocks

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def uses_attnres(self) -> bool:
        return self.residual != "standard"


def source_count_schedule(n_sublayers: int, n_blocks: int) -> list[int]:
    """The number of sources every AttnRes site attends over, in order.

    This is the paper's Eq. 6 written out. Sublayer ``i`` sits in block
    ``n = i // block_size`` and sees ``blocks[0..n]`` -- that is ``n + 1``
    sources -- plus the running partial whenever one exists, which is every
    sublayer except the first of its block. The trailing entry is the readout
    site, which sees all ``n_blocks + 1`` block representations.

    For the default 24-layer, 8-block model this is
    ``[1,2,2,2,2,2, 2,3,3,3,3,3, ..., 8,9,9,9,9,9, 9]``.
    """
    if n_sublayers <= 0 or n_blocks <= 0:
        raise ValueError("n_sublayers and n_blocks must be positive")
    if n_sublayers % n_blocks:
        raise ValueError("n_blocks must divide n_sublayers")
    block_size = n_sublayers // n_blocks
    counts = [(i // block_size) + (1 if i % block_size == 0 else 2) for i in range(n_sublayers)]
    return counts + [n_blocks + 1]


def slab_copies(n_sublayers: int, n_blocks: int, sources: str) -> int:
    """Slabs of ``B * T * D`` copied to assemble sources over one forward pass.

    The unit is deliberately a slab rather than bytes, because that is the unit
    the operator itself moves: at ``N`` sources a site reads ``N`` slabs, so a
    stacking cost of ``N`` slabs per site means the model pays for its plumbing
    exactly as much as for its arithmetic.
    """
    if sources not in SOURCE_MODES:
        raise ValueError(f"sources must be one of {SOURCE_MODES}, got {sources!r}")
    if sources == "stack":
        return sum(source_count_schedule(n_sublayers, n_blocks))
    # Arena: the embedding once, then the running partial after each sublayer.
    # The last write of a block's partial slot is that block's representation,
    # so block boundaries cost nothing extra.
    return 1 + n_sublayers


# ---------------------------------------------------------------------------
# Standard decoder pieces. Identical in all three modes, so they cancel out of
# every comparison; kept plain for exactly that reason.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return out * self.weight


def _rope_cache(max_seq_len: int, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``x: [B, H, T, head_dim]``. Rotation is done in fp32 and cast back."""
    t = x.shape[-2]
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[:t].unsqueeze(0).unsqueeze(0), sin[:t].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Plain multi-head attention with RoPE, on top of ``F.scaled_dot_product_attention``.

    Deliberately not optimized. It is byte-identical across the three residual
    modes, so whatever it costs cancels in every difference we report.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q = _apply_rope(q, *rope)
        k = _apply_rope(k, *rope)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Sublayer(nn.Module):
    """One normalization plus one operator. The unit the paper calls a "layer"."""

    def __init__(self, norm: RMSNorm, op: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.op = op

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        return self.op(self.norm(x), rope)


# ---------------------------------------------------------------------------
# The arena
# ---------------------------------------------------------------------------


def _alias(arena: Tensor, n_src: int) -> Tensor:
    """``arena[:n_src]``, but with its own autograd version counter.

    A normal slice is a view and shares the base's counter, so a later write to
    any *other* slot would make autograd reject this one at backward time even
    though its bytes are untouched. ``set_`` produces a tensor that shares
    storage without sharing the counter, which is what lets the arena be written
    incrementally. See the module docstring for why this is safe here.
    """
    out = torch.empty(0, dtype=arena.dtype, device=arena.device)
    out.set_(arena.untyped_storage(), arena.storage_offset(),
             (n_src, *arena.shape[1:]), arena.stride())
    return out


class _ArenaAttnRes(torch.autograd.Function):
    """Evaluate ``impl(arena[:n_src], w)`` and route gradients to the real sources.

    ``sources`` are the live autograd tensors mirrored in ``arena[:n_src]``; they
    are passed as inputs purely so that autograd accumulates into them. The
    arena itself carries no gradient -- it is staging for the kernel.
    """

    @staticmethod
    def forward(ctx, w, arena, n_src, has_partial, impl, eps, *sources):
        with torch.enable_grad():
            v = _alias(arena, n_src).requires_grad_(True)
            q = w.detach().requires_grad_(True)
            out = impl(v, q, eps)
        ctx.inner = (v, q, out)
        ctx.arena, ctx.n_src, ctx.has_partial = arena, n_src, has_partial
        # Only the partial needs saving: it is the one slot backward must
        # restore. Saving it also keeps it alive, which nothing else does.
        ctx.save_for_backward(*((sources[-1],) if has_partial else ()))
        return out.detach()

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.has_partial:
            with torch.no_grad():
                ctx.arena[ctx.n_src - 1].copy_(ctx.saved_tensors[0])
        v, q, out = ctx.inner
        ctx.inner = None
        dv, dw = torch.autograd.grad(out, (v, q), grad_out)
        return (dw, None, None, None, None, None, *(dv[j] for j in range(ctx.n_src)))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    """Decoder-only Transformer with a switchable residual mechanism."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        ops: list[nn.Module] = []
        for _ in range(cfg.n_layers):
            ops.append(CausalSelfAttention(cfg))
            ops.append(SwiGLU(cfg))
        self.sublayers = nn.ModuleList(
            Sublayer(RMSNorm(cfg.d_model, cfg.norm_eps), op) for op in ops
        )

        # One pseudo-query and one RMSNorm gain per sublayer, plus the readout
        # site. BlockAttnRes is reused for the parameters alone -- the operator
        # itself is dispatched through _IMPLS -- which is what guarantees the two
        # AttnRes modes have identical parameter counts by construction.
        self.attn_res = (
            nn.ModuleList(
                BlockAttnRes(cfg.d_model, eps=cfg.norm_eps)
                for _ in range(cfg.n_sublayers + 1)
            )
            if cfg.uses_attnres
            else None
        )

        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = _rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-path projections get the usual depth-scaled init so the
        # residual stream does not grow with depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for sub in self.sublayers:
            proj = sub.op.o_proj if isinstance(sub.op, CausalSelfAttention) else sub.op.down_proj
            with torch.no_grad():
                proj.weight.mul_(scale)

        #: Source count per AttnRes site from the most recent forward. Recorded
        #: so the Eq. 6 schedule can be checked against what actually ran.
        self.last_source_counts: list[int] = []

        #: Wrap the residual mechanism in ``record_function`` regions so
        #: ``torch.profiler`` can attribute its device time. Costs a little CPU
        #: per site, so it is off unless a profiling run turns it on. It only
        #: annotates the forward pass -- ``record_function`` scopes do not reach
        #: the autograd engine -- which is why ``bench/bench_model.py`` treats
        #: the profiler as a cross-check of the control difference, not as the
        #: primary measurement.
        self.profile_regions = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Only Linear and Embedding draw from the RNG, so the three residual
        # modes consume the identical random stream.
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- parameter accounting ------------------------------------------------

    def param_counts(self) -> dict[str, int]:
        """Total, and the slice of it the residual mechanism owns."""
        total = sum(p.numel() for p in self.parameters())
        attnres = (
            sum(p.numel() for m in self.attn_res for p in m.parameters())
            if self.attn_res is not None
            else 0
        )
        return {
            "total": total,
            "attnres": attnres,
            "non_attnres": total - attnres,
            "embedding": self.embed.weight.numel(),
            "tied_head": self.cfg.tie_embeddings,
        }

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        collect_alphas: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``idx: [B, T]`` int64 -> logits ``[B, T, V]``, or ``(logits, loss)``.

        ``collect_alphas`` additionally returns nothing but populates
        :attr:`last_alphas` with the mean attention weight each source received,
        per site. It is a diagnostic for the training smoke test and does real
        extra work, so leave it off when timing.
        """
        b, t = idx.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        rope = (self.rope_cos[:t], self.rope_sin[:t])

        x0 = self.embed(idx)
        self.last_alphas: list[Tensor] = [] if collect_alphas else []
        h = (
            self._forward_attnres(x0, rope, collect_alphas)
            if self.cfg.uses_attnres
            else self._forward_standard(x0, rope)
        )

        logits = self.lm_head(self.final_norm(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss

    def _forward_standard(self, x0: Tensor, rope) -> Tensor:
        x = x0
        for sub in self.sublayers:
            x = x + sub(x, rope)
        self.last_source_counts = []
        return x

    def _forward_attnres(self, x0: Tensor, rope, collect_alphas: bool) -> Tensor:
        cfg = self.cfg
        b, t, d = x0.shape
        block_size = cfg.block_size

        arena = None
        if cfg.sources == "arena":
            # Allocated per call: a second forward before backward must not
            # write over the first one's saved slots.
            arena = torch.empty(cfg.n_blocks + 1, b, t, d, device=x0.device, dtype=x0.dtype)
            with torch.no_grad():
                arena[0].copy_(x0)

        blocks: list[Tensor] = [x0]
        partial: Tensor | None = None
        counts: list[int] = []

        for i, sub in enumerate(self.sublayers):
            if i and i % block_size == 0:
                # The partial this block ended with *is* its representation, and
                # it already occupies the right arena slot.
                blocks.append(partial)
                partial = None

            sources = blocks if partial is None else [*blocks, partial]
            counts.append(len(sources))
            h = self._site(i, arena, sources, partial is not None, collect_alphas)

            out = sub(h, rope)
            partial = out if partial is None else partial + out
            if arena is not None:
                with self._region("attnres_stage"), torch.no_grad():
                    arena[len(blocks)].copy_(partial)

        blocks.append(partial)
        counts.append(len(blocks))
        self.last_source_counts = counts
        return self._site(len(self.sublayers), arena, blocks, False, collect_alphas)

    def _site(
        self,
        i: int,
        arena: Tensor | None,
        sources: list[Tensor],
        has_partial: bool,
        collect_alphas: bool,
    ) -> Tensor:
        site = self.attn_res[i]
        q = site.effective_query()
        impl = _IMPLS[self.cfg.residual]
        if collect_alphas:
            self.last_alphas.append(_alpha_stats(sources, q, self.cfg.norm_eps))
        if arena is None:
            with self._region("attnres_stage"):
                v = torch.stack(sources, 0)
            with self._region("attnres_op"):
                return impl(v, q, self.cfg.norm_eps)
        with self._region("attnres_op"):
            return _ArenaAttnRes.apply(
                q, arena, len(sources), has_partial, impl, self.cfg.norm_eps, *sources
            )

    def _region(self, name: str):
        if not self.profile_regions:
            return contextlib.nullcontext()
        return torch.profiler.record_function(name)


@torch.no_grad()
def _alpha_stats(sources: list[Tensor], q: Tensor, eps: float) -> Tensor:
    """Mean attention weight per source, averaged over batch and position.

    Computed from the reference formula in fp32, off the autograd graph. Used
    only by the training smoke test.
    """
    v = torch.stack(sources, 0).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    alpha = torch.einsum("d,nbtd->nbt", q.float(), k).softmax(0)
    return alpha.mean(dim=(1, 2))
