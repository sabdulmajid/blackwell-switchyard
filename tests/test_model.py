"""The Transformer integration: architecture fidelity and mode equivalence.

These are the tests that make the end-to-end numbers mean something. A step-time
comparison between residual modes is only evidence if the modes are genuinely
matched — same parameters, same data path, same weights — and if the AttnRes
wiring actually follows the paper. Both are checked here rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a CUDA device", allow_module_level=True)

from switchyard.model import (  # noqa: E402
    ModelConfig,
    Transformer,
    slab_copies,
    source_count_schedule,
)

DEV = "cuda"


def _cfg(**kw):
    base = dict(
        vocab_size=512, d_model=128, n_heads=4, d_ff=256,
        n_layers=6, n_blocks=3, max_seq_len=64,
    )
    base.update(kw)
    return ModelConfig(**base)


def _build(residual, sources, seed=0):
    torch.manual_seed(seed)
    return Transformer(_cfg(residual=residual, sources=sources)).to(DEV)


def _batch(b=2, t=32, vocab=512):
    torch.manual_seed(99)
    return (torch.randint(0, vocab, (b, t), device=DEV),
            torch.randint(0, vocab, (b, t), device=DEV))


# --------------------------------------------------------------------------
# Architecture fidelity: does this implement the paper?
# --------------------------------------------------------------------------


def test_source_count_schedule_matches_equation_6():
    """Paper Eq. 6: the i-th sublayer of block n attends over ``[b_0..b_{n-1}]``
    if it is the block's first, and additionally over the running partial sum
    otherwise. With the token embedding as ``b_0``, that makes the count
    ``n+1`` or ``n+2``. The trailing entry is the readout site."""
    got = source_count_schedule(12, 3)
    assert got == [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    # First sublayer of the network sees only the token embedding.
    assert got[0] == 1
    # Readout sees every block plus the embedding.
    assert got[-1] == 3 + 1


def test_model_actually_produces_that_schedule():
    """The schedule function could be right while the forward pass is not."""
    m = _build("attnres", "arena")
    idx, tgt = _batch()
    m(idx, tgt)
    assert m.last_source_counts == source_count_schedule(
        m.cfg.n_sublayers, m.cfg.n_blocks
    )


def test_two_attnres_sites_per_transformer_block():
    """A 'layer' in the paper is a sublayer, so each transformer block carries
    two AttnRes sites with their own pseudo-queries. Plus one readout site."""
    m = _build("attnres", "arena")
    assert m.cfg.n_sublayers == 2 * m.cfg.n_layers
    assert len(m.attn_res) == m.cfg.n_sublayers + 1


def test_pseudo_queries_are_zero_initialized():
    """The paper calls this crucial: it makes the initial attention uniform."""
    m = _build("attnres", "arena")
    for site in m.attn_res:
        assert torch.count_nonzero(site.w) == 0


def test_arena_saves_the_stacking_copies():
    """The paper's pseudocode stacks the sources at every site, which copies
    N slabs per site. A preallocated arena writes each source once."""
    stack = slab_copies(12, 3, "stack")
    arena = slab_copies(12, 3, "arena")
    assert stack == sum(source_count_schedule(12, 3))
    assert arena == 1 + 12
    assert stack > 2 * arena


# --------------------------------------------------------------------------
# Mode equivalence: is the comparison fair?
# --------------------------------------------------------------------------


@pytest.mark.parametrize("residual", ["attnres", "switchyard"])
def test_arena_and_stack_agree(residual):
    """The arena writes sources into one buffer in place and hands the kernel an
    alias into it, restoring the partial-sum slot during backward because later
    sublayers overwrite it. That is delicate enough that only a numerical check
    settles it, and it is the single thing most likely to be subtly wrong in the
    integration.

    Forward is required to be bit-identical: both modes hand the operator the
    same bytes in the same layout, so any difference at all would mean the arena
    is staging the wrong thing.

    Gradients are required to agree closely but not bitwise. The two modes reach
    the same mathematics by different accumulation orders -- the arena returns a
    gradient per source directly, while stacking backpropagates through
    ``torch.stack`` and sums slices -- and in fp32 those round differently. The
    observed gap is ~1e-8 absolute, which is that and nothing more.
    """
    idx, tgt = _batch()
    outs = {}
    for sources in ("stack", "arena"):
        m = _build(residual, sources)
        m.zero_grad()
        logits, loss = m(idx, tgt)
        loss.backward()
        g = torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters())])
        outs[sources] = (loss, logits, g)

    torch.testing.assert_close(outs["arena"][0], outs["stack"][0], rtol=0, atol=0)
    torch.testing.assert_close(outs["arena"][1], outs["stack"][1], rtol=0, atol=0)

    ga, gs = outs["arena"][2], outs["stack"][2]
    rel = ((ga - gs).norm() / gs.norm()).item()
    assert rel < 1e-6, f"{residual}: gradient relative L2 {rel:.2e}"
    if residual == "attnres":
        # The framework path happens to reach the identical order and is
        # bit-exact. Pinned so a change that perturbs it gets noticed.
        torch.testing.assert_close(ga, gs, rtol=0, atol=0)


def test_framework_and_fused_modes_agree():
    """Only the operator differs between these two, so they must match within
    the tolerance the fused kernel's fp32 accumulation implies."""
    idx, tgt = _batch()
    ref = _build("attnres", "arena")
    ours = _build("switchyard", "arena")
    ref.zero_grad()
    ours.zero_grad()
    _, l1 = ref(idx, tgt)
    _, l2 = ours(idx, tgt)
    l1.backward()
    l2.backward()
    torch.testing.assert_close(l2, l1, rtol=1e-5, atol=1e-5)
    g1 = torch.cat([p.grad.flatten() for _, p in sorted(ref.named_parameters())])
    g2 = torch.cat([p.grad.flatten() for _, p in sorted(ours.named_parameters())])
    assert ((g2 - g1).norm() / g1.norm()).item() < 1e-5


def test_attnres_modes_have_identical_parameter_counts():
    """If they did not, the step-time comparison would be measuring a different
    model rather than a different residual mechanism."""
    counts = {
        r: sum(p.numel() for p in _build(r, "arena").parameters())
        for r in ("attnres", "switchyard")
    }
    assert counts["attnres"] == counts["switchyard"]


def test_attnres_adds_only_the_parameters_the_paper_says():
    """One pseudo-query and one RMSNorm gain of size d_model per site, and
    nothing else. If the count drifts, the standard-vs-attnres comparison is
    quietly comparing model capacity too."""
    std = sum(p.numel() for p in _build("standard", "arena").parameters())
    att = _build("attnres", "arena")
    added = sum(p.numel() for p in att.parameters()) - std
    assert added == 2 * att.cfg.d_model * (att.cfg.n_sublayers + 1)


# --------------------------------------------------------------------------
# It trains
# --------------------------------------------------------------------------


@pytest.mark.parametrize("residual", ["standard", "attnres", "switchyard"])
def test_overfits_a_fixed_batch(residual):
    """A systems change must not break learning. Overfitting one batch is the
    cheapest test that the gradients are actually useful rather than merely
    finite."""
    torch.manual_seed(0)
    m = Transformer(_cfg(residual=residual, sources="arena")).to(DEV).float()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx, tgt = _batch()
    first = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        _, loss = m(idx, tgt)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert torch.isfinite(loss)
    assert loss.item() < first * 0.5, f"{residual}: {first:.3f} -> {loss.item():.3f}"


def test_initial_attention_is_uniform():
    """Zero-initialized queries reduce AttnRes to an equal-weight average, which
    is the property the paper says prevents training volatility."""
    m = _build("attnres", "arena")
    idx, tgt = _batch()
    m(idx, tgt, collect_alphas=True)
    for a in m.last_alphas:
        n = a.shape[0]
        torch.testing.assert_close(
            a, torch.full_like(a, 1.0 / n), rtol=1e-4, atol=1e-4
        )
