"""Correctness of the reference implementation itself.

These run on CPU in float64. Their job is to pin down the *semantics* -- if the
reference is wrong, every other test in this repository is validating against a
wrong oracle, so these are deliberately about mathematical properties rather
than about agreeing with some other implementation.
"""

from __future__ import annotations

import pytest
import torch

from switchyard.baselines import batched_folded_form
from switchyard.reference import (
    DEFAULT_EPS,
    BlockAttnRes,
    attn_with_stats,
    block_attn_res_reference,
    merge_online_softmax,
    rms_norm,
)

F64 = dict(dtype=torch.float64)


def _inputs(n=5, b=2, t=7, d=16, seed=0, scale=1.0):
    torch.manual_seed(seed)
    v = torch.randn(n, b, t, d, **F64)
    w = torch.randn(d, **F64)
    return v, w / w.norm() * scale


def test_rms_norm_matches_torch():
    """Our hand-written RMSNorm must agree with PyTorch's, gain excluded."""
    x = torch.randn(3, 4, 32, **F64)
    ours = rms_norm(x, DEFAULT_EPS)
    theirs = torch.nn.functional.rms_norm(x, (32,), None, DEFAULT_EPS)
    torch.testing.assert_close(ours, theirs, rtol=1e-12, atol=1e-12)


def test_zero_query_gives_uniform_average():
    """The paper requires zero-initialized pseudo-queries so training starts
    from an equal-weight average. If this breaks, the initialization contract
    described in Sec. 5 is violated."""
    v, _ = _inputs()
    out = block_attn_res_reference(v, torch.zeros(v.shape[-1], **F64))
    torch.testing.assert_close(out, v.mean(0))


def test_single_source_is_identity():
    v, w = _inputs(n=1)
    torch.testing.assert_close(block_attn_res_reference(v, w), v[0])


def test_output_is_convex_combination():
    """Weights sum to one, so the output must lie within the elementwise hull of
    the sources. This is the property that bounds hidden-state growth and is the
    whole point of the mechanism."""
    v, w = _inputs(scale=3.0)
    out = block_attn_res_reference(v, w)
    assert (out <= v.amax(0) + 1e-12).all()
    assert (out >= v.amin(0) - 1e-12).all()


def test_weighted_sum_uses_raw_not_normalized_sources():
    """Guards the single easiest mistake to make in this operator.

    If the weighted sum accidentally ran over the normalized states, scaling one
    source by a large constant would leave the output nearly unchanged. It must
    not: the values are raw.
    """
    v, w = _inputs()
    v2 = v.clone()
    v2[0] *= 50.0
    out1 = block_attn_res_reference(v, w)
    out2 = block_attn_res_reference(v2, w)
    assert (out1 - out2).abs().max() > 1.0


def test_normalization_makes_scores_scale_invariant():
    """Conversely, scaling a source must barely change its key -- that is what
    RMSNorm on the keys buys, and it is why a block summing many layer outputs
    does not automatically dominate the attention.

    The invariance is close but not exact, and deliberately so: ``eps`` is an
    absolute floor that does not scale with the input, so
    ``rms_norm(c*x) = c*x / sqrt(c^2*mean(x^2) + eps)``, which equals
    ``rms_norm(x)`` only in the limit of large ``c^2*mean(x^2)`` relative to
    ``eps``. The residual disagreement below is that epsilon term and nothing
    else, which is why the bound is 1e-5 rather than exact.
    """
    v, w = _inputs()
    v2 = v.clone()
    v2[1] *= 100.0
    k = rms_norm(v, DEFAULT_EPS)
    k2 = rms_norm(v2, DEFAULT_EPS)
    torch.testing.assert_close(k[1], k2[1], rtol=1e-5, atol=1e-5)

    # Scaling a source by 100x must move its attention weight far less than it
    # would without normalization.
    a1 = torch.einsum("d,nbtd->nbt", w, rms_norm(v, DEFAULT_EPS)).softmax(0)
    a2 = torch.einsum("d,nbtd->nbt", w, rms_norm(v2, DEFAULT_EPS)).softmax(0)
    torch.testing.assert_close(a1, a2, rtol=1e-4, atol=1e-4)

    # Without the norm, the same scaling swamps the softmax entirely.
    b1 = torch.einsum("d,nbtd->nbt", w, v).softmax(0)
    b2 = torch.einsum("d,nbtd->nbt", w, v2).softmax(0)
    assert (b1 - b2).abs().max() > 0.5


def test_softmax_axis_is_depth_not_sequence():
    """Each (batch, token) gets its own softmax over sources. Perturbing one
    token must not affect any other token's output."""
    v, w = _inputs()
    out1 = block_attn_res_reference(v, w)
    v2 = v.clone()
    v2[:, 0, 3, :] += 10.0
    out2 = block_attn_res_reference(v2, w)
    changed = (out1 - out2).abs().sum(-1) > 1e-9
    assert changed[0, 3]
    assert changed.sum() == 1, "exactly one token position should change"


def test_no_sqrt_d_scaling():
    """The paper defines phi(q,k) = exp(q^T RMSNorm(k)) with no temperature.
    Adding a 1/sqrt(D) would change the result, so confirm the raw dot product
    is what drives the weights."""
    v, w = _inputs(n=2, b=1, t=1, d=8)
    k = rms_norm(v, DEFAULT_EPS)
    logits = torch.einsum("d,nbtd->nbt", w, k)
    expected = logits.softmax(0)
    alpha_implied = torch.linalg.lstsq(
        v[:, 0, 0, :].T, block_attn_res_reference(v, w)[0, 0]
    ).solution
    torch.testing.assert_close(alpha_implied, expected[:, 0, 0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("split", [1, 2, 4])
def test_two_phase_online_softmax_merge_is_exact(split):
    """The paper's Algorithm 1 splits sources into groups and merges with an
    online softmax. That merge must be exact, or the two-phase schedule silently
    computes something else."""
    v, w = _inputs(n=8)
    o1, m1, l1 = attn_with_stats(v[:split], w)
    o2, m2, l2 = attn_with_stats(v[split:], w)
    o, _, ell = merge_online_softmax(o1, m1, l1, o2, m2, l2)
    torch.testing.assert_close(
        o / ell.unsqueeze(-1), block_attn_res_reference(v, w), rtol=1e-12, atol=1e-12
    )


def test_merge_survives_extreme_logit_separation():
    """Online softmax exists to be numerically safe. A source group whose logits
    are enormously larger than another's must not produce inf or nan."""
    v, w = _inputs(n=6, scale=200.0)
    o1, m1, l1 = attn_with_stats(v[:3], w)
    o2, m2, l2 = attn_with_stats(v[3:], w)
    o, _, ell = merge_online_softmax(o1, m1, l1, o2, m2, l2)
    got = o / ell.unsqueeze(-1)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, block_attn_res_reference(v, w), rtol=1e-9, atol=1e-9)


def test_gradients_match_numerical_jacobian():
    """Full gradcheck in float64. Covers d/dv and d/dw together."""
    torch.manual_seed(0)
    v = torch.randn(4, 1, 3, 6, **F64, requires_grad=True)
    w = torch.randn(6, **F64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: block_attn_res_reference(a, b, DEFAULT_EPS), (v, w), eps=1e-6,
        atol=1e-8, rtol=1e-6,
    )


def test_second_order_gradients():
    """Training with gradient clipping or higher-order methods needs these to
    exist and be finite."""
    torch.manual_seed(0)
    v = torch.randn(3, 1, 2, 5, **F64, requires_grad=True)
    w = torch.randn(5, **F64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda a, b: block_attn_res_reference(a, b, DEFAULT_EPS), (v, w),
        atol=1e-6, rtol=1e-5,
    )


def test_module_initializes_query_to_zero():
    """The paper says this is required, not optional."""
    m = BlockAttnRes(d_model=32)
    assert torch.equal(m.w, torch.zeros(32))
    v = torch.randn(4, 1, 3, 32)
    torch.testing.assert_close(m(v), v.mean(0), rtol=1e-6, atol=1e-6)


def test_batched_folded_form_matches_individual_queries():
    v, _ = _inputs(n=5, b=2, t=3, d=16)
    queries = torch.randn(4, 16, **F64)
    got = batched_folded_form(v, queries)
    expected = torch.stack([block_attn_res_reference(v, q) for q in queries])
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_rmsnorm_gain_is_redundant_with_the_query():
    """Documented claim in reference.py: a learnable gain g satisfies
    dot(w, g*v_hat) == dot(w*g, v_hat), so it adds no expressive power. If this
    ever fails, the kernels -- which only see the folded vector -- are wrong."""
    torch.manual_seed(0)
    m = BlockAttnRes(d_model=16, norm_affine=True).double()
    with torch.no_grad():
        m.w.copy_(torch.randn(16, **F64))
        m.g.copy_(torch.randn(16, **F64))
    v = torch.randn(4, 2, 3, 16, **F64)
    torch.testing.assert_close(m(v), block_attn_res_reference(v, m.w * m.g, m.eps))


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 1, 1, 3), (17, 1, 2, 63), (3, 4, 5, 129)])
def test_odd_shapes(shape):
    v, w = _inputs(*shape)
    out = block_attn_res_reference(v, w)
    assert out.shape == (shape[1], shape[2], shape[3])
    assert torch.isfinite(out).all()


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[N, B, T, D\]"):
        block_attn_res_reference(torch.randn(3, 4), torch.randn(4))


def test_rejects_mismatched_query():
    with pytest.raises(ValueError, match="D="):
        block_attn_res_reference(torch.randn(2, 1, 3, 8), torch.randn(7))


@pytest.mark.parametrize(
    ("v", "w", "eps", "error"),
    [
        (torch.empty(0, 1, 1, 8), torch.randn(8), DEFAULT_EPS, ValueError),
        (torch.ones(2, 1, 1, 8, dtype=torch.int64), torch.ones(8), DEFAULT_EPS, TypeError),
        (torch.randn(2, 1, 1, 8), torch.randn(8), 0.0, ValueError),
        (torch.randn(2, 1, 1, 8), torch.randn(8), float("nan"), ValueError),
    ],
)
def test_rejects_invalid_domain_inputs(v, w, eps, error):
    with pytest.raises(error):
        block_attn_res_reference(v, w, eps)
