"""Correctness of the fused Triton kernel against the float64 oracle.

The pass criterion is relative L2 measured against a float64 reference, with the
*dtype's own rounding error* computed alongside it. That framing matters: a bf16
kernel cannot be more accurate than bf16, so the meaningful question is not "is
the error small" but "is the error close to the floor the format imposes".
The fused kernel accumulates in fp32 and rounds once, so it sits essentially
at that floor -- better than the eager formulation, which rounds repeatedly.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a CUDA device", allow_module_level=True)

from switchyard.baselines import folded_form  # noqa: E402
from switchyard.reference import DEFAULT_EPS, block_attn_res_oracle  # noqa: E402
from switchyard.triton_op import (  # noqa: E402
    BlockAttnResTriton,
    _bwd_launch,
    _launch_config,
    block_attn_res_triton,
)

DEV = "cuda"


def _rel_l2(got: torch.Tensor, oracle: torch.Tensor) -> float:
    return ((got.float() - oracle.float()).norm() / oracle.float().norm()).item()


def _dtype_floor(oracle: torch.Tensor, dtype: torch.dtype) -> float:
    o = oracle.float()
    return ((o.to(dtype).float() - o).norm() / o.norm()).item()


def _make(n, b, t, d, dtype, seed=0, scale=1.0):
    torch.manual_seed(seed)
    v = torch.randn(n, b, t, d, device=DEV, dtype=dtype)
    w = torch.randn(d, device=DEV, dtype=torch.float32)
    w = (w / w.norm() * scale).to(dtype)
    return v, w


# Covers both dispatch strategies, non-power-of-two N, D and T, several batch
# sizes, and the boundary where the resident tile stops fitting.
SHAPES = [
    (1, 1, 32, 256), (2, 1, 64, 128), (4, 2, 128, 512), (8, 1, 256, 2048),
    (9, 1, 512, 1024), (9, 1, 256, 2048), (9, 1, 128, 4096), (9, 1, 64, 8192),
    (16, 1, 128, 1024), (17, 1, 64, 2048), (32, 1, 64, 1024), (33, 1, 64, 512),
    (3, 1, 64, 777), (5, 1, 17, 63), (8, 3, 101, 1536), (7, 2, 1, 320),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_forward_is_at_the_dtype_floor(shape, dtype):
    v, w = _make(*shape, dtype)
    oracle = block_attn_res_oracle(v, w, DEFAULT_EPS)
    got = block_attn_res_triton(v, w, DEFAULT_EPS)

    assert got.shape == oracle.shape
    assert got.dtype == dtype
    assert torch.isfinite(got).all()

    err = _rel_l2(got, oracle)
    floor = _dtype_floor(oracle, dtype)
    # Within 1.6x of what merely rounding the exact answer to this dtype costs.
    assert err <= max(floor * 1.6, 1e-6), f"rel_l2 {err:.3e} vs floor {floor:.3e}"


@pytest.mark.parametrize("shape", [(9, 1, 512, 1024), (9, 1, 256, 2048), (16, 1, 128, 1024)])
def test_more_accurate_than_the_eager_baseline(shape):
    """Fusing improves accuracy here rather than costing it, because the fused
    kernel keeps every reduction in fp32 and rounds once at the end while the
    eager chain rounds at each step. Worth asserting so a future change that
    quietly drops to bf16 accumulation is caught."""
    v, w = _make(*shape, torch.bfloat16)
    oracle = block_attn_res_oracle(v, w, DEFAULT_EPS)
    assert _rel_l2(block_attn_res_triton(v, w, DEFAULT_EPS), oracle) < _rel_l2(
        folded_form(v, w, DEFAULT_EPS), oracle
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_oracle(shape):
    n, b, t, d = shape
    torch.manual_seed(1)
    v64 = torch.randn(n, b, t, d, dtype=torch.float64)
    w64 = torch.randn(d, dtype=torch.float64)
    w64 = w64 / w64.norm()
    g64 = torch.randn(b, t, d, dtype=torch.float64)

    vv = v64.clone().requires_grad_(True)
    ww = w64.clone().requires_grad_(True)
    k = vv * torch.rsqrt(vv.pow(2).mean(-1, keepdim=True) + DEFAULT_EPS)
    torch.einsum(
        "nbt,nbtd->btd", torch.einsum("d,nbtd->nbt", ww, k).softmax(0), vv
    ).backward(g64)

    vt = v64.to(torch.bfloat16).to(DEV).requires_grad_(True)
    wt = w64.to(torch.bfloat16).to(DEV).requires_grad_(True)
    block_attn_res_triton(vt, wt, DEFAULT_EPS).backward(g64.to(torch.bfloat16).to(DEV))

    assert torch.isfinite(vt.grad).all() and torch.isfinite(wt.grad).all()
    assert _rel_l2(vt.grad.cpu(), vv.grad) < 0.03

    if n == 1:
        # With one source the softmax is identically 1 whatever the logit, so
        # the query has no effect on the output and its gradient is exactly
        # zero. The oracle agrees, which makes a relative error 0/0 -- assert
        # the actual property instead.
        assert ww.grad.abs().max() == 0.0
        assert wt.grad.abs().max() == 0.0
    else:
        assert _rel_l2(wt.grad.cpu(), ww.grad) < 0.06


def test_backward_accumulates_dw_in_fp32():
    """dw is reduced over every token with atomics. In bf16 those atomics would
    lose small contributions to swamping once the token count is large, so the
    accumulator must be fp32. This checks the result stays accurate at a token
    count large enough for that to bite."""
    n, b, t, d = 9, 1, 8192, 512
    torch.manual_seed(2)
    v64 = torch.randn(n, b, t, d, dtype=torch.float64)
    w64 = torch.randn(d, dtype=torch.float64)
    w64 = w64 / w64.norm()
    g64 = torch.randn(b, t, d, dtype=torch.float64)

    vv = v64.clone().requires_grad_(True)
    ww = w64.clone().requires_grad_(True)
    k = vv * torch.rsqrt(vv.pow(2).mean(-1, keepdim=True) + DEFAULT_EPS)
    torch.einsum(
        "nbt,nbtd->btd", torch.einsum("d,nbtd->nbt", ww, k).softmax(0), vv
    ).backward(g64)

    vt = v64.to(torch.bfloat16).to(DEV).requires_grad_(True)
    wt = w64.to(torch.bfloat16).to(DEV).requires_grad_(True)
    block_attn_res_triton(vt, wt, DEFAULT_EPS).backward(g64.to(torch.bfloat16).to(DEV))
    assert _rel_l2(wt.grad.cpu(), ww.grad) < 0.06


def test_zero_query_averages():
    v = torch.randn(6, 2, 64, 512, device=DEV, dtype=torch.float32)
    w = torch.zeros(512, device=DEV, dtype=torch.float32)
    torch.testing.assert_close(
        block_attn_res_triton(v, w, DEFAULT_EPS), v.mean(0), rtol=1e-5, atol=1e-5
    )


def test_extreme_logits_do_not_overflow():
    """The kernel subtracts the running max before exponentiating. Without that
    a large query would produce inf/nan rather than a saturated softmax."""
    v, w = _make(8, 1, 128, 1024, torch.float32, scale=500.0)
    got = block_attn_res_triton(v, w, DEFAULT_EPS)
    assert torch.isfinite(got).all()
    oracle = block_attn_res_oracle(v, w, DEFAULT_EPS)
    assert _rel_l2(got, oracle) < 1e-5


def test_handles_non_contiguous_input():
    """A transposed or sliced source stack must either work or fail loudly --
    never silently read the wrong elements."""
    base = torch.randn(4, 9, 64, 512, device=DEV, dtype=torch.bfloat16)
    v = base.transpose(0, 1)  # [9, 4, 64, 512], not contiguous
    assert not v.is_contiguous()
    w = torch.randn(512, device=DEV, dtype=torch.bfloat16) / 22.6
    torch.testing.assert_close(
        block_attn_res_triton(v, w, DEFAULT_EPS),
        block_attn_res_triton(v.contiguous(), w, DEFAULT_EPS),
    )


def test_module_matches_functional():
    m = BlockAttnResTriton(256).to(DEV)
    with torch.no_grad():
        m.w.normal_(0, 0.05)
        m.g.normal_(1, 0.05)
    v = torch.randn(5, 1, 64, 256, device=DEV)
    torch.testing.assert_close(m(v), block_attn_res_triton(v, m.w * m.g, m.eps))


def test_module_starts_from_uniform_attention():
    m = BlockAttnResTriton(128).to(DEV)
    v = torch.randn(7, 1, 32, 128, device=DEV)
    torch.testing.assert_close(m(v), v.mean(0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad", ["rank", "query", "cpu"])
def test_rejects_bad_input(bad):
    v = torch.randn(4, 1, 8, 32, device=DEV)
    w = torch.randn(32, device=DEV)
    if bad == "rank":
        with pytest.raises(ValueError, match=r"\[N, B, T, D\]"):
            block_attn_res_triton(v[0], w, DEFAULT_EPS)
    elif bad == "query":
        with pytest.raises(ValueError, match="w must be"):
            block_attn_res_triton(v, torch.randn(31, device=DEV), DEFAULT_EPS)
    else:
        with pytest.raises(ValueError, match="CUDA"):
            block_attn_res_triton(v.cpu(), w.cpu(), DEFAULT_EPS)


def test_dispatch_sends_the_representative_shape_to_the_resident_kernel():
    """N=9, D=2048 is the shape a real Block AttnRes model spends most of its
    time in (about 8 blocks, plus the embedding). An earlier threshold routed it
    to the tiled kernel and cost 2.2x. Pin the decision so it cannot regress
    silently."""
    import triton

    resident, _, _, _ = _launch_config(triton.next_power_of_2(9), 2048)
    assert resident, "N=9, D=2048 must use the resident kernel"
    usable, _, _, _ = _bwd_launch(triton.next_power_of_2(9), 2048)
    assert usable, "N=9, D=2048 backward must be fused"
