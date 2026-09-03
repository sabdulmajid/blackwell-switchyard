"""Adversarial correctness gates for complete experimental training plans."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.blackwell]

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a CUDA device", allow_module_level=True)
if torch.cuda.get_device_capability() != (12, 0):  # pragma: no cover
    pytest.skip("one-read candidates require sm_120", allow_module_level=True)

from switchyard.reference import DEFAULT_EPS, block_attn_res_oracle  # noqa: E402
from switchyard.triton_op import _block_attn_res_with_plan, block_attn_res_triton  # noqa: E402


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64 = actual.detach().to(torch.float64).cpu()
    expected64 = expected.detach().to(torch.float64).cpu()
    denominator = expected64.norm()
    if denominator == 0:
        return actual64.norm().item()
    return ((actual64 - expected64).norm() / denominator).item()


def _compare_plan(
    plan_name: str,
    values64: torch.Tensor,
    query64: torch.Tensor,
    grad64: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    oracle_values = values64.clone().requires_grad_(True)
    oracle_query = query64.clone().requires_grad_(True)
    oracle_output = block_attn_res_oracle(oracle_values, oracle_query, DEFAULT_EPS)
    oracle_dv, oracle_dw = torch.autograd.grad(oracle_output, (oracle_values, oracle_query), grad64)

    values = values64.to(device="cuda", dtype=dtype).requires_grad_(True)
    query = query64.to(device="cuda", dtype=dtype).requires_grad_(True)
    grad = grad64.to(device="cuda", dtype=dtype)
    output = _block_attn_res_with_plan(
        values,
        query,
        DEFAULT_EPS,
        plan_name=plan_name,
    )
    accepted_output = block_attn_res_triton(values.detach(), query.detach(), DEFAULT_EPS)
    torch.testing.assert_close(output, accepted_output, rtol=0.0, atol=0.0)
    dv, dw = torch.autograd.grad(output, (values, query), grad)

    dv_tolerance = 0.03 if dtype == torch.bfloat16 else 0.01
    dw_tolerance = 0.06 if dtype == torch.bfloat16 else 0.03
    assert torch.isfinite(output).all() and torch.isfinite(dv).all() and torch.isfinite(dw).all()
    assert _relative_l2(dv, oracle_dv) <= dv_tolerance
    assert _relative_l2(dw, oracle_dw) <= dw_tolerance
    return dv, dw


@pytest.mark.parametrize(
    ("plan_name", "shape"),
    [
        ("serial_recompute_atomic_t4", (9, 1, 5, 4097)),
        ("serial_saved_partials_t16", (17, 1, 3, 2049)),
        ("cuda_shared", (9, 1, 5, 4097)),
        ("cuda_cluster", (17, 1, 3, 2049)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_candidates_match_float64_oracle_on_masked_tails(plan_name, shape, dtype):
    generator = torch.Generator().manual_seed(17)
    n, b, t, d = shape
    values = torch.randn(n, b, t, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query /= query.norm()
    grad = torch.randn(b, t, d, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, dtype)


@pytest.mark.parametrize("plan_name", ["serial_saved_partials_t16", "cuda_cluster"])
def test_candidates_handle_uniform_ties_and_zero_gradient(plan_name):
    generator = torch.Generator().manual_seed(23)
    source = torch.randn(1, 2, 7, 512, generator=generator, dtype=torch.float64)
    values = source.expand(9, -1, -1, -1).clone()
    query = torch.zeros(512, dtype=torch.float64)
    grad = torch.zeros(2, 7, 512, dtype=torch.float64)
    dv, dw = _compare_plan(plan_name, values, query, grad, torch.bfloat16)
    assert torch.count_nonzero(dv) == 0
    assert torch.count_nonzero(dw) == 0


@pytest.mark.parametrize("plan_name", ["serial_saved_partials_t16", "cuda_cluster"])
def test_candidates_handle_saturated_logits(plan_name):
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(9, 1, 5, 1024, generator=generator, dtype=torch.float64)
    query = torch.randn(1024, generator=generator, dtype=torch.float64)
    query = 16.0 * query / query.norm()
    grad = torch.randn(1, 5, 1024, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, torch.bfloat16)


@pytest.mark.parametrize("plan_name", ["serial_saved_partials_t16", "cuda_cluster"])
def test_single_source_has_exact_zero_query_gradient(plan_name):
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(1, 1, 7, 511, generator=generator, dtype=torch.float64)
    query = torch.randn(511, generator=generator, dtype=torch.float64)
    grad = torch.randn(1, 7, 511, generator=generator, dtype=torch.float64)
    _, dw = _compare_plan(plan_name, values, query, grad, torch.bfloat16)
    assert torch.count_nonzero(dw) == 0
