"""Adversarial correctness gates for complete experimental training plans."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.blackwell]

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a CUDA device", allow_module_level=True)
if torch.cuda.get_device_capability() != (12, 0):  # pragma: no cover
    pytest.skip("one-read candidates require sm_120", allow_module_level=True)

from switchyard.cuda_op import (  # noqa: E402
    cuda_cluster_launch_info,
    cuda_register_launch_info,
)
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
    # Compare every implementation with the same representable low-precision
    # problem. The prior oracle used the unrounded random tensors, which mixed
    # input quantization error with kernel arithmetic error.
    values_low = values64.to(dtype)
    query_low = query64.to(dtype)
    grad_low = grad64.to(dtype)
    oracle_values = values_low.to(torch.float64).requires_grad_(True)
    oracle_query = query_low.to(torch.float64).requires_grad_(True)
    oracle_output = block_attn_res_oracle(oracle_values, oracle_query, DEFAULT_EPS)
    oracle_dv, oracle_dw = torch.autograd.grad(
        oracle_output,
        (oracle_values, oracle_query),
        grad_low.to(torch.float64),
    )

    values = values_low.cuda().requires_grad_(True)
    query = query_low.cuda().requires_grad_(True)
    grad = grad_low.cuda()
    output = _block_attn_res_with_plan(
        values,
        query,
        DEFAULT_EPS,
        plan_name=plan_name,
    )
    accepted_values = values.detach().requires_grad_(True)
    accepted_query = query.detach().requires_grad_(True)
    accepted_output = block_attn_res_triton(
        accepted_values, accepted_query, DEFAULT_EPS
    )
    torch.testing.assert_close(output, accepted_output, rtol=0.0, atol=0.0)
    dv, dw = torch.autograd.grad(output, (values, query), grad)
    accepted_dv, accepted_dw = torch.autograd.grad(
        accepted_output, (accepted_values, accepted_query), grad
    )

    dv_tolerance = 0.03 if dtype == torch.bfloat16 else 0.01
    dw_tolerance = 0.06 if dtype == torch.bfloat16 else 0.03
    assert torch.isfinite(output).all() and torch.isfinite(dv).all() and torch.isfinite(dw).all()
    candidate_dv_error = _relative_l2(dv, oracle_dv)
    candidate_dw_error = _relative_l2(dw, oracle_dw)
    assert candidate_dv_error <= dv_tolerance
    assert candidate_dw_error <= dw_tolerance
    # cuda_shared is a recomputation-based traffic control, not a promotion
    # candidate. Its absolute oracle bounds still apply. Saved-state candidates
    # must also stay at the accepted implementation's numerical floor.
    if plan_name != "cuda_shared":
        accepted_dv_error = _relative_l2(accepted_dv, oracle_dv)
        accepted_dw_error = _relative_l2(accepted_dw, oracle_dw)
        assert candidate_dv_error <= max(1.05 * accepted_dv_error, 1e-7)
        assert candidate_dw_error <= max(1.05 * accepted_dw_error, 1e-7)
    return dv, dw


@pytest.mark.parametrize(
    ("plan_name", "shape"),
    [
        ("serial_recompute_atomic_t4", (9, 1, 5, 4097)),
        ("serial_saved_partials_t16", (17, 1, 3, 2049)),
        ("cuda_shared", (9, 1, 5, 4097)),
        ("cuda_cluster", (17, 1, 3, 2049)),
        ("cuda_cluster4", (17, 1, 3, 2049)),
        ("cuda_register", (9, 1, 3, 4096)),
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


@pytest.mark.parametrize(
    ("plan_name", "d"),
    [
        ("serial_saved_partials_t16", 512),
        ("cuda_shared", 512),
        ("cuda_cluster", 512),
        ("cuda_cluster4", 512),
        ("cuda_register", 4096),
    ],
)
def test_candidates_handle_uniform_ties_with_nonzero_gradient(plan_name, d):
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(9, 1, 3, d, generator=generator, dtype=torch.float64)
    query = torch.zeros(d, dtype=torch.float64)
    grad = torch.randn(1, 3, d, generator=generator, dtype=torch.float64)
    dv, dw = _compare_plan(plan_name, values, query, grad, torch.bfloat16)
    assert torch.count_nonzero(dv) > 0
    assert torch.count_nonzero(dw) > 0


@pytest.mark.parametrize(
    "plan_name",
    [
        "serial_saved_partials_t16",
        "cuda_shared",
        "cuda_cluster",
        "cuda_cluster4",
    ],
)
def test_candidates_handle_saturated_logits(plan_name):
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(9, 1, 5, 1024, generator=generator, dtype=torch.float64)
    query = torch.randn(1024, generator=generator, dtype=torch.float64)
    query = 16.0 * query / query.norm()
    grad = torch.randn(1, 5, 1024, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, torch.bfloat16)


@pytest.mark.parametrize(
    "plan_name",
    [
        "serial_saved_partials_t16",
        "cuda_shared",
        "cuda_cluster",
        "cuda_cluster4",
    ],
)
def test_single_source_has_exact_zero_query_gradient(plan_name):
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(1, 1, 7, 511, generator=generator, dtype=torch.float64)
    query = torch.randn(511, generator=generator, dtype=torch.float64)
    grad = torch.randn(1, 7, 511, generator=generator, dtype=torch.float64)
    _, dw = _compare_plan(plan_name, values, query, grad, torch.bfloat16)
    assert torch.count_nonzero(dw) == 0


@pytest.mark.parametrize(
    ("plan_name", "cluster_blocks"), [("cuda_cluster", 2), ("cuda_cluster4", 4)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cluster_reuses_shared_tiles_across_persistent_loop(
    plan_name, cluster_blocks, dtype
):
    """Exercise more tokens than can have one resident cluster each."""
    n, d = 32, 513
    probe = torch.empty(n, 1, 1, d, device="cuda", dtype=dtype)
    tokens = (
        cuda_cluster_launch_info(probe, cluster_blocks=cluster_blocks)[
            "active_clusters"
        ]
        + 3
    )
    del probe
    generator = torch.Generator().manual_seed(37)
    values = torch.randn(n, 1, tokens, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query /= query.norm()
    grad = torch.randn(1, tokens, d, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_register_candidate_reuses_state_across_persistent_loop(dtype):
    n, d = 9, 4096
    probe = torch.empty(n, 1, 1, d, device="cuda", dtype=dtype)
    tokens = cuda_register_launch_info(probe)["active_blocks"] + 3
    del probe
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(n, 1, tokens, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query /= query.norm()
    grad = torch.randn(1, tokens, d, generator=generator, dtype=torch.float64)
    _compare_plan("cuda_register", values, query, grad, dtype)


def test_register_candidate_handles_saturated_logits():
    generator = torch.Generator().manual_seed(43)
    values = torch.randn(9, 1, 3, 4096, generator=generator, dtype=torch.float64)
    query = torch.randn(4096, generator=generator, dtype=torch.float64)
    query = 16.0 * query / query.norm()
    grad = torch.randn(1, 3, 4096, generator=generator, dtype=torch.float64)
    _compare_plan("cuda_register", values, query, grad, torch.bfloat16)
