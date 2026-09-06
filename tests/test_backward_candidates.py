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
    _load_extension,
    cuda_cluster_launch_info,
    cuda_register_cluster_forward,
    cuda_register_cluster_forward_launch_info,
    cuda_register_cluster_launch_info,
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
    *,
    odd_storage_offset: bool = False,
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

    def cuda_input(source):
        if not odd_storage_offset:
            return source.cuda()
        storage = torch.empty(source.numel() + 1, device="cuda", dtype=dtype)
        result = storage[1:].view(source.shape)
        result.copy_(source)
        assert result.is_contiguous() and result.data_ptr() % 4 == 2
        return result

    values = cuda_input(values_low).requires_grad_(True)
    query = cuda_input(query_low).requires_grad_(True)
    grad = cuda_input(grad_low)
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
    if plan_name == "cuda_register_cluster_full":
        candidate_output_error = _relative_l2(output, oracle_output)
        accepted_output_error = _relative_l2(accepted_output, oracle_output)
        output_tolerance = 0.02 if dtype == torch.bfloat16 else 0.005
        assert candidate_output_error <= output_tolerance
        assert candidate_output_error <= max(1.05 * accepted_output_error, 1e-7)
    else:
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
        ("cuda_register_cluster", (9, 1, 3, 8192)),
        ("cuda_register_cluster", (32, 1, 3, 2048)),
        ("cuda_register_cluster_full", (9, 1, 3, 8192)),
        ("cuda_register_cluster_full", (32, 1, 3, 2048)),
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
    ("plan_name", "n", "d"),
    [
        ("serial_saved_partials_t16", 9, 512),
        ("cuda_shared", 9, 512),
        ("cuda_cluster", 9, 512),
        ("cuda_cluster4", 9, 512),
        ("cuda_register", 9, 4096),
        ("cuda_register_cluster", 9, 8192),
        ("cuda_register_cluster", 32, 2048),
        ("cuda_register_cluster_full", 9, 8192),
        ("cuda_register_cluster_full", 32, 2048),
    ],
)
def test_candidates_handle_uniform_ties_with_nonzero_gradient(plan_name, n, d):
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(n, 1, 3, d, generator=generator, dtype=torch.float64)
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,d", [(9, 8192), (32, 2048)])
@pytest.mark.parametrize(
    "plan_name", ["cuda_register_cluster", "cuda_register_cluster_full"]
)
def test_register_cluster_reuses_state_across_persistent_loop(
    dtype, n, d, plan_name
):
    probe = torch.empty(n, 1, 1, d, device="cuda", dtype=dtype)
    launch = cuda_register_cluster_launch_info(probe)
    assert launch["cluster_blocks"] == (4 if (n, d) == (9, 8192) else 2)
    active_clusters = launch["active_clusters"]
    if plan_name == "cuda_register_cluster_full":
        active_clusters = max(
            active_clusters,
            cuda_register_cluster_forward_launch_info(probe)["active_clusters"],
        )
    tokens = active_clusters + 3
    del probe
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(n, 1, tokens, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query /= query.norm()
    grad = torch.randn(1, tokens, d, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("plan_name", "shape"),
    [
        ("cuda_register", (9, 2, 3, 4096)),
        ("cuda_register_cluster", (9, 2, 3, 8192)),
        ("cuda_register_cluster", (32, 2, 3, 2048)),
        ("cuda_register_cluster_full", (9, 2, 3, 8192)),
        ("cuda_register_cluster_full", (32, 2, 3, 2048)),
    ],
)
def test_register_candidates_copy_odd_offset_contiguous_inputs(plan_name, shape, dtype):
    generator = torch.Generator().manual_seed(44)
    n, b, t, d = shape
    values = torch.randn(n, b, t, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query /= query.norm()
    grad = torch.randn(b, t, d, generator=generator, dtype=torch.float64)
    _compare_plan(
        plan_name,
        values,
        query,
        grad,
        dtype,
        odd_storage_offset=True,
    )


def test_direct_cuda_register_entry_rejects_unaligned_base():
    n, b, t, d = 9, 1, 1, 4096
    storage = torch.empty(n * b * t * d + 1, device="cuda", dtype=torch.float16)
    values = storage[1:].view(n, b, t, d)
    assert values.is_contiguous() and values.data_ptr() % 4 == 2
    query = torch.zeros(d, device="cuda", dtype=torch.float16)
    grad = torch.zeros(b, t, d, device="cuda", dtype=torch.float16)
    saved = tuple(
        torch.zeros(n, b * t, device="cuda", dtype=torch.float32) for _ in range(3)
    )
    with pytest.raises(RuntimeError, match="four-byte-aligned tensor bases"):
        _load_extension().register_backward(values, query, grad, *saved)


def test_direct_cuda_register_forward_rejects_unaligned_base():
    n, b, t, d = 9, 1, 1, 8192
    storage = torch.empty(n * b * t * d + 1, device="cuda", dtype=torch.float16)
    values = storage[1:].view(n, b, t, d)
    assert values.is_contiguous() and values.data_ptr() % 4 == 2
    query = torch.zeros(d, device="cuda", dtype=torch.float16)
    output = torch.empty(b, t, d, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="four-byte-aligned tensor bases"):
        _load_extension().register_cluster_forward(
            values, query, output, DEFAULT_EPS
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,d", [(9, 8192), (32, 2048)])
def test_register_cluster_forward_saves_oracle_coefficients(dtype, n, d):
    generator = torch.Generator().manual_seed(47)
    values_low = torch.randn(n, 1, 2, d, generator=generator).to(dtype)
    query_low = torch.randn(d, generator=generator)
    query_low = (query_low / query_low.norm()).to(dtype)
    values = values_low.cuda()
    query = query_low.cuda()
    output = torch.empty(1, 2, d, device="cuda", dtype=dtype)
    alpha, rstd, norm = cuda_register_cluster_forward(
        values, query, output, DEFAULT_EPS
    )

    values64 = values_low.to(torch.float64)
    query64 = query_low.to(torch.float64)
    sum_of_squares = (values64 * values64).sum(dim=-1)
    query_dot = (values64 * query64).sum(dim=-1)
    expected_rstd = torch.rsqrt(sum_of_squares / d + DEFAULT_EPS)
    expected_alpha = torch.softmax(query_dot * expected_rstd, dim=0)
    expected_norm = query_dot * expected_rstd**3 / d
    expected_output = (expected_alpha[..., None] * values64).sum(dim=0)

    assert _relative_l2(output, expected_output) <= (0.02 if dtype == torch.bfloat16 else 0.005)
    assert _relative_l2(alpha, expected_alpha.reshape(n, 2)) <= 3e-5
    assert _relative_l2(rstd, expected_rstd.reshape(n, 2)) <= 3e-5
    assert _relative_l2(norm, expected_norm.reshape(n, 2)) <= 2e-4

    repeated_output = torch.empty_like(output)
    repeated = cuda_register_cluster_forward(
        values, query, repeated_output, DEFAULT_EPS
    )
    torch.testing.assert_close(repeated_output, output, rtol=0.0, atol=0.0)
    for actual, expected in zip(repeated, (alpha, rstd, norm), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    launch = cuda_register_cluster_forward_launch_info(values)
    assert launch["cluster_blocks"] == (4 if n == 9 else 2)
    assert launch["threads_per_block"] == 512
    assert launch["active_clusters"] > 0


def test_register_candidate_handles_saturated_logits():
    generator = torch.Generator().manual_seed(43)
    values = torch.randn(9, 1, 3, 4096, generator=generator, dtype=torch.float64)
    query = torch.randn(4096, generator=generator, dtype=torch.float64)
    query = 16.0 * query / query.norm()
    grad = torch.randn(1, 3, 4096, generator=generator, dtype=torch.float64)
    _compare_plan("cuda_register", values, query, grad, torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,d", [(9, 8192), (32, 2048)])
@pytest.mark.parametrize(
    "plan_name", ["cuda_register_cluster", "cuda_register_cluster_full"]
)
def test_register_cluster_handles_saturated_logits(dtype, n, d, plan_name):
    generator = torch.Generator().manual_seed(45)
    values = torch.randn(n, 1, 3, d, generator=generator, dtype=torch.float64)
    query = torch.randn(d, generator=generator, dtype=torch.float64)
    query = 16.0 * query / query.norm()
    grad = torch.randn(1, 3, d, generator=generator, dtype=torch.float64)
    _compare_plan(plan_name, values, query, grad, dtype)
