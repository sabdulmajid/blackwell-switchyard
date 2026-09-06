"""CPU-only invariants for complete forward/backward execution plans."""

from __future__ import annotations

import pytest

from switchyard.training_plan import (
    AUTO_PLAN,
    EXPERIMENTAL_PLANS,
    BackwardPlan,
    ForwardPlan,
    get_training_plan,
    plan_supports,
)


def test_only_accepted_plan_is_production():
    assert AUTO_PLAN.production
    assert all(not plan.production for plan in EXPERIMENTAL_PLANS)
    assert len({plan.name for plan in (AUTO_PLAN, *EXPERIMENTAL_PLANS)}) == 1 + len(
        EXPERIMENTAL_PLANS
    )


def test_forward_saved_state_is_atomic_contract():
    assert ForwardPlan().saved_fields == ()
    assert ForwardPlan(saved_state="backward_coefficients").saved_fields == (
        "alpha",
        "rstd",
        "norm_coefficient",
    )


@pytest.mark.parametrize("tokens", [0, 3, 6])
def test_token_group_must_be_a_positive_power_of_two(tokens):
    with pytest.raises(ValueError, match="power of two"):
        BackwardPlan("source_serial", tokens, "grouped_atomics")


def test_known_large_shapes_fit_two_block_cluster_shared_memory():
    cluster = get_training_plan("cuda_cluster")
    for n, d in ((9, 4096), (9, 8192), (32, 2048)):
        supported, reason = plan_supports(cluster, n, 1, 4096, d, "bfloat16")
        assert supported, reason


def test_four_block_cluster_reduces_per_block_shared_memory():
    cluster2 = get_training_plan("cuda_cluster")
    cluster4 = get_training_plan("cuda_cluster4")
    _, two_block_reason = plan_supports(cluster2, 9, 1, 4096, 8192, "bfloat16")
    supported, four_block_reason = plan_supports(
        cluster4, 9, 1, 4096, 8192, "bfloat16"
    )
    assert supported, four_block_reason
    two_block_bytes = int(two_block_reason.split()[1])
    four_block_bytes = int(four_block_reason.split()[1])
    assert four_block_bytes < two_block_bytes


def test_one_read_cluster_requires_complete_forward_coefficients():
    cluster = get_training_plan("cuda_cluster")
    assert cluster.forward.saved_fields == ("alpha", "rstd", "norm_coefficient")
    assert cluster.backward.dw_reduction == "persistent_atomics"


def test_source_serial_plan_rejects_spilling_width():
    plan = get_training_plan("serial_recompute_atomic_t4")
    assert plan_supports(plan, 16, 1, 4096, 4096, "bfloat16")[0]
    supported, reason = plan_supports(plan, 9, 1, 4096, 8192, "bfloat16")
    assert not supported
    assert "compiler spills" in reason
    supported, reason = plan_supports(plan, 32, 1, 4096, 2048, "bfloat16")
    assert not supported
    assert "16 sources" in reason


def test_register_candidate_is_fixed_shape_and_saves_coefficients():
    candidate = get_training_plan("cuda_register")
    assert candidate.forward.saved_fields == ("alpha", "rstd", "norm_coefficient")
    assert plan_supports(candidate, 9, 1, 4096, 4096, "bfloat16")[0]
    assert not plan_supports(candidate, 9, 1, 4096, 8192, "bfloat16")[0]
    assert not plan_supports(candidate, 32, 1, 4096, 2048, "bfloat16")[0]
    assert not plan_supports(candidate, 17, 1, 4096, 2048, "bfloat16")[0]


def test_register_cluster_covers_the_two_remaining_gap_shapes():
    candidate = get_training_plan("cuda_register_cluster")
    assert candidate.forward.saved_fields == ("alpha", "rstd", "norm_coefficient")
    assert plan_supports(candidate, 9, 1, 4096, 8192, "bfloat16")[0]
    assert plan_supports(candidate, 32, 1, 4096, 2048, "float16")[0]
    assert not plan_supports(candidate, 9, 1, 4096, 4096, "bfloat16")[0]
    assert not plan_supports(candidate, 17, 1, 4096, 2048, "bfloat16")[0]
    supported, reason = plan_supports(
        candidate, 9, 1, 4096, 8192, "bfloat16", cluster_launch=False
    )
    assert not supported
    assert "clusters" in reason


def test_complete_register_cluster_plan_pairs_one_read_forward_and_backward():
    candidate = get_training_plan("cuda_register_cluster_full")
    assert candidate.forward.family == "cuda_register_cluster"
    assert candidate.forward.saved_fields == ("alpha", "rstd", "norm_coefficient")
    assert candidate.backward.family == "cuda_register_cluster"
    assert plan_supports(candidate, 9, 1, 4096, 8192, "bfloat16")[0]
    assert plan_supports(candidate, 32, 1, 4096, 2048, "float16")[0]


def test_complete_register_cluster_support_includes_forward_shared_memory():
    candidate = get_training_plan("cuda_register_cluster_full")
    supported, reason = plan_supports(
        candidate,
        32,
        1,
        4096,
        2048,
        "bfloat16",
        optin_shared_bytes=10_000,
    )
    assert not supported
    assert "13440 shared bytes" in reason


def test_one_block_shared_memory_has_honest_boundary():
    shared = get_training_plan("cuda_shared")
    assert plan_supports(shared, 9, 1, 4096, 4096, "bfloat16")[0]
    assert not plan_supports(shared, 9, 1, 4096, 8192, "bfloat16")[0]
    assert not plan_supports(shared, 32, 1, 4096, 2048, "bfloat16")[0]


def test_shared_memory_plans_do_not_claim_float32_support():
    for name in (
        "cuda_shared",
        "cuda_cluster",
        "cuda_cluster4",
        "cuda_register",
        "cuda_register_cluster",
        "cuda_register_cluster_full",
    ):
        supported, reason = plan_supports(
            get_training_plan(name), 9, 1, 4096, 4096, "float32"
        )
        assert not supported
        assert "fp16 or bf16" in reason


def test_shared_memory_plans_require_sm120_and_bounded_kernel_indices():
    cluster = get_training_plan("cuda_cluster")
    supported, reason = plan_supports(
        cluster,
        9,
        1,
        4096,
        4096,
        "bfloat16",
        compute_capability=(9, 0),
    )
    assert not supported
    assert "sm_120" in reason
    supported, reason = plan_supports(cluster, 9, 2**31, 1, 4096, "bfloat16")
    assert not supported
    assert "32-bit" in reason


def test_unknown_plan_fails_loudly():
    with pytest.raises(ValueError, match="unknown training plan"):
        get_training_plan("not-a-plan")
