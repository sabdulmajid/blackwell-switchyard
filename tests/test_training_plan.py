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


def test_one_read_cluster_requires_complete_forward_coefficients():
    cluster = get_training_plan("cuda_cluster")
    assert cluster.forward.saved_fields == ("alpha", "rstd", "norm_coefficient")
    assert cluster.backward.dw_reduction == "persistent_atomics"


def test_one_block_shared_memory_has_honest_boundary():
    shared = get_training_plan("cuda_shared")
    assert plan_supports(shared, 9, 1, 4096, 4096, "bfloat16")[0]
    assert not plan_supports(shared, 9, 1, 4096, 8192, "bfloat16")[0]
    assert not plan_supports(shared, 32, 1, 4096, 2048, "bfloat16")[0]


def test_shared_memory_plans_do_not_claim_float32_support():
    for name in ("cuda_shared", "cuda_cluster"):
        supported, reason = plan_supports(
            get_training_plan(name), 9, 1, 4096, 4096, "float32"
        )
        assert not supported
        assert "fp16 or bf16" in reason


def test_unknown_plan_fails_loudly():
    with pytest.raises(ValueError, match="unknown training plan"):
        get_training_plan("not-a-plan")
