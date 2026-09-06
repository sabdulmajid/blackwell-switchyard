"""CPU-only checks for scalar contracts shared by GPU entry points."""

from __future__ import annotations

import pytest

from switchyard.triton_op import _validate_eps


@pytest.mark.parametrize("eps", [True, 2.0**-150, 4.0e38])
def test_kernel_epsilon_must_remain_positive_and_finite_in_float32(eps):
    with pytest.raises(ValueError, match="eps"):
        _validate_eps(eps)


@pytest.mark.parametrize("eps", [2.0**-149, 1.0e-6, 3.0e38])
def test_kernel_epsilon_accepts_float32_range(eps):
    _validate_eps(eps)
