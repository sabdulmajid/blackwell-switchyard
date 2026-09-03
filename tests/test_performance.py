"""CPU-only checks for the explicit backward performance model."""

from __future__ import annotations

import pytest

from switchyard.performance import backward_traffic_estimate


def test_source_serial_exposes_traffic_for_atomic_tradeoff():
    shape = (9, 1, 4096, 4096)
    tiled = backward_traffic_estimate("tiled", *shape)
    serial = backward_traffic_estimate("source_serial", *shape)

    assert serial.estimated_dram_bytes < tiled.estimated_dram_bytes
    assert serial.statistics_bytes == 0
    assert tiled.statistics_bytes == 3 * shape[0] * shape[1] * shape[2] * 4
    assert serial.dw_atomic_updates == 32 * tiled.dw_atomic_updates


def test_minimum_and_estimates_scale_linearly_with_tokens():
    small = backward_traffic_estimate("tiled", 9, 1, 1024, 2048)
    large = backward_traffic_estimate("tiled", 9, 1, 4096, 2048)
    assert large.minimum_large_tensor_bytes == 4 * small.minimum_large_tensor_bytes
    assert large.logical_large_tensor_bytes == 4 * small.logical_large_tensor_bytes
    assert large.statistics_bytes == 4 * small.statistics_bytes


@pytest.mark.parametrize("strategy", ["resident", "tiled", "source_serial"])
def test_estimate_never_beats_the_information_minimum(strategy):
    estimate = backward_traffic_estimate(strategy, 32, 2, 17, 777, itemsize=4)
    assert estimate.logical_large_tensor_bytes >= estimate.minimum_large_tensor_bytes
    assert estimate.estimated_dram_bytes >= estimate.minimum_large_tensor_bytes


@pytest.mark.parametrize(
    ("strategy", "shape", "kwargs"),
    [
        ("invalid", (9, 1, 1, 64), {}),
        ("tiled", (0, 1, 1, 64), {}),
        ("source_serial", (9, 1, 1, 64), {"l2_to_dram_ratio": 0}),
    ],
)
def test_invalid_model_inputs_fail_loudly(strategy, shape, kwargs):
    with pytest.raises(ValueError):
        backward_traffic_estimate(strategy, *shape, **kwargs)
