"""CPU-only regression tests for model profiler attribution."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from bench import bench_model


@pytest.mark.parametrize(
    ("backward_events", "expected_backward_us", "expected_share"),
    [
        pytest.param(
            {"_ArenaAttnResBackward": 300.0, "_BlockAttnResTritonBackward": 200.0},
            150.0,
            0.4,
            id="nested",
        ),
        pytest.param({"_ArenaAttnResBackward": 300.0}, 150.0, 0.4, id="arena-only"),
        pytest.param({"_BlockAttnResTritonBackward": 200.0}, 100.0, 0.3, id="triton-only"),
        pytest.param({"GenericBackward": 300.0}, None, None, id="unattributable"),
    ],
)
def test_profile_step_backward_attribution(
    monkeypatch, backward_events, expected_backward_us, expected_share
):
    # Aggregated event totals cover two steps; the arena time includes its child.
    events = [
        SimpleNamespace(
            key="kernel",
            device_type=torch.autograd.DeviceType.CUDA,
            self_device_time_total=1000.0,
        ),
        *[
            SimpleNamespace(
                key=key,
                device_type=torch.autograd.DeviceType.CPU,
                device_time_total=duration,
            )
            for key, duration in {
                "attnres_stage": 20.0,
                "attnres_op": 80.0,
                **backward_events,
            }.items()
        ],
    ]
    prof = SimpleNamespace(key_averages=lambda: events)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: nullcontext(prof))
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    model = SimpleNamespace(profile_regions=False)
    step_fn = Mock()
    device = torch.device("cuda:0")

    result = bench_model.profile_step(model, step_fn, device, iters=2)

    assert result["total_device_us"] == 500.0
    assert result["attnres_forward_us"] == 50.0
    assert result["attnres_backward_us"] == expected_backward_us
    assert result["attnres_share_of_device_time"] == expected_share
    for key, duration in backward_events.items():
        if key != "GenericBackward":
            assert result[key] == duration / 2
    assert step_fn.call_count == 5  # Three warmup steps, then two profiled steps.
    assert synchronize.call_count == 2
    assert model.profile_regions is False
