"""CPU-only validation for model configuration and source schedules."""

from __future__ import annotations

import pytest

from switchyard.model import ModelConfig, slab_copies, source_count_schedule


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_heads": 0},
        {"n_blocks": 0},
        {"n_layers": 0},
        {"d_model": 0},
        {"norm_eps": 0.0},
        {"norm_eps": float("nan")},
        {"rope_theta": 0.0},
    ],
)
def test_model_config_rejects_non_positive_domains(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_source_schedule_rejects_partial_blocks():
    with pytest.raises(ValueError, match="divide"):
        source_count_schedule(10, 3)


@pytest.mark.parametrize("args", [(0, 1), (1, 0), (-1, 1)])
def test_source_schedule_rejects_non_positive_domains(args):
    with pytest.raises(ValueError, match="positive"):
        source_count_schedule(*args)


def test_slab_copies_rejects_unknown_source_mode():
    with pytest.raises(ValueError, match="sources"):
        slab_copies(12, 3, "unknown")
