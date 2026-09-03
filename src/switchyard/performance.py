"""Transparent byte and contention models for Block AttnRes experiments.

These functions do not claim to replace hardware measurements. They make the
assumptions behind an experiment explicit before a kernel is timed. In
particular, a logical byte count is not a hardware-counter DRAM byte count.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BackwardTrafficEstimate:
    """Large-tensor traffic and ``dw`` contention for one backward strategy."""

    strategy: str
    minimum_large_tensor_bytes: int
    logical_large_tensor_bytes: int
    estimated_dram_bytes: int
    statistics_bytes: int
    dw_atomic_updates: int
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def backward_traffic_estimate(
    strategy: str,
    n: int,
    b: int,
    t: int,
    d: int,
    *,
    itemsize: int = 2,
    l2_to_dram_ratio: float = 4.15,
    tiled_tokens_per_cta: int = 32,
) -> BackwardTrafficEstimate:
    """Estimate the two tiled-backward designs before measuring them.

    ``large_tensor_bytes`` counts source, output-gradient, and source-gradient
    slabs. It excludes the small query vector and allocator metadata. The
    current two-kernel strategy writes and reads three fp32 values per
    ``(source, token)``. The source-serial candidate avoids that buffer and can
    reuse its second source pass from L2, but it performs one ``dw`` atomic per
    token and feature instead of one per 32-token chunk.

    The L2 estimate uses the machine's measured 4.15-to-1 L2/DRAM bandwidth
    ratio. It is a latency-equivalent model, not a claim that an L2 byte crossed
    the DRAM interface. The benchmark records both counts to prevent the model
    from being presented as a counter measurement.
    """
    if strategy not in {"resident", "tiled", "source_serial"}:
        raise ValueError(f"unknown backward strategy: {strategy}")
    if min(n, b, t, d, itemsize, tiled_tokens_per_cta) <= 0:
        raise ValueError("shape, item size, and token chunk must be positive")
    if l2_to_dram_ratio <= 0:
        raise ValueError("l2_to_dram_ratio must be positive")

    token_slab = b * t * d * itemsize
    source_stack = n * token_slab
    minimum = 2 * source_stack + token_slab  # read v and g; write dv
    stats = 3 * n * b * t * 4

    if strategy == "resident":
        logical = minimum
        dram = minimum
        stats = 0
        atomics = math.ceil(b * t / 4) * d
        assumptions = (
            "the complete source tile remains resident and is read once",
            "query-vector traffic is cache-resident and excluded",
        )
    elif strategy == "tiled":
        # Stats pass: v + g. Apply pass: v + g + dv. The statistics
        # materialization is exact and is reported separately.
        logical = 3 * source_stack + 2 * token_slab
        dram = logical + 2 * stats
        atomics = math.ceil(b * t / tiled_tokens_per_cta) * d
        assumptions = (
            "the source stack does not survive the kernel boundary in L2",
            "the two fp32 statistics transfers reach DRAM",
            "query-vector traffic is cache-resident and excluded",
        )
    else:
        # One CTA retains g while it visits all sources. Its immediate second
        # source pass is charged at the measured L2/DRAM latency ratio.
        logical = 3 * source_stack + token_slab
        dram = round(2 * source_stack + source_stack / l2_to_dram_ratio + token_slab)
        atomics = b * t * d
        assumptions = (
            f"the immediate second source pass is served by L2 at {l2_to_dram_ratio:.2f}x DRAM",
            "the output gradient remains resident within a token program",
            "query-vector traffic is cache-resident and excluded",
            "atomic serialization cost is not represented by the byte estimate",
        )

    return BackwardTrafficEstimate(
        strategy=strategy,
        minimum_large_tensor_bytes=minimum,
        logical_large_tensor_bytes=logical,
        estimated_dram_bytes=dram,
        statistics_bytes=stats if strategy == "tiled" else 0,
        dw_atomic_updates=atomics,
        assumptions=assumptions,
    )
