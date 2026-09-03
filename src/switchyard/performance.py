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
    saved_state_bytes: int
    workspace_bytes: int
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
    source_tokens_per_cta: int = 1,
    source_uses_partials: bool = False,
    tiled_block_d: int = 1024,
    persistent_clusters: int = 94,
) -> BackwardTrafficEstimate:
    """Estimate each backward architecture before measuring it.

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
    if strategy not in {
        "resident",
        "tiled",
        "source_serial",
        "source_serial_saved",
        "cuda_shared",
        "cuda_cluster",
    }:
        raise ValueError(f"unknown backward strategy: {strategy}")
    if min(
        n,
        b,
        t,
        d,
        itemsize,
        tiled_tokens_per_cta,
        source_tokens_per_cta,
        tiled_block_d,
        persistent_clusters,
    ) <= 0:
        raise ValueError("shape, item size, and token chunk must be positive")
    if l2_to_dram_ratio <= 0:
        raise ValueError("l2_to_dram_ratio must be positive")

    token_slab = b * t * d * itemsize
    source_stack = n * token_slab
    minimum = 2 * source_stack + token_slab  # read v and g; write dv
    stats = 3 * n * b * t * 4
    saved_state = 0
    workspace = 0

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
        # The apply grid reloads all three source scalars independently for
        # every D tile. This is smaller than source traffic but must not be
        # hidden by counting one read overall.
        statistics_transfers = stats * (1 + math.ceil(d / tiled_block_d))
        dram = logical + statistics_transfers
        stats = statistics_transfers
        atomics = math.ceil(b * t / tiled_tokens_per_cta) * d
        assumptions = (
            "the source stack does not survive the kernel boundary in L2",
            "the two fp32 statistics transfers reach DRAM",
            "query-vector traffic is cache-resident and excluded",
        )
    elif strategy in {"source_serial", "source_serial_saved"}:
        # One CTA retains g while it visits all sources. Its immediate second
        # source pass is charged at the measured L2/DRAM latency ratio.
        logical = 3 * source_stack + token_slab
        dram = round(2 * source_stack + source_stack / l2_to_dram_ratio + token_slab)
        groups = math.ceil(b * t / source_tokens_per_cta)
        if source_uses_partials:
            workspace = groups * d * 4
            atomics = 0
        else:
            atomics = groups * d
        if strategy == "source_serial_saved":
            saved_state = 3 * n * b * t * 4
        assumptions = (
            f"the immediate second source pass is served by L2 at {l2_to_dram_ratio:.2f}x DRAM",
            "the output gradient remains resident within a token program",
            "query-vector traffic is cache-resident and excluded",
            "atomic serialization cost is not represented by the byte estimate",
        )
    elif strategy == "cuda_shared":
        logical = minimum
        dram = minimum
        atomics = b * t * d
        assumptions = (
            "every source value remains in one block's shared memory",
            "the output gradient remains resident within a token program",
            "query-vector traffic is cache-resident and excluded",
        )
    else:
        logical = minimum
        dram = minimum
        saved_state = 3 * n * b * t * 4
        atomics = min(b * t, persistent_clusters) * d
        assumptions = (
            "two cluster blocks own disjoint feature shards",
            "every source and output-gradient element has one global reader",
            "FP32 forward coefficients are served from cache",
            "one dw contribution is issued per persistent cluster and feature",
        )

    return BackwardTrafficEstimate(
        strategy=strategy,
        minimum_large_tensor_bytes=minimum,
        logical_large_tensor_bytes=logical,
        estimated_dram_bytes=dram,
        statistics_bytes=stats if strategy == "tiled" else 0,
        saved_state_bytes=saved_state,
        workspace_bytes=workspace,
        dw_atomic_updates=atomics,
        assumptions=assumptions,
    )
