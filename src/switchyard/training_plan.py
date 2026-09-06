"""Complete, immutable execution plans for Block AttnRes training.

A backward algorithm is not an isolated string. It determines what forward
must save, how ``dw`` is reduced, which shapes it supports, and whether it has
earned production dispatch. Keeping those decisions in one value prevents an
experimental backward from silently using the wrong forward contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RESIDENT_TILE_MAX = 32768
TARGET_OPTIN_SHARED_BYTES = 101_376

ForwardFamily = Literal["standard"]
BackwardFamily = Literal[
    "auto",
    "source_serial",
    "cuda_shared",
    "cuda_cluster",
    "cuda_cluster4",
    "cuda_register",
]
DwReduction = Literal["grouped_atomics", "partials", "persistent_atomics"]
SavedState = Literal["none", "backward_coefficients"]


@dataclass(frozen=True)
class ForwardPlan:
    family: ForwardFamily = "standard"
    saved_state: SavedState = "none"

    @property
    def saved_fields(self) -> tuple[str, ...]:
        if self.saved_state == "none":
            return ()
        # norm_coefficient = dot(query, value) * rstd**3 / D. Together these
        # fields let a feature-sharded cluster produce exact gradients after
        # one global source read.
        return ("alpha", "rstd", "norm_coefficient")


@dataclass(frozen=True)
class BackwardPlan:
    family: BackwardFamily
    tokens_per_cta: int
    dw_reduction: DwReduction

    def __post_init__(self) -> None:
        if self.tokens_per_cta <= 0 or self.tokens_per_cta & (self.tokens_per_cta - 1):
            raise ValueError("tokens_per_cta must be a positive power of two")
        if self.family in {
            "auto",
            "cuda_shared",
            "cuda_cluster",
            "cuda_cluster4",
            "cuda_register",
        } and self.tokens_per_cta != 1:
            raise ValueError(f"{self.family} manages its own token ownership")
        if self.family == "source_serial" and self.dw_reduction == "persistent_atomics":
            raise ValueError("source_serial does not use persistent atomics")
        if self.family in {"cuda_cluster", "cuda_cluster4", "cuda_register"} and self.dw_reduction != "persistent_atomics":
            raise ValueError(f"{self.family} requires persistent atomics")
        if self.family in {"auto", "cuda_shared"} and self.dw_reduction != "grouped_atomics":
            raise ValueError(f"{self.family} requires grouped atomics")


@dataclass(frozen=True)
class TrainingPlan:
    name: str
    forward: ForwardPlan
    backward: BackwardPlan
    production: bool
    rationale: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def saves_forward_stats(self) -> bool:
        return self.forward.saved_state != "none"


AUTO_PLAN = TrainingPlan(
    name="auto",
    forward=ForwardPlan(),
    backward=BackwardPlan("auto", 1, "grouped_atomics"),
    production=True,
    rationale="accepted resident-or-split dispatch",
)

EXPERIMENTAL_PLANS = (
    TrainingPlan(
        name="serial_recompute_atomic_t1",
        forward=ForwardPlan(),
        backward=BackwardPlan("source_serial", 1, "grouped_atomics"),
        production=False,
        rationale="L2-local source passes with one dw contribution per token",
    ),
    TrainingPlan(
        name="serial_recompute_atomic_t4",
        forward=ForwardPlan(),
        backward=BackwardPlan("source_serial", 4, "grouped_atomics"),
        production=False,
        rationale="L2-local source passes and fourfold lower dw contention",
    ),
    TrainingPlan(
        name="serial_saved_atomic_t4",
        forward=ForwardPlan(saved_state="backward_coefficients"),
        backward=BackwardPlan("source_serial", 4, "grouped_atomics"),
        production=False,
        rationale="compact forward state plus grouped L2-local backward",
    ),
    TrainingPlan(
        name="serial_saved_partials_t16",
        forward=ForwardPlan(saved_state="backward_coefficients"),
        backward=BackwardPlan("source_serial", 16, "partials"),
        production=False,
        rationale="compact forward state and deterministic hierarchical dw reduction",
    ),
    TrainingPlan(
        name="cuda_shared",
        forward=ForwardPlan(),
        backward=BackwardPlan("cuda_shared", 1, "grouped_atomics"),
        production=False,
        rationale="one global source read through one block's shared memory",
    ),
    TrainingPlan(
        name="cuda_cluster",
        forward=ForwardPlan(saved_state="backward_coefficients"),
        backward=BackwardPlan("cuda_cluster", 1, "persistent_atomics"),
        production=False,
        rationale="persistent feature-sharded cluster at the one-read traffic lower bound",
    ),
    TrainingPlan(
        name="cuda_cluster4",
        forward=ForwardPlan(saved_state="backward_coefficients"),
        backward=BackwardPlan("cuda_cluster4", 1, "persistent_atomics"),
        production=False,
        rationale="four-block one-read cluster with lower per-block shared-memory pressure",
    ),
    TrainingPlan(
        name="cuda_register",
        forward=ForwardPlan(saved_state="backward_coefficients"),
        backward=BackwardPlan("cuda_register", 1, "persistent_atomics"),
        production=False,
        rationale="fixed-shape persistent CTA with packed register-resident sources",
    ),
)

_PLANS = {plan.name: plan for plan in (AUTO_PLAN, *EXPERIMENTAL_PLANS)}


def get_training_plan(name: str) -> TrainingPlan:
    try:
        return _PLANS[name]
    except KeyError as exc:
        raise ValueError(f"unknown training plan: {name}") from exc


def _source_sharded_shared_bytes(
    n: int, d: int, itemsize: int, cluster_blocks: int
) -> int:
    local_sources = (n + cluster_blocks - 1) // cluster_blocks
    values_and_grad = (local_sources + 1) * d * itemsize
    aligned_values = (values_and_grad + 15) & ~15
    # Three reduction fields, one private value per warp and source.
    return aligned_values + 4 * (5 + 3 * 8) * local_sources


def _feature_sharded_shared_bytes(
    n: int, d: int, itemsize: int, cluster_blocks: int
) -> int:
    local_width = (d + cluster_blocks - 1) // cluster_blocks
    values_and_grad = (n + 1) * local_width * itemsize
    aligned_values = (values_and_grad + 15) & ~15
    query_gradient = local_width * 4
    # Five final fields plus one g-dot-v partial per warp and source.
    return aligned_values + query_gradient + 4 * (5 + 8) * n


def plan_supports(
    plan: TrainingPlan,
    n: int,
    b: int,
    t: int,
    d: int,
    dtype: str,
    *,
    optin_shared_bytes: int = TARGET_OPTIN_SHARED_BYTES,
    cluster_launch: bool = True,
    compute_capability: tuple[int, int] = (12, 0),
) -> tuple[bool, str]:
    """Pure support check used by tests and benchmark planning."""
    if min(n, b, t, d) <= 0:
        return False, "all shape dimensions must be positive"
    if max(n, d) > 2**31 - 1 or b * t > 2**31 - 1:
        return False, "N, B*T, and D must fit signed 32-bit kernel indices"
    if dtype not in {"bfloat16", "float16", "float32"}:
        return False, f"unsupported dtype {dtype}"

    family = plan.backward.family
    if family == "auto":
        return True, "accepted dispatch"
    if family == "source_serial":
        if n > 32:
            return False, "source-serial plans support at most 32 sources"
        if d > 4096:
            return False, "source-serial plans avoid compiler spills only through width 4096"
        if not plan.saves_forward_stats and n > 16:
            return False, "recompute source-serial plans avoid compiler spills only through 16 sources"
        return True, "supported"
    if dtype == "float32":
        return False, "shared-memory CUDA plans preserve fp16 or bf16 values"
    if compute_capability != (12, 0):
        return False, f"shared-memory CUDA plans require sm_120, got sm_{compute_capability[0]}{compute_capability[1]}"

    if family == "cuda_register":
        supported_shapes = {(9, 4096)}
        if (n, d) not in supported_shapes:
            return False, "register plan has no compiled specialization for this (N,D)"
        return True, "fixed-shape packed-register specialization"

    if family == "cuda_shared" and n > 16:
        return False, "one-block shared plan supports at most 16 sources"
    if family in {"cuda_cluster", "cuda_cluster4"}:
        if not cluster_launch:
            return False, "device does not support thread-block clusters"
        if n > 32:
            return False, "feature-cluster plans support at most 32 sources"

    # cuobjdump reports 1024 bytes of static shared memory for these kernels.
    dynamic = (
        _feature_sharded_shared_bytes(
            n, d, 2, 2 if family == "cuda_cluster" else 4
        )
        if family in {"cuda_cluster", "cuda_cluster4"}
        else _source_sharded_shared_bytes(n, d, 2, 1)
    )
    required = dynamic + 1024
    if required > optin_shared_bytes:
        return False, f"needs {required} shared bytes, limit is {optin_shared_bytes}"
    return True, f"needs {required} shared bytes"
