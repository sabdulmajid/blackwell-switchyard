"""Measurement primitives shared by every benchmark in this repository.

The rules encoded here, and why each one exists:

**L2 is flushed between timed repetitions.** This GPU has 128 MiB of L2. Timing
a small shape in a tight loop leaves the entire input resident, and the second
repetition onward measures cache bandwidth rather than the DRAM traffic a real
model would pay. That inflates results by up to 4x on this part (see
``docs/machine.md``). Flushing is not free, so it happens between the timed
regions, never inside one.

**Nothing is timed until it has been checked for correctness.** A fast wrong
kernel is not a result. :func:`check_against` runs first and a failure removes
the implementation from the run rather than annotating it.

**Compilation is never counted as steady state.** Compiled implementations get
their own warmup phase and their compile time is reported as a separate number.

**Statistics are order statistics.** Median and p10/p90 rather than mean and
standard deviation, because launch hiccups produce a long right tail that a
mean quietly absorbs.
"""

from __future__ import annotations

import gc
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Large enough to evict this GPU's 128 MiB L2 several times over.
_L2_FLUSH_BYTES = 384 * 2**20
_flush_buffer: torch.Tensor | None = None


def repository_provenance(
    repo: Path, third_party: Mapping[str, Path] | None = None
) -> dict:
    """Record exact source revisions and seeds alongside raw measurements."""

    def git(path: Path, *args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revisions = {}
    for name, path in (third_party or {}).items():
        if path.is_dir():
            revisions[name] = git(path, "rev-parse", "HEAD")

    return {
        "argv": sys.argv.copy(),
        "repository_commit": git(repo, "rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(
            git(repo, "status", "--porcelain", "--untracked-files=no")
        ),
        "third_party_commits": revisions,
        "input_seed": 0,
        "query_seed": 1,
    }


def _flush_l2(device: torch.device) -> None:
    global _flush_buffer
    if _flush_buffer is None or _flush_buffer.device != device:
        _flush_buffer = torch.empty(_L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    _flush_buffer.fill_(0)


@dataclass
class Timing:
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    mean_ms: float
    cv: float
    reps: int
    warmup: int
    l2_flushed: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def measure_latency(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 25,
    reps: int = 100,
    flush_l2: bool = True,
) -> Timing:
    """Median wall-clock of ``fn`` on the GPU, via CUDA events.

    ``fn`` should perform exactly the work under test and nothing else -- build
    inputs outside it.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    samples: list[float] = []
    for _ in range(reps):
        if flush_l2:
            _flush_l2(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))

    samples.sort()
    n = len(samples)
    mean = statistics.fmean(samples)
    return Timing(
        median_ms=samples[n // 2],
        p10_ms=samples[max(0, int(0.10 * n))],
        p90_ms=samples[min(n - 1, int(0.90 * n))],
        min_ms=samples[0],
        mean_ms=mean,
        cv=(statistics.pstdev(samples) / mean) if mean > 0 else 0.0,
        reps=n,
        warmup=warmup,
        l2_flushed=flush_l2,
    )


@dataclass
class MemoryReport:
    peak_allocated_bytes: int
    #: Peak minus the bytes the caller's own inputs and outputs occupy. This is
    #: the workspace an implementation needs on top of the data it was handed,
    #: and it is the number fusion is supposed to shrink.
    workspace_bytes: int
    resident_bytes: int
    incremental_peak_bytes: int
    returned_bytes: int
    accounted_output_bytes: int
    allocation_count: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def measure_memory(
    fn: Callable[[], object], *, device: torch.device, resident_bytes: int,
    output_bytes: int | None = None,
) -> MemoryReport:
    """Peak allocator high-water mark for one call, and the workspace above the data.

    ``resident_bytes`` is what the caller already holds (inputs plus the output
    it expects back). Reporting the difference separately keeps us honest: the
    sources a Block AttnRes model must keep alive are a property of the
    architecture, not of our kernel, and eliminating a temporary does not make
    them disappear. ``output_bytes`` is needed when ``fn`` consumes its forward
    output internally (for example, a forward+backward step) and therefore
    returns no tensor from which the output size can be inferred.
    """
    fn()  # let any lazy workspace or autotune cache allocate first
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_stats(device).get("allocation.all.allocated", 0)

    out = fn()
    torch.cuda.synchronize(device)

    absolute_peak = torch.cuda.max_memory_allocated(device)
    after = torch.cuda.memory_stats(device).get("allocation.all.allocated", 0)
    incremental_peak = max(0, absolute_peak - baseline)

    def tensor_bytes(obj: object) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(tensor_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(tensor_bytes(x) for x in obj.values())
        return 0

    returned = tensor_bytes(out)
    accounted_output = returned if output_bytes is None else output_bytes
    workspace = max(0, incremental_peak - accounted_output)
    del out
    return MemoryReport(
        # Normalize away unrelated live allocations (notably the L2 flush
        # buffer and correctness oracle) while retaining the original meaning:
        # caller-owned resident data plus the implementation's workspace.
        peak_allocated_bytes=resident_bytes + workspace,
        workspace_bytes=workspace,
        resident_bytes=resident_bytes,
        incremental_peak_bytes=incremental_peak,
        returned_bytes=returned,
        accounted_output_bytes=accounted_output,
        allocation_count=after - before,
    )


@dataclass
class KernelReport:
    total_kernels: int | None
    total_cuda_us: float | None
    by_name: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total_kernels": self.total_kernels,
            "total_cuda_us": self.total_cuda_us,
            "by_name": self.by_name,
        }


def count_kernels(fn: Callable[[], object], *, device: torch.device, iters: int = 5) -> KernelReport:
    """How many CUDA kernels one call launches, and what they are.

    This is the central piece of evidence for whether a fusion opportunity
    exists at all, so it counts real device kernels from the profiler rather
    than inferring anything from the Python source.
    """
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)

    by_name: dict[str, dict] = {}
    total = 0
    total_us = 0.0
    for evt in prof.key_averages():
        # device_type 1 == CUDA; count only real device kernels, not the
        # CPU-side launch stubs or memcpy bookkeeping.
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if evt.key.startswith(("Memcpy", "Memset", "cuda")):
            continue
        n = evt.count // iters
        if n == 0 and evt.count == 0:
            continue
        by_name[evt.key] = {
            "launches_per_call": evt.count / iters,
            "cuda_us_per_call": evt.self_device_time_total / iters,
        }
        total += evt.count
        total_us += evt.self_device_time_total

    # A call that ran cannot have launched zero kernels. When the profiler is
    # invoked many times in one process it sometimes stops returning CUDA
    # events, and reporting the result as "0 kernels" would be a false claim
    # rather than a missing one. Report it as unavailable instead.
    if total == 0:
        return KernelReport(
            total_kernels=None,
            total_cuda_us=None,
            by_name={"_unavailable": "profiler returned no CUDA events for this call"},
        )

    return KernelReport(
        total_kernels=round(total / iters, 2),
        total_cuda_us=total_us / iters,
        by_name=by_name,
    )


def check_against(
    candidate: Callable[..., torch.Tensor],
    oracle: torch.Tensor,
    args: tuple,
    *,
    rel_l2_tol: float,
) -> dict:
    """Compare one implementation to a high-precision oracle.

    The pass criterion is **relative L2 error**, ``||got - oracle|| /
    ||oracle||``, not elementwise ``allclose``. Two reasons:

    * The output of this operator is a convex combination of zero-mean sources,
      so individual elements pass through zero. Elementwise relative error is
      unbounded there and reports five-figure ratios for a result that is
      correct to the last bit bf16 can represent. It measures the test's
      choice of input, not the implementation.
    * Relative L2 is the standard measure for a reduction kernel and degrades
      gracefully: it answers "how much of the signal is wrong", which is the
      question that actually matters when comparing two roundings of the same
      mathematics.

    Elementwise statistics are still reported, normalized by the oracle's RMS
    so they are readable, but they do not gate the result.

    Returns a report rather than raising, so a benchmark run can record that an
    implementation was excluded and why.
    """
    try:
        got = candidate(*args)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}

    if got.shape != oracle.shape:
        return {"ok": False, "error": f"shape {tuple(got.shape)} != {tuple(oracle.shape)}"}

    # Keep the comparison in the oracle's precision. Casting both sides to
    # fp32 would make a "float64 oracle" claim untrue and erase part of the
    # float32 candidate's own rounding floor.
    o = oracle.to(torch.float64)
    g = got.to(torch.float64)
    diff = g - o
    o_norm = o.norm().item()
    o_rms = o.pow(2).mean().sqrt().item()
    rel_l2 = (diff.norm().item() / o_norm) if o_norm > 0 else float("inf")

    # What the same tensor rounded to the candidate's dtype would already cost.
    # If our error is at this level, the implementation is as good as the dtype
    # allows and nothing is left to fix.
    round_trip = o.to(got.dtype).to(torch.float64) - o
    dtype_floor_rel_l2 = (round_trip.norm().item() / o_norm) if o_norm > 0 else 0.0

    return {
        "ok": bool(rel_l2 <= rel_l2_tol) and not bool(torch.isnan(got).any().item()),
        "rel_l2": rel_l2,
        "rel_l2_tol": rel_l2_tol,
        "dtype_floor_rel_l2": dtype_floor_rel_l2,
        "err_vs_dtype_floor": (rel_l2 / dtype_floor_rel_l2) if dtype_floor_rel_l2 > 0 else None,
        "max_abs_err": diff.abs().max().item(),
        "max_abs_err_over_rms": (diff.abs().max().item() / o_rms) if o_rms > 0 else None,
        "output_rms": o_rms,
        "has_nan": bool(torch.isnan(got).any().item()),
        "has_inf": bool(torch.isinf(got).any().item()),
    }


def achieved_bandwidth_gbps(bytes_moved: int, median_ms: float) -> float:
    return bytes_moved / (median_ms * 1e-3) / 1e9


def environment() -> dict:
    """Everything a reader needs to know a number was produced the same way."""
    import platform

    import triton

    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_name": p.name,
        "device_cc": f"{p.major}.{p.minor}",
        "device_index": dev,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
