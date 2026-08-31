"""Characterize the machine, and write the numbers every later claim rests on.

Produces ``results/machine.json``. The important output is the *achievable*
memory bandwidth, because Block AttnRes is memory-bound by a wide margin and
that number is the denominator of every efficiency figure in this repository.
Peak numbers from a spec sheet are not used anywhere; only what this script
measures.

Run::

    python bench/machine.py                 # both GPUs, write results/machine.json
    python bench/machine.py --device 0      # one GPU
    python bench/machine.py --quick         # fewer reps, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPO = Path(__file__).resolve().parent.parent
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")


# --------------------------------------------------------------------------
# Bandwidth probes. Written in Triton so the byte count is exactly known
# rather than inferred from whatever a library op decided to do.
# --------------------------------------------------------------------------


@triton.jit
def _copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, mask=m), mask=m)


@triton.jit
def _read_kernel(src, out, n, BLOCK: tl.constexpr):
    """Read-only: reduce into one value per program so nothing is written back."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + pid, tl.sum(tl.load(src + i, mask=m, other=0.0)))


@triton.jit
def _write_kernel(dst, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, 1.0, mask=m)


@triton.jit
def _empty_kernel(x):
    pass


def time_cuda(fn, warmup: int, reps: int) -> dict:
    """Median/p10/p90 milliseconds over `reps` CUDA-event-timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "min_ms": times[0],
        "p10_ms": times[max(0, int(0.10 * n))],
        "p90_ms": times[min(n - 1, int(0.90 * n))],
        "reps": n,
    }


def bandwidth_probe(nbytes: int, device: int, quick: bool, target_bytes: int = 8 * 2**30) -> dict:
    """Copy / read / write bandwidth at a given working-set size.

    Each measurement repeats the kernel enough times to move roughly
    ``target_bytes`` in total, regardless of the working-set size. Without that,
    small working sets are dominated by the ~10 us launch overhead and report a
    *lower* bandwidth than large ones, which inverts the cache hierarchy and is
    exactly the kind of measurement artifact that would poison every downstream
    efficiency claim.
    """
    warmup, reps = (3, 10) if quick else (5, 25)
    n = nbytes // 4  # fp32 elements
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    src = torch.randn(n, device=f"cuda:{device}", dtype=torch.float32)
    dst = torch.empty_like(src)
    partial = torch.empty(grid[0], device=f"cuda:{device}", dtype=torch.float32)

    out = {
        "working_set_bytes": nbytes,
        "working_set_mib": nbytes / 2**20,
    }

    def repeated(launch, per_call_bytes: int) -> dict:
        k = max(1, int(target_bytes // per_call_bytes))

        def run():
            for _ in range(k):
                launch()

        t = time_cuda(run, warmup, reps)
        return {
            **t,
            "launches_per_timing": k,
            "gbytes_per_s": per_call_bytes * k / (t["median_ms"] * 1e-3) / 1e9,
        }

    # A copy touches every byte twice: once read, once written.
    out["copy"] = repeated(lambda: _copy_kernel[grid](src, dst, n, BLOCK=BLOCK), 2 * nbytes)
    out["read"] = repeated(lambda: _read_kernel[grid](src, partial, n, BLOCK=BLOCK), nbytes)
    out["write"] = repeated(lambda: _write_kernel[grid](dst, n, BLOCK=BLOCK), nbytes)

    torch.cuda.empty_cache()
    return out


def bandwidth_sweep(device: int, quick: bool) -> dict:
    """Copy bandwidth across working-set sizes, to locate the cache hierarchy.

    The shape of this curve is the reason a two-pass formulation of Block
    AttnRes is worth considering on this part: if a token tile's sources stay
    resident in L2 between the scoring pass and the weighted-sum pass, the
    second pass costs L2 bandwidth rather than DRAM bandwidth.
    """
    sizes_mib = [4, 16, 32, 48, 64, 96, 128, 256, 1024, 2048]
    if quick:
        sizes_mib = [4, 32, 48, 96, 512, 2048]
    curve = []
    for mib in sizes_mib:
        r = bandwidth_probe(mib * 2**20, device, quick)
        curve.append({"mib": mib, "copy_gbytes_per_s": r["copy"]["gbytes_per_s"]})
        print(f"    {mib:5d} MiB  {r['copy']['gbytes_per_s']:7.0f} GB/s", flush=True)

    dram = [c for c in curve if c["mib"] >= 512]
    cache = [c for c in curve if 16 <= c["mib"] <= 64]
    return {
        "curve": curve,
        "dram_plateau_gbytes_per_s": (
            sum(c["copy_gbytes_per_s"] for c in dram) / len(dram) if dram else None
        ),
        "l2_peak_gbytes_per_s": (
            max(c["copy_gbytes_per_s"] for c in cache) if cache else None
        ),
    }


def matmul_probe(device: int, quick: bool) -> dict:
    """Achieved dense matmul throughput. Warmed up properly -- cuBLAS needs it."""
    res = {}
    sizes = [2048, 4096] if quick else [2048, 4096, 8192]
    for dtype, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
        for n in sizes:
            try:
                a = torch.randn(n, n, device=f"cuda:{device}", dtype=dtype)
                b = torch.randn(n, n, device=f"cuda:{device}", dtype=dtype)
                t = time_cuda(lambda a=a, b=b: torch.mm(a, b),
                              15 if quick else 30, 20 if quick else 50)
                res[f"{name}_{n}"] = {
                    **t,
                    "tflop_per_s": 2 * n**3 / (t["median_ms"] * 1e-3) / 1e12,
                }
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                res[f"{name}_{n}"] = {"error": str(exc)[:200]}
    return res


def launch_overhead_probe(device: int, quick: bool) -> dict:
    """Per-launch cost of a kernel that does nothing.

    Block AttnRes decomposes into several small kernels in a framework
    implementation, so this is the floor on what fusion can save.
    """
    warmup, reps = (5, 20) if quick else (20, 100)
    x = torch.zeros(1, device=f"cuda:{device}")
    chain = 200

    def many_triton():
        for _ in range(chain):
            _empty_kernel[(1,)](x)

    t = time_cuda(many_triton, warmup, reps)
    out = {"triton_empty_us": t["median_ms"] * 1000 / chain, "chain_length": chain}

    def many_torch():
        for _ in range(chain):
            x.add_(0.0)

    t = time_cuda(many_torch, warmup, reps)
    out["torch_tiny_op_us"] = t["median_ms"] * 1000 / chain

    # Same work replayed from a CUDA graph: isolates launch cost from kernel cost.
    # Capture is verified rather than assumed -- an empty graph replays in
    # nanoseconds and would otherwise be reported as a spectacular result.
    try:
        counter = torch.zeros(1, device=f"cuda:{device}")
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                for _ in range(chain):
                    counter.add_(1.0)
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)

        counter.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(chain):
                counter.add_(1.0)
        counter.zero_()
        g.replay()
        torch.cuda.synchronize(device)
        captured = int(counter.item())
        if captured != chain:
            out["cudagraph_error"] = f"graph captured {captured} of {chain} ops; discarded"
        else:
            t = time_cuda(g.replay, warmup, reps)
            out["torch_tiny_op_cudagraph_us"] = t["median_ms"] * 1000 / chain
            out["cudagraph_ops_verified"] = captured
    except Exception as exc:  # noqa: BLE001
        out["cudagraph_error"] = str(exc)[:200]
    return out


def device_info(i: int) -> dict:
    p = torch.cuda.get_device_properties(i)
    return {
        "index": i,
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "total_memory_bytes": p.total_memory,
        "multi_processor_count": p.multi_processor_count,
        "l2_cache_size_bytes": getattr(p, "L2_cache_size", None),
        "shared_memory_per_block_optin_bytes": getattr(p, "shared_memory_per_block_optin", None),
        "max_threads_per_multi_processor": getattr(p, "max_threads_per_multi_processor", None),
        "regs_per_multiprocessor": getattr(p, "regs_per_multiprocessor", None),
        "warp_size": getattr(p, "warp_size", None),
    }


def _smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return [line.strip() for line in out.splitlines()]
    except Exception:  # noqa: BLE001
        return []


def theoretical_dram_bandwidth() -> list[dict]:
    """bus width x memory clock x 2 (DDR). Reported for context only.

    Nothing in this repository divides by this number -- efficiency is always
    measured against the achieved copy bandwidth above.
    """
    clocks = _smi("clocks.max.mem")
    names = _smi("name")
    out = []
    for i, c in enumerate(clocks):
        try:
            mhz = float(c.split()[0])
        except (ValueError, IndexError):
            continue
        # GB202 is a 512-bit part. Recorded explicitly because nvidia-smi does
        # not expose bus width and torch does not either.
        bus_bits = 512
        out.append({
            "index": i,
            "name": names[i] if i < len(names) else None,
            "memory_clock_mhz": mhz,
            "assumed_bus_width_bits": bus_bits,
            "note": "bus width is an assumption, not a measurement; GB202 is a 512-bit part",
            "theoretical_gbytes_per_s": mhz * 1e6 * 2 * (bus_bits / 8) / 1e9,
        })
    return out


def p2p_probe(quick: bool) -> dict:
    if torch.cuda.device_count() < 2:
        return {"skipped": "fewer than two devices"}
    out = {
        "can_access_peer_0_1": torch.cuda.can_device_access_peer(0, 1),
        "can_access_peer_1_0": torch.cuda.can_device_access_peer(1, 0),
    }
    nbytes = 256 * 2**20
    a = torch.empty(nbytes // 4, device="cuda:0", dtype=torch.float32)
    b = torch.empty(nbytes // 4, device="cuda:1", dtype=torch.float32)
    t = time_cuda(lambda: b.copy_(a), 5 if quick else 10, 10 if quick else 30)
    out["d2d_0to1"] = {**t, "gbytes_per_s": nbytes / (t["median_ms"] * 1e-3) / 1e9}

    h = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True)
    t = time_cuda(lambda: a.copy_(h, non_blocking=True), 5 if quick else 10, 10 if quick else 30)
    out["h2d_pinned"] = {**t, "gbytes_per_s": nbytes / (t["median_ms"] * 1e-3) / 1e9}

    torch.cuda.empty_cache()
    try:
        out["topo"] = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001
        pass
    return out


def toolchain_probe() -> dict:
    """What we can actually build and profile with, as a non-root user."""
    out: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "cuda_home": CUDA_HOME,
    }
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if nvcc.exists():
        r = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=60)
        out["nvcc_version"] = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else None
        r = subprocess.run([str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, timeout=60)
        out["nvcc_gpu_archs"] = r.stdout.split() if r.returncode == 0 else None
        out["nvcc_supports_sm120"] = "compute_120" in (r.stdout or "")

    for tool in ("ncu", "nsys"):
        p = Path(CUDA_HOME) / "bin" / tool
        out[f"{tool}_path"] = str(p) if p.exists() else shutil.which(tool)
    return out


def ncu_permission_probe() -> dict:
    """Can we collect hardware counters without root?

    On many systems ``NVreg_RestrictProfilingToAdminUsers=1`` blocks this. We
    only report the result -- changing it is a host-wide modification and this
    project does not make those.
    """
    ncu = Path(CUDA_HOME) / "bin" / "ncu"
    if not ncu.exists():
        return {"available": False}
    script = "import torch; torch.randn(256, device='cuda').mul_(2.0); torch.cuda.synchronize()"
    try:
        r = subprocess.run(
            [str(ncu), "--set", "basic", "--target-processes", "all", sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        blob = (r.stdout or "") + (r.stderr or "")
        denied = "ERR_NVGPUCTRPERM" in blob or "insufficient permissions" in blob.lower()
        return {
            "available": True,
            "returncode": r.returncode,
            "counters_permitted": (not denied) and r.returncode == 0,
            "permission_denied": denied,
            "excerpt": blob[-800:],
        }
    except subprocess.TimeoutExpired:
        return {"available": True, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "error": str(exc)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None, help="probe only this device")
    ap.add_argument("--quick", action="store_true", help="fewer reps")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "machine.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")

    devices = [args.device] if args.device is not None else list(range(torch.cuda.device_count()))

    report: dict = {
        "toolchain": toolchain_probe(),
        "theoretical_dram_bandwidth": theoretical_dram_bandwidth(),
        "devices": {},
        "cpu": {"count": os.cpu_count(), "machine": platform.machine()},
    }

    for i in devices:
        torch.cuda.set_device(i)
        print(f"[device {i}] {torch.cuda.get_device_name(i)}", flush=True)
        d: dict = {"properties": device_info(i)}

        print("  bandwidth (DRAM-resident, 2 GiB)...", flush=True)
        d["bandwidth_dram"] = bandwidth_probe(2 * 2**30, i, args.quick)
        print(f"    copy {d['bandwidth_dram']['copy']['gbytes_per_s']:.0f} GB/s", flush=True)

        print("  bandwidth sweep (locating L2)...", flush=True)
        d["bandwidth_sweep"] = bandwidth_sweep(i, args.quick)

        print("  matmul...", flush=True)
        d["matmul"] = matmul_probe(i, args.quick)

        print("  launch overhead...", flush=True)
        d["launch_overhead"] = launch_overhead_probe(i, args.quick)

        report["devices"][str(i)] = d

    torch.cuda.set_device(devices[0])
    print("p2p / host transfers...", flush=True)
    report["p2p"] = p2p_probe(args.quick)
    print("ncu permission probe...", flush=True)
    report["ncu"] = ncu_permission_probe()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")

    for i, d in report["devices"].items():
        bw = d["bandwidth_dram"]
        sweep = d["bandwidth_sweep"]
        print(
            f"\ndevice {i}: copy {bw['copy']['gbytes_per_s']:.0f} GB/s  "
            f"read {bw['read']['gbytes_per_s']:.0f} GB/s  "
            f"write {bw['write']['gbytes_per_s']:.0f} GB/s"
        )
        print(
            f"  DRAM plateau {sweep['dram_plateau_gbytes_per_s']:.0f} GB/s  "
            f"| L2 peak {sweep['l2_peak_gbytes_per_s']:.0f} GB/s  "
            f"({sweep['l2_peak_gbytes_per_s'] / sweep['dram_plateau_gbytes_per_s']:.2f}x)"
        )
        if "bf16_4096" in d["matmul"] and "tflop_per_s" in d["matmul"]["bf16_4096"]:
            print(f"  bf16 4096 matmul: {d['matmul']['bf16_4096']['tflop_per_s']:.1f} TFLOP/s")
        print(f"  launch overhead: {d['launch_overhead']['triton_empty_us']:.2f} us/kernel")


if __name__ == "__main__":
    main()
