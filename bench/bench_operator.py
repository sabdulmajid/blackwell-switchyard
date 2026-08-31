"""Benchmark Block AttnRes implementations against each other and against the roofline.

Every implementation is checked against a float64 oracle before it is timed, and
every latency is converted to achieved bandwidth so it can be read against the
machine's measured DRAM ceiling rather than only against other implementations.

Run::

    python bench/bench_operator.py                    # default sweep, bf16
    python bench/bench_operator.py --set n-sweep      # one axis only
    python bench/bench_operator.py --quick            # fewer reps
    python bench/bench_operator.py --impls folded,folded_compiled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from harness import (  # noqa: E402
    achieved_bandwidth_gbps,
    check_against,
    count_kernels,
    environment,
    measure_latency,
    measure_memory,
)
from switchyard.baselines import folded_form, paper_form  # noqa: E402
from switchyard.reference import (  # noqa: E402
    DEFAULT_EPS,
    attn_res_min_bytes,
    block_attn_res_oracle,
)


@dataclass(frozen=True)
class Shape:
    n: int  # source count (stacked depth axis)
    b: int
    t: int
    d: int

    def key(self) -> str:
        return f"N{self.n}_B{self.b}_T{self.t}_D{self.d}"

    def elements(self) -> int:
        return self.n * self.b * self.t * self.d


# Shapes chosen from the paper's own setting rather than to fill a grid.
# AttnRes uses N ~ 8 blocks plus the token embedding and (for all but the first
# layer of a block) the running partial sum, so a real model produces source
# counts up to 9-10. Sequence length 8192 is the paper's training context.
SHAPE_SETS: dict[str, list[Shape]] = {
    # How does cost scale with depth history? The axis the architecture chooses.
    "n-sweep": [Shape(n, 1, 4096, 2048) for n in (2, 4, 8, 9, 16, 32)],
    # How does it scale with model width?
    "d-sweep": [Shape(9, 1, 4096, d) for d in (1024, 2048, 4096, 8192)],
    # How does it scale with context?
    "t-sweep": [Shape(9, 1, t, 2048) for t in (512, 2048, 8192, 16384)],
    # Training-like: a few real batches.
    "batch": [Shape(9, b, 2048, 2048) for b in (1, 2, 4, 8)],
    # Small shapes, where launch overhead rather than bandwidth should dominate.
    "small": [Shape(9, 1, 128, 1024), Shape(9, 1, 512, 1024), Shape(4, 1, 256, 512)],
}
SHAPE_SETS["default"] = (
    SHAPE_SETS["n-sweep"] + SHAPE_SETS["d-sweep"] + SHAPE_SETS["t-sweep"] + SHAPE_SETS["small"]
)


def build_impls(compile_mode_reps: int) -> dict:
    """Name -> (callable, needs_compile_warmup).

    Compiled variants are constructed once and reused across shapes, which means
    each new shape triggers a recompilation. Compile time is measured and
    reported separately; it never enters a steady-state number.
    """
    impls: dict[str, dict] = {
        "paper_eager": {"fn": paper_form, "compiled": False},
        "folded_eager": {"fn": folded_form, "compiled": False},
    }
    impls["paper_compiled"] = {
        "fn": torch.compile(paper_form, dynamic=False),
        "compiled": True,
    }
    impls["folded_compiled"] = {
        "fn": torch.compile(folded_form, dynamic=False),
        "compiled": True,
    }
    impls["folded_compiled_autotune"] = {
        "fn": torch.compile(folded_form, mode="max-autotune-no-cudagraphs", dynamic=False),
        "compiled": True,
    }
    impls["folded_compiled_cudagraph"] = {
        "fn": torch.compile(folded_form, mode="reduce-overhead", dynamic=False),
        "compiled": True,
    }

    # Optional: the project's own optimized path, once it exists.
    try:
        from switchyard.triton_op import block_attn_res_triton

        impls["switchyard_triton"] = {"fn": block_attn_res_triton, "compiled": False}
    except ImportError:
        pass
    return impls


def bench_one(
    name: str,
    spec: dict,
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
    *,
    oracle: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    quick: bool,
    tol: float,
) -> dict:
    fn = spec["fn"]
    rec: dict = {"impl": name, "compiled": spec["compiled"]}

    # 1. Correctness first. A fast wrong kernel is not a result.
    correctness = check_against(fn, oracle, (v, w, DEFAULT_EPS), rel_l2_tol=tol)
    rec["correctness"] = correctness
    if not correctness["ok"]:
        rec["skipped"] = "failed correctness check"
        return rec

    # 2. Compile / warmup, timed separately from steady state.
    if spec["compiled"]:
        t0 = time.perf_counter()
        fn(v, w, DEFAULT_EPS)
        torch.cuda.synchronize(device)
        rec["compile_seconds"] = time.perf_counter() - t0

    warmup, reps = (10, 30) if quick else (25, 100)

    # 3. Forward.
    fwd = measure_latency(
        lambda: fn(v, w, DEFAULT_EPS), device=device, warmup=warmup, reps=reps
    )
    rec["forward"] = fwd.as_dict()

    min_bytes = attn_res_min_bytes(shape.n, shape.b, shape.t, shape.d, v.element_size())
    rec["min_bytes"] = min_bytes
    rec["forward_achieved_gbps"] = achieved_bandwidth_gbps(min_bytes, fwd.median_ms)

    # 4. Kernel count and memory, forward.
    try:
        rec["forward_kernels"] = count_kernels(
            lambda: fn(v, w, DEFAULT_EPS), device=device
        ).as_dict()
    except Exception as exc:  # noqa: BLE001
        rec["forward_kernels"] = {"error": str(exc)[:200]}

    # Separate the time the GPU spent working from the time it spent waiting.
    # A memory-bound operator that is actually dispatch-bound looks identical in
    # a latency column, and the two call for opposite fixes: fewer kernels
    # versus fewer bytes.
    busy = rec.get("forward_kernels", {}).get("total_cuda_us")
    if busy is not None:
        rec["forward_gpu_busy_us"] = busy
        rec["forward_dispatch_gap_us"] = fwd.median_ms * 1000 - busy
        rec["forward_gpu_utilization"] = busy / (fwd.median_ms * 1000)
        # Bandwidth the kernels themselves sustained, ignoring the gaps.
        rec["forward_kernel_gbps"] = achieved_bandwidth_gbps(min_bytes, busy / 1000)

    resident = v.numel() * v.element_size() + w.numel() * w.element_size()
    resident += shape.b * shape.t * shape.d * v.element_size()  # the output
    rec["forward_memory"] = measure_memory(
        lambda: fn(v, w, DEFAULT_EPS), device=device, resident_bytes=resident
    ).as_dict()

    # 5. Forward + backward. This is a training primitive; forward alone would
    #    be an incomplete picture.
    vg = v.detach().clone().requires_grad_(True)
    wg = w.detach().clone().requires_grad_(True)
    grad_out = torch.randn(shape.b, shape.t, shape.d, device=device, dtype=dtype)

    def fwd_bwd():
        if vg.grad is not None:
            vg.grad = None
        if wg.grad is not None:
            wg.grad = None
        out = fn(vg, wg, DEFAULT_EPS)
        out.backward(grad_out)

    try:
        fb = measure_latency(fwd_bwd, device=device, warmup=warmup, reps=max(20, reps // 2))
        rec["fwd_bwd"] = fb.as_dict()
        rec["backward_only_ms"] = fb.median_ms - fwd.median_ms
        try:
            rec["fwd_bwd_kernels"] = count_kernels(fwd_bwd, device=device, iters=3).as_dict()
        except Exception as exc:  # noqa: BLE001
            rec["fwd_bwd_kernels"] = {"error": str(exc)[:200]}
        rec["fwd_bwd_memory"] = measure_memory(
            fwd_bwd, device=device, resident_bytes=resident
        ).as_dict()
        fb_busy = rec.get("fwd_bwd_kernels", {}).get("total_cuda_us")
        if fb_busy is not None:
            rec["fwd_bwd_gpu_busy_us"] = fb_busy
            rec["fwd_bwd_gpu_utilization"] = fb_busy / (fb.median_ms * 1000)
    except Exception as exc:  # noqa: BLE001
        rec["fwd_bwd"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    torch.cuda.empty_cache()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="default", choices=sorted(SHAPE_SETS))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--impls", default=None, help="comma-separated subset")
    ap.add_argument("--query-scale", type=float, default=1.0,
                    help="||w||_2, which is the logit standard deviation. 1.0 is realistic; "
                         "large values saturate the softmax and stress numerics.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    # Relative-L2 tolerances, set from each dtype's own resolution rather than
    # tuned until things pass. The reports also carry `dtype_floor_rel_l2` --
    # the error of merely rounding the exact answer to this dtype -- so it is
    # visible how much of any discrepancy is the implementation's doing and how
    # much is the format's.
    tolerances = {torch.bfloat16: 2e-2, torch.float16: 5e-3, torch.float32: 1e-5}
    tol = tolerances[dtype]

    impls = build_impls(1)
    if args.impls:
        want = set(args.impls.split(","))
        impls = {k: v for k, v in impls.items() if k in want}
        missing = want - set(impls)
        if missing:
            sys.exit(f"unknown implementations: {sorted(missing)}")

    shapes = SHAPE_SETS[args.set]
    report = {
        "environment": environment(),
        "dtype": args.dtype,
        "shape_set": args.set,
        "rel_l2_tol": tol,
        "query_scale": args.query_scale,
        "quick": args.quick,
        "results": [],
    }

    print(f"{'shape':>26} {'impl':>28} {'fwd ms':>9} {'GB/s':>8} {'krnl':>5} {'f+b ms':>9}")
    print("-" * 92)

    for shape in shapes:
        gib = shape.elements() * torch.finfo(dtype).bits / 8 / 2**30
        if gib > 24:
            print(f"skipping {shape.key()}: {gib:.1f} GiB of sources")
            continue

        v = torch.randn(shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype)
        # The pseudo-query is scaled so the logits land in a realistic range.
        # Because RMSNorm gives every key unit RMS, a logit is distributed
        # roughly as N(0, ||w||^2), so ||w|| *is* the logit scale. Drawing w
        # from N(0, 1) would put the logit standard deviation at sqrt(D) -- 64
        # at D=4096 -- which saturates the softmax to one-hot and makes the
        # output violently sensitive to rounding. That regime is an artifact of
        # the test, not of the operator: the paper zero-initializes w, and a
        # trained query produces logits of order one. Normalizing by sqrt(D)
        # gives ||w|| ~ 1 and a softmax that actually mixes.
        w = torch.randn(shape.d, device=device, dtype=torch.float32)
        w = (w / w.norm() * args.query_scale).to(dtype)
        oracle = block_attn_res_oracle(v, w, DEFAULT_EPS)

        # Speed of light: a kernel that touches exactly these bytes with the same
        # access pattern and tile geometry but skips the softmax. No correct
        # implementation can beat it, which makes it a far more honest ceiling
        # than a generic copy benchmark.
        sol_rec = None
        try:
            from switchyard.triton_op import speed_of_light

            sol_t = measure_latency(
                lambda v=v: speed_of_light(v), device=device,
                warmup=10 if args.quick else 25, reps=30 if args.quick else 100,
            )
            mb = attn_res_min_bytes(shape.n, shape.b, shape.t, shape.d, v.element_size())
            sol_rec = {
                "impl": "speed_of_light",
                "note": "same bytes and access pattern, no softmax; a ceiling, not an implementation",
                "shape": asdict(shape),
                "forward": sol_t.as_dict(),
                "min_bytes": mb,
                "forward_achieved_gbps": achieved_bandwidth_gbps(mb, sol_t.median_ms),
            }
            report["results"].append(sol_rec)
            print(
                f"{shape.key():>26} {'speed_of_light':>28} "
                f"{sol_t.median_ms:9.4f} {sol_rec['forward_achieved_gbps']:8.0f}"
                f"     1     (ceiling)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{shape.key():>26} speed_of_light failed: {str(exc)[:60]}")

        for name, spec in impls.items():
            rec = bench_one(
                name, spec, shape, dtype, device,
                oracle=oracle, v=v, w=w, quick=args.quick, tol=tol,
            )
            rec["shape"] = asdict(shape)
            if sol_rec and not rec.get("skipped"):
                rec["fraction_of_speed_of_light"] = (
                    sol_rec["forward"]["median_ms"] / rec["forward"]["median_ms"]
                )
            report["results"].append(rec)

            if rec.get("skipped"):
                print(f"{shape.key():>26} {name:>28}   SKIPPED: {rec['skipped']}")
                if "error" in rec["correctness"]:
                    print(f"{'':>26} {'':>28}   {rec['correctness']['error'][:70]}")
            else:
                fb = rec.get("fwd_bwd", {})
                fb_ms = fb.get("median_ms")
                k = rec.get("forward_kernels", {}).get("total_kernels", "?")
                print(
                    f"{shape.key():>26} {name:>28} "
                    f"{rec['forward']['median_ms']:9.4f} "
                    f"{rec['forward_achieved_gbps']:8.0f} "
                    f"{str(k):>5} "
                    f"{fb_ms:9.4f}" if fb_ms else
                    f"{shape.key():>26} {name:>28} "
                    f"{rec['forward']['median_ms']:9.4f} "
                    f"{rec['forward_achieved_gbps']:8.0f} {str(k):>5}       n/a"
                )

        torch.cuda.empty_cache()

    out = args.out or REPO / "results" / f"operator_{args.set}_{args.dtype}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
