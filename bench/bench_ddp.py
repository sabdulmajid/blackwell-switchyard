"""Two-GPU validation: does the fused operator hold up under DDP, and does it scale?

Block AttnRes is a *local* tensor operation. Nothing about it needs cross-device
communication, and inventing some so the README could say "multi-GPU" would be
contrived. So this asks the two questions that are actually worth asking:

1. **Does DDP reduce the local gradients correctly?** The benchmark computes
   local gradients with synchronization disabled. It averages those gradients
   explicitly and compares the result with a normal synchronized backward pass.
   The operator's float64-oracle tests validate the local gradient mathematics.
2. **How does it scale on this machine, and what limits it?** These two cards
   have peer access but only over PCIe at a measured 25.8 GB/s -- there is no
   NVLink. A 1.3B model gradient all-reduce moves about 2.6 GB per step in bf16,
   so the interesting number is not the speedup but where it stops.

Run::

    python bench/bench_ddp.py                 # both variants, 1 GPU then 2
    python bench/bench_ddp.py --scale small   # faster
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))


def _gradient_check(
    maximum_deviation: float,
    reference_maximum: float,
    *,
    rank_count: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict:
    tolerance = absolute_tolerance + relative_tolerance * reference_maximum
    passed = (
        math.isfinite(maximum_deviation)
        and math.isfinite(reference_maximum)
        and maximum_deviation <= tolerance
    )
    return {
        "max_abs_deviation_from_manual_average": maximum_deviation,
        "reference_max_abs": reference_maximum,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "effective_tolerance": tolerance,
        "validated_rank_count": rank_count,
        "passed": passed,
    }


def _worker(rank: int, world: int, args, out_path: str) -> None:
    from harness import environment, measure_latency
    from switchyard.model import ModelConfig, Transformer

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", args.port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)

    scales = {
        "1.3B": dict(vocab_size=32768, d_model=2048, n_layers=24, n_heads=16,
                     d_ff=5632, n_blocks=8, max_seq_len=4096),
        "small": dict(vocab_size=8192, d_model=1024, n_layers=12, n_heads=8,
                      d_ff=2816, n_blocks=6, max_seq_len=2048),
    }
    dtype = torch.bfloat16
    results = []

    for residual in ("standard", "switchyard"):
        torch.manual_seed(0)
        cfg = ModelConfig(**scales[args.scale], residual=residual, sources="arena")
        model = Transformer(cfg).to(device=device, dtype=dtype)
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[rank])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

        # Each rank gets a different shard, as it would in real data-parallel
        # training. The seed is rank-dependent so the all-reduce has something
        # to actually average rather than identical gradients.
        torch.manual_seed(1234 + rank)
        idx = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
        tgt = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

        # Bind the loop variables explicitly: the closure is only used within
        # this iteration, but late binding here would be a real bug the moment
        # anyone deferred it.
        def step(model=model, opt=opt, idx=idx, tgt=tgt):
            opt.zero_grad(set_to_none=True)
            model(idx, tgt)[1].backward()
            opt.step()

        warmup, reps = (3, 6) if args.quick else (5, 15)
        t = measure_latency(step, device=device, warmup=warmup, reps=reps, flush_l2=False)

        # Integration correctness under DDP: construct the expected average from
        # unsynchronized local gradients, then compare it with DDP's reduction.
        grad_check = None
        if world > 1:
            opt.zero_grad(set_to_none=True)
            with model.no_sync():
                model(idx, tgt)[1].backward()
            expected = torch.cat([
                p.grad.flatten() for _, p in sorted(model.module.named_parameters())
                if p.grad is not None
            ])
            local_grad_norm = expected.norm().item()
            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            expected.div_(world)

            opt.zero_grad(set_to_none=True)
            model(idx, tgt)[1].backward()
            reduced = torch.cat([
                p.grad.flatten() for _, p in sorted(model.module.named_parameters())
                if p.grad is not None
            ])
            maximum_deviation = (reduced - expected).abs().max()
            dist.all_reduce(maximum_deviation, op=dist.ReduceOp.MAX)
            reference_maximum = expected.abs().max().item()
            grad_check = {
                **_gradient_check(
                    maximum_deviation.item(),
                    reference_maximum,
                    rank_count=world,
                    absolute_tolerance=args.grad_atol,
                    relative_tolerance=args.grad_rtol,
                ),
                "local_grad_norm": local_grad_norm,
                "reduced_grad_norm": reduced.norm().item(),
            }
            opt.zero_grad(set_to_none=True)
            if not grad_check["passed"]:
                raise RuntimeError(
                    "DDP gradient reduction differs from the explicit average: "
                    f"{grad_check['max_abs_deviation_from_manual_average']:.3e} > "
                    f"{grad_check['effective_tolerance']:.3e}"
                )

        tokens = args.batch * args.seq * world
        results.append({
            "residual": residual,
            "world_size": world,
            "rank": rank,
            "step": t.as_dict(),
            "tokens_per_step_global": tokens,
            "tokens_per_second_global": tokens / (t.median_ms * 1e-3),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            "params": sum(p.numel() for p in model.parameters()),
            "grad_check": grad_check,
        })
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        Path(out_path).write_text(json.dumps(
            {"environment": environment(), "scale": args.scale, "batch": args.batch,
             "seq": args.seq, "results": results}, indent=2, default=str))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1.3B", choices=["1.3B", "small"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--port", default="29517")
    ap.add_argument("--grad-atol", type=float, default=0.0)
    ap.add_argument("--grad-rtol", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "ddp.json")
    args = ap.parse_args()

    if args.grad_atol < 0 or args.grad_rtol < 0:
        ap.error("gradient tolerances must be nonnegative")

    avail = torch.cuda.device_count()
    if avail < 2:
        sys.exit(f"needs two GPUs, found {avail}")

    scratch = Path(str(args.out) + ".tmp")
    from harness import repository_provenance

    combined = {
        "provenance": repository_provenance(REPO),
        "single": None,
        "dual": None,
    }

    for world in (1, 2):
        print(f"\n=== world_size={world} ===", flush=True)
        if world == 1:
            _worker(0, 1, args, str(scratch))
        else:
            mp.spawn(_worker, args=(world, args, str(scratch)), nprocs=world, join=True)
        combined["single" if world == 1 else "dual"] = json.loads(scratch.read_text())
        for r in combined["single" if world == 1 else "dual"]["results"]:
            gc = r.get("grad_check")
            print(f"  {r['residual']:11} step {r['step']['median_ms']:8.2f} ms  "
                  f"{r['tokens_per_second_global']:8.0f} tok/s  "
                  f"peak {r['peak_memory_bytes'] / 2**30:5.2f} GiB"
                  + (
                      f"  reduce dev {gc['max_abs_deviation_from_manual_average']:.2e}"
                      if gc
                      else ""
                  ))

    scratch.unlink(missing_ok=True)

    print("\nscaling")
    for residual in ("standard", "switchyard"):
        one = next(r for r in combined["single"]["results"] if r["residual"] == residual)
        two = next(r for r in combined["dual"]["results"] if r["residual"] == residual)
        eff = two["tokens_per_second_global"] / (2 * one["tokens_per_second_global"])
        combined.setdefault("scaling", {})[residual] = {
            "one_gpu_tokens_per_s": one["tokens_per_second_global"],
            "two_gpu_tokens_per_s": two["tokens_per_second_global"],
            "speedup": two["tokens_per_second_global"] / one["tokens_per_second_global"],
            "efficiency": eff,
            "step_ms_one": one["step"]["median_ms"],
            "step_ms_two": two["step"]["median_ms"],
        }
        print(f"  {residual:11} {one['tokens_per_second_global']:.0f} -> "
              f"{two['tokens_per_second_global']:.0f} tok/s  "
              f"({two['tokens_per_second_global'] / one['tokens_per_second_global']:.2f}x, "
              f"{100 * eff:.0f}% efficiency)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
