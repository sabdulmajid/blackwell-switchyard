"""Two-GPU validation: does the fused operator hold up under DDP, and does it scale?

Block AttnRes is a *local* tensor operation. Nothing about it needs cross-device
communication, and inventing some so the README could say "multi-GPU" would be
contrived. So this asks the two questions that are actually worth asking:

1. **Does it still train correctly under DDP?** Gradients are all-reduced across
   ranks, and a custom autograd Function that returns the wrong gradient for one
   of its inputs can pass single-GPU tests and still corrupt a distributed run.
   Rank gradients are compared explicitly after synchronization.
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

        # Correctness under DDP: after a synchronized backward every rank must
        # hold identical gradients. If our custom autograd returned a wrong or
        # rank-dependent gradient, this is where it shows.
        grad_check = None
        if world > 1:
            opt.zero_grad(set_to_none=True)
            model(idx, tgt)[1].backward()
            flat = torch.cat([
                p.grad.flatten() for _, p in sorted(model.module.named_parameters())
                if p.grad is not None
            ])
            other = flat.clone()
            dist.broadcast(other, src=0)
            grad_check = {
                "max_abs_deviation_from_rank0": (flat - other).abs().max().item(),
                "grad_norm": flat.norm().item(),
            }
            opt.zero_grad(set_to_none=True)

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
    ap.add_argument("--out", type=Path, default=REPO / "results" / "ddp.json")
    args = ap.parse_args()

    avail = torch.cuda.device_count()
    if avail < 2:
        sys.exit(f"needs two GPUs, found {avail}")

    scratch = Path(str(args.out) + ".tmp")
    combined = {"single": None, "dual": None}

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
                  + (f"  grad dev {gc['max_abs_deviation_from_rank0']:.2e}" if gc else ""))

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
