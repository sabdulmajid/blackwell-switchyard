"""End-to-end cost of Block AttnRes inside a real decoder-only Transformer.

The operator sweep in ``bench_operator.py`` answers "how fast is the kernel".
This answers the question that decides whether that matters: **what fraction of
a training step does the residual mechanism actually consume, and what does
replacing the framework implementation with the fused kernel buy at model
scale.**

Five variants are measured, differing only in the residual mechanism:

======================  =========================================================
``standard``            PreNorm ``x = x + sublayer(norm(x))``. The control.
``attnres / stack``     Block AttnRes, framework operator, paper-pseudocode
                        ``torch.stack`` at every site.
``attnres / arena``     Block AttnRes, framework operator, preallocated buffer.
``switchyard / stack``  Block AttnRes, fused Triton kernel, naive stacking.
``switchyard / arena``  Block AttnRes, fused Triton kernel, preallocated buffer.
======================  =========================================================

Model scale, and why
--------------------
24 layers, ``d_model=2048``, SwiGLU ``d_ff=5632``, 16 heads, 32k vocab with a
tied head: **1.30 B parameters**, at ``B=4, T=2048`` (8192 tokens per step).
The memory arithmetic, in bf16 with bf16 AdamW state:

* parameters 2.6 GB + gradients 2.6 GB + AdamW ``exp_avg``/``exp_avg_sq``
  5.2 GB = 10.4 GB resident;
* a ``B*T*D`` slab is 33.5 MiB, and the arena holds 9 of them (0.3 GiB) while
  the naive stacking variant materializes 265 of them and keeps them alive for
  backward (8.7 GiB);
* everything else is ordinary activation memory.

Measured peak is around 32 GiB for the arena variants and 39 GiB for the naive
ones, against 95 GiB of device memory -- comfortable, with room for the stacking
variant to be measured rather than estimated. Going larger would have forced
gradient checkpointing, which would have changed what is being measured.

Attributing the residual mechanism's share
------------------------------------------
Two independent methods, because neither alone is airtight:

1. **Control differencing** (primary). ``step(variant) - step(standard)``, on
   end-to-end wall clock. Unambiguous and complete: it captures the operator,
   the source staging, the extra autograd nodes, and any second-order effect of
   the extra memory traffic. Its one bias is that the control is not free -- a
   standard residual still runs one elementwise add per sublayer, about 144
   slabs of traffic per forward, roughly 3 ms here -- so it slightly
   *understates* the mechanism's absolute cost.

2. **Profiler attribution** (cross-check). ``record_function`` regions around
   the staging and the operator give the forward; the custom autograd node
   ``_ArenaAttnResBackward`` (or ``_BlockAttnResTritonBackward``) is a
   distinctive name that gives the backward. Reported as a fraction of summed
   device kernel time. ``record_function`` scopes do not reach the autograd
   engine, which is exactly why the backward has to be picked up by node name --
   and why the framework operator's backward under ``sources="stack"`` is not
   attributable at all: it decomposes into generic einsum and softmax nodes.

Where both methods apply they agree to within a couple of percent, which is the
reason to trust either.

Run::

    python bench/bench_model.py                  # full run, bf16, writes results/
    python bench/bench_model.py --quick          # fewer reps
    python bench/bench_model.py --scale small    # tiny model, for a fast check
    python bench/bench_model.py --skip-smoke     # no training smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

from harness import environment, measure_latency  # noqa: E402
from switchyard.model import (  # noqa: E402
    ModelConfig,
    Transformer,
    slab_copies,
    source_count_schedule,
)

#: (residual mode, source assembly). ``standard`` ignores the second field.
VARIANTS = [
    ("standard", "arena"),
    ("attnres", "stack"),
    ("attnres", "arena"),
    ("switchyard", "stack"),
    ("switchyard", "arena"),
]

SCALES = {
    # ~1.30 B parameters. See the module docstring for the memory arithmetic.
    "1.3B": dict(
        vocab_size=32768, d_model=2048, n_layers=24, n_heads=16,
        d_ff=5632, n_blocks=8, max_seq_len=4096,
    ),
    # Small enough to iterate on in seconds; used for the training smoke test.
    "small": dict(
        vocab_size=4096, d_model=512, n_layers=8, n_heads=8,
        d_ff=1408, n_blocks=4, max_seq_len=512,
    ),
}

#: ``record_function`` regions the model emits when ``profile_regions`` is set.
_REGIONS = ("attnres_stage", "attnres_op")

#: Autograd node names that isolate the mechanism's backward. Which one exists
#: depends on the variant; ``attnres/stack`` produces none of them, because the
#: framework operator's backward is a chain of generic nodes.
_BWD_NODES = ("_ArenaAttnResBackward", "_BlockAttnResTritonBackward")


def label(residual: str, sources: str) -> str:
    return residual if residual == "standard" else f"{residual}/{sources}"


def build(scale: str, residual: str, sources: str, dtype: torch.dtype, device, seed: int = 0):
    """All variants get the same seed, and only Linear/Embedding draw from it,
    so every weight the three modes share is bit-identical."""
    torch.manual_seed(seed)
    cfg = ModelConfig(**SCALES[scale], residual=residual, sources=sources)
    return cfg, Transformer(cfg).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def bench_variant(scale, residual, sources, *, dtype, device, batch, seq, quick) -> dict:
    cfg, model = build(scale, residual, sources, dtype, device)
    # lr is small but nonzero: the optimizer must do all of its real work, and
    # 25 warmup plus 20 timed steps of drift changes nothing measurable.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    idx = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    def forward():
        # Grad-enabled: this is the forward half of a training step, which is
        # what the backward and step numbers are differenced against.
        return model(idx, tgt)[1]

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        forward().backward()

    def full_step():
        model.zero_grad(set_to_none=True)
        forward().backward()
        opt.step()

    warmup, reps = (3, 8) if quick else (5, 20)
    rec: dict = {
        "variant": label(residual, sources),
        "residual": residual,
        "sources": sources if residual != "standard" else None,
        "params": model.param_counts(),
        "batch": batch,
        "seq": seq,
        "tokens_per_step": batch * seq,
    }

    fwd = measure_latency(forward, device=device, warmup=warmup, reps=reps)
    fb = measure_latency(fwd_bwd, device=device, warmup=warmup, reps=reps)
    st = measure_latency(full_step, device=device, warmup=warmup, reps=reps)
    rec["forward"] = fwd.as_dict()
    rec["fwd_bwd"] = fb.as_dict()
    rec["step"] = st.as_dict()
    rec["backward_only_ms"] = fb.median_ms - fwd.median_ms
    rec["optimizer_only_ms"] = st.median_ms - fb.median_ms
    rec["tokens_per_second"] = batch * seq / (st.median_ms * 1e-3)

    # Peak memory of one step, with the optimizer state already materialized.
    full_step()
    torch.cuda.synchronize(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    full_step()
    torch.cuda.synchronize(device)
    rec["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    rec["resident_between_steps_bytes"] = resident

    # Source plumbing, analytic and as actually executed.
    if residual != "standard":
        counts = model.last_source_counts
        slab_bytes = batch * seq * cfg.d_model * torch.finfo(dtype).bits // 8
        rec["source_counts"] = counts
        rec["source_counts_match_eq6"] = (
            counts == source_count_schedule(cfg.n_sublayers, cfg.n_blocks)
        )
        rec["source_slabs_read_by_operator"] = sum(counts)
        rec["staging_slab_copies"] = slab_copies(cfg.n_sublayers, cfg.n_blocks, sources)
        rec["staging_bytes_per_forward"] = rec["staging_slab_copies"] * slab_bytes
        rec["slab_bytes"] = slab_bytes

    rec["profile"] = profile_step(model, full_step, device, iters=2 if quick else 3)

    torch.cuda.empty_cache()
    return rec


def profile_step(model, step_fn, device, iters: int = 3) -> dict:
    """Device time of the residual mechanism inside one training step.

    Returns microseconds per step: total device kernel time, plus whatever of
    the mechanism could be isolated by region or autograd-node name.
    """
    from torch.profiler import ProfilerActivity, profile

    model.profile_regions = True
    try:
        for _ in range(3):
            step_fn()
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                step_fn()
            torch.cuda.synchronize(device)
    finally:
        model.profile_regions = False

    total_us = 0.0
    named: dict[str, float] = {}
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            # The record_function regions also surface as device-side
            # annotation events whose span covers the kernels inside them;
            # counting those as well would double-count.
            if e.key in _REGIONS or e.key.startswith(("Memcpy", "Memset", "cuda")):
                continue
            total_us += e.self_device_time_total
        elif e.key in _REGIONS or e.key in _BWD_NODES:
            named[e.key] = named.get(e.key, 0.0) + e.device_time_total

    out = {k: v / iters for k, v in named.items()}
    out["total_device_us"] = total_us / iters
    fwd_us = sum(out.get(r, 0.0) for r in _REGIONS)
    bwd_us = sum(out.get(n, 0.0) for n in _BWD_NODES)
    out["attnres_forward_us"] = fwd_us
    # None rather than 0.0 when the backward decomposes into generic nodes, so
    # an unattributable variant is never mistaken for a free one.
    out["attnres_backward_us"] = bwd_us if bwd_us > 0 else None
    out["attnres_share_of_device_time"] = (
        (fwd_us + bwd_us) / out["total_device_us"] if bwd_us > 0 else None
    )
    return out


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------


def smoke_test(residual: str, device, *, steps: int, seed: int = 0) -> dict:
    """A few hundred steps of overfitting one fixed synthetic batch.

    Random tokens are the right data for a systems measurement but the wrong
    data for "does the loss go down": the Bayes-optimal loss on i.i.d. uniform
    tokens is ``ln(V)``, which is where an untrained model already sits. Fixing
    the batch and letting the model memorize it turns that into a real signal
    while keeping the data synthetic.

    Run in fp32 on the small config: this is a correctness check, not a
    throughput measurement, and bf16 optimizer state would confound "the
    architecture does not train" with "the numerics were too coarse".
    """
    cfg, model = build("small", residual, "arena", torch.float32, device, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator(device="cpu").manual_seed(1234)
    idx = torch.randint(0, cfg.vocab_size, (8, 256), generator=g).to(device)
    tgt = torch.randint(0, cfg.vocab_size, (8, 256), generator=g).to(device)

    rec: dict = {"residual": residual, "steps": steps, "dtype": "float32",
                 "params": model.param_counts()["total"]}

    if cfg.uses_attnres:
        with torch.no_grad():
            model(idx, collect_alphas=True)
        alphas0 = [a.tolist() for a in model.last_alphas]
        # Zero-initialized pseudo-queries must give exactly uniform attention.
        worst = max(
            abs(w - 1.0 / len(a)) for a in alphas0 for w in a
        )
        rec["init_uniform_max_dev"] = worst
        rec["init_is_uniform"] = worst < 1e-5
        rec["alphas_initial"] = alphas0

    losses = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        _, loss = model(idx, tgt)
        loss.backward()
        opt.step()
        if i % max(1, steps // 12) == 0 or i == steps - 1:
            losses.append((i, loss.item()))

    rec["loss_curve"] = losses
    rec["loss_initial"] = losses[0][1]
    rec["loss_final"] = losses[-1][1]
    rec["decreased"] = losses[-1][1] < losses[0][1]
    rec["finite"] = all(torch.isfinite(torch.tensor(v)).item() for _, v in losses)

    if cfg.uses_attnres:
        with torch.no_grad():
            model(idx, collect_alphas=True)
        alphas = [a.tolist() for a in model.last_alphas]
        rec["alphas_final"] = alphas
        rec["alpha_summary"] = _alpha_summary(alphas)

    torch.cuda.empty_cache()
    return rec


def _alpha_summary(alphas: list[list[float]]) -> dict:
    """Three numbers per site, aggregated: how far from uniform did it move?"""
    import math

    ent, mx, first, last = [], [], [], []
    for a in alphas:
        n = len(a)
        if n == 1:
            continue
        h = -sum(w * math.log(max(w, 1e-30)) for w in a) / math.log(n)
        ent.append(h)
        mx.append(max(a))
        first.append(a[0])
        last.append(a[-1])
    return {
        "sites": len(ent),
        "mean_normalized_entropy": sum(ent) / len(ent),
        "min_normalized_entropy": min(ent),
        "mean_max_weight": sum(mx) / len(mx),
        "max_max_weight": max(mx),
        "mean_weight_on_embedding": sum(first) / len(first),
        "mean_weight_on_last_source": sum(last) / len(last),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1.3B", choices=sorted(SCALES))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)

    report = {
        "environment": environment(),
        "dtype": args.dtype,
        "scale": args.scale,
        "config": SCALES[args.scale],
        "batch": args.batch,
        "seq": args.seq,
        "quick": args.quick,
        "variants": [],
    }

    print(f"{'variant':>18} {'params':>10} {'fwd ms':>9} {'bwd ms':>9} {'step ms':>9} "
          f"{'tok/s':>9} {'peak GiB':>9}")
    print("-" * 82)
    for residual, sources in VARIANTS:
        rec = bench_variant(
            args.scale, residual, sources,
            dtype=dtype, device=device, batch=args.batch, seq=args.seq, quick=args.quick,
        )
        report["variants"].append(rec)
        print(
            f"{rec['variant']:>18} {rec['params']['total'] / 1e9:9.3f}B "
            f"{rec['forward']['median_ms']:9.2f} {rec['backward_only_ms']:9.2f} "
            f"{rec['step']['median_ms']:9.2f} {rec['tokens_per_second']:9.0f} "
            f"{rec['peak_memory_bytes'] / 2**30:9.2f}"
        )

    report["attribution"] = attribution(report["variants"])
    print()
    for row in report["attribution"]:
        print(
            f"{row['variant']:>18}  residual mechanism: "
            f"step +{row['step_delta_ms']:7.2f} ms = {row['step_share_pct']:5.1f}% of step "
            f"(control diff) | {_fmt(row['profiler_share_pct'])} (profiler)"
        )

    if not args.skip_smoke:
        print("\ntraining smoke test (small config, fp32, overfit one fixed batch)")
        report["smoke"] = []
        for residual in ("standard", "attnres", "switchyard"):
            rec = smoke_test(residual, device, steps=args.smoke_steps)
            report["smoke"].append(rec)
            extra = ""
            if "alpha_summary" in rec:
                s = rec["alpha_summary"]
                extra = (f" | alpha: uniform-at-init={rec['init_is_uniform']}, "
                         f"final mean H/lnN={s['mean_normalized_entropy']:.3f}, "
                         f"mean max weight={s['mean_max_weight']:.3f}")
            print(f"  {residual:>11}: loss {rec['loss_initial']:.4f} -> {rec['loss_final']:.4f}"
                  f"  finite={rec['finite']}{extra}")

    out = args.out or REPO / "results" / f"model_{args.dtype}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")


def _fmt(x) -> str:
    return "not attributable" if x is None else f"{x:5.1f}%"


def attribution(variants: list[dict]) -> list[dict]:
    """Residual-mechanism share by control differencing, with the profiler beside it."""
    base = next(v for v in variants if v["residual"] == "standard")
    rows = []
    for v in variants:
        if v["residual"] == "standard":
            continue
        share = v["profile"].get("attnres_share_of_device_time")
        rows.append({
            "variant": v["variant"],
            "control": base["variant"],
            "method": "control difference against the standard-residual model, wall clock",
            "step_delta_ms": v["step"]["median_ms"] - base["step"]["median_ms"],
            "step_share_pct": 100 * (1 - base["step"]["median_ms"] / v["step"]["median_ms"]),
            "forward_delta_ms": v["forward"]["median_ms"] - base["forward"]["median_ms"],
            "forward_share_pct": 100
            * (1 - base["forward"]["median_ms"] / v["forward"]["median_ms"]),
            "peak_memory_delta_bytes": v["peak_memory_bytes"] - base["peak_memory_bytes"],
            "tokens_per_second_ratio": v["tokens_per_second"] / base["tokens_per_second"],
            "profiler_share_pct": None if share is None else 100 * share,
            "profiler_forward_us": v["profile"].get("attnres_forward_us"),
            "profiler_backward_us": v["profile"].get("attnres_backward_us"),
            "profiler_total_device_us": v["profile"].get("total_device_us"),
        })
    return rows


if __name__ == "__main__":
    main()
