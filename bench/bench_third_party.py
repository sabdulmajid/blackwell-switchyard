"""Head-to-head against the existing public Block AttnRes kernels, on Blackwell.

Fused AttnRes kernels already exist. Comparing only against `torch.compile` would
be an incomplete claim, so this benchmarks the real alternatives on the same
hardware, the same inputs, and the same harness.

The implementations are **not vendored**. `scripts/fetch_third_party.sh` clones
them at pinned commits into a gitignored directory; anything absent is skipped with a note rather
than silently omitted.

Fairness rules applied here, each of which changes the answer:

* **Semantic equivalence is verified before anything is timed.** Each project
  parameterizes the operator slightly differently -- a learnable RMSNorm gain, an
  optional output RMSNorm, a logit scale. All are set to the identity so every
  implementation computes the same function, and that is then checked against a
  float64 oracle. A latency for an implementation computing something else is
  worse than no latency at all.
* **Everyone gets the same tensor.** `fla` takes a sequence of sources rather
  than a stacked tensor, so it is handed views into the same allocation. No copy
  is made for it and none is charged to it.
* **catswe is measured twice.** Its API is a two-phase schedule whose whole point
  is amortizing one pass over the sources across the `S` pseudo-queries of a
  block. Timing it at `S=1` compares it against its design, so it is also
  measured at `S=8` against `S` separate calls of everything else. That is the
  axis where its design should win, and the report says so.

Run::

    scripts/fetch_third_party.sh
    source scripts/env.sh
    python bench/bench_third_party.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "bench"))

# Third-party clones, if fetched.
THIRD_PARTY = Path(os.environ.get("THIRD_PARTY_DIR", REPO / "third_party"))
for sub in ("Liger-Kernel/src", "flash-linear-attention", "flash-attention-residuals/src"):
    p = THIRD_PARTY / sub
    if p.exists():
        sys.path.insert(0, str(p))

from harness import (  # noqa: E402
    achieved_bandwidth_gbps,
    check_against,
    count_kernels,
    environment,
    measure_latency,
    measure_memory,
    repository_provenance,
)
from switchyard.baselines import batched_folded_form  # noqa: E402
from switchyard.reference import (  # noqa: E402
    DEFAULT_EPS,
    attn_res_min_bytes,
    block_attn_res_oracle,
)
from switchyard.triton_op import (  # noqa: E402
    block_attn_res_batched,
    block_attn_res_triton,
    speed_of_light,
)


@dataclass(frozen=True)
class Shape:
    n: int
    b: int
    t: int
    d: int

    def key(self) -> str:
        return f"N{self.n}_B{self.b}_T{self.t}_D{self.d}"


SHAPES = [
    Shape(2, 1, 4096, 2048), Shape(4, 1, 4096, 2048), Shape(8, 1, 4096, 2048),
    Shape(9, 1, 4096, 2048), Shape(16, 1, 4096, 2048), Shape(32, 1, 4096, 2048),
    Shape(9, 1, 4096, 1024), Shape(9, 1, 4096, 4096), Shape(9, 1, 4096, 8192),
    Shape(9, 1, 8192, 2048),
]

BATCHED_CASES = [
    *((Shape(9, 1, 4096, 2048), s) for s in (1, 2, 4, 8, 16)),
    (Shape(2, 1, 4096, 2048), 8),
    (Shape(16, 1, 4096, 2048), 8),
    (Shape(9, 1, 4096, 1024), 8),
    (Shape(9, 1, 128, 1024), 8),
    (Shape(9, 4, 2048, 2048), 8),
    (Shape(9, 1, 4096, 4096), 8),
]


def build_adapters() -> tuple[dict, list[str]]:
    """Wrap each available implementation as ``fn(v, w, eps) -> [B, T, D]``.

    Returns the adapters and a list of notes about what was unavailable and why.
    """
    adapters: dict[str, dict] = {
        "switchyard": {"fn": block_attn_res_triton, "note": "this repository"},
    }
    notes: list[str] = []
    gains: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

    def unit_gain(v: torch.Tensor) -> torch.Tensor:
        """Return a cached fixed gain so adapters do not allocate in timed calls."""
        key = (v.device, v.dtype, v.shape[-1])
        if key not in gains:
            gains[key] = torch.ones(v.shape[-1], device=v.device, dtype=v.dtype)
        return gains[key]

    try:
        from liger_kernel.ops.attn_res import LigerAttnResFunction

        def liger(v, w, eps=DEFAULT_EPS):
            # Liger carries a separate RMSNorm gain; ones makes it our operator.
            return LigerAttnResFunction.apply(v, w, unit_gain(v), eps)

        adapters["liger"] = {"fn": liger, "note": "linkedin/Liger-Kernel, rms gain set to ones"}
    except Exception as exc:  # noqa: BLE001
        notes.append(f"liger unavailable: {type(exc).__name__}: {exc}"[:200])

    try:
        from fla.ops.attnres import fused_attnres

        def fla(v, w, eps=DEFAULT_EPS):
            # fla takes a sequence of sources. These are views into the caller's
            # own tensor, so no copy is made for it and none is charged to it.
            return fused_attnres(
                w, [v[i] for i in range(v.shape[0])], unit_gain(v), None, eps, 1.0
            )

        adapters["fla"] = {"fn": fla, "note": "fla-org/flash-linear-attention, fused backend"}
    except Exception as exc:  # noqa: BLE001
        notes.append(f"fla unavailable: {type(exc).__name__}: {exc}"[:200])

    try:
        from flash_attn_res import phase_1_batched_attention_triton_op

        def catswe(v, w, eps=DEFAULT_EPS):
            # Phase 1 with a single pseudo-query is our operator. Its design is
            # aimed at S > 1; see bench_batched_queries below.
            out, _lse = phase_1_batched_attention_triton_op(v, w.unsqueeze(0), eps)
            return out[0]

        adapters["catswe"] = {
            "fn": catswe,
            "note": "catswe/flash-attention-residuals, phase 1 at S=1 (see note)",
        }
    except Exception as exc:  # noqa: BLE001
        notes.append(f"catswe unavailable: {type(exc).__name__}: {exc}"[:200])

    return adapters, notes


def bench_one(name, fn, shape, v, w, oracle, device, quick, tol) -> dict:
    rec: dict = {"impl": name, "shape": asdict(shape)}
    rec["correctness"] = check_against(fn, oracle, (v, w, DEFAULT_EPS), rel_l2_tol=tol)
    if not rec["correctness"]["ok"]:
        rec["skipped"] = "failed correctness check"
        return rec

    warmup, reps = (10, 30) if quick else (25, 100)
    fwd = measure_latency(lambda: fn(v, w, DEFAULT_EPS), device=device, warmup=warmup, reps=reps)
    rec["forward"] = fwd.as_dict()
    min_bytes = attn_res_min_bytes(shape.n, shape.b, shape.t, shape.d, v.element_size())
    rec["forward_achieved_gbps"] = achieved_bandwidth_gbps(min_bytes, fwd.median_ms)
    try:
        rec["forward_kernels"] = count_kernels(
            lambda: fn(v, w, DEFAULT_EPS), device=device
        ).as_dict()
    except Exception as exc:  # noqa: BLE001
        rec["forward_kernels"] = {"error": str(exc)[:200]}

    resident = v.numel() * v.element_size() + w.numel() * w.element_size()
    resident += shape.b * shape.t * shape.d * v.element_size()
    rec["forward_memory"] = measure_memory(
        lambda: fn(v, w, DEFAULT_EPS), device=device, resident_bytes=resident
    ).as_dict()

    # Backward gets each implementation its NATIVE leaf structure.
    #
    # This is not a detail. `fla` takes a sequence of sources rather than one
    # stacked tensor, which is how a real model holds them -- each block
    # representation is produced by a different layer and is its own tensor.
    # Handing it N views of a single leaf instead makes autograd route every
    # gradient back through N slice-backwards and an accumulation, and that cost
    # is the harness's doing, not the kernel's. Measured, it is the difference
    # between fla appearing 11x slower than us at N=9 and 2.2x, and between 29x
    # and a dead heat at N=32. The first version of this file made exactly that
    # mistake; `fwd_bwd_stacked_leaf` keeps the wrong number visible, because the
    # gap between the two is itself informative about the API choice.
    go = torch.randn(shape.b, shape.t, shape.d, device=device, dtype=v.dtype)

    def time_bwd(build):
        params, call = build()
        def step():
            for p in params:
                p.grad = None
            call().backward(go)
        t = measure_latency(step, device=device, warmup=warmup, reps=max(20, reps // 2))
        k = count_kernels(step, device=device, iters=3).as_dict()
        for p in params:
            p.grad = None
        return t, k

    def stacked_leaf():
        vg = v.detach().clone().requires_grad_(True)
        wg = w.detach().clone().requires_grad_(True)
        return [vg, wg], lambda: fn(vg, wg, DEFAULT_EPS)

    def native_leaves():
        if name != "fla":
            return stacked_leaf()
        from fla.ops.attnres import fused_attnres
        leaves = [v[i].detach().clone().requires_grad_(True) for i in range(shape.n)]
        wg = w.detach().clone().requires_grad_(True)
        ones = torch.ones(shape.d, device=v.device, dtype=v.dtype)
        return [*leaves, wg], lambda: fused_attnres(
            wg, leaves, ones, None, DEFAULT_EPS, 1.0
        )

    try:
        fb, kern = time_bwd(native_leaves)
        rec["fwd_bwd"] = fb.as_dict()
        rec["fwd_bwd_kernels"] = kern
        rec["fwd_bwd_input_form"] = "separate leaves" if name == "fla" else "stacked leaf"
    except Exception as exc:  # noqa: BLE001
        rec["fwd_bwd"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    if name == "fla":
        try:
            fb2, _ = time_bwd(stacked_leaf)
            rec["fwd_bwd_stacked_leaf"] = fb2.as_dict()
        except Exception as exc:  # noqa: BLE001
            rec["fwd_bwd_stacked_leaf"] = {"error": str(exc)[:200]}

    torch.cuda.empty_cache()
    return rec


def bench_batched_queries(shape, v, device, quick, n_queries=8, tol=None) -> list[dict]:
    """Compare native and framework batched-query paths under one checked contract."""
    out: list[dict] = []
    tol = tol or (2e-2 if v.dtype == torch.bfloat16 else 5e-3)
    warmup, reps = (10, 30) if quick else (20, 60)
    torch.manual_seed(1)
    qs = torch.randn(n_queries, shape.d, device=device, dtype=v.dtype)
    qs = qs / qs.norm(dim=-1, keepdim=True)
    oracle = torch.stack(
        [block_attn_res_oracle(v, qs[i], DEFAULT_EPS) for i in range(n_queries)]
    )

    def ours_loop_native():
        return [block_attn_res_triton(v, qs[i], DEFAULT_EPS) for i in range(n_queries)]

    def ours_loop_tensor(v, queries, eps):
        return torch.stack([block_attn_res_triton(v, queries[i], eps) for i in range(n_queries)])

    candidates = [
        {
            "impl": "framework eager batched",
            "check_fn": batched_folded_form,
            "time_fn": lambda: batched_folded_form(v, qs, DEFAULT_EPS),
            "output_contract": "[S,B,T,D] tensor",
        },
        {
            "impl": f"switchyard x{n_queries} calls",
            "check_fn": ours_loop_tensor,
            "time_fn": ours_loop_native,
            "memory_fn": ours_loop_native,
            "output_contract": "list of S tensors (native per-query contract)",
            "correctness_contract": "stacked to [S,B,T,D] only for validation",
        },
        {
            "impl": "switchyard batched",
            "check_fn": block_attn_res_batched,
            "time_fn": lambda: block_attn_res_batched(v, qs, DEFAULT_EPS),
            "output_contract": "[S,B,T,D] tensor; output-only, inference-only",
        },
    ]

    # This sweep intentionally crosses more shapes than Dynamo's default
    # recompilation limit. Reset per case so later rows do not silently become
    # eager after the eighth specialization.
    torch._dynamo.reset()
    compiled = torch.compile(
        batched_folded_form, mode="max-autotune-no-cudagraphs", dynamic=False
    )
    candidates.insert(1, {
        "impl": "framework compiled batched",
        "check_fn": compiled,
        "time_fn": lambda: compiled(v, qs, DEFAULT_EPS),
        "output_contract": "[S,B,T,D] tensor",
        "compiled": True,
    })

    try:
        from flash_attn_res import phase_1_batched_attention_triton_op

        def catswe_output(v, queries, eps):
            return phase_1_batched_attention_triton_op(v, queries, eps)[0]

        candidates.append({
            "impl": f"catswe phase1 S={n_queries}",
            "check_fn": catswe_output,
            "time_fn": lambda: catswe_output(v, qs, DEFAULT_EPS),
            "output_contract": "[S,B,T,D] tensor; also computes LSE and backward auxiliaries",
        })
    except Exception as exc:  # noqa: BLE001
        out.append({"impl": "catswe phase1", "error": str(exc)[:200]})

    logical_min_bytes = (
        (shape.n + n_queries) * shape.b * shape.t * shape.d * v.element_size()
        + qs.numel() * qs.element_size()
    )
    resident = (
        v.numel() * v.element_size()
        + qs.numel() * qs.element_size()
        + n_queries * shape.b * shape.t * shape.d * v.element_size()
    )

    for spec in candidates:
        rec = {
            "impl": spec["impl"],
            "shape": asdict(shape),
            "n_queries": n_queries,
            "output_contract": spec["output_contract"],
            "logical_min_bytes": logical_min_bytes,
        }
        if "correctness_contract" in spec:
            rec["correctness_contract"] = spec["correctness_contract"]

        if spec.get("compiled"):
            start = time.perf_counter()
            spec["check_fn"](v, qs, DEFAULT_EPS)
            torch.cuda.synchronize(device)
            rec["compile_seconds"] = time.perf_counter() - start

        rec["correctness"] = check_against(
            spec["check_fn"], oracle, (v, qs, DEFAULT_EPS), rel_l2_tol=tol
        )
        if not rec["correctness"]["ok"]:
            rec["skipped"] = "failed correctness check"
            out.append(rec)
            continue

        got = spec["check_fn"](v, qs, DEFAULT_EPS)
        rec["per_query_correctness"] = [
            check_against(lambda x=x: x, oracle[i], (), rel_l2_tol=tol)
            for i, x in enumerate(got)
        ]
        del got

        timing = measure_latency(spec["time_fn"], device=device, warmup=warmup, reps=reps)
        rec["forward"] = timing.as_dict()
        rec["logical_min_achieved_gbps"] = achieved_bandwidth_gbps(
            logical_min_bytes, timing.median_ms
        )
        try:
            rec["forward_kernels"] = count_kernels(
                spec["time_fn"], device=device, iters=1
            ).as_dict()
        except Exception as exc:  # noqa: BLE001
            rec["forward_kernels"] = {"error": str(exc)[:200]}
        rec["forward_memory"] = measure_memory(
            spec.get("memory_fn", spec["time_fn"]), device=device, resident_bytes=resident
        ).as_dict()
        out.append(rec)

    del oracle
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batched-only", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    tol = {torch.bfloat16: 2e-2, torch.float16: 5e-3}[dtype]

    adapters, notes = build_adapters()
    for n in notes:
        print(f"note: {n}")
    print(f"comparing: {', '.join(adapters)}\n")

    report = {
        "environment": environment(),
        "provenance": repository_provenance(
            REPO,
            {
                name: THIRD_PARTY / name
                for name in (
                    "Liger-Kernel",
                    "flash-linear-attention",
                    "flash-attention-residuals",
                )
            },
        ),
        "dtype": args.dtype,
        "rel_l2_tol": tol,
        "unavailable": notes,
        "adapter_notes": {k: v["note"] for k, v in adapters.items()},
        "results": [],
        "batched_queries": [],
    }

    if args.batched_only:
        for shape, n_queries in BATCHED_CASES:
            torch.manual_seed(0)
            v = torch.randn(shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype)
            print(f"batched {shape.key()} S={n_queries}", flush=True)
            report["batched_queries"].extend(
                bench_batched_queries(shape, v, device, args.quick, n_queries, tol)
            )
            del v
            torch.cuda.empty_cache()

        out = args.out or REPO / "results" / f"batched_queries_{args.dtype}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out}")
        return

    print(f"{'shape':>22} {'impl':>14} {'fwd ms':>9} {'GB/s':>7} {'krnl':>5} "
          f"{'f+b ms':>9} {'xfloor':>7}")
    print("-" * 82)

    for shape in SHAPES:
        torch.manual_seed(0)
        v = torch.randn(shape.n, shape.b, shape.t, shape.d, device=device, dtype=dtype)
        w = torch.randn(shape.d, device=device, dtype=torch.float32)
        w = (w / w.norm()).to(dtype)
        oracle = block_attn_res_oracle(v, w, DEFAULT_EPS)

        sol = measure_latency(lambda v=v: speed_of_light(v), device=device,
                              warmup=10, reps=30 if args.quick else 60)
        mb = attn_res_min_bytes(shape.n, shape.b, shape.t, shape.d, v.element_size())
        report["results"].append({
            "impl": "speed_of_light", "shape": asdict(shape), "forward": sol.as_dict(),
            "forward_achieved_gbps": achieved_bandwidth_gbps(mb, sol.median_ms),
            "note": "ceiling, not an implementation",
        })
        print(f"{shape.key():>22} {'speed_of_light':>14} {sol.median_ms:9.4f} "
              f"{achieved_bandwidth_gbps(mb, sol.median_ms):7.0f}     1  (ceiling)")

        for name, spec in adapters.items():
            rec = bench_one(name, spec["fn"], shape, v, w, oracle, device, args.quick, tol)
            rec["fraction_of_speed_of_light"] = (
                sol.median_ms / rec["forward"]["median_ms"] if not rec.get("skipped") else None
            )
            report["results"].append(rec)
            if rec.get("skipped"):
                print(f"{shape.key():>22} {name:>14}   SKIPPED: "
                      f"{rec['correctness'].get('error', 'mismatch')[:40]}")
                continue
            fb = rec.get("fwd_bwd", {}).get("median_ms")
            c = rec["correctness"]
            xf = c["rel_l2"] / c["dtype_floor_rel_l2"] if c.get("dtype_floor_rel_l2") else 0
            print(f"{shape.key():>22} {name:>14} {rec['forward']['median_ms']:9.4f} "
                  f"{rec['forward_achieved_gbps']:7.0f} "
                  f"{str(rec.get('forward_kernels', {}).get('total_kernels', '?')):>5} "
                  f"{(f'{fb:9.4f}' if fb else '      n/a')} {xf:7.2f}")

        if shape.n == 9 and shape.d == 2048 and shape.t == 4096:
            report["batched_queries"].extend(
                bench_batched_queries(shape, v, device, args.quick)
            )

        del v, w, oracle
        torch.cuda.empty_cache()

    if report["batched_queries"]:
        print("\nBatched pseudo-queries (the two-phase amortization axis):")
        for r in report["batched_queries"]:
            if "forward" in r:
                print(f"  {r['impl']:28} {r['forward']['median_ms']:9.4f} ms")
            else:
                print(f"  {r['impl']:28} {r.get('error')}")

    out = args.out or REPO / "results" / f"third_party_{args.dtype}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
