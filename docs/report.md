# Block Attention Residuals on Blackwell: an engineering report

This is the working record of the project: what was established, what was
measured, what was tried and discarded, and what is left. Headline numbers live
in the README; the full tables are in [`results.md`](results.md), regenerated
from `results/` by committed scripts. Nothing here is typed by hand from memory.

---

## 1. The problem

Attention Residuals ([Kimi Team, arXiv:2603.15031](https://arxiv.org/abs/2603.15031))
replaces the fixed additive residual connection

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

with a softmax attention over the outputs of preceding layers,

```
h_l = sum_i alpha_{i->l} * v_i,     alpha_{i->l} = softmax_i( w_l . RMSNorm(v_i) )
```

so each layer aggregates earlier representations with learned, input-dependent
weights. **Block AttnRes** is the scalable variant: layers are partitioned into
`N` blocks, each block's outputs are summed into one representation, and the
attention runs over the `N` block summaries plus the token embedding rather than
over all `L` layer outputs. That takes memory and cross-stage communication from
`O(Ld)` to `O(Nd)`. The paper finds `N ≈ 8` recovers most of the benefit.

Three properties of the operator drive everything that follows.

**It is depth attention, not sequence attention.** The softmax runs over the
source axis, independently for every `(batch, token)` position. No token
interacts with any other. This is why it looks nothing like FlashAttention.

**The weighted sum is over raw sources, not normalized ones.** RMSNorm exists
only to stop a large-magnitude block from dominating the weights; the values
combined are unnormalized. Getting this backwards produces a plausible-looking
tensor with entirely wrong semantics, and no error.

**There is no `1/sqrt(D)` scaling.** The paper defines
`phi(q,k) = exp(q^T RMSNorm(k))` with no temperature term.

All three are pinned by tests in `tests/test_reference.py`, because each fails
silently.

### Shape notation

`v: [N, B, T, D]` sources, `w: [D]` pseudo-query, output `[B, T, D]`. Here `N` is
the *stacked source count* — the paper's `n` einsum subscript — not the
architectural block count. A model with 8 blocks produces operator calls with
`N` between 1 and 10, because a layer attends over the completed blocks, the
token embedding, and (for all but the first layer of a block) the running
intra-block partial sum.

---

## 2. Prior art, stated plainly

**Fused Block AttnRes kernels already exist.** This project makes no novelty
claim about being first, and the literature and code search that established
that is part of the work:

| Project | What it is | Notes |
|---|---|---|
| MoonshotAI/Attention-Residuals | The official repository | **Documentation only.** No source, no LICENSE. The paper's Figure 2 pseudocode is the only reference implementation that exists upstream. |
| linkedin/Liger-Kernel | Merged fused Triton kernel (PR #1161, 2026-03-28) | Benchmarked by its author on an RTX 5090, which is also `sm_120`. |
| fla-org/flash-linear-attention | `fla/ops/attnres`, 14 PRs, three backends | The most developed implementation, including a Gluon backend. |
| catswe/flash-attention-residuals | Triton, Apache-2.0 | Implements the paper's two-phase schedule. |
| manishklach/attnres-kernel-lab | Triton | Publishes no GPU numbers; its author had no NVIDIA runtime. |

DeepSeek's mHC additionally ships fused TileLang kernels for a near-equivalent
depth-mixing primitive, and the AttnRes paper's own Table 5 places the mechanism
in a well-populated family (DenseFormer, MUDDFormer, LAuReL, ELMo-style scalar
mixing, hyper-connections).

So the honest framing of what this repository adds is:

1. An independent head-to-head of the existing implementations **on Blackwell
   `sm_120`**, which nobody has published.
2. An implementation tuned for this part, measured against a **speed-of-light
   kernel** rather than only against other implementations, so the answer to
   "how much is left" is quantitative.
3. A reproducible measurement harness, with the failure modes documented.

Claims this project must not make are listed in the README.

---

## 3. Performance model, before any optimization

Per operator call the minimum traffic is one read of every source and one write
of the output:

```
bytes_min = (N + 1) * B * T * D * itemsize
```

The arithmetic is three passes over the source elements — sum of squares, query
dot product, weighted accumulation — at two FLOPs each:

```
flops = 6 * N * B * T * D
```

Arithmetic intensity is therefore `6N / ((N+1) * itemsize)`, which for bf16 at
`N = 9` is **2.7 FLOP/byte**, and depends only on `N`. It does not improve with
`D`, `T`, or `B`; no tiling changes it.

The measured machine balance is `299 TFLOP/s / 1462 GB/s` = **204 FLOP/byte**.

The operator sits about **75× below the ridge point**. It is memory-bound by a
margin so large that the conclusion needs no profiler to defend: the only things
worth optimizing are bytes moved and, at small shapes, kernels launched. This
was decided before writing any kernel and it held.

A corollary that shaped the design: because `bytes_min` counts *one* read of
`v`, and the softmax over the source axis must complete before the weighted sum
can begin, any implementation that does not keep `v` resident between those two
phases reads it twice and cannot exceed half the roofline.

---

## 4. The machine

Full capture in [`machine.md`](machine.md). The numbers that matter here:

| | measured |
|---|---|
| DRAM copy | 1462 GB/s |
| DRAM read | 1638 GB/s (91% of the 1792 GB/s theoretical) |
| L2 peak copy | 6064 GB/s, **4.15× DRAM** |
| L2 size | 128 MiB |
| BF16 matmul | 299 TFLOP/s |
| Triton launch | 9.3 µs |
| CUDA-graph replay | 1.07 µs |

The large, fast L2 is the single most useful architectural fact for this
operator. It means the "reads `v` twice" penalty above is not necessarily a
factor of two: if the sources touched by all concurrently-resident programs stay
under ~128 MiB, the second read is served at 4.15× DRAM bandwidth and costs
about a quarter of a DRAM read. That turns a two-pass formulation from an
obvious loss into a live design candidate, and it is why the tiled kernel
described below is competitive.

**Nsight Compute is unavailable.** `ncu` runs but counter collection fails with
`ERR_NVGPUCTRPERM`; lifting it requires a host-wide driver setting this project
will not change on a shared machine. Substitutes: `nsys`, `torch.profiler` for
kernel counts and durations, and — most usefully — achieved bandwidth derived
from measured latency and an exact byte count. For an operator this
memory-bound, that last one answers most of what a counter would.

---

## 5. Baseline profile

Two framework formulations were measured, each eager and under `torch.compile`
in three modes:

- `paper_form` — a transcription of the paper's Figure 2. Materializes the
  normalized tensor `k` in full, an extra `[N,B,T,D]` write and read.
- `folded_form` — uses `dot(w, v/rms(v)) == dot(w, v)/rms(v)` so the normalized
  tensor never exists. Both reductions over `D` are then computable in one pass.

At `N=9, B=1, T=4096, D=2048`, bf16, the best framework result was
`torch.compile` on `paper_form`: **0.432 ms in 6 kernels**, sustaining 388 GB/s
against a 1462 GB/s ceiling — about **27% of achievable bandwidth**.

Three observations from the baseline sweep were unexpected enough to record:

**`torch.compile` silently falls back at larger `N`.** At `N=16` and `N=32`,
`paper_compiled` returns latencies and kernel counts *bit-identical* to
`paper_eager` — 9 kernels, same milliseconds to four decimal places. Inductor is
declining to fuse and no error surfaces. A baseline that quietly degrades is
worth catching, because reporting it as "the compiled baseline" would overstate
our margin at exactly the shapes where we look best.

**The better formulation compiles worse.** `folded_form` is the stronger
algebraic arrangement and is faster eager, but Inductor fuses `paper_form` into
6 kernels and `folded_form` into 9, making the compiled paper form faster. The
strongest baseline is therefore not the strongest formulation, which is why both
are carried through every table.

**At small shapes the operator is dispatch-bound, not bandwidth-bound.** At
`N=9, T=512, D=1024` the eager forward spends 31 µs of GPU time inside a 55 µs
wall-clock window; the backward spends 172 µs inside 1089 µs. The harness now
reports GPU-busy time and utilization alongside latency, because the two
regimes call for opposite fixes — fewer kernels versus fewer bytes.

---

## 6. Design

The kernel folds the RMS scale into the score, so the `[N,B,T,D]` normalized
tensor is never materialized, and then picks one of two strategies:

**Resident** (`N * next_pow2(D)` fits in registers). One program per token. `v`
is loaded once into a `[BLOCK_N, BLOCK_D]` tile and stays there for the norms,
the logits, the softmax and the weighted sum. Exactly the minimum traffic, one
kernel.

**Tiled** (otherwise). One program per token, looping over `D`: accumulate the
sum of squares and the query dot product, then loop again to apply the weights.
This reads `v` twice from the memory system, but the concurrently-resident
programs touch far less than 128 MiB, so the second read is an L2 hit.

Backward mirrors the split. The resident backward recomputes the forward
statistics rather than saving them — the bytes must be read anyway to produce
`dv`, and this operator has no arithmetic to spare bytes for. The tiled backward
is two kernels, because `dw` reduces over every token while the softmax
statistics need a full-`D` reduction per token, and no single loop nesting
satisfies both.

Derivation of the backward, with `r = rsqrt(ssq/D + eps)`, `a = dot(w, v_n)`:

```
G_n      = dot(g, v_n)
S        = sum_n alpha_n G_n
dlogit_n = alpha_n * (G_n - S)
da_n     = dlogit_n * r_n
dssq_n   = -dlogit_n * a_n * r_n^3 / (2D)
dv_n     = alpha_n * g + da_n * w + 2 * dssq_n * v_n
dw      += sum_n da_n * v_n
```

`dw` accumulates in fp32 regardless of working dtype: it is reduced over every
token by atomics, and bf16 atomics would lose most contributions to swamping.

---

## 7. What was wrong along the way

Recorded because the errors are more instructive than the final code, and
because two of them would have produced flattering numbers.

**The L2 measurement inverted the cache hierarchy.** The first bandwidth probe
timed one kernel per working-set size and reported L2 as *slower* than DRAM
(1306 vs 1458 GB/s). At 32 MiB the ~10 µs launch overhead dominated. Fixed by
repeating each kernel enough times to move a fixed total volume at every size.
Had it gone unnoticed, the entire two-pass design rationale would have been
built on a number that was backwards.

**An empty CUDA graph replayed in nanoseconds.** Graph capture silently fails on
GPU 1 and PyTorch only warns. The probe reported 0.06 µs/op, a 100× "speedup"
over GPU 0. The probe now verifies the captured op count and discards the
measurement otherwise.

**The dispatch threshold was reasoned about instead of measured, and was
wrong.** A register-pressure estimate put the resident tile limit at 16384 fp32
values. The measured limit is 32768. The difference is not academic: `N=9` with
`D=2048` — the single most representative shape, since the paper uses about
eight blocks plus the embedding — has a tile of exactly 32768 and was being sent
to the tiled kernel at **616 GB/s** instead of the resident one at
**1365 GB/s**. Two further hand-picked constants were also wrong:
`num_stages=1` beats 2 or 3 nearly everywhere (the kernel is bandwidth-bound
with a short dependency chain, so extra pipeline stages buy no latency hiding
and cost occupancy), and the tiled kernel wants `BLOCK_D=2048` rather than the
small tile a reduction would suggest.

**The backward fallback was a regression, not merely an absence.** Shapes
outside the resident budget fell back to autograd, but the fallback recomputed
the forward before differentiating, so fwd+bwd ran at **0.51×** the baseline —
our "optimized" operator was twice as slow as `torch.compile` at `N=9, D=4096`,
an ordinary model shape. Fixed by the tiled backward; a test now pins both
dispatch paths.

**The correctness harness was testing an unphysical regime.** The first
benchmark drew `w ~ N(0, 0.5²)`. Because RMSNorm gives every key unit RMS, a
logit is distributed as `N(0, ||w||²)`, so at `D=1024` the logits had a standard
deviation of about 16 — a saturated, near-one-hot softmax that is violently
sensitive to rounding. Every implementation "failed" correctness. The regime is
an artifact of the test: the paper zero-initializes `w`, and a trained query
gives logits of order one. Inputs are now scaled so `||w|| ≈ 1`.

**Elementwise relative error was the wrong metric.** The operator's output is a
convex combination of zero-mean sources, so elements pass through zero and
relative error is unbounded there — it reported `max_rel_err` of 5354 for a
result correct to the last bit bf16 can represent. The pass criterion is now
relative L2 against a float64 oracle, reported alongside the error that merely
*rounding* the exact answer to that dtype would cost.

---

## 8. Correctness

Pass criterion is relative L2 against a float64 oracle, with the dtype's own
rounding floor computed beside it, so the bar is "as good as the format allows"
rather than an arbitrary tolerance.

The fused kernel is **more accurate than the framework baseline**, at 1.00× the
bf16 rounding floor across all sixteen tested shapes, where the eager chain
lands at 1.6–3.3×. It accumulates every reduction in fp32 and rounds once; the
eager chain rounds at each step. This is worth stating because fusion is usually
assumed to trade accuracy for speed, and here it does the opposite.

99 tests cover both dispatch strategies, three dtypes, non-power-of-two `N`, `D`
and `T`, `gradcheck` and `gradgradcheck` in float64, exactness of the
online-softmax merge used by the paper's two-phase schedule, overflow behaviour
at saturated logits, non-contiguous inputs, and the invariants above.

---

## 9. Results

See [`results.md`](results.md) for the full tables across 19 shapes, and the
README for the headline. Summary of the forward at
`B=1, T=4096, D=2048`, bf16: the fused kernel runs at **94–101% of speed of
light** across `N` from 2 to 32, against 22–28% for the best framework baseline.

"Speed of light" is a Triton kernel that touches exactly the operator's bytes
with the same access pattern, tile shape and launch geometry but skips the
softmax. It is a fairer ceiling than a generic copy benchmark. It is itself a
measured kernel subject to the same noise, so a figure at or slightly above
100% means "at the ceiling", not "faster than possible".

Where the kernel does *not* reach the ceiling:

- `D = 8192` on the tiled path: 84%. The second pass over `D` no longer fits the
  L2 residency argument as comfortably.
- Small shapes (`T ≤ 512`): 75–86%. Launch-bound, where a single 8 µs kernel
  launch is a large fraction of the total. Both the kernel and the ceiling are
  one launch, so there is little to win.

---

## 10. Limitations

- Measured on one GPU model. Nothing here is claimed to generalize to other
  NVIDIA parts, and the design is deliberately not `sm_120`-specific — the win
  comes from traffic and launch count, both portable. The L2 residency argument
  for the tiled path does depend on a large L2.
- No Nsight Compute counters, for the permission reason above. Occupancy,
  register counts and L2 hit rates are therefore inferred from latency and
  bandwidth rather than measured directly.
- bf16 is the primary dtype. fp16 and fp32 are tested for correctness but not
  swept for performance.
- The operator takes a pre-stacked `[N,B,T,D]` tensor. In a real model the
  sources arrive as separate tensors, and the stacking cost is a real cost —
  measured separately in the model integration rather than hidden.

---

## 11. Lessons

**Measure the machine before designing for it.** The 4.15× L2 ratio decided the
tiled-path design, and the first attempt to measure it produced a number that
was backwards.

**Do not reason about register pressure when you can sweep it.** The one
hand-derived constant in the dispatch rule was wrong, by 2.2× on the most
important shape.

**A fallback is a performance claim too.** An unoptimized path that is *slower
than the baseline* turns a win into a loss at exactly the shapes nobody checked.

**Check that the test regime is physical.** A benchmark input drawn without
thinking about what the trained model actually looks like put the softmax in a
saturated regime and made every implementation fail correctness.
