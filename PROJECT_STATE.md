# Project state

Living status document for `blackwell-switchyard`. The authoritative per-change detail lives
in the branch descriptions on issue #1; this file is the index.

**Last updated:** 2026-08-31

---

## Current best result

Three numbers, in decreasing order of how much they matter.

**End to end**, 1.3B decoder, batch 4 × seq 2048, bf16, RTX PRO 6000 Blackwell:
the residual mechanism costs **39% of a training step** framework-implemented and **11%**
fused — a 3.4× reduction in its overhead, worth **1.46× on the whole step**
(704.5 → 484.2 ms, 11627 → 16918 tokens/s).

**Operator**, `N=9 B=1 T=4096 D=2048`, median of 100 CUDA-event reps with L2 flushed:
forward **3.50×** and forward+backward **5.29×** over the fastest of six framework
configurations, 6 → 1 and 16 → 4 kernels, workspace −22%. Forward sustains
**94–101% of speed of light**.

**Against the other public kernels**: the forward is a four-way tie at the memory ceiling;
we take 7 of 10 shapes on forward+backward, Liger takes 3, and catswe is 4.3× faster on
batched pseudo-queries. See [`docs/third_party.md`](docs/third_party.md).

## Current best implementation

`src/switchyard/triton_op.py` — two forward and two backward strategies, dispatched by a
measured tile budget.

---

## DONE

- **Bootstrap** — license with attribution, src layout, tracking issue.
- **Environment characterization** (`machine-characterization`). 1462 GB/s DRAM,
  6064 GB/s L2 at 4.15×, 299 TFLOP/s bf16, 9.3 µs launch. Established that `ncu` is
  unusable here, the GPUs are PCIe-only, and the host lacks `python3.12-dev` — which breaks
  Triton *and* `torch.compile`, worked around without touching the system.
- **Upstream and prior-art investigation.** Official Moonshot repo is documentation-only.
  Fused kernels already exist in Liger-Kernel, flash-linear-attention, and two standalone
  projects. Project reframed: no novelty claim.
- **Reference + correctness suite** (`fused-operator`). Paper-faithful implementation,
  float64 oracle, two-phase online-softmax helpers.
- **Framework baselines.** Two formulations × eager and three `torch.compile` modes. Found
  Inductor silently falls back to eager at `N ≥ 16`.
- **Fused operator, forward and backward.** Tiled backward closed a fallback that had been
  *slower* than the baseline.
- **Benchmark and profiling infrastructure.** Oracle check before timing, L2 flush, order
  statistics, kernel counting, workspace memory, speed-of-light ceiling.
- **Third-party head-to-head on Blackwell** — the comparison nobody had published.
- **Transformer integration**, 1.3B, three residual modes, parameter-matched, with a
  training smoke test.
- **Two-GPU DDP validation.** 87% scaling efficiency; gradients bit-identical across ranks.
- **Regression gating** against a stored baseline, and a CPU-only CI workflow (written and
  verified locally; see BLOCKED).
- **Technical report** ([`docs/report.md`](docs/report.md)), including every measurement bug
  found and fixed.

## NEXT

Ordered by how much the measurements say they are worth.

1. **A batched-query API.** catswe is 4.3× faster on the axis the paper's own two-phase
   schedule targets, because it reads the sources once per *block* rather than once per
   *call*. Our kernel is at the ceiling for what it is asked to do; it is asked to do the
   wrong thing S times. This is the largest remaining win and it is not a kernel problem.
2. **Improve the tiled backward.** Liger beats it at `D ≥ 4096` and `N=32`. The resident
   backward is comfortably ahead everywhere it applies, so the gap is specific and local.
3. Sweep fp16 and fp32 for performance; they are currently correctness-tested only.

## BLOCKED

- **Opening pull requests.** The token has `contents:write` and `issues:write` but not
  `pull_requests:write`. Branches are pushed; descriptions are posted as comments on
  issue #1 with one-click compare links.
- **Pushing CI.** The token lacks the `workflow` scope, so `.github/workflows/` cannot be
  pushed. The workflow is written and verified locally and its contents are on issue #1.

## OPEN QUESTIONS FOR THE OWNER

None technical. The one decision that changed direction — dropping every novelty claim once
fused kernels were found to already exist, and reframing around an independent Blackwell
comparison — was made on evidence and is in the decision log.

---

## Milestones

| # | Milestone | State | Branch |
|---|-----------|-------|--------|
| 0 | Bootstrap + environment characterization | done | `machine-characterization` |
| 1 | Faithful reference + correctness suite | done | `fused-operator` |
| 2 | Benchmark & profiling infrastructure | done | `fused-operator` |
| 3 | Strong framework baselines | done | `fused-operator` |
| 4 | First optimized operator | done | `fused-operator` |
| 5 | Backward / autograd optimization | done | `fused-operator` |
| 6 | Blackwell-informed tuning | done (see note) | `fused-operator` |
| 7 | Third-party head-to-head on Blackwell | done | `fused-operator` |
| 8 | Transformer integration + training smoke test | done | `fused-operator` |
| 9 | Two-GPU validation | done | `fused-operator` |
| 10 | Benchmark study + technical report | done | `fused-operator` |

Note on milestone 6: the tuning is Blackwell-*informed*, not Blackwell-*specific*. The
128 MiB L2 at 4.15× DRAM decided the tiled-path design and the dispatch constants were swept
on this part, but the kernel uses no `sm_120`-only instruction and the wins come from traffic
and launch count, which are portable. Claiming otherwise would be cosmetic.

## Decision log

Decisions that changed direction, with the evidence that forced them. Append-only.

| Date | Decision | Evidence |
|------|----------|----------|
| 2026-08-30 | Use the existing system interpreter rather than a virtualenv. | `pip` into the shared pyenv is permission-denied and every dependency is already present. Zero installs is also lowest-risk on a shared machine. |
| 2026-08-30 | Extract Python headers into a gitignored local prefix. | Host lacks `python3.12-dev`, so neither Triton nor Inductor can build launcher shims — `torch.compile` does not work at all. Installing system-wide needs root. |
| 2026-08-30 | **Drop every novelty claim; reframe as an independent Blackwell comparison.** | Fused AttnRes Triton kernels already exist and are merged in Liger-Kernel (benchmarked on an RTX 5090, also `sm_120`) and flash-linear-attention, plus two standalone projects. Verified against the GitHub API and the sources, not taken on trust. |
| 2026-08-30 | Carry both `paper_form` and `folded_form` as baselines. | `folded_form` is the better formulation and faster eager, but Inductor fuses `paper_form` into 6 kernels versus 9, making the compiled paper form the strongest baseline. Reporting one would have understated it. |
| 2026-08-30 | Choose the dispatch threshold by sweeping, not by estimating register pressure. | The estimate said 16384 fp32 values; measurement said 32768. At the difference sits `N=9, D=2048`, running at 616 GB/s instead of 1365. |
| 2026-08-30 | Build a tiled backward rather than leaving an autograd fallback. | The fallback recomputed the forward before differentiating, so fwd+bwd was 0.51× the baseline at `N=9, D=4096` — an ordinary model shape. An unoptimized fallback was silently a regression. |
| 2026-08-30 | Judge correctness by relative L2 against a float64 oracle, with the dtype floor beside it. | Elementwise relative error is unbounded where the output crosses zero and reported `max_rel_err` of 5354 for a bit-accurate bf16 result. |
| 2026-08-30 | Scale benchmark inputs so `‖w‖₂ ≈ 1`. | Logits are distributed as `N(0, ‖w‖²)`. Drawing `w ~ N(0,1)` puts the logit standard deviation at `sqrt(D)`, saturating the softmax into a regime no trained model occupies. |
| 2026-08-31 | **Give each third-party kernel its own native input form.** | Handing fla `N` views of one leaf made autograd route gradients through `N` slice-backwards and reported it as 11× slower than us at `N=9` and 29× at `N=32`. With its actual sequence API the figures are 2.7× and 1.5×. The flattering number was one commit from being published. |
| 2026-08-31 | Report the arena *and* the stacking cost rather than only the fast path. | The paper's pseudocode stacks at every site, costing 265 slab copies per forward and 6.75 GiB. Hiding that would have compared our plumbing against their arithmetic. |
