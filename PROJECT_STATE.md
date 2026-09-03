# Project state

Living status document for `blackwell-switchyard`. The authoritative per-change detail lives
in the branch descriptions on issue #1; this file is the index.

**Last updated:** 2026-09-03

---

## Current best result

Three numbers, in decreasing order of how much they matter.

**End to end**, 1.3B decoder, batch 4 × seq 2048, bf16, RTX PRO 6000 Blackwell:
the residual mechanism costs **39% of a training step** framework-implemented and **11%**
fused — a 3.4× reduction in its overhead, worth **1.46× on the whole step**
(704.5 → 484.2 ms, 11627 → 16918 tokens/s).

**Operator**, `N=9 B=1 T=4096 D=2048`, with L2 flushed between measurements:
forward is **1.70×** faster and forward+backward is **3.71×** faster than
max-autotuned Inductor. Kernel counts are 3 → 1 and 13 → 4. Forward workspace is
0.312 → 0 MiB. Forward+backward workspace is 144.887 → 0.008 MiB. Both paths are
at 1.00× the bf16 rounding floor. The forward reaches 97% of the traffic-only limit.

**Batched-query forward**, `N=9 B=1 T=4096 D=2048 S=8`: the output-only resident
path takes 0.199 ms. This is 1.77× faster than max-autotuned Inductor and 4.70× faster
than eight separate switchyard calls. It uses one kernel, no temporary workspace, and
has rounding-floor accuracy. See [`docs/batched_queries.md`](docs/batched_queries.md).

## Current best implementation

`src/switchyard/triton_op.py` contains the accepted single-query forward and backward
strategies. It also contains one output-only batched forward strategy. Dispatch uses a
measured tile budget. Private training plans now isolate the next backward architectures from
production dispatch.

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
- **Output-only batched-query forward.** One resident kernel reuses a source tile across up
  to 16 queries. Unsupported shapes use an accurate per-query fallback. The public API
  rejects autograd because no batched backward exists.

## NEXT

Ordered by how much the measurements say they are worth.

1. **Validate the backward architecture candidates on an idle GPU.** The
   `codex/backward-architecture` branch replaces the backward-only experiment switch with
   complete immutable training plans. It contains grouped Triton paths, saved FP32 forward
   coefficients, hierarchical `dw` reduction, a one-block CUDA traffic control, and a
   persistent two-block feature-sharded CUDA cluster. The cluster targets the one-read source
   traffic lower bound. The offline `sm_120` gate reports 40 registers per cluster thread and
   no stack or local-memory spill. The benchmark uses paired trial medians and records GPU
   exclusivity at the start and end of each run. Production dispatch is unchanged. GPU
   correctness, occupancy, and latency are still pending.
2. **Complete the batched training contract.** The current batched API does not return
   merge statistics and does not implement backward. The resident forward is useful, but
   it is not the complete paper schedule.
3. **Sweep fp16 and fp32 for performance.** They are currently correctness-tested only.

## BLOCKED

- **Opening pull requests.** The token has `contents:write` and `issues:write` but not
  `pull_requests:write`. GitHub has no draft pull requests for this repository. The owner
  authorized a direct fast-forward merge of the reviewed linear branch stack on 2026-09-02.
- **Pushing CI.** The token lacks the `workflow` scope, so `.github/workflows/` cannot be
  pushed. A trackable copy is in `ci/github-actions-ci.yml`. Move it to
  `.github/workflows/ci.yml` with a token that has workflow permission.

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
| 2026-09-02 | Keep the batched-query resident forward, but limit its contract to output-only inference. | At `N=9 B=1 T=4096 D=2048 S=8`, it takes 0.199 ms in one kernel with no workspace and 1.00× floor error. It beats max-autotuned Inductor by 1.77×. At `D=4096`, the fallback loses to catswe, so the resident dispatch boundary remains explicit. |
| 2026-09-02 | Include max-autotuned Inductor in every best-baseline calculation. | The representative baseline is 0.209 ms forward and 1.354 ms forward+backward. The old public claims used 0.430 ms and 1.918 ms, which overstated speedups. |
| 2026-09-02 | Measure allocator peaks relative to live allocations. | The old method charged the 384 MiB L2 flush buffer and live oracle storage to operator workspace. The corrected representative workspaces are 0.312 MiB for max-autotuned Inductor and 0 MiB for switchyard forward. |
| 2026-09-03 | Stop tuning the split backward and prepare a one-read architecture. | The split path moves `(3N+2)X` large-tensor bytes while the exact lower bound is `(2N+1)X`. The 288–576 MiB source stacks do not fit in 128 MiB of L2. Liger is already within about 2.5–4 percent of the two-read bandwidth model, so tile changes cannot create a material lead. The new feature-sharded cluster retains source values in distributed shared memory and compiles for `sm_120` without spills. |
