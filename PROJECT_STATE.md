# Project state

Living status document for `blackwell-switchyard`. Updated as work lands. The authoritative
per-change detail lives in the branch descriptions on issue #1; this file is the index.

**Last updated:** 2026-08-30

---

## Current best result

Fused Block AttnRes operator, Triton, on RTX PRO 6000 Blackwell (`sm_120`), bf16,
`N=9 B=1 T=4096 D=2048`, median of 100 CUDA-event-timed reps with L2 flushed, steady
state after compilation:

| | best framework baseline | switchyard | |
|---|---|---|---|
| forward | 0.432 ms (`torch.compile`, paper form) | **0.123 ms** | **3.5×** |
| forward + backward | 1.864 ms | **0.362 ms** | **5.1×** |
| forward kernels | 6 | **1** | |
| fwd+bwd kernels | 16 | **4** | |

Forward runs at **94–101% of speed of light** across `N` from 2 to 32 — a kernel that
touches the same bytes with the same access pattern but skips the softmax. The forward is
finished; there is essentially nothing left to win on it.

Accuracy is **1.00× the bf16 rounding floor**, against 1.6–3.3× for the eager baseline.
Fusing improves accuracy here rather than costing it.

## Current best implementation

`src/switchyard/triton_op.py` — two forward strategies (register-resident and D-tiled,
dispatched by measured tile budget) and two matching backward strategies.

---

## DONE

- **Bootstrap.** License with attribution, src layout, project state, tracking issue.
- **Environment characterization** (`machine-characterization` branch). Measured DRAM
  1462 GB/s, L2 6064 GB/s at 4.15× DRAM, 299 TFLOP/s bf16, 9.3 µs launch overhead.
  Established that `ncu` is unusable here (`ERR_NVGPUCTRPERM`) and that the two GPUs are
  PCIe-only at 25.8 GB/s. Found and documented that the host lacks `python3.12-dev`, which
  breaks Triton *and* `torch.compile`; worked around without touching the system.
- **Upstream and prior-art investigation.** The official Moonshot repository is
  documentation-only — no code, no LICENSE. Fused AttnRes kernels already exist in
  Liger-Kernel, flash-linear-attention, and two standalone projects. The project was
  reframed accordingly: no novelty claim, and the contribution is the Blackwell study plus
  a tuned implementation measured against a speed-of-light ceiling.
- **Canonical reference + correctness suite** (`fused-operator` branch). Paper-faithful
  implementation, float64 oracle, two-phase online-softmax helpers, 99 passing tests.
- **Framework baselines.** Two formulations, eager and under three `torch.compile` modes.
  Found that Inductor silently falls back to eager at `N ≥ 16`.
- **Fused operator, forward and backward** (`fused-operator` branch), with the tiled
  backward closing a fallback that had been *slower* than the baseline.
- **Benchmark and profiling infrastructure.** `bench/harness.py` (CUDA events, L2 flush,
  order statistics, oracle check before timing, kernel counting, workspace memory),
  `bench/bench_operator.py`, and scripts that regenerate every table and figure from raw
  JSON.
- **Technical report** (`docs/report.md`), including the measurement bugs found and fixed.

## IN PROGRESS

- Head-to-head against the existing third-party kernels (Liger, fla, catswe) on this
  hardware. This is the project's actual contribution and is not optional.
- End-to-end decoder Transformer integration: step time, tokens/s, peak VRAM, and the
  share of a training step spent in the residual mechanism.

## NEXT

- Two-GPU validation (DDP), bearing in mind the PCIe-only link.
- README with the final measured headline.
- Benchmark regression checking against stored baselines.

## BLOCKED

- **Opening pull requests.** The available API token has `contents:write` and
  `issues:write` but not `pull_requests:write`, so branches are pushed but PRs cannot be
  opened programmatically. Branch descriptions are posted as comments on issue #1 with
  one-click compare links. Granting **Pull requests: write** on the fine-grained PAT would
  remove this.

## OPEN QUESTIONS FOR THE OWNER

- Nothing technical. The one decision that changed project direction — dropping any
  novelty claim once fused kernels were found to already exist, and reframing around an
  independent Blackwell comparison — was made on evidence and is documented in the
  decision log below rather than deferred.

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
| 6 | Blackwell-specific tuning | done (see note) | `fused-operator` |
| 7 | Third-party head-to-head on Blackwell | in progress | — |
| 8 | Transformer integration + training smoke test | in progress | — |
| 9 | Two-GPU validation | not started | — |
| 10 | Final benchmark study + report polish | in progress | — |

Note on milestone 6: the tuning is Blackwell-*informed* rather than Blackwell-*specific*.
The 128 MiB L2 at 4.15× DRAM decided the tiled-path design, and the dispatch constants were
swept on this part. But the kernel uses no `sm_120`-only instruction, and the wins come
from traffic and launch count, which are portable. Claiming otherwise would be cosmetic.

## Decision log

Decisions that changed direction, with the evidence that forced them. Append-only.

| Date | Decision | Evidence |
|------|----------|----------|
| 2026-08-30 | Use the existing system interpreter rather than a virtualenv. | `pip` into the shared pyenv is permission-denied, and every dependency is already present. Zero installs is also lowest-risk on a shared machine. |
| 2026-08-30 | Extract Python headers into a gitignored local prefix. | Host lacks `python3.12-dev`, so `Python.h` is absent and neither Triton nor Inductor can build launcher shims — `torch.compile` does not work at all. Installing system-wide needs root on a shared machine. |
| 2026-08-30 | **Drop every novelty claim; reframe as an independent Blackwell comparison.** | Fused AttnRes Triton kernels already exist and are merged in Liger-Kernel (benchmarked on an RTX 5090, also `sm_120`) and flash-linear-attention, plus two standalone projects. Verified directly against the GitHub API and the sources, not taken on trust. |
| 2026-08-30 | Carry both `paper_form` and `folded_form` as baselines rather than only the better one. | `folded_form` is the stronger formulation and faster eager, but Inductor fuses `paper_form` into 6 kernels versus 9, making the compiled paper form the strongest baseline. Reporting only one would have understated the baseline. |
| 2026-08-30 | Choose the dispatch threshold by sweeping, not by estimating register pressure. | The estimate said 16384 fp32 values; the measurement said 32768. At the difference sits `N=9, D=2048`, the most representative shape, which was running at 616 GB/s instead of 1365 GB/s. |
| 2026-08-30 | Build a tiled backward rather than leaving an autograd fallback. | The fallback recomputed the forward before differentiating, so fwd+bwd was 0.51× the baseline at `N=9, D=4096` — an ordinary model shape. An unoptimized fallback was silently a regression. |
| 2026-08-30 | Judge correctness by relative L2 against a float64 oracle, with the dtype's rounding floor reported beside it. | Elementwise relative error is unbounded where the output crosses zero and reported `max_rel_err` of 5354 for a bit-accurate bf16 result. |
| 2026-08-30 | Scale benchmark inputs so `‖w‖₂ ≈ 1`. | Logits are distributed as `N(0, ‖w‖²)` because RMSNorm gives keys unit RMS. Drawing `w ~ N(0, 1)` puts the logit standard deviation at `sqrt(D)`, saturating the softmax into a regime the trained model never occupies and one that no implementation can reproduce stably. |
