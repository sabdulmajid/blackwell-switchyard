# Project state

Living status document for `blackwell-switchyard`. Updated as work lands. The authoritative
per-change detail lives in pull requests; this file is the index.

**Last updated:** 2026-08-30

---

## Current best result

> _Nothing measured and reviewed yet._ This section will hold the headline number
> (implementation, shape, baseline, measurement method) as soon as one exists.

## Current best implementation

> _None yet._

---

## DONE

- Repository bootstrap: license, layout, environment capture plan, project state tracking.
- Environment ground truth established (see `docs/machine.md` once landed):
  2 x RTX PRO 6000 Blackwell Max-Q, CC 12.0 (sm_120), 188 SMs, 128 MiB L2, 95 GiB each,
  PCIe-only between GPUs (no NVLink). torch 2.9.0+cu128, triton 3.5.0, nvcc 12.8, ncu 2025.1.1, nsys 2024.6.2.
  No package installation required — the existing interpreter already has the full toolchain.

## IN PROGRESS

- Investigation phase: exact Block AttnRes semantics from arXiv:2603.15031 + upstream
  `MoonshotAI/Attention-Residuals`, prior-art/novelty check, hardware characterization,
  and a `torch.compile`/Inductor baseline probe.

## NEXT

- Canonical paper-faithful reference implementation + correctness oracle.
- Benchmark + profiling harness with machine-readable results.
- Performance model, then the first evidence-selected optimization.

## BLOCKED

- _Nothing blocked._

## OPEN QUESTIONS FOR THE OWNER

- _None requiring a decision yet._ Technical choices (Triton vs CUDA, layout, fusion boundaries)
  are being resolved by measurement, not by asking.

---

## Milestones

| # | Milestone | State | PR |
|---|-----------|-------|----|
| 0 | Bootstrap + environment characterization | in progress | — |
| 1 | Faithful reference + correctness suite | not started | — |
| 2 | Benchmark & profiling infrastructure | not started | — |
| 3 | Strong framework baselines (eager / `torch.compile`) | not started | — |
| 4 | First optimized operator (evidence-selected) | not started | — |
| 5 | Backward / autograd optimization | not started | — |
| 6 | Blackwell-specific tuning | not started | — |
| 7 | Transformer integration + training smoke test | not started | — |
| 8 | Two-GPU validation | not started | — |
| 9 | Final benchmark study + technical report | not started | — |

## Decision log

Decisions that changed direction, with the evidence that forced them. Append-only.

| Date | Decision | Evidence |
|------|----------|----------|
| 2026-08-30 | Use the existing system interpreter rather than building a virtualenv. | `pip` into the shared pyenv is permission-denied, and every dependency the project needs (torch 2.9.0+cu128, triton 3.5.0, numpy, scipy, pandas, matplotlib, pytest, cuda-python) is already present. Zero installs is also the lowest-risk option for a shared machine. |
