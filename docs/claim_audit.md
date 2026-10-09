# Historical claim audit

Audited on 2026-10-09. This audit reuses the stored measurements. It does not
remeasure unchanged kernels or validate experimental CUDA performance.
Run `python scripts/check_reported_claims.py` to recompute the arithmetic and
print the input artifact hashes, recorded environment, and available provenance.
The repository-local `verify-blackwell-switchyard` skill gives the verification recipes.

## What the stored results support

All three reports use bf16 on an NVIDIA RTX PRO 6000 Blackwell Max-Q (`sm_120`),
PyTorch 2.9.0+cu128, Triton 3.5.0, and Python 3.12.3. These are configuration-specific
historical comparisons, not claims about all Blackwell hardware or current CUDA candidates.

| Claim | Artifact and configuration | Definition and limit |
|---|---|---|
| 1.30B decoder integration | `results/model_bfloat16.json`; 24 layers, D=2048, FFN=5632, 16 heads, vocabulary=32768, eight AttnRes blocks | Each AttnRes variant has 1,300,535,296 parameters. The standard control has 1,300,334,592. Only the AttnRes variants have equal total counts. |
| 1.70x forward; 3.71x forward + backward | `results/operator_representative_bfloat16.json`; N=9, B=1, T=4096, D=2048 | `folded_compiled_autotune` versus `switchyard_triton`: 0.208896/0.122880 and 1.353696/0.364544 ms. This is an isolated operator comparison. |
| 704.54 to 484.21 ms/step; 11,627 to 16,918 tokens/s | `results/model_bfloat16.json`; B=4, T=2048 | Eager `folded_form` with arena versus fused Triton with arena. The full step includes gradient clearing, forward, loss, backward, and AdamW. Tokens/s = 8192 / seconds per step. It is not an end-to-end compiled-model comparison. |
| 39.1% to 11.4% | Same model records, standard control 428.886017 ms | Control-difference share of each variant's own step. Different denominators; neither is a profiler share nor overhead relative to the control. |
| Eight-query 1.436 TB/s | `results/batched_queries_bfloat16.json`; N=9, B=1, T=4096, D=2048, S=8 | Logical-minimum effective bandwidth: 285,245,440 bytes / 0.198655993 ms. Output-only inference; no merge statistics or backward. Not measured DRAM traffic. |

## The overhead denominators

Let `C=428.8860168457031`, `F=704.542724609375`, and `S=484.2066345214844` ms
be the stored control, framework-arena, and fused-arena step times.

| Measurement | Framework | Fused |
|---|---:|---:|
| Added step time, `variant - C` | 275.656708 ms | 55.320618 ms |
| Control-difference share, `100*(variant-C)/variant` | 39.125620% | 11.425002% |
| Overhead relative to control, `100*(variant-C)/C` | 64.272720% | 12.898676% |
| Attributed profiler region time / summed device time | 34.189513% | 7.395822% |

Total step-time reduction is `100*(F-S)/F = 31.273631%`.
Throughput improvement at the same token count is `F/S = 1.455046x`.
The former “3.4x reduction in overhead” divided the two percentages with different
denominators. Do not use it to describe the reduction in added milliseconds.

Control differencing includes source staging, autograd work, and indirect effects
against a different residual architecture. It estimates incremental cost; it is not
a direct timer around the mechanism. The profiler has narrower attribution and a
different denominator. The old artifact calls the step difference “wall clock”; the
implementation uses CUDA-event elapsed time. The original JSON is kept unchanged.

## Timing and work contracts

The common harness warms up the callable, synchronizes, flushes 384 MiB to evict
the 128 MiB L2, then records CUDA events around the callable. Cache eviction,
compilation/autotuning, oracle checks, and profiling are outside the reported region.
Device event time can include gaps between launches; it is not the sum of kernel times.
Tensor allocations and operations performed by the callable are included. Inputs
are constructed before timing. Operator training includes gradient clearing and a
new forward/backward; it does not include an optimizer. Model full-step timing does.

| Run | Warmup per timing block | Timed samples | Stored within-run spread |
|---|---:|---:|---|
| Operator forward | 25 | 100 | Inductor p10–p90 0.208864–0.210944 ms; fused 0.120832–0.126976 ms |
| Operator forward + backward | 25 | 50 | Inductor 1.336320–1.433600 ms; fused 0.360448–0.368640 ms |
| Model full step | 5 | 20 | Framework 702.520325–705.781799 ms; fused 482.934814–485.224487 ms |
| Eight-query fused forward | 20 | 60 | 0.196608–0.200704 ms |

These are sequential, within-run summaries. No per-sample times, alternating trial
schedule, or independent-run distribution is present in these historical artifacts.
The old harness used the upper-middle order statistic for even sample counts.
Future runs use the arithmetic median to match the raw-sample validators. The old
medians cannot be recalculated without their original samples.

The stored operator's `backward_only_ms` is the difference of forward-plus-backward
and forward medians. It is not a directly timed backward. The model's backward and
optimizer estimates have the same limitation. The new candidate harness times
backward directly on a retained graph, with setup outside the timed region.

Both framework formulations and their compile modes remain useful baselines. The
operator artifact identifies `folded_compiled_autotune` and contains compiled kernel
profiles (three forward kernels, 13 forward-plus-backward kernels). It lacks the
newer explicit compiler-counter verification fields. A `compiled=true` label alone
does not prove that compilation succeeded. New claims require those fields and
kernel identities; compilation fallback at other shapes must remain visible.

The eight-query compiled baseline returns the same `[S,B,T,D]` tensor. Eight separate
calls return a list of tensors; stacking is outside their timed region. Catswe phase 1
also produces merge/backward auxiliaries and is an observational comparison, not an
identical-work baseline. FLA's native independent source leaves must be preserved.
The historical operator correctness records check forward against float64. They do
not independently establish gradient accuracy; the separate backward tests are required.

## Provenance and reproducibility gaps

| Report | Timestamp (UTC) | Recorded source revision | What is missing |
|---|---|---|---|
| Model | 2026-08-31 12:50:45 | None | Commit/tree, invocation, seeds, current provenance, raw samples, paired/independent trials |
| Representative operator | 2026-09-01 22:56:45 | `43c5a156bf56fbcec583864699ec98fda9c42ad9` | Tree and full dirty-state binding, raw samples, paired/independent trials, explicit compiler verification fields |
| Batched queries | 2026-09-01 22:53:56 | `d888c1d1cb167d11b9d888b7e3118f202404da30` | Tree and full dirty-state binding, raw samples, paired/independent trials, explicit compiler verification fields |

Both recorded source SHAs resolve locally and their source was inspected. Operator
and batched provenance record invocation and `tracked_worktree_dirty=false`.
Batched provenance correctly records source seed 0 and query seed 1, and pins the
third-party revisions. The operator metadata incorrectly says query seed 1: its
source seeds once with zero, then draws sources and queries from the same stream.
Future reports describe each benchmark's actual RNG policy; the shared provenance
helper no longer invents seeds. The old operator `compile_seconds` measures a call
after the correctness check already compiled it, not cold compilation. The current
harness instead labels the combined initial-check time explicitly.
The files do not record driver or GPU clock/power conditions. Do not fill historical
gaps using today's environment or relabel these files as results from today's commit.

The stored fused-arena profiler aggregation sums inclusive parent and nested backward
regions. It can double-count device time. The 7.4% value above is the reported value,
not a validated direct attribution. The corrected aggregator needs a fresh GPU profile
before a new attribution claim. Neither old profiler share independently confirms the
control-difference percentages. The end-to-end timing is measured separately.

The N- and D-sweep artifacts contain the same operator shape with about 3.30x training
speedup, rather than 3.71x. They use a different revision and omit profiles; the D-sweep
also records a dirty worktree. These are evidence of historical variation, not controlled
replications. The operator summarizer selects one record when shapes repeat; it does
not aggregate these observations. Do not present the representative ratio as universal.

Reproduction recipes are in the verification feature map and the existing benchmark
CLIs. Write new results to new paths with a clean commit/tree, full provenance, raw
samples, timing order, correctness, profiler identities, and spread. For a new
acceptance decision, follow the benchmark checklist; the backward campaign already
uses 15 paired trials of 13 samples. A model adoption claim still needs a tuned full-model
compiled comparison and repeated full-step measurements. The 1.46x historical claim
is limited to the measured eager framework baseline.

## Pending, not accepted

- Experimental CUDA backward and one-read training plans: offline compiler evidence
  is not runtime correctness or speed evidence. The previous guarded attempt stopped
  on a collision and produced no usable candidate report. No dispatch promotion follows
  from this audit.
- Revised NCCL/DDP check: the stored DDP report only checks post-all-reduce rank
  agreement. Run the new manual-average check on both ranks under a two-device
  guard, first with the small model. Use the verification skill's all-GPU telemetry
  route with explicit two-device child visibility, not the backward campaign's
  target-device-only activity gate. Sampled collision detection is not a reservation.
- Release: regenerate affected correctness/performance evidence and review the regression
  baseline. Do not automatically accept a new baseline, merge, or weaken tolerances.
