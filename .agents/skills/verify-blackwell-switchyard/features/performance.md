# Performance evidence

Users compare the same Block AttnRes task across tuned frameworks and custom
implementations, with the configuration and limitations attached to each number.

## Sub-features

- `operator`: isolated forward and forward-plus-backward against both framework forms.
- `batched`: one output tensor from multiple queries; inference only.
- `backward`: direct backward timing and complete training-plan selection.
- `audit`: arithmetic and provenance checks on stored evidence without new GPU work.

## How to get to it (user POV)

Use `bench/bench_operator.py`, `bench/bench_third_party.py`, and
`bench/bench_backward.py`. Read `docs/claim_audit.md` before reusing old headlines.
Use `docs/backward_experiment.md` for the canonical guarded campaign command.

## Driving it with the existing benchmark tools

Preconditions: parent doctor passes, a clean committed measurement revision,
matching hardware/software/calibration, bounded external outputs, and exclusive
sampled GPU availability. Read `bench/harness.py` and the selected CLI first.

- **Historical audit:** `python scripts/check_reported_claims.py` checks the
  equations against the three stored reports and emits configuration/provenance.
  Require exit zero; inspect `docs/claim_audit.md` for evidence gaps before citing.
- **Framework anchor:** after the GPU guard, run
  `python bench/bench_operator.py --set representative --dtype bfloat16 --impls paper_eager,folded_eager,paper_compiled,folded_compiled,folded_compiled_autotune,folded_compiled_cudagraph,switchyard_triton --out "$VERIFY_EVIDENCE_DIR/operator.json"`.
  Require forward oracle success and the matching backward tests before claiming
  training correctness. Compilation flags alone do not prove compilation:
  inspect `compiled_execution_verified`, `compiled_fwd_bwd_verified`, fallback
  flags, and profiler kernel identities. Require the intended Triton kernel names.
- **Batched anchor:** use pinned dependencies from `scripts/fetch_third_party.sh`,
  then `python bench/bench_third_party.py --batched-only --dtype bfloat16 --out "$VERIFY_EVIDENCE_DIR/batched.json"`.
  Inspect each query's oracle result, kernel identity, and workspace. Require one
  compute kernel only for resident shapes; the D=4096 case exercises fallback.
  Compare against the compiled batched form with the same `[S,B,T,D]` output.
  Separate calls return a list; disclose that contract. Catswe returns additional
  statistics and is observational. The FLA comparator needs independent source
  leaves, not a list of slices from one autograd leaf.
- **Backward promotion:** use the unchanged linked-worktree campaign in
  `docs/backward_experiment.md`. It runs adversarial tests, quick bf16/fp16
  screening, the 12-shape full matrix, and fp32 portable controls. It invokes
  `check_backward_smoke.py`, `check_backward_report.py`, `evaluate_backward.py`,
  `check_evidence_binding.py`, and `check_backward_bundle.py` with the exact
  measured revision/tree and report paths. Preserve the 30-minute idle gate,
  60-second idle samples, collision monitors, three-attempt limit, and deadline.
  Use local `liger_exact` for matching two-input work; upstream Liger also
  calculates an RMS gain gradient and remains observational. A candidate must
  beat current and exact-contract Liger in every paired trial for direct backward
  and forward-plus-backward. Require at least 1.10x backward over current, 1.05x
  training over current, 1.05x backward/training over exact Liger, CV at most 5%,
  and the existing forward-regression/workspace gates. Read the evaluator's exact
  thresholds; a terminal `DROP` or `REJECT` does not authorize promotion.

## Gotchas

The common timing harness warms up, uses CUDA events and explicit synchronization,
and flushes 384 MiB outside the timed region to evict 128 MiB L2. Profiles and
allocation peaks are separate runs. Inspect each callable for included allocations,
gradient clearing, layout copies, and saved-state/recomputation costs. Legacy
operator `backward_only_ms` subtracts medians; the new backward benchmark measures
backward directly on a retained graph with setup before the flush/timing.

Full backward runs store 15 interleaved trials of 13 samples, order, median,
min/p10/p90, and CV; derive the full range from raw samples. Quick runs have only
two trials and cannot promote. Operator/batched CLIs still emit summaries without
paired raw trials. Sequential legacy
summaries lack raw samples and independent-run variation; keep them historical.
Do not replace missing repeatability with a rerun of every unchanged experiment.
Before a new release/adoption claim, add repeated evidence for the affected path.

Explain the limiter using separate profiles and measured calibration, including
bytes, launches, register/shared-memory use, and L2 reuse. A hardware model is a
bound, not an observed speedup. Report effective bandwidth as logical bytes/time.
Pair microbenchmarks with the training recipe; do not infer full-step improvement
from an operator winner. Expected unsupported-plan skips are not candidate coverage.
Profiler loss, missing required shapes/baselines, errors,
compiler fallback, contamination, or inconsistent provenance invalidate the claim.
