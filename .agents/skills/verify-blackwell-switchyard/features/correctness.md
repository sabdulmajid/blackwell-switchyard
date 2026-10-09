# Operator correctness

Users mix raw source tensors with source-axis softmax weights from RMS-normalized
keys. The public reference and optimized APIs must implement the same mathematics.

## Sub-features

- `reference`: float64 oracle, source-axis semantics, no `1/sqrt(D)` scaling.
- `training`: production forward, `dv`, and `dw` across both dispatch strategies.
- `batched`: output-only inference, resident range, and accurate fallback.
- `candidate`: experimental training plans remain isolated until all gates pass.

## How to get to it (user POV)

Use `switchyard.block_attn_res_reference`, `block_attn_res_oracle`,
`block_attn_res_triton`, `BlockAttnResTriton`, or `block_attn_res_batched`.
Experimental plans are selected explicitly by `bench/bench_backward.py`.

## Driving it with pytest

Preconditions: parent doctor passes; GPU recipes run under the existing idle and
collision guard. Keep output logs/JUnit in `VERIFY_EVIDENCE_DIR`.

- **Reference:** `env -u SWITCHYARD_TARGET_GPU_UUID CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_reference.py -q`.
  Require passing semantic, numerical Jacobian, second-order reference, odd-shape,
  invalid-input, and online-softmax merge tests.
  Numerical Jacobian uses `eps=1e-6, atol=1e-8, rtol=1e-6`; second-order checks
  use `atol=1e-6, rtol=1e-5`. Merge checks use `rtol=atol=1e-12` (`1e-9` for extremes).
- **Production:** `python -m pytest tests/test_triton_op.py -q`.
  This covers the public function/module and both launch strategies. Forward tests
  cover bf16/fp16/fp32 with `rel_L2 <= max(1.6 * dtype_floor, 1e-6)`;
  batched output and each query use `max(1.05 * dtype_floor, 1e-6)`. Production backward's 16-shape
  matrix is bf16 with `dv < 0.03`, `dw < 0.06`. Single-source `dw` must be exactly
  zero. Do not claim this is a complete fp16/fp32 production-backward sweep.
- **Candidates:** `python -m pytest tests/test_backward_candidates.py -q`.
  Require masked tails, persistent-loop reuse, aligned/unaligned inputs, saved
  coefficients, uniform/saturated scores, and dedicated single-source checks.
  The suite requires `sm_120`, even for portable plans. Most cases cover bf16/fp16;
  the Liger checks also cover fp32. Single-source Liger asserts exact `dv` and zero
  `dw`; the listed non-register plans assert zero `dw` in bf16. Do not extend
  these claims to unsupported register-plan shapes. Saved `alpha/rstd` relative-L2
  error is at most `3e-5`, norm coefficient at most `2e-4`; repeated results are exact.
  The campaign then checks output and both gradients against float64 on each timed
  shape for seeds `0,1,2`. Bounds `(output,dv,dw)` are bf16 `(0.02,0.03,0.06)`,
  fp16 `(0.005,0.01,0.03)`, fp32 `(2e-5,1e-4,1e-3)`.
  Promotion requires each relative-L2 error to be at most
  `max(1.05 * accepted_error, 1e-7)`. `cuda_shared` is a non-promotable control.
  Preserve all special zero-oracle checks.
- **Epsilon and plan contracts:** `env -u SWITCHYARD_TARGET_GPU_UUID CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_input_validation.py tests/test_training_plan.py -q`.
  Require epsilon representability and valid plan invariants/support. Tensor rank,
  size, and dtype rejection is covered by the reference and GPU production suites above.

## Gotchas

The tests exercise non-power-of-two `N/D/T`, singleton sources/tokens, batch > 1,
noncontiguous inputs, zero query, saturation, and the measured 32768 dispatch
boundary. Keep fp32 `dw` accumulation. Production backward compares with the
original unrounded float64 inputs; candidate/campaign checks use the same quantized
inputs and upstream gradient as the GPU. These include different error sources.
Do not loosen bounds to hide a mismatch.
The batched API rejects gradient-enabled inputs that require grad; `no_grad` is allowed.
Resident execution needs `S<=16` and `next_pow2(N)*next_pow2(D)<=32768`.
The numerical test matrix does not cover the 16/17-query boundary.
Its fallback is correct but can be much slower. An import-level CUDA skip is
not proof of the optimized path. The shared `check_against` helper alone does
not establish zero-norm gradient correctness; use the dedicated assertions.
