# Backward architecture experiment

This experiment targets the large-shape training path.
It does not change production dispatch.

## Why the current path loses

The accepted tiled backward uses two kernels.
`_bwd_stats` reads all source values and calculates three FP32 values for each source and
token. `_bwd_apply` starts after a grid-wide boundary. It reads the source values again,
writes `dv`, and reduces `dw`.

For one input element size `e`, define:

```text
M = B * T
X = M * D * e
```

An exact backward must read `v` and `g`, then write `dv`. Its minimum large-tensor traffic is:

```text
(2 * N + 1) * X
```

The current split path moves:

```text
(3 * N + 2) * X
```

The source stacks at the three gap shapes are 288 to 576 MiB. They do not fit in the 128 MiB
L2 cache. The grid boundary therefore turns the second source pass into device-memory traffic.
Tile-size changes cannot remove this traffic.

Stored measurements show the gap:

| Shape | switchyard forward and backward | Liger forward and backward |
|---|---:|---:|
| `N=32 B=1 T=4096 D=2048` | 1.565 ms | 1.502 ms |
| `N=9 B=1 T=4096 D=4096` | 0.920 ms | 0.774 ms |
| `N=9 B=1 T=4096 D=8192` | 1.961 ms | 1.435 ms |

Liger Kernel is LinkedIn's open-source Triton kernel library for language-model training.
Its AttnRes backward assigns one program to each token. It saves softmax weights and inverse
RMS values in the forward pass. Its two source passes occur in one program, so the second pass
can use L2. Liger also calculates an extra RMSNorm-gain gradient that this project does not
need. Its win shows that source locality is more important than the current kernel split.

Liger is already close to the measured two-read bandwidth model. A source-serial Triton kernel
can mainly match that design. The only path to a lower traffic floor is to keep each source
value on chip until both gradient equations use it.

## Complete training plans

[`training_plan.py`](../src/switchyard/training_plan.py) defines immutable forward and backward
contracts. An experiment cannot select a backward family without also selecting its required
saved state, token ownership, and `dw` reduction.

The plans are:

| Plan | Source reads | Forward state | `dw` reduction | Purpose |
|---|---:|---:|---|---|
| `auto` | accepted behavior | none | measured production strategy | production control |
| `serial_recompute_atomic_t1` | two | none | one atomic contribution per token | Liger-like diagnostic |
| `serial_recompute_atomic_t4` | two | none | one contribution per four tokens | grouped locality control |
| `serial_saved_atomic_t4` | two | three FP32 source scalars | one contribution per four tokens | saved-state control |
| `serial_saved_partials_t16` | two | three FP32 source scalars | private rows, then deterministic reduction | portable candidate |
| `cuda_shared` | one | none | one contribution per token | one-block traffic control |
| `cuda_cluster` | **one** | three FP32 source scalars | one contribution per persistent cluster | primary candidate |
| `cuda_cluster4` | **one** | three FP32 source scalars | one contribution per persistent cluster | lower shared-memory pressure |
| `cuda_register` | **one** | three FP32 source scalars | one contribution per persistent CTA | fixed-shape register candidate |

The saved fields are `alpha`, `rstd`, and:

```text
norm_coefficient = dot(query, value) * rstd**3 / D
```

The forward already calculates these values. Their total size is 0.422 MiB for `N=9, T=4096`
and 1.5 MiB for `N=32, T=4096`. They let the one-read cluster calculate the exact backward
after only one `g dot v` reduction.

## Primary one-read design

The `cuda_cluster` and `cuda_cluster4` plans use persistent two-block and four-block
thread-block clusters.

1. Each block owns a disjoint shard of `D`.
2. Each block loads its source shard into shared memory once.
3. Each block calculates its part of `g dot v`.
   Source waves use all eight warps and finish through private warp partials.
   This needs one block barrier per token instead of two barriers per source.
4. Rank zero's first warp combines the small per-source scalars through distributed shared
   memory. One lane owns each source.
5. All blocks calculate `dv` and `dw` from the retained source values.
6. The cluster repeats this work for a strided set of tokens.
7. Each cluster adds one accumulated FP32 `dw` contribution.

This layout has one global reader for each source element and each output-gradient element.
It reaches the `(2 * N + 1) * X` large-tensor traffic lower bound. Feature sharding also keeps
the shared-memory need below the measured 99 KiB per-block limit at all gap shapes. The
four-block plan roughly halves the per-block source tile. It trades two more blocks and wider
DSM aggregation for higher possible occupancy. The GPU campaign decides that trade by
measurement.

The CUDA source is in
[`shared_backward.cu`](../src/switchyard/csrc/shared_backward.cu). The package builds it only
when a private candidate entry point runs. Public imports and production dispatch do not build
the extension. The current extension contains `sm_120` code and rejects other GPU architectures
explicitly. The custom operator supports first-order training gradients. Use the framework
reference when an application requires second-order gradients.

## Packed-register candidate

The `cuda_register` plan removes the cluster's source-sized shared-memory tile.
A persistent 512-thread CTA keeps raw bf16 or fp16 source pairs in registers.
It reduces all `g dot v` values with two block barriers per token, applies both gradients from
the retained values, and accumulates `dw` in uniquely owned FP32 shared-memory slots across
its token stream. The retained source pairs stay in registers. The `(N=9, D=4096)`
specialization is spill-free. The compiler spilled the `(9,8192)` and `(32,2048)` variants, so
those variants were removed before GPU measurement. The offline compiler gate rejects any
future specialization that spills retained values to local memory.

The offline `sm_120` compiler gate reports:

| Kernel | Registers per thread | Stack | Local memory | Static shared memory |
|---|---:|---:|---:|---:|
| one-block shared, bf16/fp16 | 48 | 0 | 0 | 1024 bytes |
| two-block feature cluster, bf16/fp16 | 64 | 0 | 0 | 1024 bytes |
| four-block feature cluster, bf16/fp16 | 72 | 0 | 0 | 1024 bytes |
| `(9,4096)` register CTA, bf16/fp16 | 128 | 0 | 0 | 1024 bytes |

Dynamic shared memory depends on `N` and `D`. The runtime adds static and dynamic memory before
it accepts a launch.

## Portable candidate

The grouped source-serial Triton kernel keeps adjacent source passes in one program. It can use
L2 for its second pass. It processes several tokens in sequence and accumulates one full-width
`dw` vector.

The hierarchical variant writes one FP32 row per 16-token group. A small feature-tiled kernel
then reduces these rows in a deterministic order. This removes global atomics from the main
kernel. It does not remove the second source read, so it is a portability path and an ablation,
not the expected winner. The source-serial plans stop at `D=4096`. Recompute plans also stop at
16 sources. Larger full-width Triton programs spill, so those cases are explicitly unsupported.

## Offline gate

Run this command without a visible GPU:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES="" python scripts/compile_candidates.py
```

The gate compiles the target Triton specializations and the CUDA extension for `sm_120`. It
uses `cuobjdump` to record registers, stack, local memory, and static shared memory. It fails on
compiler-reported local storage. The command does not initialize or query a GPU.

## GPU experiment order

Use exclusive access. Do not start with the full matrix.

1. Run adversarial gradient tests for bf16 and fp16.
2. Run a short compile and launch smoke test for each supported plan.
3. Measure the gap shapes. Compare `current`, Liger, the portable candidate, and all one-read
   candidates.
4. Drop dominated plans.
5. Run the bounded crossover matrix only for the survivors.
6. Run the complete dtype, memory, kernel-count, and Transformer regressions before dispatch.

The benchmark must store the exact training plan, device UUID, kernel names, main and auxiliary
kernel launch counts, saved-state bytes, workspace, raw samples, and trial order. It must
record the GPU process state before and after each run. It must check output, `dv`, and `dw`
against the float64 oracle before timing.

## Guarded unattended campaign

[`run_when_gpu_idle.py`](../scripts/run_when_gpu_idle.py) can wait for the shared host without
using a GPU. The configured campaign does not trust an estimated finish time. It requires all
GPUs to have no compute process at every 60-second sample for 30 minutes. It then makes only
one GPU visible to the campaign.

The benchmark checks the selected GPU every 0.25 seconds while it runs. It exits immediately
if an unrelated process appears. The outer runner also checks every two seconds and stops its
process group. It then waits for a new 30-minute sampled idle interval. It makes at most three
attempts. A retry keeps each clean, complete benchmark phase and reruns only the interrupted
phase and later phases. State and logs stay outside the repository, so they cannot make
benchmark provenance dirty. The monitor stores a timestamp and a foreign-process count. It
does not store other process names. Committed command arguments redact external absolute
paths. The runner gives result validation, commit, and push up to 30 minutes after GPU work.

The campaign is fail-fast and uses this order:

1. Run all adversarial candidate tests.
2. Run quick bf16 and fp16 gate measurements for all candidate families.
3. Apply the smoke correctness, numerical-error, kernel-count, and provenance gate.
4. Run the full 12-shape bf16 and fp16 matrix for all candidates and controls.
5. Run the portable fp32 control matrix.
6. Apply the deterministic promotion evaluator to each candidate.
7. Commit and push only the raw reports, manifest, and decisions with the configured project
   identity.

The campaign stops if an evaluator requests more data. It can save only a terminal `DROP`,
`REJECT`, or `READY_FOR_DISPATCH_REVIEW` decision.

The campaign never changes production dispatch and never merges `main`. A measured result
still needs an engineering review before a separate dispatch commit.

## Promotion rules

Do not loosen a correctness tolerance to admit a candidate. The numerical error must remain
within 1.05 times the accepted path unless a documented operation-order difference explains a
smaller absolute bound.

A source-serial plan must not enter production only because it matches Liger. The primary
question is whether the one-read plan produces a material and repeatable result below Liger's
two-read floor. The expected physical ceiling is about 1.35 to 1.43 times the current backward
and about 1.10 to 1.14 times Liger at the gap shapes. Treat this as an opportunity bound, not a
performance claim.

If no one-read plan beats Liger, inspect achieved bandwidth, cluster occupancy, barriers, and
shared-memory transactions before changing tile constants. If the architecture fails after
that analysis, remove it cleanly and keep the evidence.
