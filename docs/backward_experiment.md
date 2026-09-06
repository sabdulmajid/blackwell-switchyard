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

| Plan | Backward source reads | Forward state | `dw` reduction | Purpose |
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
| `cuda_register_cluster` | **one** | three FP32 source scalars | one contribution per persistent cluster | register candidate for wide gaps |
| `cuda_register_cluster_full` | **one** | three FP32 source scalars | one contribution per persistent cluster | one-read forward and backward candidate |

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

## Packed-register candidates

The `cuda_register` plan removes the cluster's source-sized shared-memory tile.
A persistent 512-thread CTA keeps raw bf16 or fp16 source pairs in registers.
It reduces all `g dot v` values with two block barriers per token, applies both gradients from
the retained values, and accumulates `dw` in uniquely owned FP32 shared-memory slots across
its token stream. The retained source pairs stay in registers. The `(N=9, D=4096)`
specialization is spill-free. The compiler spilled the `(9,8192)` and `(32,2048)` variants, so
those variants were removed before GPU measurement. The offline compiler gate rejects any
future specialization that spills retained values to local memory.

The `cuda_register_cluster` plan covers the two remaining gap shapes. It uses four 512-thread
blocks for `(N=9, D=8192)` and two blocks for `(N=32, D=2048)`. Each block holds one feature
shard in registers. Only source-sized FP32 reductions cross distributed shared memory. The
output gradient and the persistent query-gradient accumulator use local shared memory. This
keeps each source value in registers across both gradient equations without exceeding the
128-register limit. It also avoids the source-sized shared-memory tile used by the general
feature-cluster plans.

The `cuda_register_cluster_full` plan pairs that backward with a complete one-read forward.
The accepted saved-state forward reads the source stack twice at these shapes. This extra read
makes the full-training target physically too tight at `(N=9, D=8192)`, even if the backward
reaches its traffic floor. The custom forward removes that read. It calculates the RMS and query
dot products in FP32, applies the softmax over sources, saves the three exact backward
coefficients, and produces a weighted sum of raw source values. It does not use attention
scaling.

The first register-only `(N=9, D=8192)` forward passed the spill check, but disassembly showed
that the compiler reloaded eight sources from global memory. That version was rejected. The
current specialization keeps all nine sources in a 36,864-byte shared-memory tile and uses 38,052
dynamic shared-memory bytes per block. The `(N=32, D=2048)` forward keeps 28 sources in registers
and keeps four sources in an 8,192-byte shared-memory tile. It uses 12,416 dynamic shared-memory
bytes per block. Both layouts avoid compiler spills and read each source from global memory one
time. The query shard stays in registers across the persistent token loop. Only source reduction
scalars and the final softmax weights cross distributed shared memory.

The packed kernels load two low-precision values at a time. They require four-byte base
alignment. A valid contiguous tensor can start at an odd two-byte storage offset, so the Python
operator makes an aligned copy only in that case. The CUDA entry point also rejects a misaligned
base pointer. Adversarial tests cover odd offsets, `B>1`, tied logits, saturated logits, and both
supported dtypes.

The offline `sm_120` compiler gate reports:

| Kernel | Registers per thread | Stack | Local memory | Static shared memory |
|---|---:|---:|---:|---:|
| one-block shared, bf16/fp16 | 48 | 0 | 0 | 1024 bytes |
| two-block feature cluster, bf16/fp16 | 64 | 0 | 0 | 1024 bytes |
| four-block feature cluster, bf16/fp16 | 72 | 0 | 0 | 1024 bytes |
| `(9,4096)` register CTA, bf16/fp16 | 128 | 0 | 0 | 1024 bytes |
| `(9,8192)` four-block register cluster, bf16/fp16 | 128 | 0 | 0 | 1024 bytes |
| `(32,2048)` two-block register cluster, bf16/fp16 | 128 | 0 | 0 | 1024 bytes |
| `(9,8192)` one-read forward, bf16/fp16 | 96 | 0 | 0 | 1024 bytes |
| `(32,2048)` one-read forward, bf16/fp16 | 103 / 109 | 0 | 0 | 1024 bytes |

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
uses `cuobjdump` to record registers, stack, local memory, static shared memory, and static global
load instructions. It fails on compiler-reported local storage. It also fails if the generated
register-cluster kernels do not have the exact global load count from the audited one-read
schedule. The command does not initialize or query a GPU.

## GPU experiment order

Use exclusive access. Do not start with the full matrix.

1. Run adversarial gradient tests for bf16 and fp16.
2. Run a short compile and launch smoke test for each supported plan.
3. Measure the gap shapes. Compare `current`, Liger, the portable candidate, and all one-read
   candidates.
4. Drop dominated plans.
5. Run the bounded crossover matrix only for the survivors.
6. Run the complete dtype, memory, kernel-count, and Transformer regressions before dispatch.

The evaluator makes the final decision for each measured shape and dtype. A failure in
correctness, provenance, traffic accounting, kernel structure, or measurement quality invalidates
the complete report. A valid performance loss at one shape does not erase a valid win at another
shape. Production dispatch can use only the exact shape and dtype cells that clear all gates.

The benchmark uses 15 interleaved trials with 13 samples per trial. Reversed adjacent orders
balance which implementation in each pair runs first. A dispatch cell must beat both the current
path and Liger in all 15 independent trial medians. Across the 264 candidate, shape, dtype, and
comparator hypotheses and three permitted campaign attempts, the conservative Bonferroni
sign-error bound is less than 0.05.

The benchmark must store the exact training plan, an opaque campaign device ID, kernel names,
main and auxiliary kernel launch counts, saved-state bytes, workspace, raw samples, and trial
order. The publication gate reconstructs the timing statistics from the raw samples. It rejects
missing profiler data, missing memory data, and a mismatch between the recorded schedule and the
raw trial order. The benchmark must record the GPU process state before and after each run. It
must check output, `dv`, and `dw` against the float64 oracle before timing.

`cuda_shared` is a recomputation control. It must pass the fixed float64-oracle bounds, but the
smoke gate does not require it to match the accepted path's error floor. The promotion evaluator
still rejects it if recomputation increases the error by more than 5%. All saved-state candidates
must meet both checks.

## Guarded unattended campaign

[`run_when_gpu_idle.py`](../scripts/run_when_gpu_idle.py) can wait for the shared host without
using a GPU. The configured campaign does not trust an estimated finish time. It requires all
GPUs to have no compute process at every 60-second sample for 30 minutes. It then makes only
one GPU visible to the campaign.

The benchmark samples the selected GPU every 0.05 seconds while it runs. It exits immediately
if a sample sees an unrelated process. The outer runner also samples every 0.25 seconds. It stops
its process group within a two-second grace period. It then waits for a new 30-minute sampled idle
interval. It makes at most three
attempts. A retry keeps each clean, complete benchmark phase and reruns only the interrupted
phase and later phases. State and logs stay outside the repository, so they cannot make
benchmark provenance dirty. The monitor stores a timestamp and a foreign-process count. It
does not store process names or process identifiers. Committed command arguments redact
external absolute paths. The runner gives result validation, commit, and push up to 30 minutes
after GPU work.

The runner keeps its attempt count across process restarts. A completed GPU phase uses a separate
CPU-only publication retry. This retry does not wait for another idle GPU interval and does not
consume another GPU attempt. Each decision hashes the exact byte buffers that it parsed. Before
commit, the campaign reconstructs every decision from the staged
Git blobs. It repeats that check against the committed blobs before it pushes. Recovery accepts
only the exact 16-file bundle for one campaign ID. It reconstructs every gate and decision before
it retries a push. Publication uses the literal canonical repository URL after checking that the
configured remote has one matching fetch URL and one matching push URL.

The manifest records one guard attestation for each attempt that produced reusable evidence.
Each benchmark phase records its generating attempt. Bundle validation binds the phase timestamp
to that attempt's idle interval and guarded launch. Public evidence uses a random campaign device
ID. The physical GPU UUID remains only in external guard state and is never committed.

A deterministic smoke, correctness, schema, or provenance failure stops the campaign. It does
not spend another GPU attempt on the same inputs. Only a collision, interrupted phase, or
measured statistical instability can request a fresh idle interval and another attempt.

This is high-frequency sampled detection, not hardware-enforced exclusive mode. A foreign CUDA
context that exists for less than one sample interval could escape detection. The reports mean
that no competing process appeared in any preflight, postflight, or monitor sample.

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
