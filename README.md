# blackwell-switchyard

This repository contains a fused Block Attention Residuals operator for NVIDIA Blackwell GPUs.
It also contains tests, benchmarks, raw results, and a Transformer integration.

The project measures three implementations:

- PyTorch eager mode
- PyTorch with `torch.compile` and Inductor
- Custom Triton kernels

Correctness has priority over speed.
Each timed implementation must pass a float64 oracle check.

## Current operator result

The main test shape is `N=9 B=1 T=4096 D=2048` with bf16 data.
The GPU is an NVIDIA RTX PRO 6000 Blackwell Max-Q.

`N` is the source count.
`B` is the batch size.
`T` is the token count.
`D` is the hidden dimension.

| Measurement | Max-autotuned Inductor | switchyard | Result |
|---|---:|---:|---:|
| Forward latency | 0.209 ms | **0.123 ms** | **1.70x faster** |
| Forward and backward latency | 1.354 ms | **0.365 ms** | **3.71x faster** |
| Kernel launches, forward | 3 | **1** | 2 fewer |
| Kernel launches, forward and backward | 13 | **4** | 9 fewer |
| Forward workspace | 0.312 MiB | **0 MiB** | 0.312 MiB less |
| Forward and backward workspace | 144.887 MiB | **0.008 MiB** | 144.879 MiB less |
| Error divided by the bf16 rounding floor | 1.00x | **1.00x** | equal |

The benchmark uses 100 forward measurements and 50 training measurements.
It flushes the 128 MiB L2 cache before each timed measurement.
It excludes compilation and autotuning from steady-state latency.

The switchyard forward takes 0.123 ms.
The measured traffic-only lower bound takes 0.119 ms.
Thus, the forward reaches 97 percent of this measured limit.

See [the raw representative result](results/operator_representative_bfloat16.json).

## Batched-query forward

Blocks can share one source tensor across several pseudo-queries.
The `block_attn_res_batched` function uses this reuse for supported resident shapes.

The following result uses `N=9 B=1 T=4096 D=2048` and eight queries.

| Implementation | Latency | Kernels | Temporary workspace | Error floor multiple |
|---|---:|---:|---:|---:|
| PyTorch eager | 0.891 ms | 9 | 16.6 MiB | 2.28x |
| Max-autotuned Inductor | 0.352 ms | 5 | 0.562 MiB | 1.74x |
| Eight switchyard calls | 0.934 ms | 8 | 0 MiB | 1.00x |
| **switchyard batched** | **0.199 ms** | **1** | **0 MiB** | **1.00x** |
| catswe phase 1 | 0.209 ms | 1 | 1.390 MiB | 1.00x |

The output-only switchyard path is 1.77x faster than max-autotuned Inductor.
It is 4.70x faster than eight separate switchyard calls.

This API has two important limits.
It does not return the merge statistics that the complete two-phase schedule needs.
It also does not provide a backward operation.

The resident path supports at most 16 queries.
It also requires `next_power_of_2(N) * next_power_of_2(D) <= 32768`.
Other shapes use an accurate per-query fallback.
The fallback is not a performance path.

See [the complete batched-query report](docs/batched_queries.md).

## Transformer result

The repository includes a 1.3 billion parameter decoder.
The test model has 24 layers, a hidden dimension of 2048, and eight blocks.
The test batch contains four sequences of 2048 tokens.

| Variant | Step time | Tokens per second | Peak memory | Residual mechanism share |
|---|---:|---:|---:|---:|
| Standard PreNorm residual | 428.89 ms | 19,101 | 30.33 GiB | control |
| Block AttnRes with framework operations | 704.54 ms | 11,627 | 31.87 GiB | 39 percent |
| **Block AttnRes with switchyard** | **484.21 ms** | **16,918** | **31.86 GiB** | **11 percent** |

The fused integration is 1.46x faster than the framework integration.
Block AttnRes adds 11 percent to the step time of the standard residual model.
It adds 1.53 GiB of peak memory.

The source arena avoids repeated `torch.stack` operations.
It reduces peak memory by 6.75 GiB compared with the stacking variant.

Two-GPU Distributed Data Parallel tests also pass.
The gradients are bit-identical after the all-reduce operation.
The measured scaling is 1.73x, or 87 percent efficiency.

See [the model report](docs/model.md).

## Operation definition

Attention Residuals replaces a fixed residual connection with attention over earlier states.
Block AttnRes groups the model depth into blocks.
The operator attends to block summaries for each token.

For one pseudo-query, the operation is:

```text
v: [N, B, T, D]
w: [D]

k      = v * rsqrt(mean(v * v, axis=D) + eps)
logits = dot(w, k)
alpha  = softmax(logits, axis=N)
out    = sum(alpha * v, axis=N)
```

The weighted sum uses the raw values in `v`.
Only the keys use root mean square normalization.
The softmax uses the source axis.
The operation does not use `1/sqrt(D)` attention scaling.

The operation has low arithmetic intensity.
Memory traffic and launch overhead control its performance.

## Python interface

Use the autograd operator for training:

```python
import torch

from switchyard.triton_op import block_attn_res_triton

v = torch.randn(
    9, 1, 4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
w = torch.zeros(2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = block_attn_res_triton(v, w)
out.sum().backward()
```

Use the batched function only for output-only forward inference:

```python
import torch

from switchyard.triton_op import block_attn_res_batched

v = torch.randn(9, 1, 4096, 2048, device="cuda", dtype=torch.bfloat16)
queries = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    outputs = block_attn_res_batched(v, queries)
```

## Implementation files

| File | Purpose |
|---|---|
| [`reference.py`](src/switchyard/reference.py) | Defines the operation and the float64 oracle. |
| [`baselines.py`](src/switchyard/baselines.py) | Defines framework baseline formulations. |
| [`triton_op.py`](src/switchyard/triton_op.py) | Defines two forward and two backward strategies. |
| [`model.py`](src/switchyard/model.py) | Defines the Transformer integration and source arena. |
| [`harness.py`](bench/harness.py) | Defines common timing, memory, accuracy, and kernel-count methods. |

The dispatch limit is 32768 resident values.
This limit comes from measurements on the target GPU.
The kernels use one pipeline stage.
The `dw` gradient uses fp32 accumulation.

## Correctness

The test suite compares results with a float64 oracle.
It reports relative L2 error and the data-type rounding floor.
It does not compare only with another low-precision implementation.

The suite has 129 tests with a GPU.
The CPU-only run has 23 passing tests and two skipped tests.

The tests cover these items:

- Forward and backward results
- Resident and tiled dispatch paths
- bf16, fp16, and fp32 data
- Non-power-of-two shapes
- Non-contiguous inputs
- Saturated logits
- First-order and second-order gradients
- Online softmax merging
- Transformer source schedules
- One-GPU and two-GPU integration

## Hardware and tools

The test system has two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs.
Each GPU has 95 GiB of memory and 128 MiB of L2 cache.
The GPUs use PCI Express and do not use NVLink.

Measured device values include:

- 1462 GB/s device-memory copy bandwidth
- 6064 GB/s L2 bandwidth
- 299 TFLOP/s bf16 throughput
- 9.3 microseconds of Triton launch overhead

Nsight Compute cannot access hardware counters on this host.
The project does not change host-wide driver settings.
It uses CUDA events, PyTorch Profiler, Nsight Systems, and byte models.

See [the machine report](docs/machine.md).

## Reproduction

The host does not provide the Python 3.12 development headers.
The setup script extracts matching headers into the repository.
It does not change the system Python installation.

```bash
scripts/fetch_python_headers.sh
source scripts/env.sh

python -m pytest tests/ -q
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q

python bench/bench_operator.py --set representative --dtype bfloat16
python bench/bench_third_party.py --batched-only --dtype bfloat16
python scripts/summarize_batched.py

scripts/fetch_third_party.sh
python bench/bench_third_party.py --dtype bfloat16
python scripts/summarize_third_party.py

python bench/bench_model.py
python bench/bench_ddp.py
python scripts/summarize_model.py
```

The benchmark JSON files contain the repository revision and third-party revisions.
They also contain the command, random seeds, software versions, and GPU model.

## Project status

The single-query forward is close to its measured traffic limit.
The output-only batched resident path is complete for its documented shape range.

The tiled backward path is the next optimization target.
Liger is faster for important large-width and large-source cases.
The current model points to extra source reads and weak cache reuse.

See [`PROJECT_STATE.md`](PROJECT_STATE.md) for the current work list.
See [issue 1](https://github.com/sabdulmajid/blackwell-switchyard/issues/1) for public tracking.

## Attribution

Attention Residuals is work from the Kimi Team at Moonshot AI.
See the [paper](https://arxiv.org/abs/2603.15031) and the [upstream repository](https://github.com/MoonshotAI/Attention-Residuals).

The upstream repository has no software license.
This project implements the published mathematics without upstream source code.
This project is independent from Moonshot AI.

```bibtex
@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Kimi Team},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2603.15031},
  primaryclass  = {cs.CL}
}
```

## License

This project uses the Apache License 2.0.
See [`LICENSE`](LICENSE).
