# Batched-query forward results

This report uses `results/batched_queries_bfloat16.json`.
The generator is `scripts/summarize_batched.py`.

The device is NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. The data type is bf16. Each value is the median of 60 CUDA-event measurements.
The benchmark flushes the 128 MiB L2 cache before each measurement.
The benchmark excludes compilation and autotuning from steady-state latency.
Each implementation must pass the float64 oracle check before timing.

## Query-count sweep

The source shape is `N=9 B=1 T=4096 D=2048`.
The compiled baseline uses `max-autotune-no-cudagraphs`.

| queries | eager ms | compiled ms | separate calls ms | switchyard batched ms | catswe phase 1 ms | batched vs compiled | batched vs calls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.541 | 0.213 | 0.121 | 0.121 | 0.123 | 1.76x | 1.00x |
| 2 | 0.870 | 0.281 | 0.244 | 0.133 | 0.133 | 2.11x | 1.83x |
| 4 | 0.877 | 0.299 | 0.481 | 0.152 | 0.156 | 1.97x | 3.18x |
| 8 | 0.891 | 0.352 | 0.934 | 0.199 | 0.209 | 1.77x | 4.70x |
| 16 | 0.926 | 0.455 | 1.912 | 0.369 | 0.440 | 1.23x | 5.19x |

The resident switchyard path uses one kernel and no temporary workspace.
Its error is 1.00 times the bf16 rounding floor in all measured cases.
The catswe path also has rounding-floor accuracy.
Catswe also computes merge statistics and supports its backward design.
The switchyard batched API returns only the output and has no backward.

## Shape checks at eight queries

| shape | compiled ms | switchyard ms | catswe ms | switchyard kernels | switchyard workspace MiB |
|---|---:|---:|---:|---:|---:|
| N=2 B=1 T=4096 D=2048 | 0.139 | 0.117 | 0.113 | 1 | 0.000 |
| N=16 B=1 T=4096 D=2048 | 0.571 | 0.344 | 0.399 | 1 | 0.000 |
| N=9 B=1 T=4096 D=1024 | 0.160 | 0.109 | 0.113 | 1 | 0.000 |
| N=9 B=1 T=128 D=1024 | 0.023 | 0.016 | 0.016 | 1 | 0.000 |
| N=9 B=4 T=2048 D=2048 | 0.726 | 0.393 | 0.401 | 1 | 0.000 |
| N=9 B=1 T=4096 D=4096 | 0.725 | 2.241 | 0.457 | 9 | 256.000 |

The `D=4096` case exceeds the resident tile budget.
The fallback uses eight per-query kernels and one stack kernel.
It takes 2.241 ms and uses 256 MiB of temporary workspace.
Catswe takes 0.457 ms in this case.
Use this batched API only in its documented resident dispatch range.

## Reproduction

```bash
source scripts/env.sh
scripts/fetch_third_party.sh
python bench/bench_third_party.py --batched-only --dtype bfloat16
python scripts/summarize_batched.py
```
