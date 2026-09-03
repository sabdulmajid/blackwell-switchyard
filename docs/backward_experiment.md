# Tiled backward experiment

This experiment targets the large-shape backward path.
It does not change the production dispatch.

## Current problem

The current tiled backward uses two kernels.
The first kernel computes three fp32 statistics for each source and token.
The second kernel reads those statistics and reads the source tensor again.

The source tensor is larger than L2 for the important losing shapes.
The second kernel therefore cannot rely on the first kernel's source data remaining in L2.
Stored measurements show that Liger is faster at these shapes:

| Shape | switchyard forward and backward | Liger forward and backward |
|---|---:|---:|
| `N=32 B=1 T=4096 D=2048` | 1.565 ms | 1.502 ms |
| `N=9 B=1 T=4096 D=4096` | 0.920 ms | 0.774 ms |
| `N=9 B=1 T=4096 D=8192` | 1.961 ms | 1.435 ms |

These are forward-and-backward results.
They are not direct backward measurements.
The new benchmark measures backward directly.

## Candidate

The private `_bwd_source_serial` kernel assigns one program to one token.
It visits one source at a time.
It completes the second source pass in the same program.
This schedule can serve the second pass from L2.
It also removes the three statistics buffers and one kernel launch.

The candidate keeps all reductions and `dw` atomics in fp32.
It does not use `tl.dot`.

The cost is more contention.
The candidate performs one `dw` atomic update per token and feature.
The current tiled path performs one update per 32-token chunk and feature.
The benchmark must decide whether the locality gain is larger than this cost.

## Isolation rule

Normal `block_attn_res_triton` calls cannot select the candidate.
Only the private benchmark entry point can select it.
Do not add it to production dispatch without committed GPU results.

## Test matrix

The full matrix has 12 shapes.
It covers three resident controls, the first tiled neighbors, short and long token counts,
the three known losses, and one batched training layout.

Run the full matrix for bf16, fp16, and fp32:

```bash
source scripts/env.sh

python bench/bench_backward.py --shape-set full --dtype bfloat16
python bench/bench_backward.py --shape-set full --dtype float16
python bench/bench_backward.py --shape-set full --dtype float32

python scripts/evaluate_backward.py \
  results/backward_candidates_bfloat16.json \
  results/backward_candidates_float16.json \
  results/backward_candidates_float32.json
```

Use `--quick` only to find compile or correctness failures.
Do not use a quick run for a production decision.

The benchmark does the following work:

- It checks the output, `dv`, and `dw` against a float64 oracle before timing.
- It measures backward directly on a retained graph.
- It flushes L2 after graph setup and before each timed call.
- It stores five trials of 40 raw samples.
- It records the repository revision, tree, branch, dirty paths, and diff hash.
- It records the pinned Liger revision and dirty state.
- It records kernel counts, memory, and the explicit traffic model.
- It stops if the selected GPU has an active compute process.

Do not use `--allow-busy-gpu` for an accepted result.
That option exists only for diagnostic work, and the evaluator rejects its output.

## Promotion gates

Reject the candidate if any output or gradient fails.
Do not loosen a tolerance to admit the candidate.
The candidate error must also stay within 1.05 times the current implementation's error.

A performance result is material only when it differs by at least 7 percent.
The coefficient of variation must be at most 5 percent.
The candidate needs at least a 10 percent direct-backward gain on each of the three known
losses and at least a 15 percent geometric-mean gain across those shapes.
Its forward-and-backward result must beat Liger on each anchor with a positive 95 percent
bootstrap confidence bound.
Shape-aware dispatch can select only measured wins.
It does not need to select the candidate at a losing shape.

If the evaluator returns `READY_FOR_DISPATCH_REVIEW`, make a separate dispatch commit.
Then run all operator, third-party, model, and two-GPU regression tests again.
Update public documentation only from committed raw results.

If the evaluator returns `DROP`, remove the private kernel and its entry point.
Keep this report as the record of the rejected design.
