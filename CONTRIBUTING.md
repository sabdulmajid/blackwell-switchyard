# Contributing

Thank you for improving `blackwell-switchyard`.

## Before you change a kernel

Read the operation definition and correctness tests first.
Preserve these rules:

- The weighted sum uses raw source values.
- RMS normalization applies only to keys.
- Softmax uses the source axis.
- Logits do not use `1/sqrt(D)` scaling.
- Reduction and `dw` accumulation use fp32.

Do not loosen a correctness tolerance to make a new kernel pass.

## Local checks

Run the CPU checks without access to a GPU:

```bash
env -u SWITCHYARD_TARGET_GPU_UUID CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/ -q
ruff check src bench scripts tests
```

Run the complete GPU tests only when you have exclusive access to a suitable GPU:

```bash
source scripts/env.sh
python -m pytest tests/ -q
```

## Performance changes

Use the shared benchmark tools in `bench/harness.py`.
Check correctness before timing.
Exclude compilation and autotuning from steady-state results.
Flush L2 outside the timed region.
Record raw results in JSON.

State the shape, data type, GPU, baseline, and method with every performance claim.
Include results that lose when they define a dispatch boundary.

Do not change production dispatch from an exploratory result.
Commit the candidate first, run the full evidence matrix from a clean revision, and then
make the dispatch change in a separate commit.

## Commit metadata

Use your own name and email address.
Do not add AI co-author trailers, session links, prompts, or private environment data.
Review the complete commit message before you push it.

## Local verification procedures

Use [Ponytail full](.agents/skills/ponytail/SKILL.md) for small, complete code changes.
Use the [Blackwell verification skill](.agents/skills/verify-blackwell-switchyard/SKILL.md)
to select existing tests and benchmark commands. Before a performance claim, use the
[benchmark checklist](.agents/skills/benchmark-checklist/SKILL.md) and
[Explain the Number](.agents/skills/principle-explain-the-number/SKILL.md).
Pinned revisions and license notices are in [.agents/skills/SOURCES.md](.agents/skills/SOURCES.md).

CPU-only tests must also unset `SWITCHYARD_TARGET_GPU_UUID` if it is inherited from a
guarded GPU run. Hiding CUDA alone does not disable the pytest telemetry fixture.
Run `python scripts/check_reported_claims.py` to audit the old headline arithmetic
without a GPU. This does not certify old artifacts against current release gates.
