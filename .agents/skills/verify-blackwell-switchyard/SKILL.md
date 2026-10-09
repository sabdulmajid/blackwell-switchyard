---
name: verify-blackwell-switchyard
description: Verify Blackwell Switchyard's Python operator API, numerical correctness, benchmark evidence, and Transformer/DDP integration with its existing tests and guarded GPU tools. Use after relevant code changes, for release review, or when auditing a performance claim.
---

# Verify Blackwell Switchyard

Read [the feature index](features/README.md), then the recipe affected by the task.
Use Ponytail **full**, [benchmark-checklist](../benchmark-checklist/SKILL.md), and
[Explain the Number](../principle-explain-the-number/SKILL.md). Existing project
requirements and user authorization govern execution. This skill grants no new
permissions and never merges, promotes dispatch, or accepts a baseline.

## Launch

Run from the repository root. The public surface is a Python library and short-lived
benchmark CLIs; there is no server. Use the existing environment:

```bash
source scripts/env.sh
export VERIFY_EVIDENCE_DIR=$(mktemp -d "$(dirname "$PWD")/.switchyard-verify.XXXXXX")
set -o pipefail
```

Keep evidence outside the worktree. Do not reinstall into the shared Python.
Missing Python headers require the existing local-toolchain procedure in
`HANDOFF.md`; do not silently fall back to system changes.

## Doctor

Run before a drive and after any unexpected failure:

```bash
git status --short --branch
git rev-parse HEAD
CUDA_VISIBLE_DEVICES='' python -c 'import sys, torch, triton, switchyard; print(sys.version); print(torch.__version__, torch.version.cuda, triton.__version__); print(switchyard.__file__)'
test -f "${SWITCHYARD_TOOLCHAIN_DIR:-$PWD/.local-toolchain}/usr/include/python3.12/Python.h"
uptime
nproc
df -h . /tmp
```

Require the import to resolve to this checkout. Before GPU work also inspect
`nvidia-smi` for availability, memory, and other compute users. Do not store other
users' command lines, PIDs, or GPU UUIDs in public artifacts. Source the environment
in each new shell; readiness is a successful import and passing selected test.

## Drive

The smallest CPU-only proof exercises the actual reference API and its autograd:

```bash
env -u SWITCHYARD_TARGET_GPU_UUID CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_reference.py -q \
  --junitxml="$VERIFY_EVIDENCE_DIR/reference.xml" \
  2>&1 | tee "$VERIFY_EVIDENCE_DIR/reference.log"
```

Require exit code zero and passing tests, not an all-skipped run. For broader
CPU coverage use `env -u SWITCHYARD_TARGET_GPU_UUID CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q`.
Unsetting the target prevents the pytest collision fixture from starting NVML monitoring.
Read the matching feature file for GPU correctness, performance, or training.
Do not interpret CPU passes or offline compilation as GPU validation.

## Evidence

Capture the command, exit status, revision and tree, dirty state, interpreter,
PyTorch/Triton/CUDA/driver versions, GPU model/compute capability, dtype, dimensions,
seeds, epsilon, implementation/dispatch plan, output contract, baseline settings,
raw samples, order, spread, correctness, profile, and result paths. Use the existing
JSON provenance and report validators. Record missing historical fields explicitly;
never reconstruct absent provenance from the current checkout.

Keep compilation/autotuning, oracle evaluation, cache flush, and profiling outside
steady-state timing. State whether allocations, gradient clearing, copies, retained
graphs, and optimizer steps are timed. Preserve measured machine calibration.
Repeat and interleave comparisons; distinguish within-run samples from independent
process runs. New acceptance needs at least five paired trials and stored spread;
the backward campaign already requires 15 trials with 13 samples each. Close or
unstable results need additional independent sessions before a release claim.

Use `docs/claim_audit.md` for historical definitions and gaps. In particular,
control-difference share, overhead relative to the control, profiler device-time
share, and step-time reduction are distinct. Derived logical-minimum effective
bandwidth is not a hardware traffic counter.

## Cleanup

Tests and benchmark CLIs exit themselves. The existing GPU supervisor owns and
stops its child process group on collision or failure; do not kill by name or
terminate unrelated work. Keep logs, raw JSON, and JUnit output after teardown.
Check `test -s "$VERIFY_EVIDENCE_DIR/reference.xml"` after the CPU recipe.
Delete no evidence as part of normal cleanup. Inspect `git status --short` for
unexpected tracked changes; retain user edits.

## Helpers and maintenance

All drive helpers already exist under `tests/`, `bench/`, and `scripts/`.
Their exact commands are in the feature map; no extra runner is installed.
Use [maintain-verification-skill](../maintain-verification-skill/SKILL.md) when
entry points, flags, contracts, or evidence formats change. Source-review every
mapped feature and run reachable affected paths. Report an unmet GPU idle gate or
other prerequisite as blocked coverage, not a pass. The initial creation proof
needs one small mapped case; it does not assert a complete maintenance pass.
