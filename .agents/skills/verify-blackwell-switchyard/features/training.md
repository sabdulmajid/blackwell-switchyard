# Training integration

Users can train the decoder with standard residuals, framework Block AttnRes,
or the fused operator, using stacking or an arena. DDP adds data-parallel training.

## Sub-features

- `model`: source schedule, zero-initialized queries, forward/gradient agreement.
- `arena`: source storage reuse without changing autograd results.
- `step`: loss, backward, optimizer, tokens/second, and peak memory.
- `ddp`: ordinary DDP gradients match the manual average of local gradients.

## How to get to it (user POV)

Use the decoder in `src/switchyard/model.py`, `bench/bench_model.py`, and
`bench/bench_ddp.py`. The model benchmark uses standard, framework, and fused modes.

## Driving it with pytest and model benchmarks

Preconditions: parent doctor passes and GPU availability is guarded. DDP needs
both GPUs and no concurrent campaign. The existing supervisor can guard the DDP
process tree with `all_gpus` telemetry and explicit two-device child visibility.
Its advisory lock covers one GPU; it is not an exclusive two-device reservation.

- **Small model proof:** `python -m pytest tests/test_model.py -q`.
  Require source schedule, two AttnRes sites per Transformer layer, identical
  parameter counts between the two AttnRes implementations, arena/stack forward
  and gradient equivalence, framework/fused agreement, and fixed-batch overfitting.
  Arena/stack outputs are exact and gradient relative L2 is below `1e-6`.
  Framework/fused loss uses `rtol=atol=1e-5`, gradients relative L2 below `1e-5`.
- **Full step when affected:**
  `python bench/bench_model.py --scale 1.3B --batch 4 --seq 2048 --dtype bfloat16 --out "$VERIFY_EVIDENCE_DIR/model.json"`.
  Record the measured revision, raw samples, spread, implementation, and full
  forward/loss/backward/AdamW timing. The model framework path currently uses
  eager `folded_form`; the historical 1.46x claim is against that path. The separate
  operator comparison uses max-autotuned Inductor. Do not interchange baselines.
- **DDP small first:** after both GPUs are available, run
  `CUDA_VISIBLE_DEVICES=0,1 python bench/bench_ddp.py --scale small --batch 1 --seq 32 --quick --grad-atol 0 --grad-rtol 0 --out "$VERIFY_EVIDENCE_DIR/ddp-small.json"`.
  Require `passed=true` and `max_abs_deviation_from_manual_average` within the
  recorded tolerance for every validated rank/variant. Rank agreement after an
  all-reduce alone cannot validate local gradients. Quick timing is not a scaling claim.
- **DDP full when justified:**
  `CUDA_VISIBLE_DEVICES=0,1 python bench/bench_ddp.py --scale 1.3B --batch 2 --seq 2048 --grad-atol 0 --grad-rtol 0 --out "$VERIFY_EVIDENCE_DIR/ddp.json"`.
  Preserve one-/two-device configuration, global tokens per step, communication
  topology, and correct gradient checks before reporting throughput or efficiency.

For the DDP small command, use this existing-supervisor route instead of an
unguarded launch. Source `scripts/env.sh` first. Require exactly the intended two
devices, available loopback port 29517, and no other owned campaign in flight:

```bash
python scripts/run_when_gpu_idle.py \
  --not-before "$(date -Iseconds)" --gpu-index 0 --idle-activity-scope all_gpus \
  --idle-seconds 1800 --wait-poll-seconds 60 --watchdog-seconds 0.25 \
  --deadline-hours 2 --max-attempts 1 \
  --state "$VERIFY_EVIDENCE_DIR/ddp-state.json" \
  --log "$VERIFY_EVIDENCE_DIR/ddp-small.log" \
  --lock "$VERIFY_EVIDENCE_DIR/ddp-runner.lock" --cwd "$PWD" -- \
  env -u SWITCHYARD_TARGET_GPU_UUID -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=10s 30m \
  python bench/bench_ddp.py --scale small --batch 1 --seq 32 --quick \
  --grad-atol 0 --grad-rtol 0 --port 29517 --out "$VERIFY_EVIDENCE_DIR/ddp-small.json"
```

The supervisor queries all compute processes and permits only its own descendants,
including both ranks. Do not use `target_gpu` activity scope, a GPU-complete marker,
or a backgrounded child for this route. Preserve state/logs and check both gradient
records after success. A collision, timeout, or surviving rank invalidates the drive.
This sampled smoke guard is weaker than the backward campaign's internal monitor
and evidence bundle; it is not sufficient for a new scaling-performance claim.

## Gotchas

Experimental CUDA plans are not wired into production model dispatch. A production
model benchmark does not establish their full-step benefit. Check that integration
before a candidate's end-to-end claim. Revised DDP validation and CUDA runtime
performance remain pending until real GPU checks pass.

For standard step `C`, framework step `F`, and fused step `S`, distinguish
`(F-C)/F` and `(S-C)/S` (control-difference share of each step), `(F-C)/C` and
`(S-C)/C` (overhead relative to the control), `(F-S)/F` (total step reduction),
and `F/S` (throughput ratio at equal tokens per step). Profiler region time divided
by total measured device time is a separate, narrower attribution. The standard
control lacks the AttnRes-only parameters; the two AttnRes modes are parameter-matched.

The existing model/DDP CLIs do not yet provide the full paired raw-trial evidence
required for a new performance release claim. Fill that affected evidence gap
before acceptance. Keep historical JSON intact; emit new runs under new paths.
