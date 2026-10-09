# Blackwell Switchyard working rules

Read local `HANDOFF.md` if present as a historical map, then `PROJECT_STATE.md` for current status.
Check the branch, diff, and existing artifacts before changing code or rerunning work.

## Local skills

- Use `.agents/skills/ponytail/SKILL.md` at **full** level for coding work.
- Before accepting or reporting a performance result, use both
  `.agents/skills/benchmark-checklist/SKILL.md` and
  `.agents/skills/principle-explain-the-number/SKILL.md`.
- Use `.agents/skills/verify-blackwell-switchyard/SKILL.md` to select the existing
  correctness, benchmark, and training checks relevant to a change.
- Use the local `create-verification-skill` and `maintain-verification-skill`
  procedures when the verification map needs an update. Maintenance is scoped to
  that skill's directory; report product regressions separately.

These procedures do not change model settings, permissions, merge control, or
machine configuration. Their optional upstream playbooks are not installed or
required. Source revisions, local adaptations, and licenses are in
`.agents/skills/SOURCES.md`.

## Project requirements take priority over brevity

Preserve hardware calibration, error handling, numerical bounds, both framework
formulations, and measured shape specializations. Code size is secondary to
correctness, performance, and reproducibility. Preserve raw values, source-axis
softmax, no attention scaling, and fp32 reduction/gradient accumulation.

Use `scripts/env.sh` and the existing local headers. Do not modify shared Python,
CUDA, drivers, power limits, or another user's processes. The backward campaign
requires the documented 30-minute idle gate and collision monitors. A busy GPU
does not become valid benchmark hardware by interleaving workloads.

Reuse unchanged evidence and state its limits. Historical summary-only results
do not satisfy newer raw-trial or provenance gates. Never silently relabel them
as measurements of the current revision. Keep experimental CUDA dispatch and
revised DDP validation pending until their respective GPU checks pass.

Commit as `Ayman <ayman.hasib@outlook.com>`. Do not include AI co-author trailers,
session links, prompts, GPU UUIDs, or private process data. Required third-party
license attribution is preserved. Publish reviewable branches; the owner retains
merge control. Do not accept a new performance baseline automatically.
