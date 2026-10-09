# Reviewed local procedures

Reviewed on 2026-10-09. These are text-only, project-local skills. No upstream
installer, bootstrap, hook, model configuration, or automation was executed.
The existing `.agents/skills/` directory did not exist before this integration.

| Source | Pinned revision | Files used | License |
|---|---|---|---|
| [Ponytail](https://github.com/DietrichGebert/ponytail/tree/9cc65d03aa2da1db7121b912d03596409ee340b8) | `9cc65d03aa2da1db7121b912d03596409ee340b8` | `skills/ponytail/SKILL.md` | MIT, Copyright 2026 DietrichGebert; [license](ponytail/LICENSE) |
| [Original pstack](https://github.com/cursor/plugins/tree/ccb5507cec1546dc88135c1139c811e6c59115ba/pstack) | `ccb5507cec1546dc88135c1139c811e6c59115ba` | Reviewed the four selected skills and `pstack/LICENSE` | MIT, Copyright 2026 Lauren Tan |
| [Standalone pstack mirror](https://github.com/backnotprop/pstack/tree/3a604672c46cd8187d2b19980eae0a34f9f91138) | `3a604672c46cd8187d2b19980eae0a34f9f91138` | `skills/{benchmark-checklist,principle-explain-the-number,create-verification-skill,maintain-verification-skill}/SKILL.md`; creator's three feature-map examples | MIT, Copyright 2026 Lauren Tan; [license](LICENSE-pstack) |

The mirror's creator and maintenance procedures name `.agents/skills/` for Codex.
The original versions prescribe `.cursor/skills/`. The two benchmark procedures
have the same body at these reviewed revisions. The mirror's `MIRROR.md` was also
reviewed; its repository synchronization commands are not installation steps here.

## Local adaptations

- Remove `argument-hint` from Ponytail and `disable-model-invocation` from the
  four pstack frontmatter blocks. Keep `name` and `description` for discovery.
  No explicit-only invocation policy was requested. `AGENTS.md` applies the
  benchmark procedures whenever a performance claim is made.
- Change the Explain the Number link to the uninstalled Prove It Works skill
  into a permalink at the pinned mirror revision. It is optional context.
- Keep the creator's feature-map examples unchanged so its required reference
  resolves. Other referenced upstream playbooks are optional and out of scope.
- `verify-blackwell-switchyard/` is project-authored using the reviewed creator
  and maintenance procedures. It calls existing tools instead of installing a
  second test or timing framework.

To reproduce the imported text, fetch the paths above from the pinned revisions,
apply only the listed frontmatter/link transformations, and compare with Git.
Keep both MIT notices with redistributed copies. Preserve the project's existing
Apache-2.0 and Liger BSD notices.

## Verification and updates

Skill directories contain a `SKILL.md` with a unique matching `name` and a
nonempty `description`. Relative references must resolve. The repository
`AGENTS.md` provides routing even before a client refreshes its skill catalog.
Normal project-skill discovery applies on the next task turn.

Review updates at an explicit revision, compare the selected files and licenses,
and record any changed instructions here. Do not use `setup-pstack`, poteto mode,
overnight playbooks, or global installation as an implicit part of this procedure.
