"""Compare a benchmark run against a stored baseline and fail on regressions.

Ordinary CI has no Blackwell GPU, so this is deliberately a *local* gate rather
than a GitHub Action pretending to measure performance. The intended use is::

    source scripts/env.sh
    python bench/bench_operator.py            # produces results/operator_*.json
    python scripts/check_regression.py        # compares against results/baseline/

    python scripts/check_regression.py --accept   # bless the current run

Exits non-zero if any tracked measurement regressed by more than the threshold,
so it can be wired into a pre-push hook or a release checklist.

Why a 7% default threshold: repeated runs of the same binary on this machine
show a coefficient of variation of roughly 1-3% at the shapes that matter, and
larger at the small launch-bound ones. 7% is comfortably outside that noise
while still catching the kind of change that matters -- a dispatch rule sending
a shape to the wrong kernel costs 2x, not 8%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
BASELINE = RESULTS / "baseline"

#: (metric path, human name, "lower is better")
TRACKED = [
    (("forward", "median_ms"), "forward ms", True),
    (("fwd_bwd", "median_ms"), "fwd+bwd ms", True),
    (("forward_memory", "workspace_bytes"), "workspace bytes", True),
]

#: Only these implementations gate the build. A framework baseline getting
#: slower is information, not our regression.
GATED = {"switchyard_triton"}


BASELINE_FILE = BASELINE / "operator_baseline.json"


def _key(impl: str, s: dict, dtype: str | None) -> str:
    return f"{impl}|N{s['n']}|B{s['b']}|T{s['t']}|D{s['d']}|{dtype}"


def load_current(paths: list[Path]) -> tuple[dict[str, dict], dict]:
    out: dict[str, dict] = {}
    repository_commit = None
    environment_fields = None
    source_reports = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"current report does not exist: {path}")
        data = json.loads(path.read_text())
        provenance = data.get("provenance", {})
        commit = provenance.get("repository_commit")
        argv = provenance.get("argv", [])
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or provenance.get("tracked_worktree_dirty") is not False
            or data.get("quick") is not False
            or not isinstance(argv, list)
            or "--skip-kernel-profile" in argv
        ):
            raise ValueError(
                f"current report is not a clean, full-profile release run: {path}"
            )
        if repository_commit is None:
            repository_commit = commit
        elif repository_commit != commit:
            raise ValueError("current reports do not use one repository commit")
        fields = {
            name: data.get("environment", {}).get(name)
            for name in ("device_name", "device_cc", "torch", "triton")
        }
        if environment_fields is None:
            environment_fields = fields
        elif environment_fields != fields:
            raise ValueError("current reports do not use one software and GPU environment")
        for r in data["results"]:
            s = r.get("shape")
            if not s or r.get("skipped"):
                continue
            key = _key(r["impl"], s, data.get("dtype"))
            if key in out:
                raise ValueError(f"duplicate current measurement: {key}")
            out[key] = r
        source_reports.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not out:
        raise ValueError("current reports contain no benchmark measurements")
    return out, {
        "repository_commit": repository_commit,
        "environment": environment_fields,
        "reports": source_reports,
    }


def compact(records: dict) -> dict:
    """Keep only the tracked metrics.

    The full run is a few megabytes of JSON. A baseline is consulted for exactly
    three numbers per record, so storing the rest would put megabytes of
    duplicated data into git for no benefit.
    """
    out = {}
    for key, rec in records.items():
        entry = {}
        for path, name, _ in TRACKED:
            v = dig(rec, path)
            if v is not None:
                entry[name] = v
        if entry:
            out[key] = entry
    return out


def dig(rec: dict, path: tuple):
    cur = rec
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", action="store_true", help="bless the current run as the baseline")
    ap.add_argument("--threshold", type=float, default=0.07, help="fractional regression allowed")
    ap.add_argument("--all-impls", action="store_true", help="gate on every implementation")
    ap.add_argument(
        "--current",
        nargs="+",
        type=Path,
        default=[RESULTS / "operator_default_bfloat16.json"],
        help="one or more clean, full-profile operator reports",
    )
    args = ap.parse_args()

    current_files = args.current
    try:
        cur, current_provenance = load_current(current_files)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid current regression evidence: {exc}", file=sys.stderr)
        return 2

    if args.accept:
        BASELINE.mkdir(parents=True, exist_ok=True)
        env = json.loads(current_files[0].read_text()).get("environment", {})
        BASELINE_FILE.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "environment": env,
                    "provenance": current_provenance,
                    "metrics": compact(cur),
                },
                indent=2,
                default=str,
            )
        )
        print(f"blessed {len(cur)} measurements into {BASELINE_FILE}")
        return 0

    if not BASELINE_FILE.exists():
        print(f"no baseline at {BASELINE_FILE}; create one with --accept", file=sys.stderr)
        return 2

    stored = json.loads(BASELINE_FILE.read_text())
    if stored.get("schema_version") != 2 or not isinstance(
        stored.get("provenance"), dict
    ):
        print(
            "stored baseline is legacy evidence; regenerate a clean operator run and "
            "review it before using --accept",
            file=sys.stderr,
        )
        return 2
    base = stored["metrics"]
    base_env = stored.get("environment", {})
    cur_env = json.loads(current_files[0].read_text()).get("environment", {})
    for field in ("device_name", "torch", "triton"):
        if base_env.get(field) != cur_env.get(field):
            print(f"warning: baseline {field}={base_env.get(field)!r} but current "
                  f"{field}={cur_env.get(field)!r}; comparison may be meaningless")

    gated = None if args.all_impls else GATED
    compact_cur = compact(cur)

    regressions, improvements, missing = [], [], []
    for key, b_entry in base.items():
        impl = key.split("|")[0]
        if gated is not None and impl not in gated:
            continue
        c_entry = compact_cur.get(key)
        if c_entry is None:
            missing.append(key)
            continue
        for _, name, lower_better in TRACKED:
            b, c = b_entry.get(name), c_entry.get(name)
            if b is None or c is None or b == 0:
                continue
            delta = (c - b) / b
            if not lower_better:
                delta = -delta
            shape = " ".join(key.split("|")[1:5])
            row = (shape, impl, name, b, c, delta)
            if delta > args.threshold:
                regressions.append(row)
            elif delta < -args.threshold:
                improvements.append(row)

    def show(rows, title):
        if not rows:
            return
        print(f"\n{title}")
        for shape, impl, name, b, c, d in sorted(rows, key=lambda r: -abs(r[5])):
            print(f"  {shape:28} {impl:20} {name:16} {b:12.4f} -> {c:12.4f}  {100 * d:+7.1f}%")

    show(improvements, "IMPROVED")
    show(regressions, "REGRESSED")
    if missing:
        print(f"\nMISSING from current run ({len(missing)}):")
        for key in missing[:10]:
            print(f"  {key}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if regressions:
        print(f"\nFAIL: {len(regressions)} measurement(s) regressed by more than "
              f"{100 * args.threshold:.0f}%")
        return 1
    print(f"\nOK: no regression beyond {100 * args.threshold:.0f}% "
          f"({len(improvements)} improvement(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
