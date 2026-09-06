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
import re
import subprocess
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
RELEASE_ENV_FIELDS = ("device_name", "device_cc", "torch", "triton")
EXPECTED_DEFAULT_SHAPES = {
    (2, 1, 4096, 2048),
    (4, 1, 4096, 2048),
    (8, 1, 4096, 2048),
    (9, 1, 4096, 2048),
    (16, 1, 4096, 2048),
    (32, 1, 4096, 2048),
    (9, 1, 4096, 1024),
    (9, 1, 4096, 4096),
    (9, 1, 4096, 8192),
    (9, 1, 512, 2048),
    (9, 1, 2048, 2048),
    (9, 1, 8192, 2048),
    (9, 1, 16384, 2048),
    (9, 1, 128, 1024),
    (9, 1, 512, 1024),
    (4, 1, 256, 512),
}
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}")
SAFE_REPORT_NAME_PATTERN = re.compile(
    r"operator_default_(?:bfloat16|float16|float32)\.json"
)
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
PRIVATE_TEXT_PATTERN = re.compile(
    r"(?:claude\.ai|session[_-]|co-authored-by|claude-session|/home/|/pub[0-9]+/)",
    re.IGNORECASE,
)


def _repository_tree(commit: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", f"{commit}^{{tree}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("current report revision is not in this repository") from exc


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
        if SAFE_REPORT_NAME_PATTERN.fullmatch(path.name) is None:
            raise ValueError(f"current report has a non-release filename: {path.name!r}")
        data = json.loads(path.read_text())
        provenance = data.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError(f"current report has no provenance object: {path}")
        commit = provenance.get("repository_commit")
        argv = provenance.get("argv", [])
        if (
            not isinstance(commit, str)
            or GIT_OBJECT_PATTERN.fullmatch(commit) is None
            or GIT_OBJECT_PATTERN.fullmatch(
                str(provenance.get("repository_tree", ""))
            )
            is None
            or not isinstance(provenance.get("repository_branch"), str)
            or not provenance["repository_branch"]
            or provenance.get("worktree_dirty") is not False
            or provenance.get("tracked_worktree_dirty") is not False
            or provenance.get("dirty_paths") != []
            or provenance.get("diff_sha256") != EMPTY_DIFF_SHA256
            or data.get("quick") is not False
            or not isinstance(argv, list)
            or not argv
            or argv[0] != "bench/bench_operator.py"
            or "--skip-kernel-profile" in argv
            or "--impls" in argv
            or any(
                not isinstance(argument, str)
                or not argument
                or len(argument) > 300
                or "\n" in argument
                or Path(argument).is_absolute()
                or PRIVATE_TEXT_PATTERN.search(argument) is not None
                for argument in argv
            )
        ):
            raise ValueError(
                f"current report is not a clean, full-profile release run: {path}"
            )
        if provenance["repository_tree"] != _repository_tree(commit):
            raise ValueError(f"current report tree does not match its revision: {path}")
        if data.get("shape_set") != "default":
            raise ValueError(f"current report is not the default shape matrix: {path}")
        if repository_commit is None:
            repository_commit = commit
        elif repository_commit != commit:
            raise ValueError("current reports do not use one repository commit")
        environment = data.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"current report has no environment object: {path}")
        fields = {name: environment.get(name) for name in RELEASE_ENV_FIELDS}
        if not all(isinstance(value, str) and value for value in fields.values()):
            raise ValueError(f"current report has incomplete release environment: {path}")
        if environment_fields is None:
            environment_fields = fields
        elif environment_fields != fields:
            raise ValueError("current reports do not use one software and GPU environment")
        gated_shapes = set()
        for r in data["results"]:
            s = r.get("shape")
            if not s or r.get("skipped"):
                continue
            key = _key(r["impl"], s, data.get("dtype"))
            if key in out:
                raise ValueError(f"duplicate current measurement: {key}")
            if r["impl"] in GATED:
                gated_shapes.add((s["n"], s["b"], s["t"], s["d"]))
                for metric_path, metric_name, _ in TRACKED:
                    value = dig(r, metric_path)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or value < 0
                        or (metric_name != "workspace bytes" and value == 0)
                    ):
                        raise ValueError(
                            f"current gated measurement has invalid {metric_name}: {key}"
                        )
            out[key] = r
        if gated_shapes != EXPECTED_DEFAULT_SHAPES:
            missing_shapes = EXPECTED_DEFAULT_SHAPES - gated_shapes
            extra_shapes = gated_shapes - EXPECTED_DEFAULT_SHAPES
            raise ValueError(
                "current report does not contain the exact gated default matrix: "
                f"missing={sorted(missing_shapes)}, extra={sorted(extra_shapes)}"
            )
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

    if not 0 <= args.threshold < 1:
        print("threshold must be at least 0 and less than 1", file=sys.stderr)
        return 2

    current_files = args.current
    try:
        cur, current_provenance = load_current(current_files)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid current regression evidence: {exc}", file=sys.stderr)
        return 2

    if args.accept:
        BASELINE.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "environment": current_provenance["environment"],
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
    base = stored.get("metrics")
    if not isinstance(base, dict):
        print("stored baseline has no metrics object", file=sys.stderr)
        return 2
    base_env = stored.get("environment", {})
    if (
        not isinstance(base_env, dict)
        or set(base_env) != set(RELEASE_ENV_FIELDS)
        or not all(isinstance(value, str) and value for value in base_env.values())
    ):
        print("stored baseline has invalid release environment metadata", file=sys.stderr)
        return 2
    cur_env = current_provenance["environment"]
    mismatches = [
        field for field in RELEASE_ENV_FIELDS if base_env[field] != cur_env[field]
    ]
    if mismatches:
        for field in mismatches:
            print(
                f"baseline {field}={base_env[field]!r} but current "
                f"{field}={cur_env[field]!r}",
                file=sys.stderr,
            )
        print("refusing to compare different release environments", file=sys.stderr)
        return 2

    gated = None if args.all_impls else GATED
    compact_cur = compact(cur)
    if gated is not None:
        baseline_keys = {key for key in base if key.split("|")[0] in gated}
        current_keys = {
            key for key in compact_cur if key.split("|")[0] in gated
        }
        if baseline_keys != current_keys:
            absent_from_current = sorted(baseline_keys - current_keys)
            absent_from_baseline = sorted(current_keys - baseline_keys)
            print("\nFAIL: gated baseline and current coverage differ")
            if absent_from_current:
                print(f"  missing from current: {len(absent_from_current)}")
            if absent_from_baseline:
                print(f"  missing from baseline: {len(absent_from_baseline)}")
            return 1

    regressions, improvements, missing = [], [], []
    for key, b_entry in base.items():
        impl = key.split("|")[0]
        if gated is not None and impl not in gated:
            continue
        if not isinstance(b_entry, dict):
            print(f"stored baseline has invalid metrics: {key}", file=sys.stderr)
            return 2
        c_entry = compact_cur.get(key)
        if c_entry is None:
            missing.append(key)
            continue
        for _, name, lower_better in TRACKED:
            b, c = b_entry.get(name), c_entry.get(name)
            if (
                isinstance(b, bool)
                or not isinstance(b, (int, float))
                or b < 0
            ):
                print(f"stored baseline has invalid {name}: {key}", file=sys.stderr)
                return 2
            if c is None:
                missing.append(f"{key}|{name}")
                continue
            if b == 0:
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

    if regressions or missing:
        reasons = []
        if regressions:
            reasons.append(
                f"{len(regressions)} measurement(s) regressed by more than "
                f"{100 * args.threshold:.0f}%"
            )
        if missing:
            reasons.append(f"{len(missing)} required measurement(s) are missing")
        print(f"\nFAIL: {'; '.join(reasons)}")
        return 1
    print(f"\nOK: no regression beyond {100 * args.threshold:.0f}% "
          f"({len(improvements)} improvement(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
