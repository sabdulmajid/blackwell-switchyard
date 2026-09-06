#!/usr/bin/env bash
# Run the bounded backward campaign on the GPU selected by the idle runner.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${SWITCHYARD_EXPECTED_HEAD:?set SWITCHYARD_EXPECTED_HEAD}"
: "${SWITCHYARD_CAMPAIGN_BRANCH:?set SWITCHYARD_CAMPAIGN_BRANCH}"
: "${SWITCHYARD_CAMPAIGN_DIR:?set SWITCHYARD_CAMPAIGN_DIR outside the repository}"
: "${SWITCHYARD_TARGET_GPU_UUID:?the idle runner must select one GPU UUID}"
: "${SWITCHYARD_PUBLIC_DEVICE_ID:?the idle runner must select one public device ID}"
: "${SWITCHYARD_RUN_ATTEMPT:?the idle runner must number each attempt}"
: "${SWITCHYARD_GUARD_NOT_BEFORE:?the idle runner must attest not-before time}"
: "${SWITCHYARD_GUARD_IDLE_STARTED_AT:?the idle runner must attest idle start}"
: "${SWITCHYARD_GUARD_LAUNCH_AT:?the idle runner must attest launch time}"
: "${SWITCHYARD_GUARD_IDLE_PROBE_COUNT:?the idle runner must attest idle probes}"
: "${SWITCHYARD_GUARD_MAX_IDLE_GPU_UTILIZATION_PERCENT:?the runner must attest idle utilization}"
: "${SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB:?the runner must attest idle memory}"
: "${SWITCHYARD_GUARD_GPU_COUNT:?the idle runner must attest GPU inventory size}"
: "${THIRD_PARTY_DIR:?set THIRD_PARTY_DIR to the pinned dependency directory}"
: "${SWITCHYARD_TOOLCHAIN_DIR:?set SWITCHYARD_TOOLCHAIN_DIR to the local Python headers}"

result_branch=${SWITCHYARD_RESULT_BRANCH:-codex/backward-architecture}
cpu_recovery=${SWITCHYARD_CPU_RECOVERY:-0}
expected_origin=https://github.com/sabdulmajid/blackwell-switchyard.git
forbidden_metadata='co-authored-by|claude|anthropic|wizchem|chatgpt|openai\.com|session[-_/][[:alnum:]]|file://|/(home|tmp|pub[0-9]+)/|GPU-[[:alnum:]-]+'
all_impls=current,serial_recompute_atomic_t4,serial_saved_partials_t16,cuda_shared,cuda_cluster,cuda_cluster4,cuda_register,cuda_register_cluster,cuda_register_cluster_full,liger
portable_impls=current,serial_recompute_atomic_t4,serial_saved_partials_t16,liger
candidates=(
  serial_recompute_atomic_t4
  serial_saved_partials_t16
  cuda_shared
  cuda_cluster
  cuda_cluster4
  cuda_register
  cuda_register_cluster
  cuda_register_cluster_full
)
if [[ ! "$SWITCHYARD_EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] ||
   [[ ! "$SWITCHYARD_RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$SWITCHYARD_TARGET_GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] ||
   [[ ! "$SWITCHYARD_PUBLIC_DEVICE_ID" =~ ^device-[0-9a-f]{16}$ ]] ||
   [[ ! "$cpu_recovery" =~ ^[01]$ ]] ||
   ! git check-ref-format --branch "$SWITCHYARD_CAMPAIGN_BRANCH" >/dev/null ||
   ! git check-ref-format --branch "$result_branch" >/dev/null; then
  echo "campaign identity or Git reference is malformed" >&2
  exit 2
fi

verify_origin() {
  local fetch_urls
  local push_urls
  mapfile -t fetch_urls < <(git remote get-url --all origin)
  mapfile -t push_urls < <(git remote get-url --push --all origin)
  if [[ ${#fetch_urls[@]} -ne 1 || ${fetch_urls[0]:-} != "$expected_origin" ||
        ${#push_urls[@]} -ne 1 || ${push_urls[0]:-} != "$expected_origin" ]]; then
    echo "origin must have exactly one canonical fetch URL and push URL" >&2
    return 1
  fi
}

read_result_branch() {
  verify_origin || return 1
  git ls-remote --heads "$expected_origin" "refs/heads/$result_branch" | awk '{print $1}'
}

push_result_commit() {
  local result_commit=$1
  local delay
  local observed
  for delay in 0 5 15 30 60 120 240 480; do
    if ((delay > 0)); then
      sleep "$delay"
    fi
    verify_origin || return 1
    if ! git cat-file -e "$result_commit^{commit}"; then
      echo "local result commit disappeared" >&2
      return 1
    fi
    if git push "$expected_origin" "$result_commit:refs/heads/$result_branch"; then
      return 0
    fi
    if observed=$(read_result_branch); then
      if [[ "$observed" == "$result_commit" ]]; then
        return 0
      fi
      if [[ "$observed" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
        echo "result branch advanced; refusing to retry a non-fast-forward push" >&2
        return 1
      fi
    fi
  done
  return 1
}

campaign_dir=$(realpath -m -- "$SWITCHYARD_CAMPAIGN_DIR")
third_party_dir=$(realpath -m -- "$THIRD_PARTY_DIR")
if [[ "$campaign_dir" == "$repo_dir" || "$campaign_dir" == "$repo_dir/"* ]]; then
  echo "campaign state must remain outside the repository" >&2
  exit 2
fi
if [[ "$third_party_dir" == "$repo_dir" || "$third_party_dir" == "$repo_dir/"* ]]; then
  echo "third-party dependencies must remain outside the campaign worktree" >&2
  exit 2
fi
if [[ ! -f "$SWITCHYARD_TOOLCHAIN_DIR/usr/include/python3.12/Python.h" ]]; then
  echo "repo-scoped Python headers are missing" >&2
  exit 2
fi

mkdir -p "$campaign_dir"
exec 9>"$campaign_dir/repository.lock"
if ! flock -n 9; then
  echo "another campaign owns the repository-results lock" >&2
  exit 75
fi

git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
git_dir=$(git rev-parse --path-format=absolute --git-dir)
git_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
git_index=$(git rev-parse --path-format=absolute --git-path index)
expected_tree=$(git rev-parse "$SWITCHYARD_EXPECTED_HEAD^{tree}")
export SWITCHYARD_EXPECTED_TREE="$expected_tree"
export SWITCHYARD_EXPECTED_ORIGIN="$expected_origin"

check_result_bundle() {
  local prefix=$1
  local ref=$2
  python scripts/check_backward_bundle.py "$prefix" \
    --git-ref "$ref" \
    --expected-commit "$SWITCHYARD_EXPECTED_HEAD" \
    --expected-tree "$expected_tree" \
    --expected-branch "$SWITCHYARD_CAMPAIGN_BRANCH" \
    --result-branch "$result_branch" \
    --expected-origin "$expected_origin" \
    --expected-device-id "$SWITCHYARD_PUBLIC_DEVICE_ID"
}
if [[ "$git_dir" == "$git_common_dir" || "$git_index" != "$git_dir/index" ]]; then
  echo "campaign must run from a dedicated linked worktree and index" >&2
  exit 2
fi
if [[ "$git_branch" != "$SWITCHYARD_CAMPAIGN_BRANCH" ]]; then
  echo "campaign is running from the wrong branch" >&2
  exit 2
fi
if [[ -n $(git status --porcelain) ]] || ! git diff --quiet || ! git diff --cached --quiet; then
  echo "campaign worktree must be clean" >&2
  exit 2
fi
if [[ $(git config --local user.name) != "Ayman" ]] ||
   [[ $(git config --local user.email) != "ayman.hasib@outlook.com" ]]; then
  echo "refusing to commit with unexpected Git identity" >&2
  exit 2
fi
if ! verify_origin; then
  echo "refusing to publish to an unexpected origin" >&2
  exit 2
fi
if ! remote_head=$(read_result_branch); then
  echo "cannot read the result branch" >&2
  [[ "$git_head" == "$SWITCHYARD_EXPECTED_HEAD" && "$cpu_recovery" == 0 ]] && exit 75
  exit 74
fi

# Recover a fully audited local result commit if a prior push or post-push
# verification was interrupted. No GPU phase needs to run again.
if [[ "$git_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  mapfile -t recovery_files < <(git diff-tree --no-commit-id --name-only -r "$git_head")
  recovery_prefix=""
  for path in "${recovery_files[@]}"; do
    if [[ "$path" =~ ^(results/backward_campaign_[0-9]{8}T[0-9]{6}Z)_manifest\.json$ ]]; then
      if [[ -n "$recovery_prefix" ]]; then
        recovery_prefix="invalid"
        break
      fi
      recovery_prefix=${BASH_REMATCH[1]}
    fi
  done
  if [[ $(git rev-parse "$git_head^") != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
     [[ $(git show -s --format=%an%n%ae%n%cn%n%ce "$git_head") != $'Ayman\nayman.hasib@outlook.com\nAyman\nayman.hasib@outlook.com' ]] ||
     [[ $(git show -s --format=%s "$git_head") != "Record guarded backward campaign" ]] ||
     [[ -z "$recovery_prefix" || "$recovery_prefix" == invalid ]] ||
     git show -s --format=%B "$git_head" | rg -qi "$forbidden_metadata" ||
     git grep -IinE "$forbidden_metadata" "$git_head" -- "${recovery_files[@]}" ||
     ! check_result_bundle "$recovery_prefix" "$git_head"; then
    echo "repository HEAD changed and is not a recoverable result commit" >&2
    exit 2
  fi
  if [[ "$remote_head" == "$git_head" ]]; then
    exit 0
  fi
  if [[ "$remote_head" != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
     ! push_result_commit "$git_head"; then
    echo "recoverable result commit still needs publication" >&2
    exit 74
  fi
  remote_head=$(read_result_branch) || exit 74
  [[ "$remote_head" == "$git_head" ]] && exit 0
  exit 74
fi
if [[ "$remote_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "result branch changed while the campaign waited" >&2
  exit 2
fi

liger_dir="$third_party_dir/Liger-Kernel"
pinned_liger=777799588a89d74c489ed995e3bf006427738e85
if [[ ! -d "$liger_dir/.git" ]] ||
   [[ $(git -C "$liger_dir" rev-parse HEAD) != "$pinned_liger" ]] ||
   [[ -n $(git -C "$liger_dir" status --porcelain) ]]; then
  echo "Liger must be a clean checkout at $pinned_liger" >&2
  exit 2
fi

guard_attestation="$campaign_dir/guard.json"
export SWITCHYARD_GUARD_ATTESTATION="$guard_attestation"
python - "$guard_attestation" <<'PY'
import json
import math
import os
import sys
import tempfile
from datetime import datetime

path = sys.argv[1]
attempt = {
    "attempt": int(os.environ["SWITCHYARD_RUN_ATTEMPT"]),
    "not_before": os.environ["SWITCHYARD_GUARD_NOT_BEFORE"],
    "idle_started_at": os.environ["SWITCHYARD_GUARD_IDLE_STARTED_AT"],
    "launch_at": os.environ["SWITCHYARD_GUARD_LAUNCH_AT"],
    "idle_seconds": int(os.environ["SWITCHYARD_GUARD_IDLE_SECONDS"]),
    "idle_probe_count": int(os.environ["SWITCHYARD_GUARD_IDLE_PROBE_COUNT"]),
    "max_idle_gpu_utilization_percent": int(
        os.environ["SWITCHYARD_GUARD_MAX_IDLE_GPU_UTILIZATION_PERCENT"]
    ),
    "max_idle_memory_mib": int(os.environ["SWITCHYARD_GUARD_MAX_IDLE_MEMORY_MIB"]),
    "wait_poll_seconds": int(os.environ["SWITCHYARD_GUARD_WAIT_POLL_SECONDS"]),
    "watchdog_seconds": float(os.environ["SWITCHYARD_GUARD_WATCHDOG_SECONDS"]),
    "finalize_seconds": int(os.environ["SWITCHYARD_GUARD_FINALIZE_SECONDS"]),
    "gpu_count": int(os.environ["SWITCHYARD_GUARD_GPU_COUNT"]),
    "device_id": os.environ["SWITCHYARD_PUBLIC_DEVICE_ID"],
    "target_gpu_uuid": os.environ["SWITCHYARD_TARGET_GPU_UUID"],
}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if (
        set(document) != {"schema_version", "device_id", "target_gpu_uuid", "attempts"}
        or document["schema_version"] != 1
        or document["device_id"] != attempt["device_id"]
        or document["target_gpu_uuid"] != attempt["target_gpu_uuid"]
        or not isinstance(document["attempts"], list)
    ):
        raise SystemExit("existing guard document differs from this runner")
else:
    document = {
        "schema_version": 1,
        "device_id": attempt["device_id"],
        "target_gpu_uuid": attempt["target_gpu_uuid"],
        "attempts": [],
    }

matches = [item for item in document["attempts"] if item.get("attempt") == attempt["attempt"]]
if len(matches) > 1 or (matches and matches[0] != attempt):
    raise SystemExit("existing attempt attestation differs from this launch")
if not matches:
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("attempt"), int)
        or item["attempt"] >= attempt["attempt"]
        for item in document["attempts"]
    ):
        raise SystemExit("guard attempts must be appended in increasing order")
    document["attempts"].append(attempt)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(path) or ".",
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

attestation = attempt

not_before = datetime.fromisoformat(attestation["not_before"])
idle_started = datetime.fromisoformat(attestation["idle_started_at"])
launch = datetime.fromisoformat(attestation["launch_at"])
if any(moment.tzinfo is None for moment in (not_before, idle_started, launch)):
    raise SystemExit("guard timestamps must include UTC offsets")
if (
    attestation["idle_seconds"] < 1800
    or not 30 <= attestation["wait_poll_seconds"] <= 300
    or not 0 < attestation["watchdog_seconds"] <= 0.25
    or attestation["finalize_seconds"] < 1800
    or attestation["gpu_count"] < 1
    or attestation["max_idle_gpu_utilization_percent"] != 0
    or not 0 <= attestation["max_idle_memory_mib"] <= 64
    or attestation["device_id"] != os.environ["SWITCHYARD_PUBLIC_DEVICE_ID"]
    or attestation["target_gpu_uuid"] != os.environ["SWITCHYARD_TARGET_GPU_UUID"]
    or launch < not_before
    or (launch - idle_started).total_seconds() < attestation["idle_seconds"]
    or attestation["idle_probe_count"]
    < max(2, math.floor(attestation["idle_seconds"] / attestation["wait_poll_seconds"]))
):
    raise SystemExit("guard attestation does not meet the campaign safety contract")
PY

gpu_complete_marker="$campaign_dir/gpu_complete"
attempt_dir="$campaign_dir/attempt-$SWITCHYARD_RUN_ATTEMPT"
if ! mkdir "$attempt_dir"; then
  if [[ ! -d "$attempt_dir" || ! -f "$gpu_complete_marker" ]]; then
    echo "attempt directory already exists before GPU completion: $attempt_dir" >&2
    exit 2
  fi
fi
run_dir="$campaign_dir/run"
mkdir -p "$run_dir"
if [[ "$cpu_recovery" == 0 ]]; then
  rm -f "$gpu_complete_marker"
elif [[ ! -f "$gpu_complete_marker" ]]; then
  echo "CPU recovery requires a completed GPU phase" >&2
  exit 2
fi

source scripts/env.sh
export THIRD_PARTY_DIR="$third_party_dir"
export PYTHONPATH="$liger_dir/src:$PYTHONPATH"

compile_report="$run_dir/offline_compile.json"
smoke_bf16="$run_dir/smoke_bfloat16.json"
smoke_fp16="$run_dir/smoke_float16.json"
smoke_decision="$run_dir/smoke_decision.json"
full_bf16="$run_dir/full_bfloat16.json"
full_fp16="$run_dir/full_float16.json"
full_fp32="$run_dir/full_float32.json"
manifest="$attempt_dir/manifest.json"

CUDA_VISIBLE_DEVICES="" timeout --foreground 30m \
  python scripts/compile_candidates.py --out "$compile_report"
timeout --foreground 30m python -m pytest tests/test_backward_candidates.py -q

report_reusable() {
  local report=$1
  local dtype=$2
  local shape_set=$3
  local impls=$4
  local quick_flag=${5:-}
  python scripts/check_backward_report.py "$report" \
    --dtype "$dtype" --shape-set "$shape_set" --impls "$impls" \
    --expected-commit "$SWITCHYARD_EXPECTED_HEAD" \
    --expected-branch "$SWITCHYARD_CAMPAIGN_BRANCH" \
    --expected-tree "$expected_tree" \
    --expected-device-id "$SWITCHYARD_PUBLIC_DEVICE_ID" $quick_flag
}

require_gpu_regeneration() {
  local report=$1
  if [[ "$cpu_recovery" == 1 ]]; then
    echo "CPU recovery found incomplete GPU evidence: $report" >&2
    exit 75
  fi
}

if report_reusable "$smoke_bf16" bfloat16 gate "$all_impls" --quick; then
  echo "reusing clean completed bf16 smoke phase"
else
  require_gpu_regeneration "$smoke_bf16"
  rm -f "$smoke_bf16"
  timeout --foreground 2h python bench/bench_backward.py \
    --shape-set gate --dtype bfloat16 --quick --impls "$all_impls" --out "$smoke_bf16"
fi
if report_reusable "$smoke_fp16" float16 gate "$all_impls" --quick; then
  echo "reusing clean completed fp16 smoke phase"
else
  require_gpu_regeneration "$smoke_fp16"
  rm -f "$smoke_fp16"
  timeout --foreground 2h python bench/bench_backward.py \
    --shape-set gate --dtype float16 --quick --impls "$all_impls" --out "$smoke_fp16"
fi
set +e
python scripts/check_backward_smoke.py \
  "$smoke_bf16" "$smoke_fp16" \
  --expected-commit "$SWITCHYARD_EXPECTED_HEAD" \
  --expected-branch "$SWITCHYARD_CAMPAIGN_BRANCH" \
  --expected-tree "$expected_tree" \
  --out "$smoke_decision"
smoke_exit=$?
set -e
if [[ $smoke_exit -ne 0 ]] ||
   ! python scripts/check_evidence_binding.py \
     "$smoke_decision" "$smoke_bf16" "$smoke_fp16"; then
  echo "smoke evidence failed a deterministic correctness or structure gate" >&2
  exit 3
fi

if report_reusable "$full_bf16" bfloat16 full "$all_impls"; then
  echo "reusing clean completed bf16 full phase"
else
  require_gpu_regeneration "$full_bf16"
  rm -f "$full_bf16"
  timeout --foreground 8h python bench/bench_backward.py \
    --shape-set full --dtype bfloat16 --impls "$all_impls" --out "$full_bf16"
fi
if report_reusable "$full_fp16" float16 full "$all_impls"; then
  echo "reusing clean completed fp16 full phase"
else
  require_gpu_regeneration "$full_fp16"
  rm -f "$full_fp16"
  timeout --foreground 8h python bench/bench_backward.py \
    --shape-set full --dtype float16 --impls "$all_impls" --out "$full_fp16"
fi
if report_reusable "$full_fp32" float32 full "$portable_impls"; then
  echo "reusing clean completed fp32 full phase"
else
  require_gpu_regeneration "$full_fp32"
  rm -f "$full_fp32"
  timeout --foreground 6h python bench/bench_backward.py \
    --shape-set full --dtype float32 --impls "$portable_impls" --out "$full_fp32"
fi

# GPU work is complete. Evaluation, result copying, commit, and push are CPU and
# network work. The outer runner stops GPU polling after this marker so another
# project may safely start while this campaign finalizes its evidence.
touch "$gpu_complete_marker"

decision_files=()
for candidate in "${candidates[@]}"; do
  decision="$run_dir/decision_${candidate}.json"
  evaluation_reports=("$full_bf16" "$full_fp16")
  if [[ "$candidate" == serial_* ]]; then
    evaluation_reports+=("$full_fp32")
  fi
  set +e
  python scripts/evaluate_backward.py \
    --candidate "$candidate" --json \
    --expected-commit "$SWITCHYARD_EXPECTED_HEAD" \
    --expected-branch "$SWITCHYARD_CAMPAIGN_BRANCH" \
    --expected-tree "$expected_tree" \
    "${evaluation_reports[@]}" >"$decision"
  evaluation_exit=$?
  set -e
  if ! python -m json.tool "$decision" >/dev/null; then
    echo "evaluator failed for $candidate" >&2
    exit 3
  fi
  decision_status=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$decision")
  case "$evaluation_exit:$decision_status" in
    0:DROP|0:READY_FOR_DISPATCH_REVIEW|1:REJECT) ;;
    2:MORE_DATA)
      retryable=$(python -c \
        'import json,sys; d=json.load(open(sys.argv[1])); print(int(not d["problems"] and bool(d["unstable"])))' \
        "$decision")
      if [[ "$retryable" == 1 ]]; then
        rm -f -- "$full_bf16" "$full_fp16" "$full_fp32" \
          "$decision" "${decision_files[@]}"
        echo "statistical instability requires fresh full evidence for $candidate" >&2
        exit 75
      fi
      echo "evaluator found a deterministic incomplete evidence contract for $candidate" >&2
      exit 3
      ;;
    *)
      echo "evaluator exit and status disagree for $candidate: $evaluation_exit:$decision_status" >&2
      exit 3
      ;;
  esac
  if ! python scripts/check_evidence_binding.py "$decision" "${evaluation_reports[@]}"; then
    echo "decision is not bound to the reports for $candidate" >&2
    exit 3
  fi
  decision_files+=("$decision")
done

if [[ $(git rev-parse HEAD) != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
   [[ $(git branch --show-current) != "$SWITCHYARD_CAMPAIGN_BRANCH" ]] ||
   [[ -n $(git status --porcelain) ]]; then
  echo "campaign worktree changed during measurement; results remain external" >&2
  exit 2
fi
if ! remote_head=$(read_result_branch); then
  echo "cannot recheck the result branch; results remain external" >&2
  exit 74
fi
if [[ "$remote_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "result branch changed during measurement; results remain external" >&2
  exit 2
fi

campaign_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$full_bf16")
if [[ ! "$campaign_id" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "full report has an invalid campaign ID" >&2
  exit 2
fi
result_prefix="results/backward_campaign_${campaign_id}"
result_files=(
  "${result_prefix}_offline_compile.json"
  "${result_prefix}_smoke_bfloat16.json"
  "${result_prefix}_smoke_float16.json"
  "${result_prefix}_smoke_decision.json"
  "${result_prefix}_full_bfloat16.json"
  "${result_prefix}_full_float16.json"
  "${result_prefix}_full_float32.json"
)
sources=(
  "$compile_report"
  "$smoke_bf16"
  "$smoke_fp16"
  "$smoke_decision"
  "$full_bf16"
  "$full_fp16"
  "$full_fp32"
)
decision_destinations=()
for decision in "${decision_files[@]}"; do
  candidate=${decision##*/decision_}
  candidate=${candidate%.json}
  destination="${result_prefix}_decision_${candidate}.json"
  decision_destinations+=("$destination")
  result_files+=("$destination")
done

evidence_sources=("${sources[@]}" "${decision_files[@]}")
evidence_destinations=("${result_files[@]}")
for destination in "${evidence_destinations[@]}"; do
  if [[ -e "$destination" ]]; then
    echo "refusing to overwrite existing evidence: $destination" >&2
    exit 2
  fi
done

cleanup_uncommitted_results() {
  if [[ $(git rev-parse HEAD) == "$SWITCHYARD_EXPECTED_HEAD" ]]; then
    git reset --quiet HEAD -- "${result_files[@]}" 2>/dev/null || true
    rm -f -- "${result_files[@]}"
  fi
}
trap cleanup_uncommitted_results EXIT

evidence_hashes=()
for source in "${evidence_sources[@]}"; do
  evidence_hashes+=("$(sha256sum -- "$source" | awk '{print $1}')")
done
for index in "${!evidence_sources[@]}"; do
  source=${evidence_sources[$index]}
  destination=${evidence_destinations[$index]}
  if [[ $(sha256sum -- "$source" | awk '{print $1}') != "${evidence_hashes[$index]}" ]]; then
    echo "evidence changed after evaluation: $source" >&2
    exit 2
  fi
  install -m 0644 "$source" "$destination"
  if [[ $(sha256sum -- "$destination" | awk '{print $1}') != "${evidence_hashes[$index]}" ]]; then
    echo "copied evidence does not match its evaluated bytes: $destination" >&2
    exit 2
  fi
done
for index in "${!evidence_sources[@]}"; do
  if [[ $(sha256sum -- "${evidence_sources[$index]}" | awk '{print $1}') != "${evidence_hashes[$index]}" ]]; then
    echo "evidence changed while files were copied" >&2
    exit 2
  fi
done

python scripts/check_evidence_binding.py \
  "${result_files[3]}" "${result_files[1]}" "${result_files[2]}"
for index in "${!decision_destinations[@]}"; do
  candidate=${candidates[$index]}
  copied_reports=("${result_files[4]}" "${result_files[5]}")
  if [[ "$candidate" == serial_* ]]; then
    copied_reports+=("${result_files[6]}")
  fi
  python scripts/check_evidence_binding.py \
    "${decision_destinations[$index]}" "${copied_reports[@]}"
done

python - "$manifest" <<'PY'
import json
import os
import sys

with open(os.environ["SWITCHYARD_GUARD_ATTESTATION"], encoding="utf-8") as handle:
    private_guard = json.load(handle)

guards = []
for item in private_guard["attempts"]:
    public = {key: value for key, value in item.items() if key != "target_gpu_uuid"}
    guards.append(public)

payload = {
    "schema_version": 1,
    "repository_commit": os.environ["SWITCHYARD_EXPECTED_HEAD"],
    "repository_tree": os.environ["SWITCHYARD_EXPECTED_TREE"],
    "benchmark_branch": os.environ["SWITCHYARD_CAMPAIGN_BRANCH"],
    "result_branch": os.environ.get("SWITCHYARD_RESULT_BRANCH", "codex/backward-architecture"),
    "origin": os.environ["SWITCHYARD_EXPECTED_ORIGIN"],
    "attempt": int(os.environ["SWITCHYARD_RUN_ATTEMPT"]),
    "device_id": os.environ["SWITCHYARD_PUBLIC_DEVICE_ID"],
    "guard_attestations": guards,
    "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
manifest_destination="${result_prefix}_manifest.json"
if [[ -e "$manifest_destination" ]]; then
  echo "refusing to overwrite existing evidence: $manifest_destination" >&2
  exit 2
fi
install -m 0644 "$manifest" "$manifest_destination"
if [[ $(sha256sum -- "$manifest" | awk '{print $1}') != \
      $(sha256sum -- "$manifest_destination" | awk '{print $1}') ]]; then
  echo "copied manifest does not match its source" >&2
  exit 2
fi
result_files+=("$manifest_destination")

git add -- "${result_files[@]}"
git diff --cached --check
check_result_bundle "$result_prefix" :
mapfile -t staged_files < <(git diff --cached --name-only)
mapfile -t expected_files < <(printf '%s\n' "${result_files[@]}" | sort)
mapfile -t actual_files < <(printf '%s\n' "${staged_files[@]}" | sort)
if [[ "${actual_files[*]}" != "${expected_files[*]}" ]]; then
  echo "staged result set is not exact" >&2
  exit 2
fi
if git diff --cached | rg -i "$forbidden_metadata"; then
  echo "forbidden authorship or task metadata found in staged results" >&2
  exit 2
fi

unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL
export GIT_AUTHOR_NAME=Ayman
export GIT_AUTHOR_EMAIL=ayman.hasib@outlook.com
export GIT_COMMITTER_NAME=Ayman
export GIT_COMMITTER_EMAIL=ayman.hasib@outlook.com
git commit -m "Record guarded backward campaign"
trap - EXIT

result_commit=$(git rev-parse HEAD)
if [[ $(git rev-parse HEAD^) != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
   [[ $(git show -s --format=%an%n%ae%n%cn%n%ce HEAD) != $'Ayman\nayman.hasib@outlook.com\nAyman\nayman.hasib@outlook.com' ]] ||
   git show -s --format=%B HEAD | rg -qi "$forbidden_metadata"; then
  echo "result commit failed parent, identity, or metadata audit" >&2
  exit 2
fi
mapfile -t committed_files < <(git diff-tree --no-commit-id --name-only -r HEAD | sort)
if [[ "${committed_files[*]}" != "${expected_files[*]}" ]]; then
  echo "result commit contains an unexpected path" >&2
  exit 2
fi
if git grep -IinE "$forbidden_metadata" HEAD -- "${result_files[@]}"; then
  echo "forbidden authorship, task, or host metadata found after commit" >&2
  exit 2
fi
check_result_bundle "$result_prefix" HEAD

if ! push_result_commit "$result_commit"; then
  echo "result commit is local and will be retried without rerunning GPU work" >&2
  exit 74
fi
if ! remote_head=$(read_result_branch); then
  echo "result commit was pushed but remote verification must be retried" >&2
  exit 74
fi
if [[ "$remote_head" != "$result_commit" ]] || [[ -n $(git status --porcelain) ]]; then
  echo "result push verification failed" >&2
  exit 2
fi
