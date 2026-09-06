#!/usr/bin/env bash
# Run the bounded backward campaign on the GPU selected by the idle runner.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${SWITCHYARD_EXPECTED_HEAD:?set SWITCHYARD_EXPECTED_HEAD}"
: "${SWITCHYARD_CAMPAIGN_BRANCH:?set SWITCHYARD_CAMPAIGN_BRANCH}"
: "${SWITCHYARD_CAMPAIGN_DIR:?set SWITCHYARD_CAMPAIGN_DIR outside the repository}"
: "${SWITCHYARD_TARGET_GPU_UUID:?the idle runner must select one GPU UUID}"
: "${SWITCHYARD_RUN_ATTEMPT:?the idle runner must number each attempt}"
: "${THIRD_PARTY_DIR:?set THIRD_PARTY_DIR to the pinned dependency directory}"
: "${SWITCHYARD_TOOLCHAIN_DIR:?set SWITCHYARD_TOOLCHAIN_DIR to the local Python headers}"

result_branch=${SWITCHYARD_RESULT_BRANCH:-codex/backward-architecture}
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
if [[ "$git_dir" == "$git_common_dir" || "$git_index" != "$git_dir/index" ]]; then
  echo "campaign must run from a dedicated linked worktree and index" >&2
  exit 2
fi
if [[ "$git_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "repository HEAD changed while the campaign waited" >&2
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
remote_head=$(git ls-remote --heads origin "refs/heads/$result_branch" | awk '{print $1}')
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

attempt_dir="$campaign_dir/attempt-$SWITCHYARD_RUN_ATTEMPT"
if ! mkdir "$attempt_dir"; then
  echo "attempt directory already exists: $attempt_dir" >&2
  exit 2
fi
run_dir="$campaign_dir/run"
mkdir -p "$run_dir"
gpu_complete_marker="$campaign_dir/gpu_complete"
rm -f "$gpu_complete_marker"

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

all_impls=current,serial_recompute_atomic_t4,serial_saved_partials_t16,cuda_shared,cuda_cluster,cuda_cluster4,cuda_register,liger
portable_impls=current,serial_recompute_atomic_t4,serial_saved_partials_t16,liger

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
    --expected-gpu-uuid "$SWITCHYARD_TARGET_GPU_UUID" $quick_flag
}

if report_reusable "$smoke_bf16" bfloat16 gate "$all_impls" --quick; then
  echo "reusing clean completed bf16 smoke phase"
else
  rm -f "$smoke_bf16"
  timeout --foreground 2h python bench/bench_backward.py \
    --shape-set gate --dtype bfloat16 --quick --impls "$all_impls" --out "$smoke_bf16"
fi
if report_reusable "$smoke_fp16" float16 gate "$all_impls" --quick; then
  echo "reusing clean completed fp16 smoke phase"
else
  rm -f "$smoke_fp16"
  timeout --foreground 2h python bench/bench_backward.py \
    --shape-set gate --dtype float16 --quick --impls "$all_impls" --out "$smoke_fp16"
fi
python scripts/check_backward_smoke.py \
  "$smoke_bf16" "$smoke_fp16" \
  --expected-commit "$SWITCHYARD_EXPECTED_HEAD" \
  --expected-branch "$SWITCHYARD_CAMPAIGN_BRANCH" \
  --out "$smoke_decision"

if report_reusable "$full_bf16" bfloat16 full "$all_impls"; then
  echo "reusing clean completed bf16 full phase"
else
  rm -f "$full_bf16"
  timeout --foreground 8h python bench/bench_backward.py \
    --shape-set full --dtype bfloat16 --impls "$all_impls" --out "$full_bf16"
fi
if report_reusable "$full_fp16" float16 full "$all_impls"; then
  echo "reusing clean completed fp16 full phase"
else
  rm -f "$full_fp16"
  timeout --foreground 8h python bench/bench_backward.py \
    --shape-set full --dtype float16 --impls "$all_impls" --out "$full_fp16"
fi
if report_reusable "$full_fp32" float32 full "$portable_impls"; then
  echo "reusing clean completed fp32 full phase"
else
  rm -f "$full_fp32"
  timeout --foreground 6h python bench/bench_backward.py \
    --shape-set full --dtype float32 --impls "$portable_impls" --out "$full_fp32"
fi

# GPU work is complete. Evaluation, result copying, commit, and push are CPU and
# network work. The outer runner stops GPU polling after this marker so another
# project may safely start while this campaign finalizes its evidence.
touch "$gpu_complete_marker"

candidates=(
  serial_recompute_atomic_t4
  serial_saved_partials_t16
  cuda_shared
  cuda_cluster
  cuda_cluster4
  cuda_register
)
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
    "${evaluation_reports[@]}" >"$decision"
  evaluation_exit=$?
  set -e
  if [[ $evaluation_exit -gt 1 ]] || ! python -m json.tool "$decision" >/dev/null; then
    echo "evaluator failed for $candidate" >&2
    exit 3
  fi
  decision_status=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$decision")
  case "$decision_status" in
    DROP|REJECT|READY_FOR_DISPATCH_REVIEW) ;;
    *)
      echo "evaluator did not reach a terminal decision for $candidate: $decision_status" >&2
      exit 3
      ;;
  esac
  decision_files+=("$decision")
done

if [[ $(git rev-parse HEAD) != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
   [[ $(git branch --show-current) != "$SWITCHYARD_CAMPAIGN_BRANCH" ]] ||
   [[ -n $(git status --porcelain) ]]; then
  echo "campaign worktree changed during measurement; results remain external" >&2
  exit 2
fi
remote_head=$(git ls-remote --heads origin "refs/heads/$result_branch" | awk '{print $1}')
if [[ "$remote_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "result branch changed during measurement; results remain external" >&2
  exit 2
fi

campaign_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$full_bf16")
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
for index in "${!sources[@]}"; do
  install -m 0644 "${sources[$index]}" "${result_files[$index]}"
done
for decision in "${decision_files[@]}"; do
  candidate=${decision##*/decision_}
  candidate=${candidate%.json}
  destination="${result_prefix}_decision_${candidate}.json"
  install -m 0644 "$decision" "$destination"
  result_files+=("$destination")
done

python - "$manifest" <<'PY'
import json
import os
import sys

payload = {
    "schema_version": 1,
    "repository_commit": os.environ["SWITCHYARD_EXPECTED_HEAD"],
    "benchmark_branch": os.environ["SWITCHYARD_CAMPAIGN_BRANCH"],
    "result_branch": os.environ.get("SWITCHYARD_RESULT_BRANCH", "codex/backward-architecture"),
    "attempt": int(os.environ["SWITCHYARD_RUN_ATTEMPT"]),
    "gpu_uuid": os.environ["SWITCHYARD_TARGET_GPU_UUID"],
    "guard": {
        "not_before": os.environ.get("SWITCHYARD_GUARD_NOT_BEFORE"),
        "idle_seconds": int(os.environ.get("SWITCHYARD_GUARD_IDLE_SECONDS", "0")),
        "wait_poll_seconds": int(os.environ.get("SWITCHYARD_GUARD_WAIT_POLL_SECONDS", "0")),
        "watchdog_seconds": int(os.environ.get("SWITCHYARD_GUARD_WATCHDOG_SECONDS", "0")),
        "finalize_seconds": int(os.environ.get("SWITCHYARD_GUARD_FINALIZE_SECONDS", "0")),
    },
    "liger_commit": "777799588a89d74c489ed995e3bf006427738e85",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
manifest_destination="${result_prefix}_manifest.json"
install -m 0644 "$manifest" "$manifest_destination"
result_files+=("$manifest_destination")

git add -- "${result_files[@]}"
git diff --cached --check
mapfile -t staged_files < <(git diff --cached --name-only)
mapfile -t expected_files < <(printf '%s\n' "${result_files[@]}" | sort)
mapfile -t actual_files < <(printf '%s\n' "${staged_files[@]}" | sort)
if [[ "${actual_files[*]}" != "${expected_files[*]}" ]]; then
  echo "staged result set is not exact" >&2
  exit 2
fi
forbidden_metadata='co-authored-by|claude|anthropic|wizchem|chatgpt|openai\.com|session[-_/][[:alnum:]]|file://|/(home|tmp|pub[0-9]+)/'
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

git push origin "HEAD:refs/heads/$result_branch"
remote_head=$(git ls-remote --heads origin "refs/heads/$result_branch" | awk '{print $1}')
if [[ "$remote_head" != "$result_commit" ]] || [[ -n $(git status --porcelain) ]]; then
  echo "result push verification failed" >&2
  exit 2
fi
