#!/usr/bin/env bash
# Run the staged backward campaign on the one GPU selected by the idle runner.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${SWITCHYARD_EXPECTED_HEAD:?set SWITCHYARD_EXPECTED_HEAD}"
: "${SWITCHYARD_CAMPAIGN_DIR:?set SWITCHYARD_CAMPAIGN_DIR outside the repository}"
: "${SWITCHYARD_TARGET_GPU_UUID:?the idle runner must select one GPU UUID}"

campaign_dir=$(realpath -m -- "$SWITCHYARD_CAMPAIGN_DIR")
if [[ "$campaign_dir" == "$repo_dir" || "$campaign_dir" == "$repo_dir/"* ]]; then
  echo "campaign state must remain outside the repository" >&2
  exit 2
fi
SWITCHYARD_CAMPAIGN_DIR=$campaign_dir

if [[ $(git rev-parse HEAD) != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "repository HEAD changed while the campaign waited" >&2
  exit 2
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "repository must be clean before the campaign" >&2
  exit 2
fi
remote_head=$(git ls-remote --heads origin refs/heads/codex/backward-architecture | awk '{print $1}')
if [[ "$remote_head" != "$SWITCHYARD_EXPECTED_HEAD" ]]; then
  echo "remote branch changed while the campaign waited" >&2
  exit 2
fi
if [[ $(git config --local user.name) != "Ayman" ]] ||
   [[ $(git config --local user.email) != "ayman.hasib@outlook.com" ]]; then
  echo "refusing to commit with unexpected Git identity" >&2
  exit 2
fi

mkdir -p "$SWITCHYARD_CAMPAIGN_DIR"
gpu_complete_marker="$SWITCHYARD_CAMPAIGN_DIR/gpu_complete"
rm -f "$gpu_complete_marker"
source scripts/env.sh

smoke_bf16="$SWITCHYARD_CAMPAIGN_DIR/smoke_bfloat16.json"
smoke_fp16="$SWITCHYARD_CAMPAIGN_DIR/smoke_float16.json"
smoke_decision="$SWITCHYARD_CAMPAIGN_DIR/smoke_decision.json"
full_bf16="$SWITCHYARD_CAMPAIGN_DIR/full_bfloat16.json"
full_fp16="$SWITCHYARD_CAMPAIGN_DIR/full_float16.json"
full_decision="$SWITCHYARD_CAMPAIGN_DIR/full_decision_cuda_cluster.json"

timeout --foreground 30m python -m pytest tests/test_backward_candidates.py -q

timeout --foreground 2h python bench/bench_backward.py \
  --shape-set gate --dtype bfloat16 --quick --out "$smoke_bf16"
timeout --foreground 2h python bench/bench_backward.py \
  --shape-set gate --dtype float16 --quick --out "$smoke_fp16"
python scripts/check_backward_smoke.py \
  "$smoke_bf16" "$smoke_fp16" --out "$smoke_decision"

full_impls=current,cuda_cluster,liger
timeout --foreground 6h python bench/bench_backward.py \
  --shape-set full --dtype bfloat16 --impls "$full_impls" --out "$full_bf16"
timeout --foreground 6h python bench/bench_backward.py \
  --shape-set full --dtype float16 --impls "$full_impls" --out "$full_fp16"

set +e
python scripts/evaluate_backward.py --candidate cuda_cluster --json \
  "$full_bf16" "$full_fp16" >"$full_decision"
evaluation_exit=$?
set -e
if [[ $evaluation_exit -gt 1 ]] || ! python -m json.tool "$full_decision" >/dev/null; then
  echo "full evaluator did not produce a valid decision" >&2
  exit 3
fi
touch "$gpu_complete_marker"

# Benchmark provenance was captured while the worktree was clean. Copy results
# only after every GPU measurement is complete.
if [[ $(git rev-parse HEAD) != "$SWITCHYARD_EXPECTED_HEAD" ]] ||
   [[ -n $(git status --porcelain) ]]; then
  echo "repository changed during the campaign; results remain outside the repository" >&2
  exit 2
fi

campaign_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$full_bf16")
result_prefix="results/backward_campaign_${campaign_id}"
install -m 0644 "$smoke_bf16" "${result_prefix}_smoke_bfloat16.json"
install -m 0644 "$smoke_fp16" "${result_prefix}_smoke_float16.json"
install -m 0644 "$smoke_decision" "${result_prefix}_smoke_decision.json"
install -m 0644 "$full_bf16" "${result_prefix}_full_bfloat16.json"
install -m 0644 "$full_fp16" "${result_prefix}_full_float16.json"
install -m 0644 "$full_decision" "${result_prefix}_decision_cuda_cluster.json"

git add \
  "${result_prefix}_smoke_bfloat16.json" \
  "${result_prefix}_smoke_float16.json" \
  "${result_prefix}_smoke_decision.json" \
  "${result_prefix}_full_bfloat16.json" \
  "${result_prefix}_full_float16.json" \
  "${result_prefix}_decision_cuda_cluster.json"
git diff --cached --check
decision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$full_decision")
git commit -m "Record guarded backward campaign: $decision"
git push origin codex/backward-architecture
local_head=$(git rev-parse HEAD)
remote_head=$(git ls-remote --heads origin refs/heads/codex/backward-architecture | awk '{print $1}')
[[ "$local_head" == "$remote_head" ]]
[[ -z $(git status --porcelain) ]]
