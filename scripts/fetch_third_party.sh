#!/usr/bin/env bash
#
# Clone the public fused Block AttnRes implementations that
# bench/bench_third_party.py compares against.
#
#   scripts/fetch_third_party.sh                  # the pinned commits the results were measured at
#   FETCH_LATEST=1 scripts/fetch_third_party.sh   # track each project's default branch instead
#
# They are deliberately NOT vendored. Each carries its own licence and its own
# release cadence, and a head-to-head is only meaningful against the upstream
# source rather than a copy of it that has quietly drifted. Pinning the commits
# here is what makes the numbers in results/third_party_bfloat16.json
# reproducible; FETCH_LATEST=1 is the way to re-run against current upstream.
#
# Nothing is installed and nothing is built. All three are pure Python plus
# Triton, so bench/bench_third_party.py just puts these directories on sys.path.
# That also means the benchmark degrades to "skipped, clone absent" rather than
# failing when this script has not been run.
#
# The destination is gitignored. Override it with THIRD_PARTY_DIR, and pass the
# same value to bench/bench_third_party.py.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${THIRD_PARTY_DIR:-$REPO/third_party}"

# Commits that results/third_party_bfloat16.json was measured against.
LIGER_REF="777799588a89d74c489ed995e3bf006427738e85"
FLA_REF="35dceaee5408e69a555fec34cb215c93c375dabe"
CATSWE_REF="ff92865e4e1b18809da7a8f0c0c5252039cded7c"

fetch() {
  local name="$1" url="$2" ref="$3"
  local dir="$DEST/$name"

  if [ ! -d "$dir/.git" ]; then
    # A blobless partial clone: full history graph, file contents fetched on
    # demand. Fast, and unlike --depth 1 it can still check out a pinned commit.
    echo "== $name: cloning $url"
    git clone --quiet --filter=blob:none "$url" "$dir"
  else
    echo "== $name: reusing $dir"
    git -C "$dir" fetch --quiet origin
  fi

  if [ -n "${FETCH_LATEST:-}" ]; then
    git -C "$dir" checkout --quiet "$(git -C "$dir" symbolic-ref --short refs/remotes/origin/HEAD | cut -d/ -f2-)"
    git -C "$dir" pull --quiet --ff-only
  else
    git -C "$dir" checkout --quiet --detach "$ref"
  fi

  echo "   $(git -C "$dir" rev-parse --short HEAD)  $(git -C "$dir" log -1 --format='%cs  %s')"
}

mkdir -p "$DEST"
fetch Liger-Kernel            https://github.com/linkedin/Liger-Kernel.git            "$LIGER_REF"
fetch flash-linear-attention  https://github.com/fla-org/flash-linear-attention.git   "$FLA_REF"
fetch flash-attention-residuals https://github.com/catswe/flash-attention-residuals.git "$CATSWE_REF"

cat <<EOF

Fetched into $DEST. Now run:

    source scripts/env.sh
    python bench/bench_third_party.py
EOF
