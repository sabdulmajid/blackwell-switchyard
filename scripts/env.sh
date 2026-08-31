#!/usr/bin/env bash
# Environment for running this project on the target machine.
#
#   source scripts/env.sh
#
# Why this exists: the host has python3.12 but not python3.12-dev, so there is
# no Python.h. Triton builds a small C launcher shim for every kernel it
# compiles, and Inductor does the same, so *nothing* GPU-related works without
# those headers -- including torch.compile.
#
# Installing python3.12-dev system-wide would need root on a shared machine, so
# instead scripts/fetch_python_headers.sh unpacks the matching Ubuntu .deb into
# .local-toolchain/ (gitignored) and this script points the compiler at it. No
# system path is written to and no package is installed.
#
# Caches are redirected to the repo's filesystem because $HOME lives on a
# volume that is 97% full.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="$REPO/.local-toolchain"

if [ ! -f "$TOOLCHAIN/usr/include/python3.12/Python.h" ]; then
  echo "note: Python headers missing; run scripts/fetch_python_headers.sh" >&2
else
  export CPATH="$TOOLCHAIN/usr/include/python3.12:$TOOLCHAIN/usr/include${CPATH:+:$CPATH}"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$REPO/.cache/inductor}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "switchyard env: CUDA_HOME=$CUDA_HOME  headers=$([ -n "$CPATH" ] && echo local || echo system)"
