#!/usr/bin/env bash
# Unpack the Python development headers into .local-toolchain/ without
# installing anything system-wide.
#
# The target machine has python3.12 (3.12.3-1ubuntu0.15) but not
# python3.12-dev, so Python.h is absent. Triton and Inductor both build a small
# C launcher for every compiled kernel and need that header. Rather than ask
# for root on a shared machine, we download the matching Ubuntu package and
# extract it locally; `apt-get download` needs no privileges and `dpkg-deb -x`
# writes only where we tell it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/.local-toolchain"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "downloading python3.12 headers..."
(cd "$WORK" && apt-get download libpython3.12-dev python3.12-dev)

mkdir -p "$DEST"
for deb in "$WORK"/*.deb; do
  echo "extracting $(basename "$deb")"
  dpkg-deb -x "$deb" "$DEST"
done

HDR="$DEST/usr/include/python3.12/Python.h"
[ -f "$HDR" ] || { echo "failed: $HDR not present" >&2; exit 1; }
echo "ok: $HDR"
echo "now run: source scripts/env.sh"
