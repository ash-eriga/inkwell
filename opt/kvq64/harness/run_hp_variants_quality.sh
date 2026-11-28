#!/usr/bin/env bash
# HP window variants real 4K quality, one job: find the minimum fp16 local
# window that preserves the HP1024 quality win (mean 0.00094/top1 1.0).
# Mechanism hypothesis: local>=1024 makes all SWA layers (cap 1024) exact fp16.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
runvar() {
  local tag=$1 hp=$2
  local dst=$ART/quality-$tag-ctx4096
  mkdir -p "$dst"
  for f in ref.bin repeat.bin noise64.bin teacher.ids; do
    if [ -s "$SRC/$f" ] && [ ! -e "$dst/$f" ]; then ln "$SRC/$f" "$dst/$f" 2>/dev/null || cp "$SRC/$f" "$dst/$f"; fi
  done
  echo "== K8V6 HP$hp 4K quality =="
  env INKWELL_KV_HP=$hp "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 "$tag" k8v6 || return $?
}
runvar v6hp0_1024 0:1024 || exit $?
runvar v6hp0_512 0:512 || exit $?
runvar v6hp0_256 0:256 || exit $?
runvar v6hp8_1024 8:1024 || exit $?
