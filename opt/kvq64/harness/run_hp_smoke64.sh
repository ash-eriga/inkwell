#!/usr/bin/env bash
# K8V6+HP 64K smoke: correctness, peak VRAM, indicative decode speed.
# HP memory is a fixed sliding window (hp_cap*8*128*2*2 per pass) independent
# of context length; verify peak stays <=5120 and decode is coherent.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
M=$ROOT/opt/kvq64/harness/measure_ctx.sh
HP=${HP:-0:1024}
echo "== K8V6 HP$HP 64K smoke =="
env INKWELL_KV_HP=$HP "$M" "$BIN" 65536 "v6hp${HP//:/_}-smoke" INKWELL_KV=k8v6 INKWELL_WARM_GENS=16 || exit $?
