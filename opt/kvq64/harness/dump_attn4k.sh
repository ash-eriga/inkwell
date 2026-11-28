#!/usr/bin/env bash
# Dump real F16 K/V/Q/bias/mask for full-attention layers at the 4096-token
# power-of-two decode step.  Run inside one gpu-run job.
set -u
BIN=${1:-/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v26-rowptr-jcache}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=${2:-/home/ash/.cache/inkwell-intkv/attn4k-v26}
rm -rf "$OUT"
mkdir -p "$OUT"
export GGML_PASCAL_GRAPHS=1 INKWELL_NO_GCACHE=1 INKWELL_MAX_CTX=4352 INKWELL_PROMPT_IDS=$ROOT/opt/kvq64/prompts/p4096.ids INKWELL_GENS=65
export INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10 INKWELL_KV=f16
export INKWELL_ATTN_DUMP_DIR=$OUT INKWELL_ATTN_DUMP_FILTER='5:*:4096,11:*:4096,17:*:4096,23:*:4096'
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
"$BIN" "$HOME/models/inkling" --run > "$OUT/run.log" 2>&1
rc=$?
echo "dump_rc=$rc out=$OUT"
find "$OUT" -type f -printf '%P %s\n' | sort | head -80
exit $rc
