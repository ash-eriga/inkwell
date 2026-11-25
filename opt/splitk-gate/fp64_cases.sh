#!/bin/bash
# Compare old and split attention against fp64 CPU references on dumped real tensors.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
BASE="$ART/dumps/q8192-old"
OUT="$ART/results/fp64_attention.txt"
: > "$OUT"
for spec in 1024/l5p0 4096/l5p0 8192/l5p0 1024/l11p0 4096/l17p0 8192/l23p0; do
  nkv=${spec%%/*}
  lp=${spec##*/}
  for chunk in 32 64; do
    echo "== nkv=$nkv case=$lp chunk=$chunk ==" | tee -a "$OUT"
    nice -n 19 flock /tmp/inkwell-gpu.lock \
      env INKWELL_ATTNSPLIT_CHUNK=$chunk \
      LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64" \
      "$ROOT/opt/splitk-gate/bin/attn_ctx_probe" load "$BASE/$nkv/$lp" "$nkv" 2>&1 | tee -a "$OUT"
  done
done
