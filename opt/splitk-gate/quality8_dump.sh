#!/bin/bash
# 8K divergence runs; the old-kernel run also captures real attention tensors.
set -eu
CTX=8192
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
PROMPT=$(cat "$ART/prompts/p$CTX.txt")
MAXCTX=8448
FILTER='5:0:1024,5:0:4096,5:0:8192,11:0:1024,17:0:4096,23:0:8192'
"$ROOT/opt/splitk-gate/run_infer.sh" q8192-old \
  INKWELL_MAX_CTX=$MAXCTX INKWELL_GENS=256 \
  INKWELL_LOGIT_DUMP="$ART/logits/q8192-old.bin" \
  INKWELL_PROMPT="$PROMPT" INKWELL_NO_ATTNSPLIT=1 \
  INKWELL_ATTN_DUMP_DIR="$ART/dumps/q8192-old" \
  INKWELL_ATTN_DUMP_FILTER="$FILTER"
"$ROOT/opt/splitk-gate/run_infer.sh" q8192-sdef \
  INKWELL_MAX_CTX=$MAXCTX INKWELL_GENS=256 \
  INKWELL_LOGIT_DUMP="$ART/logits/q8192-sdef.bin" \
  INKWELL_PROMPT="$PROMPT" INKWELL_ATTNSPLIT_MIN=320
"$ROOT/opt/splitk-gate/run_infer.sh" q8192-c64 \
  INKWELL_MAX_CTX=$MAXCTX INKWELL_GENS=256 \
  INKWELL_LOGIT_DUMP="$ART/logits/q8192-c64.bin" \
  INKWELL_PROMPT="$PROMPT" INKWELL_ATTNSPLIT_MIN=320 INKWELL_ATTNSPLIT_CHUNK=64
