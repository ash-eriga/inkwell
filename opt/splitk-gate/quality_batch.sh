#!/bin/bash
# Sequential full-logit divergence runs for one exact prompt length.
set -eu
CTX=$1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
PROMPT=$(cat "$ART/prompts/p$CTX.txt")
if [ "$CTX" = 4096 ]; then MAXCTX=4352; elif [ "$CTX" = 8192 ]; then MAXCTX=8448; else echo "bad ctx" >&2; exit 2; fi
run() {
  tag=$1; shift
  "$ROOT/opt/splitk-gate/run_infer.sh" "$tag" \
    INKWELL_MAX_CTX=$MAXCTX INKWELL_GENS=256 \
    INKWELL_LOGIT_DUMP="$ART/logits/$tag.bin" \
    INKWELL_PROMPT="$PROMPT" "$@"
}
run "q$CTX-old" INKWELL_NO_ATTNSPLIT=1
run "q$CTX-sdef" INKWELL_ATTNSPLIT_MIN=320
run "q$CTX-c64" INKWELL_ATTNSPLIT_MIN=320 INKWELL_ATTNSPLIT_CHUNK=64
