#!/bin/bash
# REXL3 decode bench: N runs of INKWELL_GENS tokens → JSONL lines
# usage: rexl3_bench.sh <config-name> <N>
set -u
CFG=$1; N=${2:-6}
cd ~/inkwell
export LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin
export GGML_PASCAL_GRAPHS=1 INKWELL_GENS=8
for i in $(seq 1 "$N"); do
    out=$(timeout 300 ./target/release/inkwell ~/models/inkling --run 2>&1)
    s=$(echo "$out" | grep -oP 'decode 8 tok in \K[0-9.]+(ms)?' | head -1)
    txt=$(echo "$out" | grep -oP 'generated text: "\K[^"]+')
    gold=no; echo "$txt" | grep -q 'capital of France' && gold=yes
    if [ -n "$s" ]; then
        ms=$(echo "$s" | awk '{ if ($0 ~ /ms$/) { printf "%.1f", ($1+0)/8 } else { printf "%.1f", $1*1000/8 } }')
        printf '{"ts":"%s","config":"%s","i":%d,"decode_s":%s,"ms_tok":%s,"golden":"%s","text":"%s"}\n' \
            "$(date -Is)" "$CFG" "$i" "$s" "$ms" "$gold" "$txt" | tee -a ~/inkwell/opt/ledger.jsonl
    else
        printf '{"ts":"%s","config":"%s","i":%d,"error":"no decode line","tail":"%s"}\n' \
            "$(date -Is)" "$CFG" "$i" "$(echo "$out" | tail -3 | tr '\n' '|')" | tee -a ~/inkwell/opt/ledger.jsonl
    fi
done
