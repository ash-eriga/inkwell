#!/bin/bash
# hill-climb harness: N decode samples of 64 tokens, one JSONL line each
# usage: bench.sh <config-name> <N>
set -u
CFG=$1
N=${2:-6}
cd ~/inkwell
export LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin
export GGML_PASCAL_GRAPHS=1
GOLDEN='正确答案：A'
for i in $(seq 1 "$N"); do
    out=$(timeout 180 ./target/release/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run 2>&1)
    s=$(echo "$out" | grep -oP 'decode 8 tok in \K[0-9.]+')
    gold=no; echo "$out" | grep -q "$GOLDEN" && gold=yes
    if [ -n "$s" ]; then
        ms=$(echo "$s" | awk '{printf "%.1f", $1*1000/64}')
        ts=$(echo "$s" | awk '{printf "%.2f", 64/$1}')
        printf '{"ts":"%s","config":"%s","i":%d,"decode_s":%s,"ms_tok":%s,"tok_s":%s,"golden":"%s"}\n' \
            "$(date -Is)" "$CFG" "$i" "$s" "$ms" "$ts" "$gold" | tee -a ~/inkwell/opt/ledger.jsonl
    else
        printf '{"ts":"%s","config":"%s","i":%d,"error":"no decode line"}\n' \
            "$(date -Is)" "$CFG" "$i" | tee -a ~/inkwell/opt/ledger.jsonl
        echo "$out" | tail -5
    fi
done
