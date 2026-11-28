#!/usr/bin/env bash
# 64K KVQ attention attribution trace; CUDA graphs disabled so kernel executions are visible.
set -u
BIN=${1:-/home/ash/.cache/inkwell-front2/bin/inkwell-f2-t532}
TAG=${2:-t532}
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
OUT=$ROOT/opt/kvq64/artifacts/nsys-$TAG-64k
mkdir -p "$OUT"
STATE=$ROOT/opt/kvq64/artifacts/mv2-k8v4-ctx65536/state.bin
export INKWELL_MAX_CTX=65536 INKWELL_PROMPT_IDS=$ROOT/opt/kvq64/prompts/p65536.ids INKWELL_GENS=64 INKWELL_WARM_GENS=16 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10 INKWELL_KV=k8v4 INKWELL_STATE_LOAD=$STATE INKWELL_KVQ_DP4A=2 INKWELL_VRAM_CAP_MB=5180
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys profile -o "$OUT/trace" --force-overwrite true -t cuda --cpuctxsw=none "$BIN" "$HOME/models/inkling" --run >"$OUT/run.log" 2>&1
rc=$?
echo "nsys_rc=$rc"
if [ $rc -eq 0 ]; then
 /home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys stats --report cuda_gpu_kern_sum "$OUT/trace.nsys-rep" > "$OUT/kernsum.txt" 2>&1
 tail -12 "$OUT/kernsum.txt"
fi
exit $rc
