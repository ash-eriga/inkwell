#!/usr/bin/env bash
# kernel-attribution trace at 64K on the final binary, CUDA graphs disabled so
# individual kernel executions are visible. usage inside gpu-run: run_nsys64_v13.sh <fmt>
set -u
FMT=${1:-k4v2}
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg
OUT=$ROOT/opt/kvq64/artifacts/nsys-v13-$FMT-64k
mkdir -p "$OUT"
STATE=$ROOT/opt/kvq64/artifacts/mv2-$FMT-ctx65536/state.bin
export INKWELL_MAX_CTX=65536 INKWELL_PROMPT_IDS=$ROOT/opt/kvq64/prompts/p65536.ids INKWELL_GENS=64 INKWELL_WARM_GENS=16 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10 INKWELL_KV=$FMT INKWELL_STATE_LOAD=$STATE
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys profile -o "$OUT/trace" --force-overwrite true -t cuda --cpuctxsw=none "$BIN" "$HOME/models/inkling" --run >"$OUT/run.log" 2>&1
echo "nsys_rc=$?"
/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys stats --report cuda_gpu_kern_sum "$OUT/trace.nsys-rep" > "$OUT/kernsum.txt" 2>&1
/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys stats --report cuda_gpu_mem_time_sum "$OUT/trace.nsys-rep" > "$OUT/memsum.txt" 2>&1
tail -5 "$OUT/kernsum.txt"
