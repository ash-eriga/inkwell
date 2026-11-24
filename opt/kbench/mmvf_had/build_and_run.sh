#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-$HOME/lfm-gtx1060-bench/cuda-12.6}"
NVCC="$CUDA_HOME/bin/nvcc"
CCBIN="${CCBIN:-$HOME/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++}"
GOLD="${1:-/tmp/sgemv_gold}"
BIN="/tmp/bench_mmvf_had"

echo "[1/3] Compiling bench_mmvf_had.cu for sm_61..."
"$NVCC" -O3 -std=c++17 --use_fast_math -arch=sm_61 -ccbin="$CCBIN" -Xptxas -v \
    -I"$DIR" "$DIR/bench_mmvf_had.cu" -L"$CUDA_HOME/lib64" -lcublas -o "$BIN" \
    2> /tmp/mmvf_had_ptxas.log

echo "[2/3] Kernel register / smem / spill summary:"
python3 -c '
import re
txt = open("/tmp/mmvf_had_ptxas.log").read()
cur = None
for line in txt.splitlines():
    m = re.search(r"Compiling entry function '\''([^'\'']+)'\''", line)
    if m:
        cur = m.group(1)
    m2 = re.search(r"Used (\d+) registers(?:, (\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads)?(?:, (\d+) bytes smem)?", line)
    if cur and ("Used " in line and "registers" in line):
        if any(k in cur for k in ["mmvf_bf16", "router", "gate_router", "rms2down", "gf_apply", "sink_relbias", "kv_sconv", "had_in", "had_out", "situ_had", "gemv_moe_s"]):
            print(f"  {cur[:68]:68s} | {line.strip()}")
'

echo "[3/3] Running CUDA-Graph interleaved benchmark under flock /tmp/inkwell-gpu.lock..."
LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
    flock /tmp/inkwell-gpu.lock "$BIN" "$GOLD" | tee /tmp/mmvf_had_bench.log
