#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-$HOME/lfm-gtx1060-bench/cuda-12.6/bin/nvcc}"
CUOBJ="${CUOBJ:-$HOME/lfm-gtx1060-bench/cuda-12.6/bin/cuobjdump}"
CCBIN="${CCBIN:-$HOME/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++}"
GOLD="${1:-/tmp/sgemv_dense_gold}"
BIN="/tmp/bench_w4dense"

echo "[1/3] Compiling bench_w4dense.cu for sm_61..."
"$NVCC" -O3 -arch=sm_61 -ccbin="$CCBIN" --use_fast_math -Xptxas -v \
    -I"$DIR" "$DIR/bench_w4dense.cu" -o "$BIN" 2> /tmp/w4dense_ptxas.log

echo "[2/3] Inspecting SASS & occupancy (cuobjdump -sass)..."
"$CUOBJ" -sass "$BIN" > /tmp/w4dense.sass
python3 "$DIR/sass_inspect.py" /tmp/w4dense.sass /tmp/w4dense_ptxas.log | tee /tmp/w4dense_sass_table.txt

echo "[3/3] Running CPU-verified interleaved GPU benchmark under flock /tmp/inkwell-gpu.lock..."
flock /tmp/inkwell-gpu.lock "$BIN" "$GOLD" | tee /tmp/w4dense_bench.log
