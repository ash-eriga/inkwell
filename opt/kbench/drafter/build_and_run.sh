#!/usr/bin/env bash
# Reproducible build & run script for opt/kbench/drafter/.
# Always wraps GPU runs in `flock /tmp/inkwell-gpu.lock`.
set -euo pipefail

CUDA=~/lfm-gtx1060-bench/cuda-12.6
CCBIN=~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++
NVCC_FLAGS="-O3 -std=c++17 --use_fast_math -Xptxas -v -ccbin $CCBIN -gencode arch=compute_61,code=sm_61"

echo "=== 1. Question 1: Cost Model (CPU) ==="
python3 -I opt/kbench/drafter/cost_model.py

echo "=== 2. Question 2: Independent Corpus Vocab Ranking (CPU) ==="
python3 -I opt/kbench/drafter/build_vocab_rank.py
if [ -d /tmp/drafter_data/serial ]; then
  python3 -I opt/kbench/drafter/vocab_coverage.py
fi

echo "=== 3. Compile sm_61 Drafter Microbenchmarks & Acceptance Proxy ==="
$CUDA/bin/nvcc $NVCC_FLAGS opt/kbench/drafter/bench_drafter.cu -o /tmp/bench_drafter
$CUDA/bin/nvcc -O3 -std=c++17 -ccbin "$CCBIN" -gencode arch=compute_61,code=sm_61 \
  opt/kbench/drafter/acceptance_proxy.cu -o /tmp/acceptance_proxy

echo "=== 4. Run GPU Microbenchmarks + Acceptance Proxy under flock ==="
flock /tmp/inkwell-gpu.lock bash -c '
  /tmp/bench_drafter
  if [ -f /tmp/drafter_data/proxy_inputs/manifest.txt ] && [ -d /tmp/mtp_dense ]; then
    /tmp/acceptance_proxy
  else
    echo "Skipping acceptance_proxy (run collect_data.sh + prep_proxy_data.py first)"
  fi
'
