#!/bin/sh
# Build and run the decode-attention context probe on the GTX 1060.
# Usage: opt/kbench/attn_ctx/run.sh   (GPU part runs under the shared lock)
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../../crates/inkwell-cuda/src
cuda=/home/ash/lfm-gtx1060-bench/cuda-12.6
"$cuda/bin/nvcc" -O3 -std=c++17 --use_fast_math \
  -ccbin /home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -I"$src" "$here/probe.cu" -o /tmp/attn_ctx_probe
LD_LIBRARY_PATH=$cuda/lib64 flock /tmp/inkwell-gpu.lock /tmp/attn_ctx_probe
