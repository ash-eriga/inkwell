#!/bin/sh
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../../../crates/inkwell-cuda/src
cuda=/home/ash/lfm-gtx1060-bench/cuda-12.6
"$cuda/bin/nvcc" -O3 -std=c++17 --use_fast_math \
  -ccbin /home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -I"$src" "$here/probe_relpos.cu" -L"$cuda/lib64" -lcuda -o /tmp/dyn_nkv_relpos_probe
LD_LIBRARY_PATH=$cuda/lib64 nice -n 19 flock /tmp/inkwell-gpu.lock /tmp/dyn_nkv_relpos_probe
