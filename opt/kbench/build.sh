#!/bin/bash
# build a kbench .cu with the same toolchain/flags as crates/inkwell-cuda/build.rs
# usage: build.sh <name> [extra nvcc flags...]  -> opt/kbench/<name>
set -eu
name=$1; shift
cd "$(dirname "$0")"
CUDA=~/lfm-gtx1060-bench/cuda-12.6
$CUDA/bin/nvcc -O3 -std=c++17 --use_fast_math \
  -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -lineinfo -I../../crates/inkwell-cuda/src \
  "$@" -o "$name" "$name.cu"
echo "built opt/kbench/$name"
