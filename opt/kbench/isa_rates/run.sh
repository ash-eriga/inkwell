#!/bin/bash
set -eu
cd "$(dirname "$0")"
CUDA=~/lfm-gtx1060-bench/cuda-12.6
$CUDA/bin/nvcc -O3 -std=c++17 \
  -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -o isa_rates isa_rates.cu
$CUDA/bin/cuobjdump -sass isa_rates > /tmp/isa_rates.sass
./loop_ops.py /tmp/isa_rates.sass
nice -n 19 flock /tmp/inkwell-gpu.lock ./isa_rates
