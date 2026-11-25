#!/bin/sh
set -eu
cd "$(dirname "$0")"
NVCC=$HOME/lfm-gtx1060-bench/cuda-12.6/bin/nvcc
CUOBJ=$HOME/lfm-gtx1060-bench/cuda-12.6/bin/cuobjdump
GXX=$HOME/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++
$NVCC -O3 -std=c++17 --use_fast_math -ccbin "$GXX" -gencode arch=compute_61,code=sm_61 \
  -DCH2_USE_I2F ch2_perk.cu -o ch2_perk_i2f -lcudart
python3 patch_sass.py ./ch2_perk "$CUOBJ"
