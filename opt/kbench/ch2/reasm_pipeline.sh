#!/bin/sh
# Regenerate the challenge-2 reassembly probe cubins.  Not production integration.
# Usage: opt/kbench/ch2/reasm_pipeline.sh [output-dir]
# Default output-dir: ~/.cache/inkwell-sass-reasm/out
set -eu

CH2=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$CH2/../../.." && pwd)
OUT=${1:-$HOME/.cache/inkwell-sass-reasm/out}
CUDA=${CUDA_HOME:-$HOME/lfm-gtx1060-bench/cuda-12.6}
GXX=${GXX:-$HOME/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++}
CUASMROOT=${CUASMROOT:-$HOME/.cache/inkwell-sass-reasm/CuAssembler}
VENVPY=${VENVPY:-$HOME/.cache/inkwell-sass-reasm/cuasm-venv/bin/python}
NVCC=$CUDA/bin/nvcc
CUOBJ=$CUDA/bin/cuobjdump

mkdir -p "$OUT/extract-unpatched"
cd "$OUT"

# 1. Compile kernels.cu without patch_sass.py; extract the embedded sm_61 cubin.
"$NVCC" -O3 -std=c++17 --use_fast_math -ccbin "$GXX" \
  -gencode arch=compute_61,code=sm_61 \
  -c "$ROOT/crates/inkwell-cuda/src/kernels.cu" -o kernels_unpatched.o
rm -f extract-unpatched/kernels_unpatched.1.sm_61.cubin
(cd extract-unpatched && "$CUOBJ" -xelf all ../kernels_unpatched.o >/dev/null)
cp extract-unpatched/kernels_unpatched.1.sm_61.cubin kernels_unpatched.sm_61.cubin

# 2. Dump baseline SASS and augment the sm_61 assembler repository from it.
"$CUOBJ" -sass kernels_unpatched.sm_61.cubin > kernels_unpatched.sass
PYTHONPATH="$CUASMROOT" "$VENVPY" "$CH2/reasm_augment_repos.py"

# 3. Copy the cubin, patch it in place, dump/check SASS, and disassemble to cuasm.
cp kernels_unpatched.sm_61.cubin kernels_patched_nops.sm_61.cubin
python3 "$CH2/patch_sass.py" kernels_patched_nops.sm_61.cubin "$CUOBJ"
"$CUOBJ" -sass kernels_patched_nops.sm_61.cubin > kernels_patched_nops.sass
python3 "$CH2/count_loops.py" kernels_patched_nops.sass > count_patched_nops.txt
python3 "$CH2/reasm_loop_ranges.py" kernels_patched_nops.sass loop_ranges.json > loop_ranges.txt
PATH="$CUDA/bin:$PATH" PYTHONPATH="$CUASMROOT" \
  "$VENVPY" "$CUASMROOT/bin/cuasm.py" kernels_patched_nops.sm_61.cubin \
  -o kernels_patched_nops.cuasm

# 4. Delete the 32 retired-XMAD NOPs in each main K=1.5..4 loop.
python3 "$CH2/reasm_remove_loop_nops.py" kernels_patched_nops.cuasm kernels_nopless.cuasm loop_ranges.json

# 5. Assemble with the augmented repository and count the timed cubin's loops.
PYTHONPATH="$CUASMROOT" "$VENVPY" \
  "$CH2/reasm_assemble_with_repos.py" kernels_nopless.cuasm \
  Repos.sm_61.production.txt kernels_patched_nopless.sm_61.cubin
"$CUOBJ" -sass kernels_patched_nopless.sm_61.cubin > kernels_patched_nopless.sass
python3 "$CH2/count_loops.py" kernels_patched_nopless.sass > count_patched_nopless.txt

# 6. Build the driver-API validator/benchmark.  Run it only under the GPU lock:
# nice -n 19 flock /tmp/inkwell-gpu.lock "$OUT/reasm_drv_perk" \
#   "$OUT/kernels_unpatched.sm_61.cubin" "$OUT/kernels_patched_nops.sm_61.cubin" \
#   "$OUT/kernels_patched_nopless.sm_61.cubin"
"$NVCC" -O3 -std=c++17 -ccbin "$GXX" \
  "$CH2/reasm_drv_perk.cc" -o reasm_drv_perk -lcuda -lcudart_static -ldl -lrt -lpthread

sha256sum kernels_unpatched.sm_61.cubin kernels_patched_nops.sm_61.cubin kernels_patched_nopless.sm_61.cubin
cat count_patched_nops.txt count_patched_nopless.txt
