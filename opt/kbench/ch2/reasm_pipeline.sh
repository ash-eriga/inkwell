#!/bin/sh
# Probe pipeline used on t3/sass-reasm-probe.  Not production integration.
# Requires: CUDA 12.6 at ~/lfm-gtx1060-bench/cuda-12.6, CuAssembler pinned at
# 96a9f72baf00f40b9b299653fcef8d3e2b4a3d49 plus cuasm-96a9f72-local.patch,
# and a venv with sympy==1.14.0 pyelftools==0.32 (mpmath==1.3.0).
set -eu
# 1. Compile kernels.cu without patch_sass.py; extract the embedded sm_61 cubin.
# 2. cuobjdump -sass cubin > baseline.sass; augment the sm_61 repository from it.
# 3. Copy cubin, run patch_sass.py on the copy, disassemble it with CuAssembler.
# 4. Run reasm_remove_loop_nops.py patched.cuasm nopsremoved.cuasm.
# 5. Assemble with reasm_assemble_with_repos.py using the augmented repository.
# 6. Validate/benchmark with reasm_drv_perk under: nice -n 19 flock /tmp/inkwell-gpu.lock
