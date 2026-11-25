NVCC=/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nvcc
CXXBIN=/home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++
FLAGS=-O3 -std=c++17 --use_fast_math -ccbin $(CXXBIN) -gencode arch=compute_61,code=sm_61
prefill_tests: prefill_tests.cu decode_attn_ref.cu ../../../crates/inkwell-cuda/src/prefill.cu
	$(NVCC) $(FLAGS) prefill_tests.cu decode_attn_ref.cu -o $@
test: prefill_tests
	gpu-run -p normal -t kbench-prefill -- ./prefill_tests all
gemm: prefill_tests
	gpu-run -p normal -t kbench-prefill -- ./prefill_tests gemm
moe: prefill_tests
	gpu-run -p normal -t kbench-prefill -- ./prefill_tests moe
conv: prefill_tests
	gpu-run -p normal -t kbench-prefill -- ./prefill_tests conv
flash: prefill_tests
	gpu-run -p normal -t kbench-prefill -- ./prefill_tests flash
