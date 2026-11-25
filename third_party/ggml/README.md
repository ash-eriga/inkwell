# Build dependencies outside this repo

inkwell links a patched ggml/llama.cpp build and needs CUDA 12.6 (sm_61). On the
owner's GPU host these live under `~/lfm-gtx1060-bench`. Other hosts (Devbox,
cloud agents) rebuild them from this recipe and point the build at them with env vars.

## 1. ggml fork

- Upstream: `https://github.com/ggml-org/llama.cpp.git`, base commit
  `cbb1c1c50bba81929387dea473873d1df87a6212`.
- Patches: `patches/0001…0008` are `git format-patch` output (`git am`).
  `0009-uncommitted-ggml-cuda.patch` is a plain diff (`git apply`).

```sh
git clone https://github.com/ggml-org/llama.cpp.git ggml-src
cd ggml-src && git checkout cbb1c1c50bba81929387dea473873d1df87a6212
git am /path/to/inkwell/third_party/ggml/patches/000[1-8]-*.patch
git apply /path/to/inkwell/third_party/ggml/patches/0009-uncommitted-ggml-cuda.patch
cmake -B build-inkwell-dp4a -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build-inkwell-dp4a -j --target ggml ggml-base ggml-cuda ggml-cpu
```

## 2. CUDA 12.6

nvcc 12.6 is required (sm_61 SASS patterns in `opt/kbench/ch2` are pinned to it).
No GPU is needed to compile.

## 3. CuAssembler (optional; without it the build falls back to the plain ptxas kernel)

- `https://github.com/cloudcores/CuAssembler` at `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49`
  plus `opt/kbench/ch2/cuasm-96a9f72-local.patch`.
- Python venv with `sympy==1.14.0 pyelftools==0.32 mpmath==1.3.0`.
- Discovery: `INKWELL_CUASM_ROOT` / `INKWELL_CUASM_PYTHON`, or
  `~/.cache/inkwell-sass-reasm/{CuAssembler,cuasm-venv/bin/python}`.

## 4. Env vars read by build.rs (defaults = the GPU host's paths)

| var | meaning |
|---|---|
| `INKWELL_CUDA_HOME` | CUDA 12.6 root (`bin/nvcc`) |
| `INKWELL_CXX` | host C++ compiler for nvcc `-ccbin` and the ggml shim (GCC 13 used on the GPU host) |
| `INKWELL_GGML_SRC` | ggml/llama.cpp source root (contains `ggml/include`) |
| `INKWELL_GGML_LIB` | dir with `libggml*.so` |

## 5. GPU work stays on the GPU host

Hosts without the GTX 1060 compile, inspect SASS, and run CPU harnesses only.
Every GPU job (bitwise gates, md5, A/B) runs on the GPU host under
`nice -n 19 flock /tmp/inkwell-gpu.lock`, queued by the orchestrator.
