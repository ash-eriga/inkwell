# Building Inkwell's ggml dependency

Inkwell v0.1 links against a small patched `ggml` build (for CUDA graphs on Pascal and auxiliary ops) and requires CUDA Toolkit 12.x (`12.6` recommended for the optional Pascal `sm_61` SASS NOP-stripping pass).

## 1. Build patched ggml

Base upstream commit: `https://github.com/ggml-org/llama.cpp.git` @ `cbb1c1c50bba81929387dea473873d1df87a6212`.

```sh
git clone https://github.com/ggml-org/llama.cpp.git third_party/ggml/ggml-src
cd third_party/ggml/ggml-src
git checkout cbb1c1c50bba81929387dea473873d1df87a6212
git am ../patches/000[1-8]-*.patch
git apply ../patches/0009-uncommitted-ggml-cuda.patch

# Set CMAKE_CUDA_ARCHITECTURES to your GPU(s), e.g. "61;75;80;86;89;90"
cmake -B build-inkwell-dp4a -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="61;75;80;86;89;90" \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build-inkwell-dp4a -j --target ggml ggml-base ggml-cuda ggml-cpu
```

When placed at `third_party/ggml/ggml-src`, `cargo build --release` discovers it automatically.

## 2. Optional: CuAssembler (Pascal `sm_61` +18% decode boost)

Without CuAssembler, Inkwell builds and runs cleanly using the standard `ptxas` EXL3 kernels (~30 tok/s on GTX 1060). To enable the fail-closed Pascal `sm_61` SASS NOP-stripping pass (~36.9 tok/s):

- Clone `https://github.com/cloudcores/CuAssembler` at `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49` and apply `opt/kbench/ch2/cuasm-96a9f72-local.patch`.
- Install Python deps in a venv: `sympy==1.14.0 pyelftools==0.32 mpmath==1.3.0`.
- Point `INKWELL_CUASM_ROOT` / `INKWELL_CUASM_PYTHON` (or place them under `~/.cache/inkwell-sass-reasm/{CuAssembler,cuasm-venv/bin/python}`).

## 3. Environment variables (all optional if using default paths)

| Variable | Meaning |
|---|---|
| `INKWELL_CUDA_HOME` | CUDA Toolkit root containing `bin/nvcc` (falls back to `CUDA_HOME`, `/usr/local/cuda`, `/opt/cuda`) |
| `INKWELL_CXX` | Host C++17 compiler for `nvcc -ccbin` and the ggml shim (falls back to `CXX`, `g++`) |
| `INKWELL_GGML_SRC` | Patched `ggml` source root containing `ggml/include` (falls back to `third_party/ggml/ggml-src`) |
| `INKWELL_GGML_LIB` | Directory containing `libggml*.so` (falls back to `<INKWELL_GGML_SRC>/build-inkwell-dp4a/bin`) |
| `INKWELL_CUDA_ARCHS` | Comma-separated SM targets for Inkwell kernels (default: `61,75,80,86,89,90`) |
